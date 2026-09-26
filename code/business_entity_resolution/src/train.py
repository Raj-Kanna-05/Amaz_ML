"""
Training Pipeline for Business Entity Resolution.
Executes candidate generation on training data, builds hard-negative pairs,
extracts pairwise features, performs GroupKFold cross-validation grouped by S1,
tunes the macro F0.5 threshold, and trains the final production model.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from .blocking import CandidateBlocker
    from .evaluation import compute_candidate_recall, evaluate_predictions
    from .features import FeatureExtractor
    from .models import EntityMatchingModel
    from .postprocess import generate_final_matches
    from .preprocessing import preprocess_dataframe
    from .utils import (
        get_logger,
        load_ground_truth,
        load_tsv,
        sample_representative_dataset,
        timer,
    )
except (ImportError, ValueError):
    from src.blocking import CandidateBlocker
    from src.evaluation import compute_candidate_recall, evaluate_predictions
    from src.features import FeatureExtractor
    from src.models import EntityMatchingModel
    from src.postprocess import generate_final_matches
    from src.preprocessing import preprocess_dataframe
    from src.utils import (
        get_logger,
        load_ground_truth,
        load_tsv,
        sample_representative_dataset,
        timer,
    )

logger = get_logger("Train")


def split_group_kfold(groups: np.ndarray, n_splits: int = 5, seed: int = 42):
    """Pure NumPy implementation of GroupKFold cross-validation splitter."""
    unique_groups = list(dict.fromkeys(groups))
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits <= 1:
        yield np.arange(len(groups)), np.arange(len(groups))
        return

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    group_to_fold = {g: i % actual_splits for i, g in enumerate(unique_groups)}
    for fold in range(actual_splits):
        val_idx = np.array([i for i, g in enumerate(groups) if group_to_fold[g] == fold], dtype=int)
        train_idx = np.array([i for i, g in enumerate(groups) if group_to_fold[g] != fold], dtype=int)
        if len(train_idx) == 0:
            train_idx = val_idx
        yield train_idx, val_idx


def prepare_training_pairs(
    s1_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    ground_truth_map: Dict[str, List[str]],
    negative_ratio: float = 10.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Constructs labeled dataset of (S1, Candidate) pairs.
    Positives: pairs in ground_truth_map.
    Negatives: candidate pairs not in ground_truth_map (hard negatives).
    Subsamples negatives to negative_ratio : 1 if necessary.
    Accelerated with vectorized column iteration.
    """
    logger.info("Constructing labeled training pairs from candidates and ground truth...")

    gt_pair_set = set()
    for s1, matched_list in ground_truth_map.items():
        for m in matched_list:
            gt_pair_set.add((s1, m))

    cand_records = []
    seen_cand_pairs = set()

    if len(candidate_pairs_df) > 0:
        s1_col = candidate_pairs_df["source1_entity_id"].astype(str).tolist()
        cand_col = candidate_pairs_df["candidate_entity_id"].astype(str).tolist()
        tgt_col = candidate_pairs_df.get("target_source", pd.Series(["S2"] * len(candidate_pairs_df))).astype(str).tolist()
        score_col = candidate_pairs_df.get("blocking_score", pd.Series([0.0] * len(candidate_pairs_df))).astype(float).tolist()
        rank_col = candidate_pairs_df.get("blocking_rank", pd.Series([1] * len(candidate_pairs_df))).astype(int).tolist()

        for s1, cand, tgt, score, rank in zip(s1_col, cand_col, tgt_col, score_col, rank_col):
            pair_key = (s1, cand)
            seen_cand_pairs.add(pair_key)
            is_pos = 1 if pair_key in gt_pair_set else 0
            cand_records.append({
                "source1_entity_id": s1,
                "candidate_entity_id": cand,
                "target_source": tgt if tgt else ("S2" if cand.startswith("S2-") else "S3"),
                "blocking_score": score,
                "blocking_rank": rank,
                "retrieved_by_blocking": 1,
                "label": is_pos,
            })

    # Ensure all true positives are included in training pairs (patching missing positives for classifier fit only)
    missed_pos = 0
    for s1, m in gt_pair_set:
        if (s1, m) not in seen_cand_pairs:
            missed_pos += 1
            cand_records.append({
                "source1_entity_id": s1,
                "candidate_entity_id": m,
                "target_source": "S2" if m.startswith("S2-") else "S3",
                "blocking_score": 0.0,
                "blocking_rank": 999,
                "retrieved_by_blocking": 0,
                "label": 1,
            })

    if missed_pos > 0:
        logger.warning(
            f"Added {missed_pos} ground-truth pairs missed during blocking to train set "
            f"(flagged as retrieved_by_blocking=0 for realistic evaluation)."
        )

    if not cand_records:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "target_source", "blocking_score", "blocking_rank", "retrieved_by_blocking", "label"]), np.array([])

    full_pairs_df = pd.DataFrame(cand_records)

    pos_df = full_pairs_df[full_pairs_df["label"] == 1]
    neg_df = full_pairs_df[full_pairs_df["label"] == 0]

    logger.info(f"Total True Positives: {len(pos_df)}, Available Hard Negatives: {len(neg_df)}")

    # Subsample negatives if requested
    max_negs = int(len(pos_df) * negative_ratio) if len(pos_df) > 0 else len(neg_df)
    if len(neg_df) > max_negs and max_negs > 0:
        neg_df_sorted = neg_df.sort_values(by="blocking_score", ascending=False)
        hard_top = neg_df_sorted.iloc[: max_negs // 2]
        remaining_neg = neg_df_sorted.iloc[max_negs // 2 :]
        random_neg = remaining_neg.sample(
            n=min(len(remaining_neg), max_negs - len(hard_top)),
            random_state=random_state,
        )
        neg_df = pd.concat([hard_top, random_neg])

    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    labels = np.asarray(balanced_df["label"].values, dtype=int)

    logger.info(f"Final training dataset size: {len(balanced_df)} pairs ({len(pos_df)} pos, {len(neg_df)} neg).")
    return balanced_df, labels


def optimize_threshold(
    oof_df: pd.DataFrame,
    ground_truth_map: Dict[str, List[str]],
    all_s1_ids: List[str],
    enforce_one_to_one: bool = True,
) -> Tuple[float, float, float, Dict, Dict]:
    """
    Sweeps probability thresholds to maximize REALISTIC macro F0.5
    (evaluated strictly on candidate pairs actually retrieved by blocking).
    Simultaneously tracks and reports ORACLE macro F0.5 (with injected ground truth).
    """
    logger.info("Sweeping thresholds to optimize realistic macro F0.5 score...")

    # Realistic evaluation: candidates that were actually retrieved by blocking
    if "retrieved_by_blocking" in oof_df.columns:
        realistic_mask = oof_df["retrieved_by_blocking"] == 1
        realistic_oof = oof_df.loc[realistic_mask].copy()
    else:
        realistic_oof = oof_df.copy()

    best_thresh = 0.50
    best_realistic_f05 = -1.0
    best_realistic_metrics: Dict = {}
    best_oracle_f05 = -1.0
    best_oracle_metrics: Dict = {}

    thresholds = np.arange(0.20, 0.85, 0.02)
    for thresh in thresholds:
        thresh = round(float(thresh), 3)

        # 1. Realistic predictions (only candidates blocking retrieved)
        r_pred_map = generate_final_matches(
            scored_pairs_df=realistic_oof,
            all_s1_ids=all_s1_ids,
            threshold=thresh,
            enforce_one_to_one=enforce_one_to_one,
        )
        r_metrics = evaluate_predictions(ground_truth_map, r_pred_map)
        r_f05 = r_metrics["macro_f05"]

        # 2. Oracle predictions (includes injected ground-truth pairs)
        o_pred_map = generate_final_matches(
            scored_pairs_df=oof_df,
            all_s1_ids=all_s1_ids,
            threshold=thresh,
            enforce_one_to_one=enforce_one_to_one,
        )
        o_metrics = evaluate_predictions(ground_truth_map, o_pred_map)
        o_f05 = o_metrics["macro_f05"]

        if r_f05 > best_realistic_f05:
            best_realistic_f05 = r_f05
            best_thresh = thresh
            best_realistic_metrics = r_metrics
            best_oracle_f05 = o_f05
            best_oracle_metrics = o_metrics

    logger.info(
        f"Optimal threshold: {best_thresh:.3f} | "
        f"Realistic Macro F0.5: {best_realistic_f05:.4f} "
        f"(Precision: {best_realistic_metrics.get('macro_precision', 0):.4f}, "
        f"Recall: {best_realistic_metrics.get('macro_recall', 0):.4f}, "
        f"Singleton Acc: {best_realistic_metrics.get('singleton_accuracy', 0):.4f}) | "
        f"Oracle Macro F0.5: {best_oracle_f05:.4f} "
        f"(Precision: {best_oracle_metrics.get('macro_precision', 0):.4f}, "
        f"Recall: {best_oracle_metrics.get('macro_recall', 0):.4f})"
    )
    return best_thresh, best_realistic_f05, best_oracle_f05, best_realistic_metrics, best_oracle_metrics


def train_pipeline(
    train_dir: str,
    output_dir: str,
    model_type: str = "lightgbm",
    top_k_candidates: int = 80,
    n_splits: int = 5,
    enforce_one_to_one: bool = True,
    sample_size: Optional[int] = None,
) -> Tuple[EntityMatchingModel, float, Dict]:
    """
    Executes full training pipeline on the training directory.
    Uses representative sampling preserving exact singleton rate if sample_size is passed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    if sample_size and sample_size > 0:
        logger.info(f"Subsampling training data to representative sample of {sample_size} S1 entities...")
        s1_raw, s2_raw, s3_raw, gt_map = sample_representative_dataset(
            train_dir=train_dir,
            sample_size=sample_size,
            random_state=42,
        )
    else:
        with timer("Loading full training datasets", logger):
            s1_raw = load_tsv(os.path.join(train_dir, "train_source1.tsv"), ["entity_id", "business_name", "business_address", "country"])
            s2_raw = load_tsv(os.path.join(train_dir, "train_source2.tsv"), ["entity_id", "business_name", "business_address", "country"])
            s3_raw = load_tsv(os.path.join(train_dir, "train_source3.tsv"), ["entity_id", "business_name", "business_address", "country"])
            gt_map = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    # 2. Preprocess dataframes
    with timer("Preprocessing datasets", logger):
        s1_df = preprocess_dataframe(s1_raw)
        s2_df = preprocess_dataframe(s2_raw)
        s3_df = preprocess_dataframe(s3_raw)

    # 3. Candidate Generation, Cross-Validation, & Full Feature Engineering
    import gc
    extractor = FeatureExtractor()

    if sample_size and sample_size > 0:
        # Standard representative sample execution (fast prototyping)
        with timer("Candidate generation (Blocking)", logger):
            blocker = CandidateBlocker(top_k=top_k_candidates)
            cand_pairs_df = blocker.block(s1_df, s2_df, s3_df)
            cand_map = blocker.candidate_df_to_mapping(cand_pairs_df, list(s1_df["entity_id"]))
            recall_stats = compute_candidate_recall(gt_map, cand_map)
            logger.info(f"Training Blocking Recall: {recall_stats['candidate_recall']:.4f} ({recall_stats['retrieved_true_pairs']}/{recall_stats['total_true_pairs']} true pairs)")

        train_pairs_df, labels = prepare_training_pairs(
            s1_df=s1_df,
            candidate_pairs_df=cand_pairs_df,
            ground_truth_map=gt_map,
        )

        if len(train_pairs_df) == 0:
            raise ValueError("No training pairs available to train model.")

        with timer("Pairwise feature extraction", logger):
            X = extractor.extract_features(train_pairs_df, s1_df, s2_df, s3_df)
            feature_names = list(X.columns)

        groups = np.asarray(train_pairs_df["source1_entity_id"].values)
        logger.info(f"Starting {n_splits}-fold GroupKFold CV (grouped by source1_entity_id, {len(set(groups))} groups)...")
        oof_preds = np.zeros(len(train_pairs_df))

        for fold, (train_idx, val_idx) in enumerate(split_group_kfold(groups, n_splits=n_splits), start=1):
            train_indices: List[int] = [int(i) for i in train_idx]
            val_indices: List[int] = [int(i) for i in val_idx]
            X_train, y_train = X.iloc[train_indices], labels[train_indices]
            X_val = X.iloc[val_indices]

            fold_model = EntityMatchingModel(model_type=model_type)
            fold_model.fit(X_train, y_train, feature_names=feature_names)
            oof_preds[val_indices] = fold_model.predict_proba(X_val)
            logger.info(f"Completed Fold {fold}/{n_splits}.")

        oof_df = train_pairs_df[["source1_entity_id", "candidate_entity_id", "retrieved_by_blocking"]].copy()
        oof_df["probability"] = list(oof_preds)

        all_s1_ids = list(s1_df["entity_id"])
        best_thresh, best_r_f05, best_o_f05, r_metrics, o_metrics = optimize_threshold(
            oof_df=oof_df,
            ground_truth_map=gt_map,
            all_s1_ids=all_s1_ids,
            enforce_one_to_one=enforce_one_to_one,
        )
    else:
        # Full-scale training across all 2.2M Source 1 entities with memory-safe chunking
        logger.info(f"Running FULL-SCALE training across all {len(s1_df):,} S1 entities...")

        # Step 1: Run CV & threshold optimization on a 50,000 representative entity split
        logger.info("Step 1: Running GroupKFold CV on representative 50,000-entity split for threshold tuning...")
        cv_s1_raw, cv_s2_raw, cv_s3_raw, cv_gt_map = sample_representative_dataset(
            train_dir=train_dir,
            sample_size=50000,
            random_state=42,
        )
        cv_s1_df = preprocess_dataframe(cv_s1_raw)
        cv_s2_df = preprocess_dataframe(cv_s2_raw)
        cv_s3_df = preprocess_dataframe(cv_s3_raw)

        cv_blocker = CandidateBlocker(top_k=top_k_candidates)
        cv_cands = cv_blocker.block(cv_s1_df, cv_s2_df, cv_s3_df)
        cv_pairs, cv_labels = prepare_training_pairs(cv_s1_df, cv_cands, cv_gt_map)
        X_cv = extractor.extract_features(cv_pairs, cv_s1_df, cv_s2_df, cv_s3_df)
        feature_names = list(X_cv.columns)

        groups = np.asarray(cv_pairs["source1_entity_id"].values)
        oof_preds = np.zeros(len(cv_pairs))
        for fold, (train_idx, val_idx) in enumerate(split_group_kfold(groups, n_splits=n_splits), start=1):
            train_indices = [int(i) for i in train_idx]
            val_indices = [int(i) for i in val_idx]
            fold_m = EntityMatchingModel(model_type=model_type)
            fold_m.fit(X_cv.iloc[train_indices], cv_labels[train_indices], feature_names=feature_names)
            oof_preds[val_indices] = fold_m.predict_proba(X_cv.iloc[val_indices])

        cv_oof_df = cv_pairs[["source1_entity_id", "candidate_entity_id", "retrieved_by_blocking"]].copy()
        cv_oof_df["probability"] = list(oof_preds)
        best_thresh, best_r_f05, best_o_f05, r_metrics, o_metrics = optimize_threshold(
            oof_df=cv_oof_df,
            ground_truth_map=cv_gt_map,
            all_s1_ids=list(cv_s1_df["entity_id"]),
            enforce_one_to_one=enforce_one_to_one,
        )
        del cv_pairs, cv_labels, X_cv, cv_s1_df, cv_s2_df, cv_s3_df, cv_blocker, cv_cands
        gc.collect()

        # Step 2: Chunked full-scale feature extraction across all 2.2M entities
        logger.info(f"Step 2: Processing all {len(s1_df):,} S1 entities in chunks of 50,000 for full model fitting...")
        full_blocker = CandidateBlocker(top_k=top_k_candidates)
        with timer("Pre-indexing target sources (S2 and S3) once for training chunks", logger):
            full_blocker.index_targets(s2_df, s3_df)
        chunk_size = 50000
        n_chunks = (len(s1_df) + chunk_size - 1) // chunk_size
        all_X_list = []
        all_y_list = []

        for c_idx in range(n_chunks):
            start_i = c_idx * chunk_size
            end_i = min(start_i + chunk_size, len(s1_df))
            s1_chunk = s1_df.iloc[start_i:end_i].copy()
            logger.info(f"Training chunk {c_idx + 1}/{n_chunks}: S1 entities {start_i:,} to {end_i:,}...")

            c_cands = full_blocker.block(s1_chunk, s2_df, s3_df)
            c_pairs, c_labels = prepare_training_pairs(s1_chunk, c_cands, gt_map, negative_ratio=2.0)
            if len(c_pairs) > 0:
                c_feats = extractor.extract_features(c_pairs, s1_chunk, s2_df, s3_df).astype(np.float32)
                all_X_list.append(c_feats)
                all_y_list.append(c_labels)

            del s1_chunk, c_cands, c_pairs, c_labels
            gc.collect()

        X = pd.concat(all_X_list, ignore_index=True)
        labels = np.concatenate(all_y_list)
        recall_stats = {"note": "Trained across full 2,206,821 S1 entities"}
        del all_X_list, all_y_list
        gc.collect()

    # 8. Train final production model on all data
    with timer("Fitting final production model", logger):
        final_model = EntityMatchingModel(model_type=model_type)
        final_model.fit(X, labels, feature_names=feature_names)

    # Save model and artifacts
    model_save_path = os.path.join(output_dir, "entity_matching_model.joblib")
    final_model.save(model_save_path)
    logger.info(f"Saved trained model to {model_save_path}")

    metadata = {
        "model_type": model_type,
        "best_threshold": best_thresh,
        "best_cv_realistic_macro_f05": best_r_f05,
        "best_cv_oracle_macro_f05": best_o_f05,
        "cv_realistic_metrics": r_metrics,
        "cv_oracle_metrics": o_metrics,
        "blocking_recall": recall_stats,
        "feature_names": feature_names,
        "top_features": list(final_model.get_feature_importances().items())[:15],
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {meta_path}")

    return final_model, best_thresh, metadata
