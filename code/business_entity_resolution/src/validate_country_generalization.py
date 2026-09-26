"""
Leave-One-Country-Out Cross-Validation Benchmark.
Evaluates model generalization across unseen countries (US <-> India).
Validates open-set transferability required for unseen France test entities.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

try:
    from .blocking import CandidateBlocker
    from .evaluation import evaluate_predictions
    from .features import FeatureExtractor
    from .models import EntityMatchingModel
    from .postprocess import generate_final_matches
    from .preprocessing import preprocess_dataframe
    from .train import prepare_training_pairs
    from .utils import get_logger, sample_representative_dataset, timer
except (ImportError, ValueError):
    from src.blocking import CandidateBlocker
    from src.evaluation import evaluate_predictions
    from src.features import FeatureExtractor
    from src.models import EntityMatchingModel
    from src.postprocess import generate_final_matches
    from src.preprocessing import preprocess_dataframe
    from src.train import prepare_training_pairs
    from src.utils import get_logger, sample_representative_dataset, timer

logger = get_logger("CountryValidation")


def run_country_validation():
    train_dir = "student_resource/dataset/train"
    logger.info("Sampling 2,000 entities for cross-country validation...")
    s1_raw, s2_raw, s3_raw, gt_map = sample_representative_dataset(
        train_dir, sample_size=2000, random_state=42
    )

    s1_df = preprocess_dataframe(s1_raw)
    s2_df = preprocess_dataframe(s2_raw)
    s3_df = preprocess_dataframe(s3_raw)

    blocker = CandidateBlocker(top_k=80)
    cand_pairs_df = blocker.block(s1_df, s2_df, s3_df)

    pairs_df, labels = prepare_training_pairs(
        s1_df=s1_df,
        candidate_pairs_df=cand_pairs_df,
        ground_truth_map=gt_map,
    )

    extractor = FeatureExtractor()
    X = extractor.extract_features(pairs_df, s1_df, s2_df, s3_df)

    s1_country_map = dict(zip(s1_df["entity_id"], s1_df["norm_country"]))
    pairs_df["s1_country"] = pairs_df["source1_entity_id"].map(s1_country_map)

    # Countries present in train data: "us" / "in" as pure integer index lists (no np.where overload issues)
    us_indices: List[int] = [i for i, c in enumerate(pairs_df["s1_country"]) if c == "us"]
    in_indices: List[int] = [i for i, c in enumerate(pairs_df["s1_country"]) if c == "in"]

    logger.info(f"Pairs breakdown: US={len(us_indices)}, India={len(in_indices)}")

    all_s1_us = [eid for eid, c in s1_country_map.items() if c == "us"]
    all_s1_in = [eid for eid, c in s1_country_map.items() if c == "in"]

    gt_us = {k: v for k, v in gt_map.items() if k in all_s1_us}
    gt_in = {k: v for k, v in gt_map.items() if k in all_s1_in}

    results = {}

    # --- Experiment 1: Train on India, Test on Held-out US ---
    if len(in_indices) > 0 and len(us_indices) > 0:
        logger.info("\n--- Experiment 1: Train on India, Test on Held-out US ---")
        X_train_in, y_train_in = X.iloc[in_indices], labels[in_indices]
        X_test_us = X.iloc[us_indices]

        model_in = EntityMatchingModel(model_type="lightgbm")
        model_in.fit(X_train_in, y_train_in, feature_names=list(X.columns))
        us_preds = model_in.predict_proba(X_test_us)

        us_oof = pairs_df.iloc[us_indices][["source1_entity_id", "candidate_entity_id", "retrieved_by_blocking"]].copy()
        us_oof["probability"] = list(us_preds)
        realistic_us = us_oof.loc[us_oof["retrieved_by_blocking"] == 1]

        us_pred_map = generate_final_matches(
            scored_pairs_df=realistic_us,
            all_s1_ids=all_s1_us,
            threshold=0.60,
            enforce_one_to_one=True,
        )
        res_exp1 = evaluate_predictions(gt_us, us_pred_map)
        logger.info(f"India -> US Zero-Shot Macro F0.5: {res_exp1['macro_f05']:.4f} (Precision: {res_exp1['macro_precision']:.4f}, Recall: {res_exp1['macro_recall']:.4f})")
        results["India_to_US"] = res_exp1

    # --- Experiment 2: Train on US, Test on Held-out India ---
    if len(us_indices) > 0 and len(in_indices) > 0:
        logger.info("\n--- Experiment 2: Train on US, Test on Held-out India ---")
        X_train_us, y_train_us = X.iloc[us_indices], labels[us_indices]
        X_test_in = X.iloc[in_indices]

        model_us = EntityMatchingModel(model_type="lightgbm")
        model_us.fit(X_train_us, y_train_us, feature_names=list(X.columns))
        in_preds = model_us.predict_proba(X_test_in)

        in_oof = pairs_df.iloc[in_indices][["source1_entity_id", "candidate_entity_id", "retrieved_by_blocking"]].copy()
        in_oof["probability"] = list(in_preds)
        realistic_in = in_oof.loc[in_oof["retrieved_by_blocking"] == 1]

        in_pred_map = generate_final_matches(
            scored_pairs_df=realistic_in,
            all_s1_ids=all_s1_in,
            threshold=0.60,
            enforce_one_to_one=True,
        )
        res_exp2 = evaluate_predictions(gt_in, in_pred_map)
        logger.info(f"US -> India Zero-Shot Macro F0.5: {res_exp2['macro_f05']:.4f} (Precision: {res_exp2['macro_precision']:.4f}, Recall: {res_exp2['macro_recall']:.4f})")
        results["US_to_India"] = res_exp2

    return results


if __name__ == "__main__":
    run_country_validation()
