"""
Inference and Prediction Pipeline for Business Entity Resolution.
Generates candidate pairs, extracts pairwise features, runs model scoring,
applies one-to-one post-processing, and exports both required submission TSVs.
"""

import json
import os
from typing import Dict, List, Optional
import pandas as pd

try:
    from .blocking import CandidateBlocker
    from .features import FeatureExtractor
    from .models import EntityMatchingModel
    from .postprocess import generate_final_matches
    from .preprocessing import preprocess_dataframe
    from .utils import (
        get_logger,
        load_tsv,
        save_submission_candidates,
        save_submission_matching,
        timer,
    )
except (ImportError, ValueError):
    from src.blocking import CandidateBlocker
    from src.features import FeatureExtractor
    from src.models import EntityMatchingModel
    from src.postprocess import generate_final_matches
    from src.preprocessing import preprocess_dataframe
    from src.utils import (
        get_logger,
        load_tsv,
        save_submission_candidates,
        save_submission_matching,
        timer,
    )

logger = get_logger("Predict")


def predict_pipeline(
    test_dir: str,
    model_path: str,
    metadata_path: str,
    output_dir: str,
    top_k_candidates: int = 80,
    threshold: Optional[float] = None,
    enforce_one_to_one: bool = True,
    chunk_size: int = 50000,
) -> Dict[str, str]:
    """
    Executes end-to-end inference on the test dataset and writes:
    - output/candidate_pairs.tsv
    - output/matching_results.tsv
    Supports chunked processing for large datasets to keep memory usage under 4 GB.
    """
    import gc
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load trained model & metadata
    logger.info(f"Loading trained model from {model_path}")
    model = EntityMatchingModel.load(model_path)

    if threshold is None:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                threshold = meta.get("best_threshold", 0.50)
                logger.info(f"Loaded optimal threshold {threshold:.3f} from metadata.")
        else:
            threshold = 0.50
            logger.warning("Metadata not found, falling back to default threshold 0.50.")

    th: float = float(threshold) if threshold is not None else 0.50

    # 2. Load test source files
    with timer("Loading test datasets", logger):
        s1_raw = load_tsv(
            os.path.join(test_dir, "test_source1.tsv"),
            ["entity_id", "business_name", "business_address", "country"],
        )
        s2_raw = load_tsv(
            os.path.join(test_dir, "test_source2.tsv"),
            ["entity_id", "business_name", "business_address", "country"],
        )
        s3_raw = load_tsv(
            os.path.join(test_dir, "test_source3.tsv"),
            ["entity_id", "business_name", "business_address", "country"],
        )

    all_s1_ordered_ids = list(s1_raw["entity_id"])

    # 3. Preprocess test files
    with timer("Preprocessing test datasets", logger):
        s1_df = preprocess_dataframe(s1_raw)
        s2_df = preprocess_dataframe(s2_raw)
        s3_df = preprocess_dataframe(s3_raw)

    total_s1 = len(s1_df)
    blocker = CandidateBlocker(top_k=top_k_candidates)
    extractor = FeatureExtractor()

    cand_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ordered_ids}
    scored_records: List[Dict] = []
    min_keep_prob = max(0.15, th - 0.25)

    # 4. Pre-index target sources once for high-throughput chunked candidate queries
    with timer("Pre-indexing target sources (S2 and S3) once", logger):
        blocker.index_targets(s2_df, s3_df)

    # 5. Chunked candidate generation, feature extraction, and scoring
    if total_s1 <= chunk_size:
        # Single-pass execution for small/dev test sets
        with timer("Generating candidate pairs (Blocking)", logger):
            cand_pairs_df = blocker.block(s1_df, s2_df, s3_df)
            cand_map = blocker.candidate_df_to_mapping(cand_pairs_df, all_s1_ordered_ids)

        if len(cand_pairs_df) > 0:
            with timer("Extracting pairwise features for test candidates", logger):
                X_test = extractor.extract_features(cand_pairs_df, s1_df, s2_df, s3_df)
                for col in model.feature_names:
                    if col not in X_test.columns:
                        X_test[col] = 0.0
                X_test = X_test[model.feature_names]

            with timer("Scoring candidate pairs", logger):
                probs = model.predict_proba(X_test)
                scored_df = cand_pairs_df[["source1_entity_id", "candidate_entity_id"]].copy()
                scored_df["probability"] = list(probs)
        else:
            scored_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "probability"])
    else:
        # Memory-safe chunked execution for full 1.73M production test set
        n_chunks = (total_s1 + chunk_size - 1) // chunk_size
        logger.info(f"Processing {total_s1:,} S1 entities in {n_chunks} chunks (chunk_size={chunk_size})...")

        for chunk_idx in range(n_chunks):
            start_i = chunk_idx * chunk_size
            end_i = min(start_i + chunk_size, total_s1)
            s1_chunk = s1_df.iloc[start_i:end_i].copy()
            chunk_s1_ids = list(s1_chunk["entity_id"])

            logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: S1 entities {start_i:,} to {end_i:,}...")

            # Block chunk
            chunk_cands_df = blocker.block(s1_chunk, s2_df, s3_df)
            chunk_cand_map = blocker.candidate_df_to_mapping(chunk_cands_df, chunk_s1_ids)
            for k, v in chunk_cand_map.items():
                cand_map[k] = v

            if len(chunk_cands_df) > 0:
                # Extract features for chunk
                X_chunk = extractor.extract_features(chunk_cands_df, s1_chunk, s2_df, s3_df)
                for col in model.feature_names:
                    if col not in X_chunk.columns:
                        X_chunk[col] = 0.0
                X_chunk = X_chunk[model.feature_names]

                # Score chunk
                probs_chunk = model.predict_proba(X_chunk)
                s1_c = chunk_cands_df["source1_entity_id"].tolist()
                t_c = chunk_cands_df["candidate_entity_id"].tolist()

                for s1_id, t_id, prob in zip(s1_c, t_c, probs_chunk):
                    if prob >= min_keep_prob:
                        scored_records.append({
                            "source1_entity_id": s1_id,
                            "candidate_entity_id": t_id,
                            "probability": float(prob),
                        })

                del X_chunk, probs_chunk, chunk_cands_df
            del s1_chunk
            gc.collect()

        scored_df = pd.DataFrame(scored_records) if scored_records else pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "probability"]
        )

    # 5. Export candidate_pairs.tsv
    cand_output_file = os.path.join(output_dir, "candidate_pairs.tsv")
    save_submission_candidates(cand_map, cand_output_file, s1_order=all_s1_ordered_ids)
    logger.info(f"Saved candidate pairs to {cand_output_file}")

    # 6. Post-processing matches
    with timer("Post-processing final matches", logger):
        final_matches_map = generate_final_matches(
            scored_pairs_df=scored_df,
            all_s1_ids=all_s1_ordered_ids,
            threshold=th,
            enforce_one_to_one=enforce_one_to_one,
            candidate_mapping=cand_map,
        )

    # 7. Export matching_results.tsv
    matching_output_file = os.path.join(output_dir, "matching_results.tsv")
    save_submission_matching(final_matches_map, matching_output_file, s1_order=all_s1_ordered_ids)
    logger.info(f"Saved final matches to {matching_output_file}")

    return {
        "matching_file": matching_output_file,
        "candidate_file": cand_output_file,
    }
