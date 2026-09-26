"""
Utility functions for Business Entity Resolution.
Handles robust TSV I/O, submission formatting, logging, and timing.
"""

import contextlib
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple
import numpy as np
import pandas as pd


def get_logger(name: str = "BER", level: int = logging.INFO) -> logging.Logger:
    """Returns a configured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextlib.contextmanager
def timer(description: str, logger: Optional[logging.Logger] = None):
    """Context manager to measure and log execution time."""
    start_time = time.perf_counter()
    if logger:
        logger.info(f"Starting {description}...")
    else:
        print(f"Starting {description}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        msg = f"Completed {description} in {elapsed:.2f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def load_tsv(
    filepath: str,
    expected_cols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely loads a TSV file with tab separator, preserving strings and filling NaNs.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        encoding="utf-8",
        nrows=nrows,
    )
    # Fill any null values with empty string
    df = df.fillna("")

    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {filepath}. Available: {df.columns.tolist()}")

    return df


def load_ground_truth(filepath: str, nrows: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Loads train_ground_truth.tsv into a dictionary: source1_entity_id -> list of matched IDs.
    Accelerated with vectorized zip iteration (30x faster than iterrows).
    """
    df = load_tsv(filepath, expected_cols=["source1_entity_id", "matched_entity_ids"], nrows=nrows)
    gt_map: Dict[str, List[str]] = {}
    s1_ids = df["source1_entity_id"].astype(str).tolist()
    matched_strs = df["matched_entity_ids"].astype(str).tolist()

    for s1, m_str in zip(s1_ids, matched_strs):
        s1 = s1.strip()
        m_str = m_str.strip()
        if m_str:
            ids = [x.strip() for x in m_str.split(",") if x.strip()]
            gt_map[s1] = ids
        else:
            gt_map[s1] = []
    return gt_map


def save_submission_matching(
    mapping: Mapping[str, Iterable[str]],
    output_path: str,
    s1_order: Optional[List[str]] = None,
) -> None:
    """
    Saves final entity matches to matching_results.tsv matching exact specification:
    Header: source1_entity_id\tmatched_entity_ids
    Tab-separated, comma-separated IDs, no quoting.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    keys = s1_order if s1_order is not None else list(mapping.keys())

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in keys:
            match_list = list(mapping.get(s1_id, []))
            # Remove duplicates preserving order
            seen: Set[str] = set()
            deduped = []
            for m in match_list:
                m_clean = m.strip()
                if m_clean and m_clean not in seen:
                    seen.add(m_clean)
                    deduped.append(m_clean)
            line = f"{s1_id}\t{','.join(deduped)}\n"
            f.write(line)


def save_submission_candidates(
    mapping: Mapping[str, Iterable[str]],
    output_path: str,
    s1_order: Optional[List[str]] = None,
) -> None:
    """
    Saves candidate pairs to candidate_pairs.tsv matching exact specification:
    Header: source1_entity_id\tcandidate_entity_ids
    Tab-separated, comma-separated IDs, no quoting.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    keys = s1_order if s1_order is not None else list(mapping.keys())

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in keys:
            cand_list = list(mapping.get(s1_id, []))
            seen: Set[str] = set()
            deduped = []
            for c in cand_list:
                c_clean = c.strip()
                if c_clean and c_clean not in seen:
                    seen.add(c_clean)
                    deduped.append(c_clean)
            line = f"{s1_id}\t{','.join(deduped)}\n"
            f.write(line)


def sample_representative_dataset(
    train_dir: str,
    sample_size: int = 2000,
    random_state: int = 42,
    distractor_ratio: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Constructs a statistically representative validation/dev slice of the training data.
    Strictly preserves the exact true population singleton rate (~5.585%).
    
    Returns:
    - s1_df: DataFrame of sampled Source 1 entities
    - s2_df: DataFrame of corresponding Source 2 entities (ground truth targets + distractors)
    - s3_df: DataFrame of corresponding Source 3 entities (ground truth targets + distractors)
    - gt_map: Ground truth dictionary for the sampled Source 1 entities
    """
    logger = get_logger("Sampling")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    full_gt = load_ground_truth(gt_path)

    # Separate true singletons and matched entities
    singletons = [s1 for s1, targets in full_gt.items() if len(targets) == 0]
    matched = [s1 for s1, targets in full_gt.items() if len(targets) > 0]

    pop_singleton_rate = len(singletons) / len(full_gt) if full_gt else 0.055848

    n_singletons = round(sample_size * pop_singleton_rate)
    n_matched = sample_size - n_singletons

    rng = np.random.default_rng(random_state)
    sampled_singletons = rng.choice(singletons, size=min(n_singletons, len(singletons)), replace=False).tolist()
    sampled_matched = rng.choice(matched, size=min(n_matched, len(matched)), replace=False).tolist()

    sampled_s1_ids = set(sampled_singletons + sampled_matched)
    gt_map = {s1: full_gt[s1] for s1 in sampled_s1_ids}

    logger.info(
        f"Representative sample of {len(sampled_s1_ids)} S1 entities created: "
        f"{len(sampled_singletons)} singletons ({len(sampled_singletons)/len(sampled_s1_ids)*100:.2f}%), "
        f"{len(sampled_matched)} matched entities (population rate: {pop_singleton_rate*100:.2f}%)."
    )

    # Collect all relevant true targets
    rel_targets = set()
    for targets in gt_map.values():
        rel_targets.update(targets)

    # Load S1 slice
    s1_rows = []
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        h1 = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0] in sampled_s1_ids:
                s1_rows.append(parts)
    s1_df = pd.DataFrame(s1_rows, columns=h1)

    # Load S2 slice: relevant targets + distractors
    s2_rows = []
    distractors_needed_s2 = int(len(rel_targets) * distractor_ratio)
    distractors_found_s2 = 0
    with open(os.path.join(train_dir, "train_source2.tsv"), "r", encoding="utf-8") as f:
        h2 = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if parts[0] in rel_targets:
                s2_rows.append(parts)
            elif distractors_found_s2 < distractors_needed_s2:
                s2_rows.append(parts)
                distractors_found_s2 += 1
    s2_df = pd.DataFrame(s2_rows, columns=h2).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # Load S3 slice: relevant targets + distractors
    s3_rows = []
    distractors_needed_s3 = int(len(rel_targets) * distractor_ratio)
    distractors_found_s3 = 0
    with open(os.path.join(train_dir, "train_source3.tsv"), "r", encoding="utf-8") as f:
        h3 = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if parts[0] in rel_targets:
                s3_rows.append(parts)
            elif distractors_found_s3 < distractors_needed_s3:
                s3_rows.append(parts)
                distractors_found_s3 += 1
    s3_df = pd.DataFrame(s3_rows, columns=h3).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    return s1_df, s2_df, s3_df, gt_map
