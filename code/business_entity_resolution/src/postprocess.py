"""
Post-Processing Module for Business Entity Resolution.
Applies:
1. Thresholding tuned for macro F0.5.
2. Global One-to-One Conflict Resolution (assigns candidate to highest-scoring S1).
3. Singleton preservation and deduplication.
4. Verification that matched IDs are a strict subset of candidates.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple
import pandas as pd

try:
    from .utils import get_logger
except (ImportError, ValueError):
    from src.utils import get_logger

logger = get_logger("Postprocess")


def resolve_one_to_one(
    scored_pairs: List[Dict],
) -> List[Dict]:
    """
    Enforces the global 1-to-1 constraint across sources:
    If a target candidate ID (S2 or S3) is matched to multiple S1 records,
    it is assigned exclusively to the S1 record with the highest predicted probability.
    """
    # Sort pairs by probability descending
    sorted_pairs = sorted(scored_pairs, key=lambda x: x["probability"], reverse=True)

    claimed_targets: Set[str] = set()
    accepted_pairs: List[Dict] = []

    for pair in sorted_pairs:
        target_id = pair["candidate_entity_id"]
        if target_id not in claimed_targets:
            claimed_targets.add(target_id)
            accepted_pairs.append(pair)

    return accepted_pairs


def generate_final_matches(
    scored_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    threshold: Optional[float] = 0.50,
    enforce_one_to_one: bool = True,
    candidate_mapping: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Produces final mapping of source1_entity_id -> list of matched IDs.
    
    Parameters:
        scored_pairs_df: DataFrame containing:
            ['source1_entity_id', 'candidate_entity_id', 'probability']
        all_s1_ids: complete list of test/evaluation S1 IDs
        threshold: probability cutoff for positive classification
        enforce_one_to_one: whether to resolve target collisions
        candidate_mapping: mapping of S1 -> candidate IDs to guarantee subset constraint
    """
    th = threshold if threshold is not None else 0.50
    # Filter pairs exceeding probability threshold
    filtered_df = scored_pairs_df[scored_pairs_df["probability"] >= th].copy()

    records = filtered_df.to_dict(orient="records")

    if enforce_one_to_one and records:
        records = resolve_one_to_one(records)

    # Group by S1 entity ID
    matches_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}

    # Sort remaining candidates by score per S1
    records.sort(key=lambda x: x["probability"], reverse=True)

    for item in records:
        s1 = item["source1_entity_id"]
        cand = item["candidate_entity_id"]
        if s1 in matches_map:
            # Check candidate subset constraint
            if candidate_mapping is not None:
                cand_pool = set(candidate_mapping.get(s1, []))
                if cand not in cand_pool:
                    continue
            if cand not in matches_map[s1]:
                matches_map[s1].append(cand)

    # Log statistics
    total_s1 = len(all_s1_ids)
    singletons = sum(1 for v in matches_map.values() if len(v) == 0)
    total_matched = sum(len(v) for v in matches_map.values())

    logger.info(
        f"Post-processing complete (threshold={threshold:.3f}, one_to_one={enforce_one_to_one}): "
        f"{total_matched} total matches assigned, {singletons}/{total_s1} singletons "
        f"({singletons / total_s1 * 100:.2f}%)."
    )

    return matches_map
