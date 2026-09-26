"""
Pairwise Feature Engineering Module for Business Entity Resolution.
Computes comprehensive string similarity metrics (Levenshtein, token sort,
token set, n-gram Jaccard), address number overlap, cross-field signals,
and country equality features (open-set safe).
"""

from typing import Dict, List, Set
import difflib
import numpy as np
import pandas as pd

# Safe import for rapidfuzz with pure-Python difflib fallback
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

    class _PurePythonFuzz:
        """Pure Python fallback for rapidfuzz using standard difflib."""

        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            return float(difflib.SequenceMatcher(None, s1, s2).ratio() * 100.0)

        @staticmethod
        def partial_ratio(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            len1 = len(s1)
            len2 = len(s2)
            if len1 == len2:
                return _PurePythonFuzz.ratio(s1, s2)
            blocks = difflib.SequenceMatcher(None, s1, s2).get_matching_blocks()
            best = 0.0
            for i, j, k in blocks:
                start = max(0, j - i)
                end = start + len1
                sub = s2[start:end]
                r = difflib.SequenceMatcher(None, s1, sub).ratio() * 100.0
                if r > best:
                    best = r
            return float(best)

        @staticmethod
        def token_sort_ratio(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            sorted_s1 = " ".join(sorted(s1.split()))
            sorted_s2 = " ".join(sorted(s2.split()))
            return _PurePythonFuzz.ratio(sorted_s1, sorted_s2)

        @staticmethod
        def token_set_ratio(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            set1 = set(s1.split())
            set2 = set(s2.split())
            inter = sorted(list(set1 & set2))
            diff1 = sorted(list(set1 - set2))
            diff2 = sorted(list(set2 - set1))

            inter_str = " ".join(inter)
            str1 = (inter_str + " " + " ".join(diff1)).strip()
            str2 = (inter_str + " " + " ".join(diff2)).strip()

            r1 = _PurePythonFuzz.ratio(inter_str, str1)
            r2 = _PurePythonFuzz.ratio(inter_str, str2)
            r3 = _PurePythonFuzz.ratio(str1, str2)
            return float(max(r1, r2, r3))

    fuzz = _PurePythonFuzz()

try:
    from .utils import get_logger, timer
except (ImportError, ValueError):
    from src.utils import get_logger, timer

logger = get_logger("Features")


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates set of character n-grams."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / float(union) if union > 0 else 0.0


def compute_pair_features(
    s1_row: Dict,
    target_row: Dict,
    blocking_score: float = 0.0,
    blocking_rank: int = 1,
    target_source: str = "S2",
) -> Dict[str, float]:
    """
    Computes all pairwise features for a single (S1, Candidate) pair.
    """
    # Names
    s1_name = str(s1_row.get("norm_name", ""))
    t_name = str(target_row.get("norm_name", ""))
    s1_core = str(s1_row.get("core_name", ""))
    t_core = str(target_row.get("core_name", ""))

    s1_ntoks = set(s1_row.get("name_tokens", []))
    t_ntoks = set(target_row.get("name_tokens", []))

    # Addresses
    s1_addr = str(s1_row.get("norm_address", ""))
    t_addr = str(target_row.get("norm_address", ""))
    s1_atoks = set(s1_row.get("address_tokens", []))
    t_atoks = set(target_row.get("address_tokens", []))

    s1_nums = set(str(x) for x in s1_row.get("address_numbers", []))
    t_nums = set(str(x) for x in target_row.get("address_numbers", []))

    # Countries
    s1_country = str(s1_row.get("norm_country", ""))
    t_country = str(target_row.get("norm_country", ""))

    feat: Dict[str, float] = {}

    # --- 1. Business Name Features ---
    has_names = bool(s1_name and t_name)
    feat["name_fuzz_ratio"] = (fuzz.ratio(s1_name, t_name) / 100.0) if has_names else 0.0
    feat["name_partial_ratio"] = (fuzz.partial_ratio(s1_name, t_name) / 100.0) if has_names else 0.0
    feat["name_token_sort_ratio"] = (fuzz.token_sort_ratio(s1_name, t_name) / 100.0) if has_names else 0.0
    feat["name_token_set_ratio"] = (fuzz.token_set_ratio(s1_name, t_name) / 100.0) if has_names else 0.0

    has_core = bool(s1_core and t_core)
    feat["core_name_ratio"] = (fuzz.ratio(s1_core, t_core) / 100.0) if has_core else 0.0
    feat["core_name_token_sort"] = (fuzz.token_sort_ratio(s1_core, t_core) / 100.0) if has_core else 0.0

    # Token overlap
    feat["name_token_jaccard"] = jaccard_similarity(s1_ntoks, t_ntoks)
    feat["name_token_overlap_count"] = float(len(s1_ntoks & t_ntoks))
    feat["name_token_diff_count"] = float(abs(len(s1_ntoks) - len(t_ntoks)))

    # First token match (very strong business name signal)
    first_tok_s1 = s1_name.split()[0] if s1_name.split() else ""
    first_tok_t = t_name.split()[0] if t_name.split() else ""
    feat["name_first_token_match"] = 1.0 if (first_tok_s1 and first_tok_s1 == first_tok_t) else 0.0

    # Character 3-gram Jaccard
    s1_ngrams = char_ngrams(s1_name, 3)
    t_ngrams = char_ngrams(t_name, 3)
    feat["name_char_3gram_jaccard"] = jaccard_similarity(s1_ngrams, t_ngrams)

    # Length features
    len_s1 = len(s1_name)
    len_t = len(t_name)
    feat["name_len_diff"] = float(abs(len_s1 - len_t))
    feat["name_len_ratio"] = (min(len_s1, len_t) / float(max(len_s1, len_t))) if (len_s1 > 0 and len_t > 0) else 0.0
    feat["name_exact_match"] = 1.0 if (s1_name and s1_name == t_name) else 0.0

    # --- 2. Business Address Features ---
    has_addrs = bool(s1_addr and t_addr)
    feat["addr_fuzz_ratio"] = (fuzz.ratio(s1_addr, t_addr) / 100.0) if has_addrs else 0.0
    feat["addr_partial_ratio"] = (fuzz.partial_ratio(s1_addr, t_addr) / 100.0) if has_addrs else 0.0
    feat["addr_token_sort_ratio"] = (fuzz.token_sort_ratio(s1_addr, t_addr) / 100.0) if has_addrs else 0.0
    feat["addr_token_set_ratio"] = (fuzz.token_set_ratio(s1_addr, t_addr) / 100.0) if has_addrs else 0.0

    feat["addr_token_jaccard"] = jaccard_similarity(s1_atoks, t_atoks)
    feat["addr_token_overlap_count"] = float(len(s1_atoks & t_atoks))
    feat["addr_exact_match"] = 1.0 if (s1_addr and s1_addr == t_addr) else 0.0

    # Address numbers (building numbers / postal codes)
    num_overlap = len(s1_nums & t_nums)
    has_both_nums = bool(s1_nums and t_nums)
    feat["addr_num_overlap_count"] = float(num_overlap)
    feat["addr_num_jaccard"] = jaccard_similarity(s1_nums, t_nums)
    feat["addr_num_exact_match"] = 1.0 if (has_both_nums and len(s1_nums) == len(t_nums) == num_overlap) else 0.0
    feat["addr_num_any_match"] = 1.0 if num_overlap > 0 else 0.0
    feat["addr_num_diff"] = float(abs(len(s1_nums) - len(t_nums)))

    # --- 3. Country Features (Open-Set Compatible) ---
    has_both_countries = bool(s1_country and t_country)
    feat["country_exact_match"] = 1.0 if (has_both_countries and s1_country == t_country) else 0.0
    feat["country_mismatch"] = 1.0 if (has_both_countries and s1_country != t_country) else 0.0
    feat["country_missing_either"] = 1.0 if not has_both_countries else 0.0

    # --- 4. Cross-Field Features ---
    # Name tokens appearing in address or vice versa
    feat["s1_name_in_t_addr"] = float(len(s1_ntoks & t_atoks))
    feat["t_name_in_s1_addr"] = float(len(t_ntoks & s1_atoks))

    # --- 5. Candidate Generation Signals ---
    feat["is_s2"] = 1.0 if target_source == "S2" else 0.0
    feat["blocking_score"] = float(blocking_score)
    feat["blocking_rank"] = float(blocking_rank)

    # Combined composite heuristics
    feat["composite_score"] = (
        0.55 * feat["name_token_set_ratio"]
        + 0.35 * feat["addr_token_set_ratio"]
        + 0.10 * feat["country_exact_match"]
    )

    return feat


class FeatureExtractor:
    """Extracts pairwise feature matrix from candidate pairs dataframe."""

    def __init__(self):
        self.feature_columns: List[str] = []

    def extract_features(
        self,
        pairs_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes pairwise features for all pairs in pairs_df.
        Returns a DataFrame of features aligned with pairs_df.
        """
        if len(pairs_df) == 0:
            return pd.DataFrame()

        with timer(f"Feature extraction for {len(pairs_df)} candidate pairs", logger):
            s1_id_col = pairs_df["source1_entity_id"].astype(str).tolist()
            cand_id_col = pairs_df["candidate_entity_id"].astype(str).tolist()
            tgt_col = pairs_df["target_source"].astype(str).tolist() if "target_source" in pairs_df.columns else [
                "S2" if c.startswith("S2-") else "S3" for c in cand_id_col
            ]
            score_col = pairs_df["blocking_score"].astype(float).tolist() if "blocking_score" in pairs_df.columns else [0.0] * len(pairs_df)
            rank_col = pairs_df["blocking_rank"].astype(int).tolist() if "blocking_rank" in pairs_df.columns else [1] * len(pairs_df)

            active_s1_ids = set(s1_id_col)
            active_cand_ids = set(cand_id_col)

            # Subset to active entities only — saves 4+ GB RAM and takes milliseconds
            s1_sub = s1_df[s1_df["entity_id"].isin(active_s1_ids)]
            s2_sub = s2_df[s2_df["entity_id"].isin(active_cand_ids)]
            s3_sub = s3_df[s3_df["entity_id"].isin(active_cand_ids)]

            s1_lookup = s1_sub.set_index("entity_id").to_dict(orient="index")
            s2_lookup = s2_sub.set_index("entity_id").to_dict(orient="index")
            s3_lookup = s3_sub.set_index("entity_id").to_dict(orient="index")

            feature_records = []
            for s1_id, cand_id, tgt_src, b_score, b_rank in zip(
                s1_id_col, cand_id_col, tgt_col, score_col, rank_col
            ):
                s1_row = s1_lookup.get(s1_id, {})
                t_lookup = s2_lookup if tgt_src == "S2" else s3_lookup
                t_row = t_lookup.get(cand_id, {})

                feat = compute_pair_features(
                    s1_row=s1_row,
                    target_row=t_row,
                    blocking_score=b_score,
                    blocking_rank=b_rank,
                    target_source=tgt_src,
                )
                feature_records.append(feat)

            feat_df = pd.DataFrame(feature_records)
            self.feature_columns = list(feat_df.columns)
            logger.info(f"Generated {len(self.feature_columns)} pairwise features.")
            return feat_df
