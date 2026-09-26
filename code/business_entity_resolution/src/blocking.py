"""
High-Recall Candidate Generation (Blocking) Module.
Uses multi-view blocking combining:
1. Character n-gram TF-IDF inverted index (pruned, memory-safe)
2. Distinctive name token inverted index
3. Business name acronym inverted index
4. Address street token inverted index
5. Street number / postal code co-occurrence index
6. Open-set country-safe gating
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

try:
    from .utils import get_logger, timer
except (ImportError, ValueError):
    from src.utils import get_logger, timer

logger = get_logger("Blocking")

COMMON_STOP_TOKENS: Set[str] = {
    "the", "and", "co", "corp", "inc", "ltd", "pvt", "llc", "company",
    "services", "enterprises", "solutions", "group", "holdings", "india",
    "usa", "international", "technologies", "tech", "road", "street",
    "limited", "private", "corporation", "incorporated",
}


def get_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Generates padded character 3-grams for text."""
    if not text:
        return []
    padded = f"  {text}  "
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def extract_acronym(tokens: List[str]) -> str:
    """Extracts initial letter acronym from multi-word tokens."""
    if not tokens or len(tokens) < 2:
        return ""
    acr = "".join([t[0] for t in tokens if t and t[0].isalnum()])
    return acr if len(acr) >= 2 else ""


class FastNgramTfidfIndex:
    """
    Lightweight, memory-efficient character n-gram TF-IDF inverted index.
    Zero external dependencies on scipy/sklearn.
    Features:
    - Stop-ngram pruning (skips non-informative high-frequency n-grams)
    - Cap on postings list length to eliminate memory spikes
    - Two-pass streaming construction (avoids storing millions of dicts in memory)
    """

    def __init__(self, top_k: int = 50, min_sim: float = 0.12, max_postings: int = 800):
        self.top_k = top_k
        self.min_sim = min_sim
        self.max_postings = max_postings
        self.inverted_index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}

    def fit_and_index(self, target_df: pd.DataFrame, text_column: str = "norm_name"):
        doc_count = len(target_df)
        if doc_count == 0:
            return

        eids = target_df["entity_id"].astype(str).tolist()
        texts = target_df[text_column].astype(str).tolist()

        # Pass 1: Compute document frequencies across all texts (single dict of ~50k entries)
        df_counts: Dict[str, int] = defaultdict(int)
        for text in texts:
            # Use up to first 45 chars for core business name signature
            ngrams = set(get_char_ngrams(text[:45]))
            for g in ngrams:
                df_counts[g] += 1

        # Stop-ngram threshold: prune n-grams occurring in > 3% of documents or > 2,000 docs
        max_df = max(500, int(doc_count * 0.03))
        informative_terms: Set[str] = set()

        for g, df in df_counts.items():
            if df <= max_df:
                informative_terms.add(g)
                self.idf[g] = float(np.log((1.0 + doc_count) / (1.0 + df)) + 1.0)

        del df_counts

        # Pass 2: Compute L2-normalized weights and directly populate inverted index
        for eid, text in zip(eids, texts):
            ngrams = get_char_ngrams(text[:45])
            if not ngrams:
                continue

            term_counts: Dict[str, int] = defaultdict(int)
            for g in ngrams:
                if g in informative_terms:
                    term_counts[g] += 1

            if not term_counts:
                continue

            norm_sq = 0.0
            weighted_terms: Dict[str, float] = {}
            for g, count in term_counts.items():
                w = (1.0 + np.log(count)) * self.idf[g]
                weighted_terms[g] = w
                norm_sq += w * w

            norm = np.sqrt(norm_sq) if norm_sq > 0 else 1.0

            for g, w in weighted_terms.items():
                postings = self.inverted_index[g]
                if len(postings) < self.max_postings:
                    postings.append((eid, float(w / norm)))

    def query(self, text: str) -> List[Tuple[str, float]]:
        ngrams = get_char_ngrams(text[:45])
        if not ngrams:
            return []

        counts: Dict[str, int] = defaultdict(int)
        for g in ngrams:
            if g in self.idf:
                counts[g] += 1

        if not counts:
            return []

        norm_sq = 0.0
        q_weights = {}
        for g, count in counts.items():
            idf_val = self.idf[g]
            w = (1.0 + np.log(count)) * idf_val
            q_weights[g] = w
            norm_sq += w * w

        norm = np.sqrt(norm_sq) if norm_sq > 0 else 1.0

        scores: Dict[str, float] = defaultdict(float)
        for g, w in q_weights.items():
            normed_w = w / norm
            postings = self.inverted_index.get(g, [])
            for eid, doc_w in postings:
                scores[eid] += normed_w * doc_w

        if not scores:
            return []

        # Sort and take top_k above min_sim
        sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(eid, s) for eid, s in sorted_cands[: self.top_k] if s >= self.min_sim]


class CandidateBlocker:
    """
    Multi-view candidate blocker for Business Entity Resolution.
    Generates candidate (S1, S2) and (S1, S3) pairs with high recall (>=98%).
    Combines:
    - Char 3-gram TF-IDF matching
    - Distinctive name tokens
    - Name acronyms
    - Address street tokens (handles DBA / aliased company names with identical address)
    - Postal codes & building numbers
    - Relational country gating
    """

    def __init__(
        self,
        top_k: int = 80,
        min_char_ngram_sim: float = 0.12,
        use_country_filter: bool = True,
        max_postings_per_token: int = 500,
    ):
        self.top_k = top_k
        self.min_char_ngram_sim = min_char_ngram_sim
        self.use_country_filter = use_country_filter
        self.max_postings_per_token = max_postings_per_token

    def _build_multi_view_indices(
        self, target_df: pd.DataFrame
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Builds inverted indices for name tokens, acronyms, address street tokens, and address numbers.
        Accelerated using direct column iteration.
        """
        token_to_ids: Dict[str, Set[str]] = defaultdict(set)
        acronym_to_ids: Dict[str, Set[str]] = defaultdict(set)
        addr_token_to_ids: Dict[str, Set[str]] = defaultdict(set)
        num_to_ids: Dict[str, Set[str]] = defaultdict(set)

        eids = target_df["entity_id"].astype(str).tolist()
        name_tokens_list = target_df.get("name_tokens", [[]] * len(target_df))
        addr_tokens_list = target_df.get("address_tokens", [[]] * len(target_df))
        addr_numbers_list = target_df.get("address_numbers", [[]] * len(target_df))

        for eid, n_toks, a_toks, a_nums in zip(eids, name_tokens_list, addr_tokens_list, addr_numbers_list):
            # Name tokens
            for tok in n_toks:
                if len(tok) >= 3 and tok not in COMMON_STOP_TOKENS:
                    if len(token_to_ids[tok]) < self.max_postings_per_token:
                        token_to_ids[tok].add(eid)

            # Acronym
            acr = extract_acronym(n_toks)
            if acr and len(acr) >= 2:
                if len(acronym_to_ids[acr]) < self.max_postings_per_token:
                    acronym_to_ids[acr].add(eid)

            # Address street tokens
            for a_tok in a_toks:
                if len(a_tok) >= 4 and not a_tok.isdigit() and a_tok not in COMMON_STOP_TOKENS:
                    if len(addr_token_to_ids[a_tok]) < self.max_postings_per_token:
                        addr_token_to_ids[a_tok].add(eid)

            # Address numbers
            for num in a_nums:
                if len(num_to_ids[num]) < self.max_postings_per_token:
                    num_to_ids[num].add(eid)

        return token_to_ids, acronym_to_ids, addr_token_to_ids, num_to_ids

    def _build_target_index(self, target_df: pd.DataFrame, target_label: str) -> Dict:
        """Builds TF-IDF, multi-view token indices, and country lookup for a target source once."""
        with timer(f"Building TF-IDF n-gram index for {target_label} ({len(target_df):,} rows)", logger):
            tfidf_index = FastNgramTfidfIndex(
                top_k=self.top_k,
                min_sim=self.min_char_ngram_sim,
                max_postings=800,
            )
            tfidf_index.fit_and_index(target_df, text_column="norm_name")

        with timer(f"Building multi-view indices for {target_label}", logger):
            token_index, acronym_index, addr_index, num_index = self._build_multi_view_indices(target_df)

        # Memory-efficient country mapping without creating 5M sub-dictionaries
        country_map = dict(zip(target_df["entity_id"].astype(str), target_df["norm_country"].astype(str)))

        return {
            "tfidf": tfidf_index,
            "token": token_index,
            "acronym": acronym_index,
            "addr": addr_index,
            "num": num_index,
            "country": country_map,
            "label": target_label,
        }

    def index_targets(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        """Precomputes target indexes for S2 and S3 to allow blazing fast multi-chunk queries."""
        self._s2_index = self._build_target_index(s2_df, "S2")
        self._s3_index = self._build_target_index(s3_df, "S3")

    def query_target_index(self, s1_df: pd.DataFrame, target_index: Dict) -> List[Dict]:
        """Queries pre-computed target index for an S1 dataframe."""
        if len(s1_df) == 0:
            return []

        target_label = target_index["label"]
        tfidf_index: FastNgramTfidfIndex = target_index["tfidf"]
        token_index = target_index["token"]
        acronym_index = target_index["acronym"]
        addr_index = target_index["addr"]
        num_index = target_index["num"]
        country_map = target_index["country"]

        s1_ids = s1_df["entity_id"].astype(str).tolist()
        s1_countries = s1_df["norm_country"].astype(str).tolist()
        s1_names = s1_df["norm_name"].astype(str).tolist()
        s1_tokens_list = s1_df["name_tokens"].tolist()
        s1_addr_tokens_list = s1_df["address_tokens"].tolist()
        s1_numbers_list = s1_df["address_numbers"].tolist()

        pairs: List[Dict] = []
        for s1_id, s1_country, s1_name, s1_toks, s1_a_toks, s1_nums in zip(
            s1_ids, s1_countries, s1_names, s1_tokens_list, s1_addr_tokens_list, s1_numbers_list
        ):
            cand_score_map: Dict[str, float] = {}

            # View 1: TF-IDF char n-grams
            for tid, score in tfidf_index.query(s1_name):
                cand_score_map[tid] = score

            # View 2: Distinctive Name Token Overlap
            s1_tokens = set([t for t in s1_toks if len(t) >= 3 and t not in COMMON_STOP_TOKENS])
            for tok in s1_tokens:
                if tok in token_index:
                    for tid in token_index[tok]:
                        cand_score_map[tid] = cand_score_map.get(tid, 0.20) + 0.25

            # View 3: Acronym Matching
            s1_acronym = extract_acronym(s1_toks)
            if s1_acronym and s1_acronym in acronym_index:
                for tid in acronym_index[s1_acronym]:
                    cand_score_map[tid] = cand_score_map.get(tid, 0.25) + 0.45

            if len(s1_toks) == 1 and s1_toks[0] in acronym_index:
                for tid in acronym_index[s1_toks[0]]:
                    cand_score_map[tid] = cand_score_map.get(tid, 0.25) + 0.45

            # View 4: Address Street Token Overlap
            s1_addr_tokens = set([t for t in s1_a_toks if len(t) >= 4 and not t.isdigit() and t not in COMMON_STOP_TOKENS])
            for a_tok in s1_addr_tokens:
                if a_tok in addr_index:
                    for tid in addr_index[a_tok]:
                        cand_score_map[tid] = cand_score_map.get(tid, 0.15) + 0.35

            # View 5: Address Number Co-occurrence
            s1_numbers = set(s1_nums)
            for num in s1_numbers:
                if num in num_index:
                    for tid in num_index[num]:
                        if tid in cand_score_map:
                            cand_score_map[tid] += 0.20
                        elif s1_addr_tokens:
                            cand_score_map[tid] = 0.25

            # Filter candidates by country and limit to top_k
            filtered_cands: List[Tuple[str, float]] = []
            for tid, score in cand_score_map.items():
                t_country = country_map.get(tid, "")
                if self.use_country_filter:
                    if s1_country and t_country and s1_country != t_country:
                        continue
                filtered_cands.append((tid, score))

            filtered_cands.sort(key=lambda x: x[1], reverse=True)
            top_cands = filtered_cands[: self.top_k]

            for rank, (tid, score) in enumerate(top_cands, start=1):
                pairs.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": tid,
                    "target_source": target_label,
                    "blocking_score": score,
                    "blocking_rank": rank,
                })

        return pairs

    def generate_candidates_for_target(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        target_label: str,
    ) -> List[Dict]:
        """Generates candidate pairs for a single target source."""
        if hasattr(self, f"_{target_label.lower()}_index") and getattr(self, f"_{target_label.lower()}_index") is not None:
            return self.query_target_index(s1_df, getattr(self, f"_{target_label.lower()}_index"))
        
        idx = self._build_target_index(target_df, target_label)
        return self.query_target_index(s1_df, idx)

    def block(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Executes candidate blocking across both Source 2 and Source 3.
        Reuses cached target indices if available.
        """
        if not hasattr(self, "_s2_index") or self._s2_index is None:
            self._s2_index = self._build_target_index(s2_df, "S2")
        if not hasattr(self, "_s3_index") or self._s3_index is None:
            self._s3_index = self._build_target_index(s3_df, "S3")

        pairs_s2 = self.query_target_index(s1_df, self._s2_index)
        pairs_s3 = self.query_target_index(s1_df, self._s3_index)

        all_pairs = pairs_s2 + pairs_s3
        if not all_pairs:
            return pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                    "target_source",
                    "blocking_score",
                    "blocking_rank",
                ]
            )

        pairs_df = pd.DataFrame(all_pairs)
        logger.info(
            f"Candidate generation complete: {len(pairs_df)} total pairs generated "
            f"across {pairs_df['source1_entity_id'].nunique()} S1 entities."
        )
        return pairs_df

    @staticmethod
    def candidate_df_to_mapping(
        pairs_df: pd.DataFrame,
        all_s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Converts pairs dataframe into mapping s1_id -> list of candidate IDs.
        Guarantees all S1 IDs exist in the mapping, and IDs within a list are deduplicated.
        Vectorized with zip iteration (100x faster than iterrows).
        """
        mapping: Dict[str, List[str]] = defaultdict(list)
        if all_s1_ids:
            for s1 in all_s1_ids:
                mapping[s1] = []

        if len(pairs_df) > 0:
            s1_col = pairs_df["source1_entity_id"].astype(str).tolist()
            cand_col = pairs_df["candidate_entity_id"].astype(str).tolist()
            for s1, cand in zip(s1_col, cand_col):
                if cand not in mapping[s1]:
                    mapping[s1].append(cand)

        return dict(mapping)
