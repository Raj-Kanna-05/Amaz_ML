"""
Unit tests for postprocessing and one-to-one conflict resolution.
Verifies:
1. Multi-match retention for Source 1 entities (S1 can match >=1 targets).
2. Global 1-to-1 conflict resolution (target assigned strictly to highest-prob S1).
3. Singleton preservation below threshold.
4. Candidate subset constraint satisfaction.
"""

import sys
import unittest
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from postprocess import generate_final_matches, resolve_one_to_one


class TestPostprocess(unittest.TestCase):

    def test_multi_match_retention(self):
        """Source-1 entity with multiple valid matches across S2 and S3 retains all of them."""
        scored_pairs = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-1", "probability": 0.95},
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S3-1", "probability": 0.90},
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S3-2", "probability": 0.85},
        ])
        all_s1 = ["S1-100"]

        result = generate_final_matches(
            scored_pairs_df=scored_pairs,
            all_s1_ids=all_s1,
            threshold=0.50,
            enforce_one_to_one=True,
        )

        self.assertIn("S1-100", result)
        self.assertEqual(len(result["S1-100"]), 3)
        self.assertListEqual(result["S1-100"], ["S2-1", "S3-1", "S3-2"])

    def test_target_conflict_resolution(self):
        """When two S1 entities claim the same target, assign strictly to the higher probability S1."""
        scored_pairs = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-CONFLICT", "probability": 0.92},
            {"source1_entity_id": "S1-200", "candidate_entity_id": "S2-CONFLICT", "probability": 0.78},
            {"source1_entity_id": "S1-200", "candidate_entity_id": "S3-OTHER", "probability": 0.88},
        ])
        all_s1 = ["S1-100", "S1-200"]

        result = generate_final_matches(
            scored_pairs_df=scored_pairs,
            all_s1_ids=all_s1,
            threshold=0.50,
            enforce_one_to_one=True,
        )

        # S2-CONFLICT must belong to S1-100, NOT S1-200
        self.assertIn("S2-CONFLICT", result["S1-100"])
        self.assertNotIn("S2-CONFLICT", result["S1-200"])
        # S1-200 gets its uncontested target S3-OTHER
        self.assertIn("S3-OTHER", result["S1-200"])

    def test_singleton_preservation(self):
        """Entities whose candidate probabilities are below threshold remain singletons (empty list)."""
        scored_pairs = pd.DataFrame([
            {"source1_entity_id": "S1-SINGLETON", "candidate_entity_id": "S2-WEAK", "probability": 0.35},
        ])
        all_s1 = ["S1-SINGLETON", "S1-NO_CANDS"]

        result = generate_final_matches(
            scored_pairs_df=scored_pairs,
            all_s1_ids=all_s1,
            threshold=0.50,
            enforce_one_to_one=True,
        )

        self.assertEqual(result["S1-SINGLETON"], [])
        self.assertEqual(result["S1-NO_CANDS"], [])

    def test_candidate_subset_constraint(self):
        """Predictions are strictly restricted to the provided candidate pool."""
        scored_pairs = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-VALID", "probability": 0.95},
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-INVALID", "probability": 0.99},
        ])
        cand_mapping = {
            "S1-100": ["S2-VALID"]  # S2-INVALID not in candidate mapping
        }

        result = generate_final_matches(
            scored_pairs_df=scored_pairs,
            all_s1_ids=["S1-100"],
            threshold=0.50,
            enforce_one_to_one=True,
            candidate_mapping=cand_mapping,
        )

        self.assertIn("S2-VALID", result["S1-100"])
        self.assertNotIn("S2-INVALID", result["S1-100"])


if __name__ == "__main__":
    unittest.main()
