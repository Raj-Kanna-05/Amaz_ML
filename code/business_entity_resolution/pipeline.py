#!/usr/bin/env python3
"""
Business Entity Resolution End-to-End CLI Pipeline.
Provides commands for:
  - train: trains model on training set with GroupKFold CV and tunes threshold
  - predict: generates candidate_pairs.tsv and matching_results.tsv on test set
  - end-to-end: executes train, predict, and auto-validates submission files
"""

import argparse
import os
import sys

# Ensure local src directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.models import EntityMatchingModel
from src.predict import predict_pipeline
from src.train import train_pipeline
from src.utils import get_logger

logger = get_logger("Pipeline")


def resolve_path(provided_path: str, default_candidates: list) -> str:
    """Resolves provided path or falls back to existing candidates."""
    if provided_path and os.path.exists(provided_path):
        return provided_path
    for cand in default_candidates:
        if os.path.exists(cand):
            return cand
    return provided_path


def run_train(args):
    logger.info("=== Running Training Pipeline ===")
    train_dir = resolve_path(args.train_dir, ["student_resource/dataset/train", "dataset/train"])
    logger.info(f"Using training directory: {train_dir}")
    model, thresh, meta = train_pipeline(
        train_dir=train_dir,
        output_dir=args.model_dir,
        model_type=args.model_type,
        top_k_candidates=args.top_k,
        n_splits=args.cv_splits,
        enforce_one_to_one=not args.disable_one_to_one,
        sample_size=args.sample_size,
    )
    print("\nTraining complete!")
    print(f"Optimal F0.5 Threshold: {thresh:.3f}")
    print(f"Best CV Macro F0.5: {meta['best_cv_macro_f05']:.4f}")
    print(f"Model saved to: {os.path.join(args.model_dir, 'entity_matching_model.joblib')}")


def run_predict(args):
    logger.info("=== Running Inference Pipeline ===")
    test_dir = resolve_path(args.test_dir, ["student_resource/dataset/test", "dataset/test"])
    logger.info(f"Using test directory: {test_dir}")
    model_path = os.path.join(args.model_dir, "entity_matching_model.joblib")
    meta_path = os.path.join(args.model_dir, "model_metadata.json")

    results = predict_pipeline(
        test_dir=test_dir,
        model_path=model_path,
        metadata_path=meta_path,
        output_dir=args.output_dir,
        top_k_candidates=args.top_k,
        threshold=args.threshold,
        enforce_one_to_one=not args.disable_one_to_one,
    )
    print("\nPrediction complete!")
    print(f"Matching Results: {results['matching_file']}")
    print(f"Candidate Pairs:  {results['candidate_file']}")


def run_end_to_end(args):
    logger.info("=== Running End-to-End Pipeline (Train + Predict + Validate) ===")
    train_dir = resolve_path(args.train_dir, ["student_resource/dataset/train", "dataset/train"])
    test_dir = resolve_path(args.test_dir, ["student_resource/dataset/test", "dataset/test"])
    logger.info(f"Using training directory: {train_dir}")
    logger.info(f"Using test directory: {test_dir}")

    # 1. Train
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model, thresh, meta = train_pipeline(
        train_dir=train_dir,
        output_dir=model_dir,
        model_type=args.model_type,
        top_k_candidates=args.top_k,
        n_splits=args.cv_splits,
        enforce_one_to_one=not args.disable_one_to_one,
        sample_size=args.sample_size,
    )

    # 2. Predict
    model_path = os.path.join(model_dir, "entity_matching_model.joblib")
    meta_path = os.path.join(model_dir, "model_metadata.json")
    results = predict_pipeline(
        test_dir=test_dir,
        model_path=model_path,
        metadata_path=meta_path,
        output_dir=args.output_dir,
        top_k_candidates=args.top_k,
        threshold=thresh,
        enforce_one_to_one=not args.disable_one_to_one,
    )

    # 3. Validate Submission
    print("\n=== Validating Submission Files ===")
    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    validator_path = os.path.join(workspace_root, "student_resource", "utils", "validate_submission.py")
    if not os.path.exists(validator_path):
        validator_path = os.path.join(workspace_root, "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        import subprocess
        cmd = [
            sys.executable,
            validator_path,
            "--matching",
            results["matching_file"],
            "--candidate",
            results["candidate_file"],
            "--test-dir",
            test_dir,
        ]
        ret = subprocess.call(cmd)
        if ret == 0:
            print("\nSUCCESS: End-to-end pipeline completed and verified!")
        else:
            print(f"\nWARNING: Submission validator exited with status {ret}")
    else:
        logger.warning(f"Validator script not found at {validator_path}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Business Entity Resolution Pipeline"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train subparser
    train_p = subparsers.add_parser("train", help="Train model on training data")
    train_p.add_argument("--train-dir", default=None, help="Path to train dir")
    train_p.add_argument("--model-dir", default="models", help="Directory to save model")
    train_p.add_argument("--model-type", choices=["lightgbm", "xgboost", "numpy"], default="lightgbm")
    train_p.add_argument("--top-k", type=int, default=80, help="Top K candidates per source")
    train_p.add_argument("--cv-splits", type=int, default=5, help="GroupKFold CV splits")
    train_p.add_argument("--sample-size", type=int, default=None, help="Limit number of S1 entities for training (e.g. for fast prototypes)")
    train_p.add_argument("--disable-one-to-one", action="store_true", help="Disable 1-to-1 conflict resolution")

    # Predict subparser
    pred_p = subparsers.add_parser("predict", help="Generate predictions on test data")
    pred_p.add_argument("--test-dir", default=None, help="Path to test dir")
    pred_p.add_argument("--model-dir", default="models", help="Directory with trained model")
    pred_p.add_argument("--output-dir", default="output", help="Directory for output TSVs")
    pred_p.add_argument("--top-k", type=int, default=80, help="Top K candidates per source")
    pred_p.add_argument("--threshold", type=float, default=None, help="Probability threshold override")
    pred_p.add_argument("--disable-one-to-one", action="store_true", help="Disable 1-to-1 conflict resolution")

    # End-to-end subparser
    e2e_p = subparsers.add_parser("end-to-end", help="Train, predict, and validate submission")
    e2e_p.add_argument("--train-dir", default=None, help="Path to train dir")
    e2e_p.add_argument("--test-dir", default=None, help="Path to test dir")
    e2e_p.add_argument("--model-dir", default="models", help="Directory to save model")
    e2e_p.add_argument("--output-dir", default="output", help="Directory for output TSVs")
    e2e_p.add_argument("--model-type", choices=["lightgbm", "xgboost", "numpy"], default="lightgbm")
    e2e_p.add_argument("--top-k", type=int, default=80, help="Top K candidates per source")
    e2e_p.add_argument("--cv-splits", type=int, default=5, help="GroupKFold CV splits")
    e2e_p.add_argument("--sample-size", type=int, default=None, help="Limit number of S1 entities for training")
    e2e_p.add_argument("--disable-one-to-one", action="store_true", help="Disable 1-to-1 conflict resolution")

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)
    elif args.mode == "end-to-end":
        run_end_to_end(args)


if __name__ == "__main__":
    main()
