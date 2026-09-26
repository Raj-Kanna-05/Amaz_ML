# Business Entity Resolution Pipeline

Self-contained, reproducible machine learning solution for resolving noisy business entities across three independent data sources (`Source 1`, `Source 2`, and `Source 3`) evaluated on **Macro F0.5 Score**.

---

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── __init__.py           # Package marker
│   ├── utils.py              # TSV I/O, submission writing, logging & timing
│   ├── evaluation.py         # Exact macro F0.5 & candidate recall scoring
│   ├── preprocessing.py      # Unicode, accents, legal & street abbreviation normalizers
│   ├── blocking.py           # Multi-view candidate generation & country gating
│   ├── features.py           # Pairwise fuzzy string & open-set country features
│   ├── models.py             # LightGBM / XGBoost classifier wrapper
│   ├── train.py              # Training pipeline with GroupKFold CV & threshold sweep
│   ├── predict.py            # Inference pipeline & candidate pair generator
│   └── postprocess.py        # One-to-one constraint & singleton thresholding
├── pipeline.py               # Unified CLI runner (train, predict, end-to-end)
├── README.md                 # Pipeline documentation
└── requirements.txt          # Pinned dependencies
```

---

## Installation

Ensure Python 3.10+ is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage Instructions

All commands can be run from the root of this project or the repository root.

### 1. End-to-End Pipeline (Train + Predict + Validate)
Trains the model on training data, optimizes the macro F0.5 threshold via GroupKFold CV, runs inference on the test set, writes both `output/candidate_pairs.tsv` and `output/matching_results.tsv`, and validates the outputs automatically:

```bash
python code/business_entity_resolution/pipeline.py end-to-end \
  --train-dir dataset/train \
  --test-dir dataset/test \
  --output-dir output \
  --model-dir models \
  --model-type lightgbm
```

### 2. Training Only
Trains the model with 5-fold GroupKFold cross-validation grouped by `source1_entity_id`, optimizes the decision threshold for macro F0.5, and saves the trained model and metadata:

```bash
python code/business_entity_resolution/pipeline.py train \
  --train-dir dataset/train \
  --model-dir models \
  --model-type lightgbm \
  --top-k 50
```

### 3. Inference Only
Applies a previously trained model to new test data, producing the required candidate pairs and matching results TSV files:

```bash
python code/business_entity_resolution/pipeline.py predict \
  --test-dir dataset/test \
  --model-dir models \
  --output-dir output \
  --top-k 50
```

### 4. Submission Validation
Validate formatting, headers, ID constraints, singleton formats, and candidate subset guarantees using the standalone validator:

```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## Key Design Principles

1. **Macro F0.5 Objective:** Threshold is tuned directly to maximize macro F0.5 rather than standard accuracy or F1.
2. **Open-Set Country Modeling:** Country features rely purely on equality/mismatch flags and missingness indicators without hard-coding specific country categories, ensuring generalization to unseen countries like France.
3. **Hard-Negative Mining:** Negative pairs are drawn directly from candidates that pass the blocking phase, forcing the model to learn fine-grained discriminative boundaries.
4. **Permissive Licensing:** All core dependencies (LightGBM, XGBoost, Scikit-learn, RapidFuzz) comply with MIT/Apache 2.0 licenses.
