# Phase 10: Submission & Report

## Overview
Package model predictions into the CodaBench submission format, upload to the competition, and write the required validation report.

## Step 10.1: Generate Test Predictions

Run the final model on the **test split** (goalstep_test.json):
- Test set uses flat `step_segments` (no goal hierarchy, no timestamps)
- Each query gets exactly 5 predicted temporal windows ranked by confidence

## Step 10.2: Format Submission

Use `scripts/format_submission.py` to validate and package predictions:

```bash
# From real predictions
python scripts/format_submission.py results/test_predictions.json

# Or generate a dummy submission for format testing
python scripts/format_submission.py --dummy
```

Output: `results/submission.zip` ready for CodaBench upload.

## Step 10.3: Submit to CodaBench

1. Go to [CodaBench competition page](https://www.codabench.org/competitions/14878/)
2. Login and navigate to "My Submissions"
3. Upload `submission.zip`
4. Wait for evaluation to complete
5. Check results on the "Results" tab

> [!IMPORTANT]
> **Deadline: May 14, 2026 at 5:00 PM PDT.** Submit early to catch any format issues.

## Step 10.4: Validation Report

Participants must submit a short validation report (up to 4 pages) describing:
- Method overview
- Training details (data, features, hyperparameters)
- Results on validation set
- Ablation studies / analysis
- Positive results and limitations

## Verification Strategy
- **Format Test:** Submit a dummy `.zip` to CodaBench and verify it's accepted before the real submission.
- **Schema Validation:** Assert `submission.json` matches the exact schema (version, challenge, results with 5 predictions each).
- **Query Coverage:** Assert every query in the test annotations has a corresponding prediction entry.
