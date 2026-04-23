# Phase 9: Local Evaluation

## Overview
Evaluate model predictions on the **validation split** using the exact same metrics as the CodaBench competition. This enables iterating on our approach locally before submitting to the leaderboard.

## Official Metrics

The leaderboard is ranked by **r@1, IoU=0.3**. Ties are broken by **r@1, IoU=0.5**.

| Metric | Description | Role |
|---|---|---|
| **r@1, IoU=0.3** | Recall at 1 of ground truth window at IoU ≥ 0.3 | ✅ Primary |
| **r@1, IoU=0.5** | Recall at 1 of ground truth window at IoU ≥ 0.5 | Tiebreaker |
| **r@5, IoU=0.3** | Recall at 5 of ground truth window at IoU ≥ 0.3 | Secondary |
| **r@5, IoU=0.5** | Recall at 5 of ground truth window at IoU ≥ 0.5 | Secondary |
| **Mean r@1** | Average of r@1 at IoU=0.3 and IoU=0.5 | Secondary |

## Target Performance

Based on the previous EvalAI leaderboard:

| Rank | Team | r@1, IoU=0.3 | r@1, IoU=0.5 |
|------|------|--------------|--------------|
| 1 | iLearn2.0 | **42.02** | 32.83 |
| 2 | CarLor (BayesianVSLNet) | 35.18 | 20.49 |
| 3 | 123456ABCD | 34.06 | 26.97 |

Our initial goal: match or exceed the VSLNet baseline (~30% r@1 IoU=0.3). Stretch goal: beat 42.02%.

## Evaluation Code

```python
import json
import os
import numpy as np

SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"


def compute_iou(pred_start, pred_end, gt_start, gt_end):
    """Compute temporal IoU between predicted and ground-truth segments."""
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_predictions(predictions, ground_truth, iou_thresholds=[0.3, 0.5]):
    """
    Evaluate predictions against ground truth using competition metrics.

    Args:
        predictions: List of dicts with clip_uid, query_idx, predicted_times (5 windows)
        ground_truth: Dict mapping (video_uid, query_idx) → (gt_start, gt_end)
        iou_thresholds: IoU thresholds to evaluate at

    Returns:
        Dict of metric_name → value (as percentage)
    """
    metrics = {}
    for k in [1, 5]:
        for t in iou_thresholds:
            key = f"r@{k}_iou{t}"
            hits = 0
            total = 0

            for pred in predictions:
                vid = pred["clip_uid"]
                qidx = pred["query_idx"]
                gt_key = (vid, qidx)

                if gt_key not in ground_truth:
                    continue

                gt_start, gt_end = ground_truth[gt_key]
                total += 1

                # Check top-k predictions
                times = pred["predicted_times"][:k]
                for (ps, pe) in times:
                    iou = compute_iou(ps, pe, gt_start, gt_end)
                    if iou >= t:
                        hits += 1
                        break  # Only count once per query

            metrics[key] = round(hits / total * 100, 2) if total > 0 else 0.0

    # Mean r@1
    metrics["mean_r@1"] = round(
        (metrics.get("r@1_iou0.3", 0) + metrics.get("r@1_iou0.5", 0)) / 2, 2
    )

    return metrics


def load_val_ground_truth():
    """Load ground truth from the validation split annotations."""
    from parse_annotations import parse_goalstep_annotations

    videos = parse_goalstep_annotations("val")
    gt = {}
    for vid, ann in videos.items():
        for q in ann.queries:
            if q.gt_start_time >= 0:
                gt[(vid, q.query_idx)] = (q.gt_start_time, q.gt_end_time)
    return gt


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Load predictions
    pred_path = os.path.join(SSD_BASE, "results/val_predictions.json")
    if not os.path.exists(pred_path):
        print(f"No predictions found at {pred_path}")
        print("Run the model first, then save predictions in submission format.")
        exit(1)

    with open(pred_path, 'r') as f:
        submission = json.load(f)
    predictions = submission.get("results", submission)

    # Load ground truth
    gt = load_val_ground_truth()
    print(f"Loaded {len(gt)} ground truth entries")

    # Evaluate
    metrics = evaluate_predictions(predictions, gt)

    print("\n" + "=" * 50)
    print("LOCAL EVALUATION RESULTS (Validation Split)")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        primary = " ← PRIMARY" if k == "r@1_iou0.3" else ""
        print(f"  {k:<15} {v:>7.2f}%{primary}")
    print("=" * 50)
```

## Verification Strategy
- **Known-Answer Test:** Create predictions where predicted = ground truth exactly. Assert r@1 IoU=0.3 = 100%.
- **Zero-Overlap Test:** Predictions that don't overlap GT at all. Assert all metrics = 0%.
- **Format Check:** Verify predictions have exactly 5 windows per query.
