# Phase 5.5: Local Evaluation Pipeline

## Overview
Before submitting to CodaBench, we must validate our predictions against the ground truth on the **validation split**. This phase runs the official Ego4D evaluation metrics locally and provides diagnostic breakdowns to identify failure modes.

> [!IMPORTANT]
> **Never submit to CodaBench without running this phase first.** The leaderboard has limited submission attempts, and blind submissions waste them.

## Official Metrics
The Ego4D Goal-Step Step Grounding task uses:

| Metric | Description | Primary? |
|---|---|---|
| **r@1, IoU=0.3** | % of queries where the top-1 prediction has IoU ≥ 0.3 with ground truth | ✅ Primary |
| **r@1, IoU=0.5** | % of queries where the top-1 prediction has IoU ≥ 0.5 with ground truth | Tie-breaker |
| **r@5, IoU=0.3** | % of queries where any of top-5 predictions has IoU ≥ 0.3 | Secondary |
| **r@5, IoU=0.5** | % of queries where any of top-5 predictions has IoU ≥ 0.5 | Secondary |

## Pseudocode Implementation

```python
import json
import os
import numpy as np
from collections import defaultdict

SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"

def compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute Intersection over Union between predicted and ground truth temporal segments."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    intersection = max(0, intersection_end - intersection_start)
    
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def evaluate_predictions(predictions_path: str, ground_truth_path: str) -> dict:
    """
    Evaluate predictions against ground truth using official metrics.
    
    Args:
        predictions_path: Path to submission JSON (our Phase 6 output)
        ground_truth_path: Path to val split annotations
    
    Returns:
        Dict with all metric values and per-query diagnostics
    """
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Build GT lookup: (clip_uid, query_idx) -> (start, end)
    gt_lookup = {}
    for video_entry in ground_truth.get("videos", ground_truth):
        video_uid = video_entry["video_uid"]
        query_idx = 0
        for goal in video_entry.get("segments", []):
            for step in goal.get("segments", []):
                if "start_time" in step and "end_time" in step:
                    gt_lookup[(video_uid, query_idx)] = (
                        step["start_time"], step["end_time"]
                    )
                query_idx += 1
    
    # Evaluate
    iou_thresholds = [0.3, 0.5]
    k_values = [1, 5]
    
    results = {f"r@{k}_iou{t}": 0 for k in k_values for t in iou_thresholds}
    total_queries = 0
    diagnostics = []
    
    for pred_entry in predictions.get("results", []):
        clip_uid = pred_entry["clip_uid"]
        query_idx = pred_entry["query_idx"]
        predicted_times = pred_entry["predicted_times"]
        
        gt_key = (clip_uid, query_idx)
        if gt_key not in gt_lookup:
            print(f"[WARN] No ground truth for {clip_uid} query {query_idx}")
            continue
        
        gt_start, gt_end = gt_lookup[gt_key]
        total_queries += 1
        
        # Compute IoU for each prediction
        ious = []
        for pred_start, pred_end in predicted_times:
            iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
            ious.append(iou)
        
        # Check recall at different k and IoU thresholds
        for t in iou_thresholds:
            for k in k_values:
                top_k_ious = ious[:k]
                if any(iou >= t for iou in top_k_ious):
                    results[f"r@{k}_iou{t}"] += 1
        
        # Diagnostic entry
        diagnostics.append({
            "clip_uid": clip_uid,
            "query_idx": query_idx,
            "gt": [gt_start, gt_end],
            "pred_top1": predicted_times[0] if predicted_times else None,
            "iou_top1": ious[0] if ious else 0,
            "max_iou_top5": max(ious[:5]) if ious else 0,
            "hit_03": ious[0] >= 0.3 if ious else False,
            "hit_05": ious[0] >= 0.5 if ious else False,
        })
    
    # Compute percentages
    if total_queries > 0:
        for key in results:
            results[key] = round(results[key] / total_queries * 100, 2)
    
    # Sort diagnostics by IoU (worst first) for debugging
    diagnostics.sort(key=lambda x: x["iou_top1"])
    
    return {
        "metrics": results,
        "total_queries": total_queries,
        "diagnostics": diagnostics
    }

def print_evaluation_report(eval_result: dict):
    """Pretty-print the evaluation results."""
    metrics = eval_result["metrics"]
    total = eval_result["total_queries"]
    diagnostics = eval_result["diagnostics"]
    
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total queries evaluated: {total}")
    print()
    
    print("📊 Official Metrics:")
    print(f"  r@1, IoU=0.3: {metrics['r@1_iou0.3']}%  ← PRIMARY METRIC")
    print(f"  r@1, IoU=0.5: {metrics['r@1_iou0.5']}%  ← TIE-BREAKER")
    print(f"  r@5, IoU=0.3: {metrics['r@5_iou0.3']}%")
    print(f"  r@5, IoU=0.5: {metrics['r@5_iou0.5']}%")
    print()
    
    # Failure analysis
    failures_03 = [d for d in diagnostics if not d["hit_03"]]
    failures_05 = [d for d in diagnostics if not d["hit_05"]]
    
    print(f"❌ Failures (IoU < 0.3): {len(failures_03)} queries")
    print(f"⚠️  Failures (IoU < 0.5): {len(failures_05)} queries")
    print()
    
    # Show worst 10 predictions
    print("🔍 Worst 10 Predictions (by IoU):")
    for d in diagnostics[:10]:
        print(f"  {d['clip_uid']}[q{d['query_idx']}]: "
              f"GT={d['gt']}, Pred={d['pred_top1']}, IoU={d['iou_top1']:.3f}")
    
    print("=" * 60)

if __name__ == '__main__':
    eval_result = evaluate_predictions(
        os.path.join(SSD_BASE, "submissions/submission_ego4d_goalstep.json"),
        os.path.join(SSD_BASE, "annotations/goalstep_val.json")
    )
    print_evaluation_report(eval_result)
    
    # Save full diagnostics for analysis
    diag_path = os.path.join(SSD_BASE, "submissions/val_diagnostics.json")
    with open(diag_path, 'w') as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nFull diagnostics saved to {diag_path}")
```

## Verification Strategy
- **Known-Answer Test:** Create a mock prediction file where `predicted_times[0] == ground_truth` exactly. Assert r@1 IoU=0.3 and r@1 IoU=0.5 both equal 100%.
- **Zero-Overlap Test:** Create predictions that don't overlap with any GT. Assert all metrics are 0%.
- **IoU Edge Cases:** Test overlapping, containing, and adjacent segments to verify the IoU computation handles boundary conditions.
