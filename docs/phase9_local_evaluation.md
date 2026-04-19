# Phase 9: Local Evaluation & Comparative Analysis

## Overview
This phase is the heart of the research contribution. We run **both** the ToC pipeline (Phases 2–7) and the full-video baseline (Phase 8) on each dataset's validation split, then compare them on accuracy and efficiency. The goal is to demonstrate that the ToC approach achieves competitive accuracy while processing significantly fewer features.

> [!IMPORTANT]
> **This is not just an evaluation phase — it's the primary experimental result.** The comparison between the ToC pipeline and the full-video baseline is what validates (or invalidates) the core hypothesis: that structured ToC narrowing is more efficient than brute-force temporal grounding.

## Official Metrics
We use standard temporal grounding metrics across all three datasets:

| Metric | Description | Primary? |
|---|---|---|
| **r@1, IoU=0.3** | % of queries where the top-1 prediction has IoU ≥ 0.3 with ground truth | ✅ Primary |
| **r@1, IoU=0.5** | % of queries where the top-1 prediction has IoU ≥ 0.5 with ground truth | Tie-breaker |
| **r@5, IoU=0.3** | % of queries where any of top-5 predictions has IoU ≥ 0.3 | Secondary |
| **r@5, IoU=0.5** | % of queries where any of top-5 predictions has IoU ≥ 0.5 | Secondary |

## Efficiency Metrics

| Metric | Description |
|---|---|
| **Features Processed** | Total feature vectors fed to BayesianVSLNet (ToC vs. Baseline) |
| **Speed-Up Ratio** | `baseline_features / toc_features` per query |
| **Wall-Clock Time** | End-to-end inference time per query (seconds) |
| **Accuracy Delta** | `toc_r@1 - baseline_r@1` (target: ≥ 0, i.e., no accuracy loss) |

## Pseudocode Implementation

```python
import json
import os
import numpy as np
from collections import defaultdict

SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"

def compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute Intersection over Union between predicted and ground truth temporal segments."""
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    intersection = max(0, intersection_end - intersection_start)
    
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def evaluate_pipeline(predictions: list, method_name: str) -> dict:
    """
    Evaluate a set of predictions against ground truth.
    
    Each prediction dict must contain:
    - predicted_start, predicted_end
    - gt_start, gt_end
    - features_processed
    - wall_clock_sec
    - dataset_source (for per-dataset breakdown)
    """
    iou_thresholds = [0.3, 0.5]
    k_values = [1, 5]
    
    # Per-dataset results
    dataset_results = defaultdict(lambda: {
        "total_queries": 0,
        "metrics": {f"r@{k}_iou{t}": 0 for k in k_values for t in iou_thresholds},
        "total_features": 0,
        "total_wall_time": 0,
        "ious": []
    })
    
    for pred in predictions:
        ds = pred.get("dataset_source", "unknown")
        r = dataset_results[ds]
        r["total_queries"] += 1
        r["total_features"] += pred.get("features_processed", 0)
        r["total_wall_time"] += pred.get("wall_clock_sec", 0)
        
        iou = compute_iou(
            pred["predicted_start"], pred["predicted_end"],
            pred["gt_start"], pred["gt_end"]
        )
        r["ious"].append(iou)
        
        # r@1 check
        for t in iou_thresholds:
            if iou >= t:
                r["metrics"][f"r@1_iou{t}"] += 1
                r["metrics"][f"r@5_iou{t}"] += 1  # If r@1 hits, r@5 auto-hits
            # TODO: handle top-5 predictions when we have multiple candidates
    
    # Compute percentages
    summary = {}
    for ds, r in dataset_results.items():
        n = r["total_queries"]
        if n > 0:
            metrics_pct = {k: round(v / n * 100, 2) for k, v in r["metrics"].items()}
            summary[ds] = {
                "method": method_name,
                "total_queries": n,
                "metrics": metrics_pct,
                "mean_iou": round(np.mean(r["ious"]), 4),
                "median_iou": round(np.median(r["ious"]), 4),
                "total_features_processed": r["total_features"],
                "avg_features_per_query": round(r["total_features"] / n, 1),
                "total_wall_time_sec": round(r["total_wall_time"], 2),
                "avg_wall_time_per_query_sec": round(r["total_wall_time"] / n, 4)
            }
    
    return summary


def compare_pipelines(toc_predictions: list, baseline_predictions: list) -> dict:
    """
    Compare ToC pipeline vs. full-video baseline across all datasets.
    
    Returns a structured comparison report.
    """
    toc_summary = evaluate_pipeline(toc_predictions, "toc_pipeline")
    baseline_summary = evaluate_pipeline(baseline_predictions, "baseline_full_video")
    
    comparison = {}
    all_datasets = set(list(toc_summary.keys()) + list(baseline_summary.keys()))
    
    for ds in all_datasets:
        toc = toc_summary.get(ds, {})
        base = baseline_summary.get(ds, {})
        
        if not toc or not base:
            continue
        
        # Speed-up ratio
        toc_feats = toc.get("avg_features_per_query", 1)
        base_feats = base.get("avg_features_per_query", 1)
        speed_up = round(base_feats / toc_feats, 2) if toc_feats > 0 else float('inf')
        
        # Accuracy delta
        toc_r1 = toc.get("metrics", {}).get("r@1_iou0.3", 0)
        base_r1 = base.get("metrics", {}).get("r@1_iou0.3", 0)
        accuracy_delta = round(toc_r1 - base_r1, 2)
        
        comparison[ds] = {
            "toc": toc,
            "baseline": base,
            "speed_up_ratio": speed_up,
            "accuracy_delta_r1_03": accuracy_delta,
            "wall_time_speed_up": round(
                base.get("avg_wall_time_per_query_sec", 1) / 
                max(toc.get("avg_wall_time_per_query_sec", 1), 0.001), 2
            )
        }
    
    return comparison


def print_comparison_report(comparison: dict):
    """Pretty-print the comparative evaluation results."""
    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION REPORT: ToC Pipeline vs. Full-Video Baseline")
    print("=" * 80)
    
    for ds, comp in comparison.items():
        toc = comp["toc"]
        base = comp["baseline"]
        
        print(f"\n📊 Dataset: {ds}")
        print(f"   Queries evaluated: {toc['total_queries']}")
        print()
        
        print(f"   {'Metric':<20} {'ToC Pipeline':>15} {'Baseline':>15} {'Delta':>10}")
        print(f"   {'─'*20} {'─'*15} {'─'*15} {'─'*10}")
        
        for metric in ["r@1_iou0.3", "r@1_iou0.5", "r@5_iou0.3", "r@5_iou0.5"]:
            toc_val = toc["metrics"].get(metric, 0)
            base_val = base["metrics"].get(metric, 0)
            delta = round(toc_val - base_val, 2)
            marker = "← PRIMARY" if metric == "r@1_iou0.3" else ""
            print(f"   {metric:<20} {toc_val:>14.1f}% {base_val:>14.1f}% {delta:>+9.1f}%  {marker}")
        
        print()
        print(f"   Avg features/query: {toc['avg_features_per_query']:>8.0f} (ToC)  vs  {base['avg_features_per_query']:>8.0f} (Baseline)")
        print(f"   Speed-Up Ratio:     {comp['speed_up_ratio']}×")
        print(f"   Wall-Time Speed-Up: {comp['wall_time_speed_up']}×")
        print(f"   Mean IoU:           {toc['mean_iou']:.4f} (ToC)  vs  {base['mean_iou']:.4f} (Baseline)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Load predictions from both pipelines
    toc_path = os.path.join(SSD_BASE, "results/toc_predictions.json")
    baseline_path = os.path.join(SSD_BASE, "results/baseline_predictions.json")
    
    with open(toc_path, 'r') as f:
        toc_preds = json.load(f)
    with open(baseline_path, 'r') as f:
        baseline_preds = json.load(f)
    
    comparison = compare_pipelines(toc_preds, baseline_preds)
    print_comparison_report(comparison)
    
    # Save full comparison
    report_path = os.path.join(SSD_BASE, "results/comparison_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nFull comparison report saved to {report_path}")
```

## Cross-Dataset Analysis

Each dataset exercises different aspects of the pipeline:

| Dataset | Expected Strengths | Expected Challenges |
|---|---|---|
| **EPIC-KITCHENS-100** | Strong acoustic signals (kitchen sounds); rich narrations | Long videos; many overlapping actions |
| **Charades-Ego** | Scripted activities; clear action boundaries | Generic class labels (not natural language) |
| **EgoProceL** | Procedural structure; clear step ordering | Composite dataset; variable video quality |

## Ablation Study Design

To quantify the contribution of each pipeline component, run these ablation configurations:

| Configuration | Description | What it measures |
|---|---|---|
| **Full ToC** | Phases 2 + 3 + 4 + 5 + 6 + 7 | Complete pipeline performance |
| **No Event Annotation** | Skip Phase 2 | Value of dense event pre-annotation |
| **No Acoustic** | Skip Phases 3–4 (only event samples) | Value of acoustic trigger detection |
| **No Librarian** | Random chapter selection in Phase 6 | Value of LLM-based reasoning |
| **Baseline** | Phase 8 only | Cost of no ToC narrowing |

## Verification Strategy
- **Known-Answer Test:** Create a mock prediction file where `predicted_times == ground_truth` exactly. Assert r@1 IoU=0.3 and r@1 IoU=0.5 both equal 100%.
- **Zero-Overlap Test:** Create predictions that don't overlap with any GT. Assert all metrics are 0%.
- **IoU Edge Cases:** Test overlapping, containing, and adjacent segments to verify the IoU computation handles boundary conditions.
- **Speed-Up Sanity:** Assert that `speed_up_ratio >= 1.0` for all datasets (ToC should always process fewer features than baseline).
