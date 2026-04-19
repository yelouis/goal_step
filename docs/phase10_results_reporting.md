# Phase 10: Results & Reporting

## Overview
The final phase aggregates the comparative evaluation results from Phase 9 into publication-ready tables, charts, and analysis. Since this project is a research study demonstrating the efficiency of the ToC approach (not a competition submission), the output is a structured report rather than a CodaBench JSON payload.

## Step 10.1: Aggregate Results Table

Build a comprehensive comparison table across all three datasets:

```
| Dataset              | Method              | r@1 IoU=0.3 | r@1 IoU=0.5 | Speed-Up | Avg Features |
|----------------------|---------------------|-------------|-------------|----------|--------------|
| EPIC-KITCHENS-100    | Baseline (Full)     |    XX.X%    |    XX.X%    |   1.0×   |     XXXX     |
| EPIC-KITCHENS-100    | ToC Pipeline        |    XX.X%    |    XX.X%    |   X.X×   |     XXX      |
| Charades-Ego         | Baseline (Full)     |    XX.X%    |    XX.X%    |   1.0×   |     XXXX     |
| Charades-Ego         | ToC Pipeline        |    XX.X%    |    XX.X%    |   X.X×   |     XXX      |
| EgoProceL            | Baseline (Full)     |    XX.X%    |    XX.X%    |   1.0×   |     XXXX     |
| EgoProceL            | ToC Pipeline        |    XX.X%    |    XX.X%    |   X.X×   |     XXX      |
```

## Step 10.2: Efficiency Analysis

Compute and visualize:
- **Mean/Median Speed-Up Ratios** across all queries and datasets
- **Speed-Up Distribution:** Histogram of per-query speed-up ratios (are some queries much faster? are outliers dragging the average?)
- **Accuracy vs. Speed-Up Scatter:** Plot per-query accuracy (IoU) against speed-up ratio to identify trade-off regions
- **Video Length Analysis:** Does the speed-up improve for longer videos? (Expected: yes, since longer videos = more features saved)

## Step 10.3: Ablation Study

Run the ablation configurations defined in Phase 9 and report results:

| Configuration        | r@1 IoU=0.3 | Speed-Up | Contribution |
|----------------------|-------------|----------|--------------|
| Full ToC Pipeline    |    XX.X%    |   X.X×   | —            |
| No Event Annotation  |    XX.X%    |   X.X×   | Phase 2      |
| No Acoustic Triggers |    XX.X%    |   X.X×   | Phases 3–4   |
| No Librarian         |    XX.X%    |   X.X×   | Phase 6      |
| Baseline (Full Video)|    XX.X%    |   1.0×   | —            |

## Step 10.4: Failure Analysis Summary

From Phase 9's worst-performing queries, categorize failure modes:
- **Librarian Mis-Mapping:** ToC exists but the Librarian selected the wrong chapter
- **Acoustic Blind Spot:** Silent step with no acoustic trigger and weak event annotation
- **Caption Hallucination:** VLM produced inaccurate captions that misled the Librarian
- **Feature Misalignment:** Cross-dataset feature backbone mismatch
- **Short/Ambiguous Steps:** Steps too brief or vaguely described for accurate grounding

## Step 10.5: Export & Documentation

```python
import json
import os

SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"

def generate_report(comparison: dict, ablation: dict = None):
    """
    Generate the final structured report.
    
    Outputs:
    - results/final_report.json — Machine-readable full results
    - results/summary_table.md — Human-readable markdown table
    """
    report = {
        "project": "ToC-Accelerated Step Grounding",
        "datasets": ["EPIC-KITCHENS-100", "Charades-Ego", "EgoProceL"],
        "comparison": comparison,
        "ablation": ablation or {},
        "hardware": {
            "chip": "Apple M4 Pro",
            "memory": "24GB Unified",
            "storage": "2TB External SSD",
            "models": {
                "vlm": "Qwen2.5-VL-3B-Instruct (4-bit, ~2GB)",
                "llm": "Gemma 4 26B MoE (4-bit, ~14GB)",
                "grounding": "BayesianVSLNet (~4-8GB)"
            }
        }
    }
    
    out_dir = os.path.join(SSD_BASE, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    # JSON report
    with open(os.path.join(out_dir, "final_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Markdown summary
    md_lines = ["# ToC-Accelerated Step Grounding: Results Summary\n"]
    md_lines.append("## Comparison: ToC Pipeline vs. Full-Video Baseline\n")
    md_lines.append("| Dataset | Method | r@1 IoU=0.3 | r@1 IoU=0.5 | Speed-Up | Avg Features |")
    md_lines.append("|---|---|---|---|---|---|")
    
    for ds, comp in comparison.items():
        toc = comp.get("toc", {})
        base = comp.get("baseline", {})
        
        base_r1_03 = base.get("metrics", {}).get("r@1_iou0.3", "—")
        base_r1_05 = base.get("metrics", {}).get("r@1_iou0.5", "—")
        base_feats = base.get("avg_features_per_query", "—")
        
        toc_r1_03 = toc.get("metrics", {}).get("r@1_iou0.3", "—")
        toc_r1_05 = toc.get("metrics", {}).get("r@1_iou0.5", "—")
        toc_feats = toc.get("avg_features_per_query", "—")
        speed_up = comp.get("speed_up_ratio", "—")
        
        md_lines.append(f"| {ds} | Baseline | {base_r1_03}% | {base_r1_05}% | 1.0× | {base_feats} |")
        md_lines.append(f"| {ds} | **ToC Pipeline** | {toc_r1_03}% | {toc_r1_05}% | **{speed_up}×** | {toc_feats} |")
    
    md_content = "\n".join(md_lines)
    with open(os.path.join(out_dir, "summary_table.md"), 'w') as f:
        f.write(md_content)
    
    print(f"Report exported to {out_dir}/")
    print(f"  - final_report.json")
    print(f"  - summary_table.md")


if __name__ == '__main__':
    # Load comparison from Phase 9
    report_path = os.path.join(SSD_BASE, "results/comparison_report.json")
    with open(report_path, 'r') as f:
        comparison = json.load(f)
    
    generate_report(comparison)
```

## Verification Strategy
- **Schema Validation:** Assert that `final_report.json` contains all required keys (`comparison`, `hardware`, `datasets`).
- **Markdown Rendering:** Open `summary_table.md` in a markdown viewer and verify the table renders correctly with all cells populated.
- **Completeness Check:** Assert that results exist for all three datasets in the comparison table. If any dataset is missing, log a warning but don't fail.
