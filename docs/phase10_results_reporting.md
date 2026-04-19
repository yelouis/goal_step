# Phase 10: Results, Reporting & HuggingFace Upload

## Overview
The final phase aggregates the comparative evaluation results from Phase 9 into publication-ready tables, charts, and analysis, then **uploads the complete findings to HuggingFace** as a public dataset + model card. Since this project is a research study demonstrating the efficiency of the ToC approach, the end goal is a reproducible, open-access artifact on HuggingFace Hub.

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
    
    return report
```

## Step 10.6: HuggingFace Upload

> [!IMPORTANT]
> **This is the terminal deliverable of the entire project.** The HuggingFace upload makes the research findings publicly reproducible and citable.

### What Gets Uploaded

| Artifact | HuggingFace Type | Contents |
|---|---|---|
| **Dataset** | `datasets` repo | Per-query predictions, ground truth, efficiency metrics, ablation results |
| **Model Card** | README.md in dataset repo | Full methodology, results tables, hardware specs, reproduction instructions |
| **Plots** | Embedded in Model Card | Speed-up distributions, accuracy vs. efficiency scatter |

### Upload Pseudocode

```python
import json
import os
from huggingface_hub import HfApi, create_repo

SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"
HF_REPO_ID = "yelouis/toc-step-grounding"  # Adjust to your HF username

def upload_to_huggingface():
    """Upload all findings to HuggingFace as a dataset repository."""
    api = HfApi()
    
    # 1. Create the repo (or confirm it exists)
    create_repo(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False
    )
    
    results_dir = os.path.join(SSD_BASE, "results")
    
    # 2. Upload the machine-readable results
    api.upload_file(
        path_or_fileobj=os.path.join(results_dir, "final_report.json"),
        path_in_repo="final_report.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    print("✅ Uploaded final_report.json")
    
    # 3. Upload per-pipeline predictions (for reproducibility)
    for pred_file in ["toc_predictions.json", "baseline_predictions.json", "comparison_report.json"]:
        filepath = os.path.join(results_dir, pred_file)
        if os.path.exists(filepath):
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"predictions/{pred_file}",
                repo_id=HF_REPO_ID,
                repo_type="dataset"
            )
            print(f"✅ Uploaded {pred_file}")
    
    # 4. Upload the summary table
    api.upload_file(
        path_or_fileobj=os.path.join(results_dir, "summary_table.md"),
        path_in_repo="summary_table.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    print("✅ Uploaded summary_table.md")
    
    # 5. Generate and upload the Dataset Card (README.md)
    readme_content = generate_dataset_card()
    readme_path = os.path.join(results_dir, "README_hf.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    print("✅ Uploaded README.md (Dataset Card)")
    
    # 6. Upload plots (if generated)
    plots_dir = os.path.join(results_dir, "plots")
    if os.path.exists(plots_dir):
        api.upload_folder(
            folder_path=plots_dir,
            path_in_repo="plots",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print("✅ Uploaded plots/")
    
    print(f"\n🎉 All findings uploaded to: https://huggingface.co/datasets/{HF_REPO_ID}")


def generate_dataset_card() -> str:
    """Generate a HuggingFace-compatible Dataset Card (README.md)."""
    
    # Load the final report for dynamic content
    report_path = os.path.join(SSD_BASE, "results/final_report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    card = """---
language:
- en
license: mit
task_categories:
- video-text-to-text
tags:
- temporal-grounding
- egocentric-video
- step-localization
- bayesian-inference
- table-of-contents
pretty_name: ToC-Accelerated Step Grounding Results
size_categories:
- 1K<n<10K
---

# ToC-Accelerated Step Grounding: Results & Findings

## Summary
This dataset contains the complete experimental results for the **Table-of-Contents (ToC) Accelerated Step Grounding** research project. We demonstrate that a structured ToC approach to temporal step grounding in egocentric videos is significantly more efficient than running BayesianVSLNet on entire videos, while maintaining competitive accuracy.

## Methodology
The ToC pipeline processes egocentric videos through:
1. **Event Annotation** (Qwen2.5-VL-3B) — dense VLM captioning
2. **Acoustic Characterization** — audio-based activity segmentation  
3. **Adaptive Sentry** — bidirectional magnitude trigger detection
4. **Visual Captioning** — trigger-aware frame captioning → ToC construction
5. **Librarian** (Gemma 4 26B MoE) — LLM-based chapter selection
6. **Focused Bayesian Grounding** (BayesianVSLNet) — precise temporal localization within selected chapters

## Datasets Evaluated
- **EPIC-KITCHENS-100** — Kitchen activities
- **Charades-Ego** — Daily indoor activities  
- **EgoProceL** — Procedural tasks

## Hardware
- Apple M4 Pro (24GB Unified Memory)
- 2TB External SSD for data storage
- All inference runs locally (no cloud GPU)

## Files
- `final_report.json` — Complete machine-readable results
- `predictions/` — Per-query predictions from both pipelines
- `summary_table.md` — Human-readable results table
- `plots/` — Visualizations (speed-up distributions, accuracy scatter)

## Citation
If you use these results, please cite this repository.
"""
    return card


if __name__ == '__main__':
    # Load comparison from Phase 9
    report_path = os.path.join(SSD_BASE, "results/comparison_report.json")
    with open(report_path, 'r') as f:
        comparison = json.load(f)
    
    # Step 10.5: Generate local report
    report = generate_report(comparison)
    
    # Step 10.6: Upload everything to HuggingFace
    upload_to_huggingface()
```

## Session Resilience

| Item | Detail |
|---|---|
| **Input Dependencies** | `results/comparison_report.json` (Phase 9), `results/toc_predictions.json`, `results/baseline_predictions.json` |
| **Output Artifacts** | `/Volumes/Extreme SSD/goal_step_data/results/final_report.json`, `results/summary_table.md`, `results/README_hf.md` |
| **HuggingFace Artifact** | `https://huggingface.co/datasets/yelouis/toc-step-grounding` |
| **Cache Check** | On entry, check if `final_report.json` exists. If HF upload already succeeded (check via `api.repo_info()`), skip upload |
| **Verification Checkpoint** | After upload, write `results/_hf_upload_manifest.json` with upload timestamps and file checksums |
| **Resume Strategy** | `generate_report()` is idempotent. `upload_to_huggingface()` uses `exist_ok=True` for the repo and overwrites files on re-upload |

## Verification Strategy
- **Schema Validation:** Assert that `final_report.json` contains all required keys (`comparison`, `hardware`, `datasets`).
- **Markdown Rendering:** Open `summary_table.md` in a markdown viewer and verify the table renders correctly with all cells populated.
- **Completeness Check:** Assert that results exist for all three datasets in the comparison table. If any dataset is missing, log a warning but don't fail.
- **HuggingFace Validation:** After upload, use `api.repo_info(HF_REPO_ID, repo_type="dataset")` to verify the repo exists and contains the expected files.
- **Dataset Card Check:** Verify the README.md renders correctly on the HuggingFace web UI with proper YAML frontmatter.
