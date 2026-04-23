# Ego4D Goal-Step Competition: Master Implementation Plan

This document outlines our strategy for the **Ego4D Goal-Step Step Grounding** challenge on [CodaBench](https://www.codabench.org/competitions/14878/). The competition deadline is **May 14, 2026**.

## Competition Task

**Step Grounding:** Given an untrimmed egocentric video and a natural language description of a step (keystep), predict the temporal segment `(start_time, end_time)` corresponding to that step.

## Dataset

| Item | Detail |
|---|---|
| **Dataset** | Ego4D Goal-Step (NeurIPS 2023) |
| **Scale** | 2,724 hours of video with goal labels; 422 hours with fine-grained step/substep labels |
| **Hierarchy** | Goal → Step → Substep (train/val); flat `step_segments` (test) |
| **Splits** | `goalstep_train.json`, `goalstep_valid.json`, `goalstep_test.json` |
| **Features** | Pre-extracted Omnivore video features (SwinL backbone) |
| **Access** | [ego4d-data.org](https://ego4d-data.org/) via Ego4D CLI |

> [!IMPORTANT]
> **Data Storage:** All data stored on external SSD at `/Volumes/Extreme SSD/ego4d_data/`.

## Evaluation Metrics

Evaluated on **top-1 and top-5 predicted temporal windows** against ground truth:

| Metric | Description | Role |
|---|---|---|
| **r@1, IoU=0.3** | Recall at 1, IoU threshold 0.3 | ✅ **Primary** (leaderboard ranking) |
| **r@1, IoU=0.5** | Recall at 1, IoU threshold 0.5 | Tiebreaker |
| **r@5, IoU=0.3** | Recall at 5, IoU threshold 0.3 | Secondary |
| **r@5, IoU=0.5** | Recall at 5, IoU threshold 0.5 | Secondary |
| **Mean r@1** | Average of r@1 at IoU=0.3 and IoU=0.5 | Secondary |

## Submission Format

Submit a `.zip` containing a JSON file:
```json
{
  "version": "1.0",
  "challenge": "ego4d_goalstep_challenge",
  "results": [
    {
      "clip_uid": "<video_uid>",
      "annotation_uid": "<video_uid>",
      "query_idx": 0,
      "predicted_times": [
        [start_1, end_1],
        [start_2, end_2],
        [start_3, end_3],
        [start_4, end_4],
        [start_5, end_5]
      ]
    }
  ]
}
```

> [!WARNING]
> Each query **must** have exactly **5 predicted temporal windows** `[start, end]`, ranked by confidence (most confident first). `clip_uid` and `annotation_uid` should both be set to `video_uid` from the test annotations. `query_idx` is the index of the keystep within the `step_segments` list.

## Previous Leaderboard (EvalAI)

| Rank | Team | r@1, IoU=0.3 | r@1, IoU=0.5 |
|------|------|--------------|--------------|
| 1 | iLearn2.0 | **42.02** | 32.83 |
| 2 | CarLor (BayesianVSLNet) | 35.18 | 20.49 |
| 3 | 123456ABCD (our_clip) | 34.06 | 26.97 |
| 4 | iLearn | 33.00 | 26.37 |

## Strategy: Hybrid Approach

**Phase A** — Get the official VSLNet/NaQ baseline running and producing a valid submission.
**Phase B** — Layer our ToC innovation (acoustic triggers + VLM captioning + LLM reasoning) on top as a pre-filter to narrow the search window for VSLNet.

## Alternative Strategy: Late-Fusion Ensemble (To try later)

If the Hybrid Approach (ToC as a hard pre-filter) suffers from cascading errors where the Librarian drops accuracy, we will pivot to a **Late-Fusion Ensemble**:
1. Run the baseline VSLNet on the **full video** to get raw probability distributions (logits) for start/end frames.
2. Run the ToC Pipeline to generate the Librarian's proposed temporal window.
3. Instead of cropping the video, use the Librarian's window as a **Gaussian prior/mask** over the VSLNet logits.
   - If the Librarian is right, the correct peak is amplified, suppressing distractors.
   - If the Librarian is wrong or uncertain, we fall back gracefully to the raw VSLNet predictions.
This prevents the ToC pipeline from becoming a single point of failure.

> [!IMPORTANT]
> **Memory Budget (24GB M4 Pro):** Models must be loaded sequentially. Qwen2.5-VL-3B (~2GB) → unload → Gemma 4 26B (~14GB) → unload → VSLNet (~4GB). Explicit `del` + `gc.collect()` between phases.

---

## Phase 0: Environment Setup ✅
- Python venv with `mlx`, `librosa`, `torch`, `opencv-python`, `transformers`.
- SSD routing to `/Volumes/Extreme SSD/ego4d_data/`.
- VLM (Qwen2.5-VL-3B) verified.

## Phase 1: Data Acquisition & Annotation Parsing
- Download videos, annotations, and pre-extracted features via Ego4D CLI.
- Parse train/val/test annotations (note: test uses different schema).
- Build unified query index and video UID lists.

## Phase 1.5: Official Baseline Setup (NEW — Critical Path)
- Clone `facebookresearch/ego4d-goalstep` and `srama2512/NaQ`.
- Convert goalstep annotations to NaQ format.
- Aggregate Omnivore features for GoalStep videos.
- Train VSLNet on GoalStep (may require cloud GPU).
- Run inference on test set → produce baseline predictions.

## Phase 2: Event Annotation Pass (ToC Enhancement)
- Uniform frame sampling + VLM captioning → dense event log per video.
- Only run on videos we're actually evaluating.

## Phase 3: Acoustic Characterization
- Global noise floor calculation, stationary noise masking.

## Phase 4: Adaptive Sentry (Acoustic Triggers)
- Bidirectional magnitude change detection → "Interest Zones".

## Phase 5: Visual Captioning & ToC Construction
- Trigger-aware captions + event annotations → structured Table of Contents.

## Phase 6: The Librarian (LLM Reasoning)
- Gemma 4 26B maps target step queries to ToC chapters.
- Output: narrowed temporal windows for each query.

## Phase 7: Focused Grounding
- Run VSLNet on the narrowed windows from Phase 6 instead of full video.
- Compare efficiency and accuracy vs. full-video baseline.

## Phase 8: Submission Formatting
- Format predictions into CodaBench JSON schema.
- Ensure exactly 5 ranked predictions per query.
- Package as `.zip` and submit.

## Phase 9: Local Evaluation
- Compute r@1/r@5 at IoU=0.3/0.5 on validation split.
- Compare against leaderboard baselines.
- Iterate on model/pipeline improvements.

## Phase 10: Final Submission & Report
- Submit best model predictions to CodaBench.
- Write validation report (up to 4 pages) describing method.
