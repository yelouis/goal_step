# Egocentric Step Grounding: Master Implementation Plan

This document outlines the architecture and execution strategy for the **ToC-Accelerated Step Grounding** research project. The goal is to demonstrate that a Table-of-Contents (ToC) approach to temporal step grounding is significantly more efficient than running Bayesian Grounding on the entire video, while maintaining competitive accuracy.

## Datasets

We evaluate on three publicly available egocentric video benchmarks:

| Dataset | Domain | Annotation Style | Access |
|---|---|---|---|
| **EPIC-KITCHENS-100** | Kitchen activities | Narrated action segments (verb + noun) with start/end times | [epic-kitchens.github.io](https://epic-kitchens.github.io/2024) |
| **Charades-Ego** | Daily indoor activities | Activity classes with temporal intervals; paired ego/third-person | [allenai.org/plato/charades](http://allenai.org/plato/charades/) |
| **EgoProceL** | Procedural tasks (cooking, assembly) | Key-step annotations with start/end seconds and step labels (CSV) | [sid2697.github.io/egoprocel](https://sid2697.github.io/egoprocel/) |

> [!IMPORTANT]
> **Data Storage Requirement:**
> All raw video, audio extractions, cached features, and local MLX models should be persistently stored on the 2TB external SSD. The target base directory is `/Volumes/Extreme SSD/goal_step_data/`.

> [!WARNING]
> **Memory Budget (24GB M4 Pro):**
> Models must be loaded sequentially, not concurrently. Qwen2.5-VL-3B-Instruct-4bit (~2GB) → unload → Gemma 4 26B (~14GB) → unload → BayesianVSLNet (~4-8GB). Explicit `del` + `gc.collect()` between phases.

## Phase 0: Installation and Model Verification
- Set up base package dependencies (`mlx`, `librosa`, `torch`).
- Run a baseline environment check to verify inference hardware (Apple Silicon unified memory constraints) and ensure models are caching correctly to `/Volumes/Extreme SSD/`.

## Phase 1: Data Download & Annotation Parsing
- Download EPIC-KITCHENS-100, Charades-Ego, and EgoProceL datasets and their annotation files.
- Parse each dataset's native annotation format into a **unified internal schema** so downstream phases are dataset-agnostic.
- Extract `video_id`, `dataset_source`, `query_idx`, `step_description`, and `goal_description` for every query.
- Verify data integrity (all referenced videos exist on SSD and are readable).

## Phase 2: Event Annotation Pass (What Happened?)
Since EPIC-KITCHENS-100, Charades-Ego, and EgoProceL do not have the same hierarchical `Goal → Step → Substep` annotations that Ego4D provided, we must first produce a dense event listing for each video before the ToC pipeline can operate.

- **Step 2.1: Uniform Temporal Sampling:** Sample frames at a fixed interval (e.g., every 2–3 seconds) across the entire video.
- **Step 2.2: VLM Event Captioning:** Pass each sampled frame through **Qwen2.5-VL-3B-Instruct-4bit** with a neutral prompt: *"Describe exactly what the person is doing in this frame. Be specific about objects and actions."*
- **Step 2.3: Event Deduplication & Merging:** Consecutive frames with semantically identical captions are merged into a single event span with `(start_time, end_time, caption)`.
- **Step 2.4: Event Log Export:** Store the dense event log as `{video_id}_events.json` on the SSD. This log serves as the raw input for the Acoustic Characterization and Adaptive Sentry passes.

> [!NOTE]
> **Relationship to Phase 5 ToC:** The event log from Phase 2 provides a coarse "what happened" overview. Phases 3–5 then refine this into a structured ToC by applying acoustic analysis to discover more precise trigger points and generate higher-quality, trigger-aware captions. Phase 2 ensures we have baseline coverage even for acoustically silent steps.

## Phase 3: The Acoustic Characterization Pass (Global Scan)
- **Step 3.1: Global Baseline Calculation:** Calculate the **Global Noise Floor ($N_g$)** and the **Median Absolute Deviation (MAD)** of the energy envelope.
- **Step 3.2: Stationary Noise Masking:** Identify persistent frequency bands (e.g., HVAC hum) to create a spectral mask, ensuring the trigger logic isn't distracted by constant environmental noise.
- **Step 3.3: Pattern Discovery:** Map "Active vs. Passive" acoustic regions. If a specific volume level remains constant for long durations, it is flagged as a "baseline" for that segment of the video.

## Phase 4: The Adaptive Sentry (Bidirectional Magnitude Triggers)
This stage identifies "Interest Zones" for visual captioning by detecting significant shifts in the acoustic state.

- **Step 4.1: Bidirectional Change Detection:** Trigger a **Qwen2.5-VL-3B** capture if the Short-Term Average Energy ($E_{sta}$) deviates significantly from the Long-Term Average Energy ($E_{lta}$):
  - **The Spike:** Trigger if $E_{sta} / E_{lta} > k_{up}$ (e.g., $k_{up} = 2.5$).
  - **The Drop:** Trigger if $E_{sta} / E_{lta} < k_{down}$ (e.g., $k_{down} = 0.3$). This captures the exact moment a noisy action ceases.
- **Step 4.2: Semantic Rate Limiting:**
  - **Temporal Masking:** After a trigger, initiate a 3-second "Cool-down" to prevent redundant captions during stuttering noises.
  - **Saturation Logic:** If 3 captions are triggered within 15 seconds, exponentially increase the trigger thresholds ($k_{up}$ and $k_{down}$) with per-trigger decay to gradually relax back to baseline.
- **Step 4.3: The Silence Watcher (Forced Detection):** If no audio trigger (spike or drop) occurs for a period of $T_{force}$ (set between 30s and 60s), force a visual caption to ensure silent, steady-state actions are indexed.

> **Known Limitation:** Audio-only triggers are unreliable for acoustically silent procedural steps. A visual change-detection fallback (frame differencing / optical flow) is documented as a future enhancement.

## Phase 5: Visual Captioning & Table of Contents (ToC)
- **Step 5.1: Trigger-Aware Captioning:** For every trigger, pass the frame to **Qwen2.5-VL-3B** (via `mlx_vlm` with `apply_chat_template`) using a prompt adapted to the trigger type:
  - `spike` → "What action is the user starting?"
  - `drop` → "What action did the user just finish?"
  - `forced_silence` → "What is the user currently doing?"
- **Step 5.2: ToC Construction:** Merge Phase 2 event annotations with Phase 4 trigger-aware captions. Store as a JSON index on the SSD. Each entry includes:
  - `timestamp`: $[s]$
  - `trigger_type`: ["spike", "drop", "forced_silence", "event_sample"]
  - `caption`: "User is assembling the metal bracket."
- **Step 5.3: Caption Quality Filtering:** Retry with a fallback prompt if the caption is too short or generic.

> **Caching:** Phases 2–5 run **once per video** regardless of how many queries target that video.

## Phase 6: The Librarian (Gemma 4 26B MoE Reasoning)
### The Librarian System Prompt
> You are the "Procedural Video Librarian." Map a "Target Step" to the most likely "Chapters" within a video's Table of Contents (ToC).
> 
> **Context:** Receives the video's goal category, goal description, and full ToC.
> 
> **Logical Rules:**
> 1. **Explicit Identification:** Prioritize chapters where the caption matches the Target Step.
> 2. **Silent-Step Interpolation:** If the step is missing, identify the **Pre-Condition** and **Post-Condition**. The Target Step is located in the temporal gap between them.
> 3. **Change Detection Logic:** Treat "drop" triggers as potential "Step Completion" markers and "spike" triggers as "Step Start" markers.
> 4. **Temporal Ordering:** Use the goal description to reason about step ordering.
> 5. **Output Format:** Return a JSON object with `"top_chapters"`, `"confidence"`, and `"reasoning"`.
> 
> **Robustness:** JSON extraction uses balanced-brace parsing with retry logic (up to 3 attempts with increasing temperature).

## Phase 7: Focused Bayesian Grounding (High-Res Pass — ToC Pipeline)
- **Step 7.1: Windowed Feature Extraction:** Extract visual features **only** for the chapters identified by the Librarian. Use dataset-native features when available (e.g., EPIC-KITCHENS slowfast features), otherwise extract on-the-fly with a lightweight backbone.
- **Step 7.2: Bayesian Head Inference:** Run the **BayesianVSLNet** head on this specific slice, using text features from the step description query.
- **Step 7.3: Timestamp Refinement:** Use the posterior distribution with non-causal temporal-order prior to calculate the final `(start, end)` timestamps.

> **Feature Alignment:** BayesianVSLNet expects aligned visual and text feature dimensions. When switching datasets, verify that the feature pipeline produces the expected concatenated vector size.

## Phase 8: Full-Video Bayesian Baseline (Control Condition)
To quantify the efficiency gain of the ToC approach, we run BayesianVSLNet on the **entire video** without any ToC narrowing.

- **Step 8.1: Full Feature Loading:** Extract or load visual features for the complete video duration.
- **Step 8.2: Full-Video Inference:** Run BayesianVSLNet on all features with the same text query.
- **Step 8.3: Baseline Timestamps:** Record the predicted `(start, end)` timestamps as the baseline prediction.
- **Step 8.4: Compute Cost Logging:** Log wall-clock time, total features processed, and peak memory usage for direct comparison with Phase 7.

> [!IMPORTANT]
> **This phase exists solely as a comparison baseline.** It demonstrates the cost of running BayesianVSLNet naively on full-length videos. Phases 1–7 represent the optimized ToC pipeline we are evaluating.

## Phase 9: Local Evaluation & Comparative Analysis
- **Step 9.1: Run Both Pipelines on Validation Splits:** Execute the full ToC pipeline (Phases 2–7) and the baseline pipeline (Phase 8) on each dataset's validation split.
- **Step 9.2: Compute Grounding Metrics:** r@1 IoU=0.3 (primary), r@1 IoU=0.5 (tie-breaker), r@5 IoU=0.3, r@5 IoU=0.5 — computed per-dataset.
- **Step 9.3: Compute Efficiency Metrics:** Compare the two pipelines on:
  - **Wall-clock time** per query (ToC vs. Baseline)
  - **Features processed** (number of feature vectors fed to BayesianVSLNet)
  - **Inference Speed-Up ratio:** `baseline_features / toc_features`
  - **Accuracy delta:** `toc_r@1 - baseline_r@1` (we want this ≥ 0, i.e., no accuracy loss)
- **Step 9.4: Cross-Dataset Analysis:** Report metrics separately for EPIC-KITCHENS-100, Charades-Ego, and EgoProceL to identify domain-specific strengths/weaknesses.
- **Step 9.5: Failure Analysis:** Identify worst-performing queries, analyze failure modes, and iterate on Phases 4–6 thresholds.
- **Step 9.6: Iterate:** Repeat until val metrics are satisfactory.

## Phase 10: Results & Reporting
- **Step 10.1: Aggregate Results Table:** Build a comparison table across all three datasets:
  | Dataset | Method | r@1 IoU=0.3 | r@1 IoU=0.5 | Speed-Up | Features Processed |
  |---|---|---|---|---|---|
  | EPIC-KITCHENS-100 | Baseline (Full Video) | — | — | 1.0× | — |
  | EPIC-KITCHENS-100 | ToC Pipeline | — | — | —× | — |
  | ... | ... | ... | ... | ... | ... |
- **Step 10.2: Efficiency Analysis:** Compute mean/median speed-up ratios across all queries and datasets. Report the distribution of speed-ups (are some queries much faster? are outliers dragging the average?).
- **Step 10.3: Ablation Study:** Run with individual phases disabled to measure their contribution:
  - ToC without Phase 2 (no event annotation) 
  - ToC without Phase 3–4 (no acoustic triggers, only event sampling)
  - ToC without Librarian (random chapter selection)
- **Step 10.4: Export & Documentation:** Package all results, plots, and analysis into a reproducible report.

