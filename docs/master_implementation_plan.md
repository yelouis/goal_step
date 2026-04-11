# Ego4D Goal-Step Challenge: Master Implementation Plan

This document outlines the architecture and execution strategy for participating in the Ego4D competition for the `goal_step` challenge (Step Grounding task).

> [!IMPORTANT]
> **Data Storage Requirement:**
> All raw video, audio extractions, cached features (Omnivore-L, EgoVLPv2), and local MLX models should be persistently stored on the 2TB external SSD. The target base directory is `/Volumes/Extreme SSD/ego4d_data/`.

> [!WARNING]
> **Memory Budget (24GB M4 Pro):**
> Models must be loaded sequentially, not concurrently. Moondream2 (~3.7GB) → unload → Gemma 4 26B (~14GB) → unload → BayesianVSLNet (~4-8GB). Explicit `del` + `gc.collect()` between phases.

## Phase 0: Installation and Model Verification
- Set up base package dependencies (`mlx`, `librosa`, `torch`).
- Run a baseline environment check to verify inference hardware (Apple Silicon unified memory constraints) and ensure models are caching correctly to `/Volumes/Extreme SSD/`.

## Phase 0.5: Data Download & Annotation Parsing
- Obtain Ego4D data license and download dataset via the Ego4D CLI.
- Download Goal-Step annotations (train/val/test splits) and referenced videos.
- Parse the hierarchical annotation structure (`Goal → Steps → Substeps`) into a flat query index.
- Extract `video_uid`, `annotation_uid`, `query_idx`, `step_description`, and `goal_description` for every test query.
- Verify data integrity (all referenced videos exist on SSD and are readable).

## Phase 1: The Acoustic Characterization Pass (Global Scan)
- **Step 1.1: Global Baseline Calculation:** Calculate the **Global Noise Floor ($N_g$)** and the **Median Absolute Deviation (MAD)** of the energy envelope.
- **Step 1.2: Stationary Noise Masking:** Identify persistent frequency bands (e.g., HVAC hum) to create a spectral mask, ensuring the trigger logic isn't distracted by constant environmental noise.
- **Step 1.3: Pattern Discovery:** Map "Active vs. Passive" acoustic regions. If a specific volume level remains constant for long durations, it is flagged as a "baseline" for that segment of the video.

## Phase 2: The Adaptive Sentry (Bidirectional Magnitude Triggers)
This stage identifies "Interest Zones" for visual captioning by detecting significant shifts in the acoustic state.

- **Step 2.1: Bidirectional Change Detection:** Trigger a **Moondream2** capture if the Short-Term Average Energy ($E_{sta}$) deviates significantly from the Long-Term Average Energy ($E_{lta}$):
  - **The Spike:** Trigger if $E_{sta} / E_{lta} > k_{up}$ (e.g., $k_{up} = 2.5$).
  - **The Drop:** Trigger if $E_{sta} / E_{lta} < k_{down}$ (e.g., $k_{down} = 0.3$). This captures the exact moment a noisy action ceases.
- **Step 2.2: Semantic Rate Limiting:**
  - **Temporal Masking:** After a trigger, initiate a 3-second "Cool-down" to prevent redundant captions during stuttering noises.
  - **Saturation Logic:** If 3 captions are triggered within 15 seconds, exponentially increase the trigger thresholds ($k_{up}$ and $k_{down}$) with per-trigger decay to gradually relax back to baseline.
- **Step 2.3: The Silence Watcher (Forced Detection):** If no audio trigger (spike or drop) occurs for a period of $T_{force}$ (set between 30s and 60s), force a visual caption to ensure silent, steady-state actions are indexed.

> **Known Limitation:** Audio-only triggers are unreliable for acoustically silent procedural steps. A visual change-detection fallback (frame differencing / optical flow) is documented as a future enhancement.

## Phase 3: Visual Captioning & Table of Contents (ToC)
- **Step 3.1: Trigger-Aware Captioning:** For every trigger, pass the frame to **Moondream2** using a prompt adapted to the trigger type:
  - `spike` → "What action is the user starting?"
  - `drop` → "What action did the user just finish?"
  - `forced_silence` → "What is the user currently doing?"
- **Step 3.2: ToC Construction:** Store as a JSON index on the SSD. Each entry includes:
  - `timestamp`: $[s]$
  - `trigger_type`: ["spike", "drop", "forced_silence"]
  - `caption`: "User is assembling the metal bracket."
- **Step 3.3: Caption Quality Filtering:** Retry with a fallback prompt if the caption is too short or generic.

> **Caching:** Phases 1-3 run **once per video** regardless of how many queries target that video.

## Phase 4: The Librarian (Gemma 4 26B MoE Reasoning)
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

## Phase 5: Focused Bayesian Grounding (High-Res Pass)
- **Step 5.1: Windowed Feature Loading:** Load pre-extracted **Omnivore-L + EgoVLPv2** features (concatenated, 1024-d) only for the chapters identified by the Librarian. Use Ego4D pre-extracted features when available.
- **Step 5.2: Bayesian Head Inference:** Run the **BayesianVSLNet** head on this specific slice, using **EgoVLPv2** text features (with RoBERTa fallback if EgoVLPv2 weights unavailable).
- **Step 5.3: Timestamp Refinement:** Use the posterior distribution with non-causal temporal-order prior to calculate the final `(start, end)` timestamps.

> **Feature Alignment:** The BayesianVSLNet head was trained on Omnivore-L + EgoVLPv2 features. Using mismatched backbones will produce degraded predictions.

## Phase 5.5: Local Evaluation Pipeline
- **Step 5.5.1: Run on Validation Split:** Execute the full pipeline (Phases 1-5) on the val split.
- **Step 5.5.2: Compute Official Metrics:** r@1 IoU=0.3 (primary), r@1 IoU=0.5 (tie-breaker), r@5 IoU=0.3, r@5 IoU=0.5.
- **Step 5.5.3: Failure Analysis:** Identify worst-performing queries, analyze failure modes, and iterate on Phases 2-4 thresholds.
- **Step 5.5.4: Iterate:** Repeat until val metrics are satisfactory before touching the test split.

## Phase 6: Submission Formatting
- **Step 6.1: Batch Processing:** Iterate over ALL test annotations, running cached Phases 1-3 per-video and Phases 4-5 per-query.
- **Step 6.2: Ranking:** For each query, produce exactly **5 predicted temporal windows** ranked by confidence.
- **Step 6.3: Export:** Format to the official CodaBench JSON schema (`clip_uid`, `annotation_uid`, `query_idx`, `predicted_times`).
- **Step 6.4: Package:** Compress to `.zip` archive for CodaBench upload.
- **Step 6.5: Efficiency Metrics:** Calculate the "Inference Speed-up" by comparing the duration of processed chapters vs. total video length.
