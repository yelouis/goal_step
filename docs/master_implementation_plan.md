# Ego4D Goal-Step Challenge: Master Implementation Plan

This document outlines the architecture and execution strategy for participating in the Ego4D competition for the `goal_step` challenge.

> [!IMPORTANT]
> **Data Storage Requirement:**
> All raw video, audio extractions, cached OmniMAE features, and local MLX models should be persistently stored on the 2TB external SSD. The target base directory is `/Volumes/Extreme SSD/ego4d_data/`.

## Phase 0: Installation and Model Verification
- Set up base package dependencies (`mlx`, `librosa`, `torch`).
- Run a baseline environment check to verify inference hardware (Apple Silicon unified memory constraints) and ensure models are caching correctly to `/Volumes/Extreme SSD/`.

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
  - **Saturation Logic:** If 3 captions are triggered within 15 seconds, exponentially increase the trigger thresholds ($k_{up}$ and $k_{down}$) to force the system to prioritize only the most drastic acoustic shifts.
- **Step 2.3: The Silence Watcher (Forced Detection):** If no audio trigger (spike or drop) occurs for a period of $T_{force}$ (set between 30s and 60s), force a visual caption to ensure silent, steady-state actions are indexed.

## Phase 3: Visual Captioning & Table of Contents (ToC)
- **Step 3.1: Sparse Captioning:** For every trigger, pass the frame to **Moondream2**.
- **Step 3.2: ToC Construction:** Store as a JSON index on the SSD. Each entry includes:
  - `timestamp`: $[s]$
  - `trigger_type`: ["spike", "drop", "forced_silence"]
  - `caption`: "User is assembling the metal bracket."

## Phase 4: The Librarian (Gemma 4 26B MoE Reasoning)
### The Librarian System Prompt
> You are the "Procedural Video Librarian." Map a "Target Step" to the most likely "Chapters" within a video's Table of Contents (ToC).
> 
> **Logical Rules:**
> 1. **Explicit Identification:** Prioritize chapters where the caption matches the Target Step.
> 2. **Silent-Step Interpolation:** If the step is missing, identify the **Pre-Condition** and **Post-Condition**. The Target Step is located in the temporal gap between them.
> 3. **Change Detection Logic:** Treat "drop" triggers as potential "Step Completion" markers and "spike" triggers as "Step Start" markers.
> 4. **Output Format:** Return a JSON object with `"top_chapters"`, `"confidence"`, and `"reasoning"`.

## Phase 5: Focused Bayesian Grounding (High-Res Pass)
- **Step 5.1: Windowed Feature Extraction:** Extract high-resolution **OmniMAE** features only for the chapters identified by the Librarian. Cache to SSD.
- **Step 5.2: Bayesian Head Inference:** Run the **BayesianVSLNet** head on this specific slice.
- **Step 5.3: Timestamp Refinement:** Use the posterior distribution to calculate the final `(start, end)` timestamps.

## Phase 6: Submission Formatting
- **Step 6.1: Ranking:** Select the top 5 distinct predictions based on confidence.
- **Step 6.2: Export:** Format to the `ego4d_goalstep_challenge` JSON schema logic.
- **Step 6.3: Efficiency Metrics:** Calculate the "Inference Speed-up" by comparing the duration of processed chapters vs. total video length.
