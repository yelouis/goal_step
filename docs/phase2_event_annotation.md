# Phase 2: Event Annotation Pass (What Happened?)

## Overview
While Ego4D provides goal/step annotations, these are often sparse. To build a robust **Table of Contents (ToC)**, we need a dense event listing that describes the "what happened" across the entire video timeline. This listing provides context for the Librarian (Phase 6) and ensures coverage for steps that might not have a strong acoustic trigger.

This phase produces that event listing by uniformly sampling frames and captioning them with a VLM (Qwen2.5-VL-3B), then deduplicating consecutive identical descriptions.

> [!IMPORTANT]
> **This phase runs once per video and is cached.** The event log is stored on the SSD and reused across all queries targeting the same video.

## Relationship to Ground Truth
The event annotation pass is **NOT** generating ground truth labels. It is producing a coarse timeline that:
1. Provides baseline coverage for silent steps.
2. Gives the Librarian (Phase 6) reasoning context about the video's procedural flow.
3. Fills temporal gaps between acoustic triggers identified in Phase 4.

## Theoretical Pipeline

1. **Uniform Temporal Sampling:** Extract frames at a fixed interval (e.g., every 2.5 seconds).
2. **VLM Event Captioning:** Run each frame through **Qwen2.5-VL-3B** with a neutral, action-focused prompt.
3. **Semantic Deduplication:** Merge consecutive frames with equivalent captions into a single event span.
4. **Event Log Export:** Write the result as a JSON file indexed by `video_uid`.

## Pseudocode Implementation

```python
import cv2
import json
import os
import gc
from mlx_vlm import load, generate

SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
SAMPLING_INTERVAL_SEC = 2.5

EVENT_PROMPT = (
    "Describe exactly what the person is doing in this frame. "
    "Be specific about the objects being used and the action being performed. "
    "Answer in one sentence."
)

# Implementation follows the pattern in scripts/parse_annotations.py
# Using video_uid as the primary key for Ego4D consistency.
```

## Memory & Performance Considerations

- **VLM Throughput:** Estimated ~2 minutes per 10-minute video on M4 Pro.
- **Caching:** Skip processing if `{video_uid}_events.json` already exists in `cache/phase2/`.

## Verification Strategy
- **Coverage Test:** Assert event spans cover ≥80% of video duration.
- **Caption Quality:** Manually inspect sample captions for accuracy on procedural tasks.
- **Cache Integrity:** Verify that re-running the phase correctly uses cached results.
