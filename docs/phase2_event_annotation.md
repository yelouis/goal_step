# Phase 2: Event Annotation Pass (What Happened?)

## Overview
Unlike Ego4D's hierarchical `Goal → Step → Substep` annotations, our three evaluation datasets (EPIC-KITCHENS-100, Charades-Ego, EgoProceL) provide only sparse action segment labels — they don't include a dense narrative of everything happening in the video. Before the acoustic analysis and ToC pipeline can operate effectively, we need a **dense event listing** that describes "what happened" across the entire video timeline.

This phase produces that event listing by uniformly sampling frames and captioning them with a VLM, then deduplicating consecutive identical descriptions into merged event spans.

> [!IMPORTANT]
> **This phase runs once per video and is cached.** The event log is stored on the SSD and reused across all queries targeting the same video. It also serves as input to Phase 5's ToC construction (merged with acoustic trigger captions).

## Relationship to Ground Truth
The event annotation pass is **NOT** generating ground truth labels. It is producing a coarse "what happened" timeline that:
1. Provides baseline coverage for steps that are acoustically silent (no spike/drop trigger)
2. Gives Phase 6 (the Librarian) additional context about the video's content
3. Fills temporal gaps between acoustic triggers

The actual temporal grounding evaluation uses the dataset's native ground-truth `(start, end)` annotations.

## Theoretical Pipeline

1. **Uniform Temporal Sampling:** Extract frames at a fixed interval (e.g., every 2–3 seconds) across the full video duration.
2. **VLM Event Captioning:** Run each frame through Qwen2.5-VL-3B with a neutral, action-focused prompt.
3. **Semantic Deduplication:** Merge consecutive frames with equivalent captions into a single event span.
4. **Event Log Export:** Write the result as a JSON file indexed by video ID.

## Pseudocode Implementation

```python
import cv2
import json
import os
import gc
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"
MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
SAMPLING_INTERVAL_SEC = 2.5   # One frame every 2.5 seconds
MIN_EVENT_DURATION_SEC = 1.0  # Minimum span before we consider it a real event

# Neutral captioning prompt — no trigger-type bias
EVENT_PROMPT = (
    "Describe exactly what the person is doing in this frame. "
    "Be specific about the objects being used and the action being performed. "
    "Answer in one sentence."
)

class EventAnnotator:
    def __init__(self):
        print("Loading VLM for event annotation...")
        os.environ['HF_HOME'] = os.path.join(SSD_BASE, "models/huggingface")
        self.model, self.processor = load(MODEL_ID)
        self.config = load_config(MODEL_ID)
    
    def unload(self):
        """Free model memory before next phase."""
        del self.model
        del self.processor
        gc.collect()
        print("Event annotator VLM unloaded.")
    
    def _extract_frame(self, video_path: str, timestamp_sec: float) -> str:
        """Extract a single frame and save to cache. Returns image path."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame at {timestamp_sec}s")
        
        safe_ts = str(timestamp_sec).replace('.', '_')
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        img_path = os.path.join(
            SSD_BASE, f"cache/phase2/frames/{video_id}_{safe_ts}.jpg"
        )
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, frame)
        return img_path
    
    def _caption_frame(self, img_path: str) -> str:
        """Run VLM inference on a single frame."""
        try:
            formatted = apply_chat_template(
                self.processor,
                self.config,
                EVENT_PROMPT,
                num_images=1
            )
            output = generate(
                self.model,
                self.processor,
                img_path,
                formatted,
                max_tokens=64,
                temperature=0.2
            )
            return output.text.strip() if hasattr(output, 'text') else str(output).strip()
        except Exception as e:
            print(f"[ERROR] Caption failed for {img_path}: {e}")
            return "[CAPTION_FAILED]"
    
    def _captions_are_similar(self, a: str, b: str) -> bool:
        """
        Heuristic similarity check for consecutive captions.
        
        We use a simple word-overlap ratio. For production, consider
        using sentence embeddings (e.g., from Gemma) but that would
        require loading a second model.
        """
        if a == b:
            return True
        if a == "[CAPTION_FAILED]" or b == "[CAPTION_FAILED]":
            return False
        
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return False
        
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = overlap / union if union > 0 else 0
        
        return jaccard > 0.6  # Threshold: >60% word overlap = same event
    
    def annotate_video(self, video_id: str, video_path: str) -> list:
        """
        Produce a dense event log for one video.
        
        Returns list of event dicts:
        [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "caption": "The person is chopping an onion on a cutting board.",
                "source": "event_sample",
                "frame_count": 2
            },
            ...
        ]
        """
        # Check cache first
        cache_path = os.path.join(SSD_BASE, f"cache/phase2/{video_id}_events.json")
        if os.path.exists(cache_path):
            print(f"[CACHED] Event log for {video_id} already exists.")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration <= 0:
            print(f"[ERROR] Cannot determine duration for {video_id}")
            return []
        
        print(f"Annotating {video_id}: {duration:.1f}s @ {fps:.1f} fps")
        
        # Step 1: Uniform temporal sampling
        timestamps = []
        t = 0.0
        while t < duration:
            timestamps.append(round(t, 2))
            t += SAMPLING_INTERVAL_SEC
        
        print(f"  Sampling {len(timestamps)} frames at {SAMPLING_INTERVAL_SEC}s intervals...")
        
        # Step 2: Caption each frame
        raw_captions = []
        for ts in timestamps:
            try:
                img_path = self._extract_frame(video_path, ts)
                caption = self._caption_frame(img_path)
                raw_captions.append({"timestamp": ts, "caption": caption})
                
                # Clean up frame file to save space (optional)
                os.remove(img_path)
            except ValueError as e:
                print(f"  [SKIP] {ts}s: {e}")
                continue
        
        # Step 3: Deduplicate consecutive identical captions
        events = []
        if raw_captions:
            current_event = {
                "start_time": raw_captions[0]["timestamp"],
                "end_time": raw_captions[0]["timestamp"] + SAMPLING_INTERVAL_SEC,
                "caption": raw_captions[0]["caption"],
                "source": "event_sample",
                "frame_count": 1
            }
            
            for i in range(1, len(raw_captions)):
                if self._captions_are_similar(
                    current_event["caption"], raw_captions[i]["caption"]
                ):
                    # Extend current event
                    current_event["end_time"] = raw_captions[i]["timestamp"] + SAMPLING_INTERVAL_SEC
                    current_event["frame_count"] += 1
                else:
                    # Close current event and start new one
                    if (current_event["end_time"] - current_event["start_time"]) >= MIN_EVENT_DURATION_SEC:
                        events.append(current_event)
                    
                    current_event = {
                        "start_time": raw_captions[i]["timestamp"],
                        "end_time": raw_captions[i]["timestamp"] + SAMPLING_INTERVAL_SEC,
                        "caption": raw_captions[i]["caption"],
                        "source": "event_sample",
                        "frame_count": 1
                    }
            
            # Don't forget the last event
            if (current_event["end_time"] - current_event["start_time"]) >= MIN_EVENT_DURATION_SEC:
                events.append(current_event)
        
        # Step 4: Export
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(events, f, indent=2)
        
        print(f"  Event log: {len(raw_captions)} frames → {len(events)} merged events")
        return events


if __name__ == '__main__':
    annotator = EventAnnotator()
    
    # Example: annotate a single EPIC-KITCHENS video
    events = annotator.annotate_video(
        "P01_101",
        "/Volumes/Extreme SSD/goal_step_data/datasets/epic_kitchens_100/videos/P01_101.MP4"
    )
    
    # Print summary
    for e in events[:5]:
        print(f"  [{e['start_time']:.1f}s - {e['end_time']:.1f}s] {e['caption']}")
    
    annotator.unload()
```

## Memory & Performance Considerations

> [!WARNING]
> **VLM Throughput:** At ~0.5s per caption (measured in Phase 0 verification), a 10-minute video sampled at 2.5s intervals produces ~240 frames. Estimated annotation time: **~2 minutes per video**. For the full dataset, consider batching videos and using the caching to resume interrupted runs.

> [!NOTE]
> **Deduplication Quality:** The word-overlap Jaccard heuristic is fast but imperfect. If the Librarian (Phase 6) struggles with noisy event logs, consider upgrading to sentence-embedding similarity using the Gemma model (at the cost of loading a second model during this phase).

## Session Resilience

| Item | Detail |
|---|---|
| **Input Dependencies** | Video files from Phase 1 at `datasets/{dataset}/videos/` |
| **Output Artifact** | `/Volumes/Extreme SSD/goal_step_data/cache/phase2/{video_id}_events.json` |
| **Cache Check** | On entry, `annotate_video()` checks if `{video_id}_events.json` exists and returns cached results |
| **Verification Checkpoint** | After completing all videos, write `cache/phase2/_manifest.json` listing all processed video IDs and event counts |
| **Resume Strategy** | On re-run, skip any video whose `_events.json` already exists on the SSD |

## Verification Strategy
- **Coverage Test:** For a sample video, assert that the union of all event spans covers ≥80% of the video's total duration. Gaps (uncovered time) should be logged for investigation.
- **Deduplication Sanity:** Manually inspect 10 consecutive raw captions and their merged events. Verify that semantically identical descriptions are properly collapsed.
- **Caption Quality Baseline:** Select 20 random event captions and manually rate them as "accurate", "vague", or "wrong". Target: ≥70% accurate.
- **Cache Consistency:** Run `annotate_video()` twice for the same video. Assert the second call returns cached results instantly.

