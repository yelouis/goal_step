# Phase 6: Submission Formatting

## Overview
The final phase constructs the official JSON payload expected by the Ego4D Goal-Step challenge submission portal on CodaBench. It maps refined predictions from the Bayesian Grounder back to the official annotation schema, ranks the top-5 temporal windows per query, and packages the result as a `.zip` archive.

## Official Submission Schema
The CodaBench challenge requires a very specific JSON format. Each entry maps to a specific keystep query identified by `clip_uid`, `annotation_uid`, and `query_idx`.

```json
{
  "version": "1.0",
  "challenge": "ego4d_goalstep_challenge",
  "results": [
    {
      "clip_uid": "video_uid_string",
      "annotation_uid": "video_uid_string",
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

> [!IMPORTANT]
> **Key Requirements:**
> - `results` is a **list**, not a dict.
> - Each entry must have exactly **5 predicted `[start, end]` pairs**, ranked by confidence.
> - `clip_uid` and `annotation_uid` should both be set to the `video_uid` from the test annotations.
> - `query_idx` is the index of the keystep description in the `step_segments` hierarchy.
> - The final JSON must be compressed into a **`.zip` archive** before uploading to CodaBench.

## Pseudocode Implementation

```python
import json
import os
import zipfile

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
SUBMISSION_DIR = os.path.join(SSD_BASE, "submissions")
SUBMISSION_JSON = os.path.join(SUBMISSION_DIR, "submission_ego4d_goalstep.json")
SUBMISSION_ZIP = os.path.join(SUBMISSION_DIR, "submission_ego4d_goalstep.zip")

def get_video_duration(video_path: str) -> float:
    """Get actual video duration from file metadata instead of hardcoding."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0.0

def calculate_speed_up_metric(video_length_sec: float, processed_window_sec: float) -> str:
    """
    Since we only pass segments to BayesianVSLNet instead of the whole video,
    we calculate our reduction in FLOP overhead.
    """
    if processed_window_sec == 0:
        return "100.0x (No windows processed)"
        
    ratio = video_length_sec / processed_window_sec
    return f"{ratio:.1f}x reduction in processed frames"

def generate_submission_entry(
    video_uid: str, 
    query_idx: int, 
    refined_bounds: list,
    librarian_confidences: list
) -> dict:
    """
    Build a single submission entry for one query.
    
    Args:
        video_uid: The video UID from the Ego4D annotations.
        query_idx: Index of the keystep description in step_segments.
        refined_bounds: List of dicts from Phase 5 with refined_start/refined_end.
        librarian_confidences: Confidence scores from Phase 4 for ranking.
    """
    # Sort by confidence (descending) for ranking
    ranked = sorted(
        zip(refined_bounds, librarian_confidences),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Build exactly 5 predicted_times entries
    predicted_times = []
    for i in range(5):
        if i < len(ranked):
            bounds, _ = ranked[i]
            predicted_times.append([
                bounds["refined_start"],
                bounds["refined_end"]
            ])
        else:
            # Pad with fallback predictions if fewer than 5 candidates
            # Use the last valid prediction with slight temporal jitter
            if predicted_times:
                last = predicted_times[-1]
                jitter = (i - len(ranked) + 1) * 0.5
                predicted_times.append([
                    max(0, last[0] - jitter),
                    last[1] + jitter
                ])
            else:
                # Absolute fallback: predict the full video
                predicted_times.append([0.0, 300.0])  # Will be replaced with actual duration
    
    return {
        "clip_uid": video_uid,
        "annotation_uid": video_uid,
        "query_idx": query_idx,
        "predicted_times": predicted_times
    }

def build_full_submission(test_annotations_path: str):
    """
    Iterate over ALL test queries and assemble the complete submission.
    
    Phases 1-3 are cached per-video. Phases 4-5 run per-query.
    This function orchestrates the full batch.
    """
    with open(test_annotations_path, 'r') as f:
        test_data = json.load(f)
    
    all_results = []
    total_processed_sec = 0.0
    total_video_sec = 0.0
    
    for annotation in test_data:
        video_uid = annotation["video_uid"]
        video_path = os.path.join(SSD_BASE, f"videos/{video_uid}.mp4")
        video_duration = get_video_duration(video_path)
        total_video_sec += video_duration
        
        # Phases 1-3 are cached per-video (run once, read many)
        # Phase 4-5 run per query
        for query_idx, step_segment in enumerate(annotation.get("step_segments", [])):
            target_step = step_segment["step_description"]
            
            # Load Phase 4 results for this query
            hypo_path = os.path.join(
                SSD_BASE, f"cache/phase4/{video_uid}_q{query_idx}_hypothesis.json"
            )
            
            try:
                with open(hypo_path, 'r') as f:
                    hypothesis = json.load(f)
            except FileNotFoundError:
                print(f"[SKIP] Missing hypothesis for {video_uid} query {query_idx}")
                continue
            
            # Load Phase 5 refined bounds
            bounds_path = os.path.join(
                SSD_BASE, f"cache/phase5/{video_uid}_q{query_idx}_refined.json"
            )
            
            try:
                with open(bounds_path, 'r') as f:
                    refined_bounds = json.load(f)
            except FileNotFoundError:
                print(f"[SKIP] Missing refined bounds for {video_uid} query {query_idx}")
                continue
            
            # Track processed duration for efficiency metric
            for b in refined_bounds:
                total_processed_sec += (b["refined_end"] - b["refined_start"])
            
            # Build the entry
            confidences = [
                hypothesis.get("confidence", 0.5) * (0.9 ** i) 
                for i in range(len(refined_bounds))
            ]
            
            entry = generate_submission_entry(
                video_uid=video_uid,
                query_idx=query_idx,
                refined_bounds=refined_bounds,
                librarian_confidences=confidences
            )
            all_results.append(entry)
    
    # Assemble final payload
    submission = {
        "version": "1.0",
        "challenge": "ego4d_goalstep_challenge",
        "results": all_results
    }
    
    # Write JSON
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    with open(SUBMISSION_JSON, 'w') as f:
        json.dump(submission, f, indent=4)
    
    # Compress to .zip for CodaBench upload
    with zipfile.ZipFile(SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUBMISSION_JSON, "submission_ego4d_goalstep.json")
    
    print(f"Submission generated: {len(all_results)} entries across test set.")
    print(f"ZIP archive ready at: {SUBMISSION_ZIP}")
    
    # Efficiency Metrics
    speed_up = calculate_speed_up_metric(total_video_sec, total_processed_sec)
    print(f"Efficiency Metric: {speed_up} compared to full baseline scanning.")

if __name__ == '__main__':
    build_full_submission(
        os.path.join(SSD_BASE, "annotations/goalstep_test_unannotated.json")
    )
```

## Verification Strategy
- **Schema Validation:** Use `jsonschema` to validate `submission_ego4d_goalstep.json` against the official contest format. Assert that every entry has exactly 5 `predicted_times` pairs, and that all `clip_uid`/`annotation_uid` values exist in the test annotations.
- **Dry Run:** Before final submission, run against the **validation split** (which has ground truth) to compute local metrics. See Phase 5.5.
- **ZIP Integrity:** Verify the `.zip` archive can be extracted and the JSON inside parses correctly.
