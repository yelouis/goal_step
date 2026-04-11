# Phase 6: Submission Formatting

## Overview
The final phase constructs the official JSON payload expected by the Ego4D challenge submission portal. It ranks the top predictions from the Bayesian Grounder, formats them according to the `ego4d_goalstep_challenge` schema, and calculates internal metrics to validate our Hybrid VLA speed-up logic.

## Formatting Rules
The challenge expects dicts mapping `video_id` to a list of predictions.
Each prediction requires:
- `segment`: `[start_time_sec, end_time_sec]`
- `score`: Confidence value (from Phase 4 or 5)

## Pseudocode Implementation

```python
import json
import os

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
SUBMISSION_PATH = os.path.join(SSD_BASE, "submissions/submission_ego4d_goalstep.json")

def calculate_speed_up_metric(video_length_sec: float, processed_window_sec: float) -> str:
    """
    Since we only pass segments to BayesianVSLNet instead of the whole video,
    we calculate our reduction in FLOP overhead.
    """
    if processed_window_sec == 0:
        return "100.0x (No windows processed)"
        
    ratio = video_length_sec / processed_window_sec
    return f"{ratio:.1f}x reduction in processed frames"

def generate_submission(video_id: str, challenge_query_id: str):
    # Load Phase 4 Confidence
    hypo_path = os.path.join(SSD_BASE, "cache/phase4/sample_hypothesis.json")
    with open(hypo_path, 'r') as f:
        hypothesis = json.load(f)
        
    base_confidence = hypothesis.get("confidence", 0.0)
    
    # Load Phase 5 Refined Bounds
    bounds_path = os.path.join(SSD_BASE, "cache/phase5/refined_bounds.json")
    with open(bounds_path, 'r') as f:
        refined_bounds = json.load(f)
        
    # Sort and rank (In our architecture, the LLM Librarian ranked them first)
    # We will just map them directly and assign decaying confidences if there are multiple.
    predictions = []
    total_processed_sec = 0.0
    
    for i, bounds in enumerate(refined_bounds):
        if i >= 5: # Challenge typically caps top-k predictions
            break 
            
        start_t = bounds["refined_start"]
        end_t = bounds["refined_end"]
        
        # Calculate metric
        total_processed_sec += (end_t - start_t)
        
        predictions.append({
            "segment": [start_t, end_t],
            "score": base_confidence * (0.9 ** i) # slight penalty for lower rank chapters
        })
        
    # The submission schema is usually a mapping of Query ID -> Video Predictions
    # For Ego4d Goal Step, they test specific queries against specific videos
    submission_payload = {
        "version": "1.0",
        "challenge": "ego4d_goal_step",
        "results": {
            # Usually mapped by a unique query UUID, using video_id here for pseudocode
            challenge_query_id: predictions
        }
    }
    
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    
    # Append to or overwrite the submission file
    if os.path.exists(SUBMISSION_PATH):
        with open(SUBMISSION_PATH, 'r') as f:
            full_submission = json.load(f)
            full_submission["results"].update(submission_payload["results"])
    else:
        full_submission = submission_payload
        
    with open(SUBMISSION_PATH, 'w') as f:
        json.dump(full_submission, f, indent=4)
        
    print(f"Successfully generated submission for Query {challenge_query_id}.")
    
    # Optional Metric Logging
    video_length = 300.0 # MOCK: Assume a 5 min video
    speed_up = calculate_speed_up_metric(video_length, total_processed_sec)
    print(f"Efficiency Metric: {speed_up} compared to full baseline scanning.")
    
if __name__ == '__main__':
    generate_submission("sample_ego4d_vid_001", "query_assemble_bracket_001")
```

## Verification Strategy
- **Linter Check:** Use `jsonschema` to validate `submission_ego4d_goalstep.json` against the official contest JSON format schema to ensure zero formatting rejections.
