# Phase 0.5: Data Download & Annotation Parsing

## Overview
Before any processing can begin, we must obtain the Ego4D dataset, accept the data license, and build a local annotation index that maps the challenge's hierarchical `Goal → Step → Substep` structure into a format our pipeline can iterate over.

> [!IMPORTANT]
> **Ego4D requires a signed data license agreement.** You must request access at [ego4d-data.org](https://ego4d-data.org/) and be approved before downloading any data. This can take 1-2 business days.

## Data Storage Layout

All data lives on the 2TB external SSD:

```
/Volumes/Extreme SSD/ego4d_data/
├── annotations/
│   ├── goalstep_train.json
│   ├── goalstep_val.json
│   └── goalstep_test_unannotated.json
├── videos/
│   └── {video_uid}.mp4
├── features/                    # Pre-extracted features (Omnivore, EgoVLPv2)
│   ├── omnivore/
│   └── egovlpv2/
├── models/
│   ├── huggingface/
│   ├── moondream2/
│   └── gemma/
├── cache/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   └── phase5/
└── submissions/
```

## Step 0.5.1: Install Ego4D CLI & Download Data

```bash
# Install the Ego4D CLI tool
pip install ego4d

# Set the download directory to SSD
export EGO4D_DIR="/Volumes/Extreme SSD/ego4d_data"

# Download Goal-Step annotations
ego4d --output_directory="$EGO4D_DIR/annotations" \
      --datasets goalstep \
      --yes

# Download videos referenced in the Goal-Step annotations
# NOTE: This can be hundreds of GB. Consider downloading a subset first.
ego4d --output_directory="$EGO4D_DIR/videos" \
      --datasets full_scale \
      --video_uids_file="$EGO4D_DIR/annotations/goalstep_video_uids.txt" \
      --yes

# (Optional) Download pre-extracted Omnivore features if available
ego4d --output_directory="$EGO4D_DIR/features" \
      --datasets omnivore_video_features \
      --yes
```

## Step 0.5.2: Parse Annotation Hierarchy

The annotations follow a nested structure: `Goal → Steps → Substeps`. The Step Grounding task requires us to localize specific `step_description` queries within untrimmed videos.

```python
import json
import os
from dataclasses import dataclass, field

SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"

@dataclass
class StepQuery:
    """A single grounding query extracted from the annotation hierarchy."""
    video_uid: str
    annotation_uid: str
    query_idx: int
    goal_description: str
    goal_category: str
    step_description: str
    # Ground truth (only available in train/val)
    gt_start_time: float = -1.0
    gt_end_time: float = -1.0

@dataclass
class VideoAnnotation:
    """All queries associated with a single video."""
    video_uid: str
    video_duration: float
    queries: list = field(default_factory=list)

def parse_goalstep_annotations(split: str = "val") -> dict:
    """
    Parse the hierarchical Goal-Step annotations into a flat list of 
    StepQuery objects, grouped by video_uid.
    
    Args:
        split: One of 'train', 'val', 'test_unannotated'
    
    Returns:
        Dict mapping video_uid -> VideoAnnotation
    """
    ann_path = os.path.join(SSD_BASE, f"annotations/goalstep_{split}.json")
    
    with open(ann_path, 'r') as f:
        raw = json.load(f)
    
    videos = {}
    
    for video_entry in raw.get("videos", raw):  # Handle both list and dict formats
        video_uid = video_entry["video_uid"]
        
        if video_uid not in videos:
            videos[video_uid] = VideoAnnotation(
                video_uid=video_uid,
                video_duration=video_entry.get("duration", 0.0)
            )
        
        query_idx = 0
        
        # Walk the goal -> step hierarchy
        for goal in video_entry.get("segments", []):
            goal_desc = goal.get("goal_description", "Unknown goal")
            goal_cat = goal.get("goal_category", "Unknown category")
            
            for step in goal.get("segments", []):
                step_desc = step.get("step_description", "")
                
                if not step_desc:
                    continue
                
                query = StepQuery(
                    video_uid=video_uid,
                    annotation_uid=video_uid,  # Per challenge spec
                    query_idx=query_idx,
                    goal_description=goal_desc,
                    goal_category=goal_cat,
                    step_description=step_desc,
                )
                
                # Ground truth (only in train/val splits)
                if "start_time" in step and "end_time" in step:
                    query.gt_start_time = step["start_time"]
                    query.gt_end_time = step["end_time"]
                
                videos[video_uid].queries.append(query)
                query_idx += 1
    
    total_queries = sum(len(v.queries) for v in videos.values())
    print(f"Parsed {split} split: {len(videos)} videos, {total_queries} step queries")
    
    return videos

def build_video_uid_list(split: str = "val") -> None:
    """
    Export a text file of video UIDs for selective downloading.
    """
    videos = parse_goalstep_annotations(split)
    out_path = os.path.join(SSD_BASE, f"annotations/goalstep_{split}_video_uids.txt")
    
    with open(out_path, 'w') as f:
        for uid in sorted(videos.keys()):
            f.write(uid + "\n")
    
    print(f"Exported {len(videos)} video UIDs to {out_path}")

if __name__ == '__main__':
    # Parse all available splits
    for split in ["train", "val"]:
        try:
            videos = parse_goalstep_annotations(split)
            
            # Print a sample for sanity checking
            sample_vid = next(iter(videos.values()))
            if sample_vid.queries:
                q = sample_vid.queries[0]
                print(f"  Sample query: [{q.goal_category}] {q.step_description}")
                if q.gt_start_time >= 0:
                    print(f"  Ground truth: [{q.gt_start_time:.1f}s - {q.gt_end_time:.1f}s]")
        except FileNotFoundError:
            print(f"  {split} annotations not downloaded yet -- skipping")
    
    # Build UID list for selective video download
    build_video_uid_list("val")
```

## Step 0.5.3: Verify Data Integrity

```python
def verify_data_integrity(split: str = "val"):
    """Check that all referenced videos exist on disk and are readable."""
    import cv2
    
    videos = parse_goalstep_annotations(split)
    missing = []
    corrupt = []
    
    for video_uid, annotation in videos.items():
        video_path = os.path.join(SSD_BASE, f"videos/{video_uid}.mp4")
        
        if not os.path.exists(video_path):
            missing.append(video_uid)
            continue
        
        # Quick readability check
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            corrupt.append(video_uid)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps if fps > 0 else 0
            print(f"  {video_uid}: {duration:.1f}s @ {fps:.1f} fps, {len(annotation.queries)} queries")
        cap.release()
    
    if missing:
        print(f"\n[WARNING] {len(missing)} videos missing from disk:")
        for uid in missing[:10]:
            print(f"  - {uid}")
    
    if corrupt:
        print(f"\n[ERROR] {len(corrupt)} videos unreadable:")
        for uid in corrupt:
            print(f"  - {uid}")
    
    return len(missing) == 0 and len(corrupt) == 0
```

## Verification Strategy
- **Annotation Parse Test:** Assert that `parse_goalstep_annotations("val")` returns a non-empty dict, and that every `StepQuery` has a non-empty `step_description` and valid `gt_start_time < gt_end_time`.
- **UID Coverage:** Assert that every `video_uid` referenced in the annotations has a corresponding `.mp4` file on the SSD.
- **Schema Spot-Check:** Print 5 random queries and manually verify they match the raw JSON structure.
