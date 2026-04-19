# Phase 1: Data Download & Annotation Parsing

## Overview
Before any processing can begin, we must obtain the three evaluation datasets (EPIC-KITCHENS-100, Charades-Ego, EgoProceL), download their annotation files, and build a **unified internal query index** so that all downstream phases are dataset-agnostic.

> [!IMPORTANT]
> **No special data license is required.** Unlike Ego4D (which requires a signed agreement), all three datasets are publicly available:
> - **EPIC-KITCHENS-100**: Academic download via University of Bristol.
> - **Charades-Ego**: Open download from Allen AI.
> - **EgoProceL**: Open GitHub repository + links to source datasets.

## Data Storage Layout

All data lives on the 2TB external SSD:

```
/Volumes/Extreme SSD/goal_step_data/
├── datasets/
│   ├── epic_kitchens_100/
│   │   ├── annotations/          # CSV: EPIC_100_{train,val}.csv
│   │   └── videos/               # P{xx}_{yyy}.MP4
│   ├── charades_ego/
│   │   ├── annotations/          # Charades_v1_{train,test}.csv, CharadesEgo_v1_{train,test}.csv
│   │   └── videos/               # {video_id}.mp4
│   └── egoprocel/
│       ├── annotations/          # {task_name}/{video_id}.csv (start_sec, end_sec, label)
│       └── videos/               # {task_name}/{video_id}.mp4
├── models/
│   ├── huggingface/
│   └── bayesian_vslnet/
├── cache/
│   ├── phase2/                # Event annotation logs (formerly 0.75)
│   ├── phase3/                # Acoustic characterization (formerly 1)
│   ├── phase4/                # Adaptive sentry (formerly 2)
│   ├── phase5/                # Visual captioning (formerly 3)
│   ├── phase6/                # Librarian (formerly 4)
│   ├── phase7/                # Bayesian grounding (formerly 5)
│   └── phase8/                # Full-video baseline results (formerly 5b)
└── results/
```

## Step 1.1: Download EPIC-KITCHENS-100

```bash
# Clone the annotation repository
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git \
    "/Volumes/Extreme SSD/goal_step_data/datasets/epic_kitchens_100/annotations"

# Download videos via the official download scripts
# See: https://github.com/epic-kitchens/epic-kitchens-download-scripts
pip install epic-kitchens-downloader
python -m epic_kitchens.download \
    --output-path="/Volumes/Extreme SSD/goal_step_data/datasets/epic_kitchens_100/videos" \
    --splits=validation
```

## Step 1.2: Download Charades-Ego

```bash
# Download from Allen AI
# Annotations: http://ai2-website.s3.amazonaws.com/data/Charades_Ego.zip
# Videos: direct links from the Charades website

mkdir -p "/Volumes/Extreme SSD/goal_step_data/datasets/charades_ego"
cd "/Volumes/Extreme SSD/goal_step_data/datasets/charades_ego"

# Download annotations
curl -L -o charades_ego_annotations.zip \
    "http://ai2-website.s3.amazonaws.com/data/Charades_Ego.zip"
unzip charades_ego_annotations.zip -d annotations/

# Videos must be downloaded separately from:
# http://allenai.org/plato/charades/
# Follow the instructions for egocentric video download links
```

## Step 1.3: Download EgoProceL

```bash
# Clone the repository for annotations
git clone https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning.git \
    "/Volumes/Extreme SSD/goal_step_data/datasets/egoprocel/repo"

# Annotations are CSV files in the repository
# Videos are sourced from CMU-MMAC, EGTEA Gaze+, MECCANO, EPIC-Tents
# Follow the download instructions in the repo README for each source
```

## Step 1.4: Parse Annotations into Unified Schema

Each dataset has a different native annotation format. We normalize them into a common `StepQuery` dataclass.

```python
import csv
import json
import os
from dataclasses import dataclass, field
from enum import Enum

SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"

class DatasetSource(Enum):
    EPIC_KITCHENS = "epic_kitchens_100"
    CHARADES_EGO = "charades_ego"
    EGOPROCEL = "egoprocel"

@dataclass
class StepQuery:
    """A single grounding query in the unified schema."""
    video_id: str
    dataset_source: str        # One of DatasetSource values
    query_idx: int
    step_description: str      # Natural language description of the step
    goal_description: str      # High-level goal/task context
    # Ground truth (only available in train/val)
    gt_start_time: float = -1.0
    gt_end_time: float = -1.0
    # Original metadata preserved for traceability
    original_metadata: dict = field(default_factory=dict)

@dataclass
class VideoAnnotation:
    """All queries associated with a single video."""
    video_id: str
    dataset_source: str
    video_duration: float
    video_path: str
    queries: list = field(default_factory=list)


# ---- EPIC-KITCHENS-100 Parser ----

def parse_epic_kitchens(split: str = "validation") -> dict:
    """
    Parse EPIC-KITCHENS-100 action segment annotations.
    
    Each narrated action becomes a StepQuery. The narration text is the
    step_description, and the participant+video context forms the goal.
    
    Annotation CSV columns: narration_id, participant_id, video_id,
    narration_timestamp, start_timestamp, stop_timestamp, start_frame,
    stop_frame, narration, verb, verb_class, noun, noun_class, all_nouns,
    all_noun_classes
    """
    ann_dir = os.path.join(SSD_BASE, "datasets/epic_kitchens_100/annotations")
    csv_path = os.path.join(ann_dir, f"EPIC_100_{split}.csv")
    
    videos = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            
            if video_id not in videos:
                videos[video_id] = VideoAnnotation(
                    video_id=video_id,
                    dataset_source=DatasetSource.EPIC_KITCHENS.value,
                    video_duration=0.0,  # Will be filled during integrity check
                    video_path=os.path.join(
                        SSD_BASE, f"datasets/epic_kitchens_100/videos/{video_id}.MP4"
                    )
                )
            
            # Parse timestamps (HH:mm:ss.SS format)
            start_sec = _timestamp_to_seconds(row["start_timestamp"])
            stop_sec = _timestamp_to_seconds(row["stop_timestamp"])
            
            query = StepQuery(
                video_id=video_id,
                dataset_source=DatasetSource.EPIC_KITCHENS.value,
                query_idx=len(videos[video_id].queries),
                step_description=row["narration"],
                goal_description=f"Kitchen activity by participant {row['participant_id']}",
                gt_start_time=start_sec,
                gt_end_time=stop_sec,
                original_metadata={
                    "verb": row["verb"],
                    "noun": row["noun"],
                    "narration_id": row["narration_id"]
                }
            )
            videos[video_id].queries.append(query)
    
    total_queries = sum(len(v.queries) for v in videos.values())
    print(f"Parsed EPIC-KITCHENS-100 {split}: {len(videos)} videos, {total_queries} queries")
    return videos


# ---- Charades-Ego Parser ----

def parse_charades_ego(split: str = "train") -> dict:
    """
    Parse Charades-Ego annotations.
    
    Charades provides activity class intervals per video. Each activity 
    interval becomes a StepQuery with the class name as step_description.
    """
    ann_dir = os.path.join(SSD_BASE, "datasets/charades_ego/annotations")
    csv_path = os.path.join(ann_dir, f"CharadesEgo_v1_{split}.csv")
    
    # Load class labels mapping
    classes = _load_charades_classes(ann_dir)
    
    videos = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["id"]
            
            if video_id not in videos:
                videos[video_id] = VideoAnnotation(
                    video_id=video_id,
                    dataset_source=DatasetSource.CHARADES_EGO.value,
                    video_duration=float(row.get("length", 0)),
                    video_path=os.path.join(
                        SSD_BASE, f"datasets/charades_ego/videos/{video_id}.mp4"
                    )
                )
            
            # Parse actions field: "c012 1.0 5.0;c034 3.2 8.1;..."
            actions_str = row.get("actions", "")
            if not actions_str:
                continue
                
            for action_entry in actions_str.split(";"):
                parts = action_entry.strip().split()
                if len(parts) != 3:
                    continue
                class_id, start_sec, end_sec = parts
                class_name = classes.get(class_id, class_id)
                
                query = StepQuery(
                    video_id=video_id,
                    dataset_source=DatasetSource.CHARADES_EGO.value,
                    query_idx=len(videos[video_id].queries),
                    step_description=class_name,
                    goal_description=row.get("script", "Daily indoor activity"),
                    gt_start_time=float(start_sec),
                    gt_end_time=float(end_sec),
                    original_metadata={"class_id": class_id}
                )
                videos[video_id].queries.append(query)
    
    total_queries = sum(len(v.queries) for v in videos.values())
    print(f"Parsed Charades-Ego {split}: {len(videos)} videos, {total_queries} queries")
    return videos


# ---- EgoProceL Parser ----

def parse_egoprocel() -> dict:
    """
    Parse EgoProceL key-step annotations.
    
    Annotations are per-video CSV files with columns:
    start_second, end_second, key_step_label
    """
    base_dir = os.path.join(SSD_BASE, "datasets/egoprocel")
    ann_dir = os.path.join(base_dir, "annotations")
    
    videos = {}
    
    for task_dir in os.listdir(ann_dir):
        task_path = os.path.join(ann_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        
        for csv_file in os.listdir(task_path):
            if not csv_file.endswith(".csv"):
                continue
            
            video_id = csv_file.replace(".csv", "")
            video_path = os.path.join(base_dir, f"videos/{task_dir}/{video_id}.mp4")
            
            videos[video_id] = VideoAnnotation(
                video_id=video_id,
                dataset_source=DatasetSource.EGOPROCEL.value,
                video_duration=0.0,  # Will be filled during integrity check
                video_path=video_path
            )
            
            with open(os.path.join(task_path, csv_file), 'r') as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if len(row) < 3:
                        continue
                    start_sec, end_sec, label = float(row[0]), float(row[1]), row[2].strip()
                    
                    query = StepQuery(
                        video_id=video_id,
                        dataset_source=DatasetSource.EGOPROCEL.value,
                        query_idx=idx,
                        step_description=label,
                        goal_description=f"Procedural task: {task_dir}",
                        gt_start_time=start_sec,
                        gt_end_time=end_sec,
                        original_metadata={"task": task_dir}
                    )
                    videos[video_id].queries.append(query)
    
    total_queries = sum(len(v.queries) for v in videos.values())
    print(f"Parsed EgoProceL: {len(videos)} videos, {total_queries} queries")
    return videos


# ---- Helpers ----

def _timestamp_to_seconds(ts: str) -> float:
    """Convert HH:mm:ss.SS to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s

def _load_charades_classes(ann_dir: str) -> dict:
    """Load Charades class ID → name mapping."""
    classes = {}
    classes_path = os.path.join(ann_dir, "Charades_v1_classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    classes[parts[0]] = parts[1]
    return classes
```

## Step 1.5: Verify Data Integrity

```python
def verify_data_integrity(videos: dict):
    """Check that all referenced videos exist on disk and are readable."""
    import cv2
    
    missing = []
    corrupt = []
    
    for video_id, annotation in videos.items():
        video_path = annotation.video_path
        
        if not os.path.exists(video_path):
            missing.append(video_id)
            continue
        
        # Quick readability check
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            corrupt.append(video_id)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frames / fps if fps > 0 else 0
            annotation.video_duration = duration
            print(f"  {video_id}: {duration:.1f}s @ {fps:.1f} fps, {len(annotation.queries)} queries")
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

## Step 1.6: Build Combined Query Index

```python
def build_combined_index() -> dict:
    """
    Parse all three datasets and return a unified dict of video_id -> VideoAnnotation.
    
    Video IDs are prefixed with dataset name to avoid collisions:
    e.g., 'epic_P01_101', 'charades_ABC123', 'egoprocel_cooking_01'
    """
    all_videos = {}
    
    # EPIC-KITCHENS-100
    epic = parse_epic_kitchens("validation")
    for vid, ann in epic.items():
        all_videos[f"epic_{vid}"] = ann
    
    # Charades-Ego
    charades = parse_charades_ego("train")
    for vid, ann in charades.items():
        all_videos[f"charades_{vid}"] = ann
    
    # EgoProceL
    egoprocel = parse_egoprocel()
    for vid, ann in egoprocel.items():
        all_videos[f"egoprocel_{vid}"] = ann
    
    total_videos = len(all_videos)
    total_queries = sum(len(v.queries) for v in all_videos.values())
    print(f"\nCombined index: {total_videos} videos, {total_queries} step queries")
    
    return all_videos

if __name__ == '__main__':
    all_videos = build_combined_index()
    
    # Print sample from each dataset
    for prefix in ["epic_", "charades_", "egoprocel_"]:
        sample = next((v for k, v in all_videos.items() if k.startswith(prefix)), None)
        if sample and sample.queries:
            q = sample.queries[0]
            print(f"\n  [{q.dataset_source}] {q.step_description}")
            if q.gt_start_time >= 0:
                print(f"  Ground truth: [{q.gt_start_time:.1f}s - {q.gt_end_time:.1f}s]")
```

## Verification Strategy
- **Annotation Parse Test:** Assert that each parser returns a non-empty dict, and that every `StepQuery` has a non-empty `step_description` and valid `gt_start_time < gt_end_time`.
- **Cross-Dataset UID Uniqueness:** Assert that prefixed video IDs (`epic_*`, `charades_*`, `egoprocel_*`) produce zero collisions in `build_combined_index()`.
- **Schema Spot-Check:** Print 5 random queries per dataset and manually verify they match the raw annotation files.

---

## Session Resilience

| Item | Detail |
|---|---|
| **Input Dependencies** | Dataset downloads (manual), annotation files |
| **Output Artifact** | `/Volumes/Extreme SSD/goal_step_data/cache/phase1/unified_query_index.json` |
| **Cache Check** | On entry, check if `unified_query_index.json` exists and skip parsing if so |
| **Verification Checkpoint** | After building the combined index, write `cache/phase1/_manifest.json` with video counts, query counts, and integrity status per dataset |
| **Resume Strategy** | On re-run, load the cached index. If new datasets are added, re-parse only the new sources and merge |

```python
def save_combined_index(all_videos: dict):
    """Persist the unified query index to SSD for cross-session resilience."""
    import json
    from dataclasses import asdict
    
    serializable = {}
    for vid, ann in all_videos.items():
        serializable[vid] = {
            "video_id": ann.video_id,
            "dataset_source": ann.dataset_source,
            "video_duration": ann.video_duration,
            "video_path": ann.video_path,
            "queries": [
                {
                    "video_id": q.video_id,
                    "dataset_source": q.dataset_source,
                    "query_idx": q.query_idx,
                    "step_description": q.step_description,
                    "goal_description": q.goal_description,
                    "gt_start_time": q.gt_start_time,
                    "gt_end_time": q.gt_end_time,
                    "original_metadata": q.original_metadata
                }
                for q in ann.queries
            ]
        }
    
    out_path = os.path.join(SSD_BASE, "cache/phase1/unified_query_index.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Unified query index saved: {len(serializable)} videos")
```

---

## Phase 1 Completion Summary & Future Considerations

### What was Completed
- Extracted the python parsing logic into an executable artifact `scripts/parse_annotations.py`.
- Configured data hierarchy definitions for the 2TB External SSD (`/Volumes/Extreme SSD/goal_step_data`).
- Validated logic for annotation parsing and data integrity processing using a mock dataset structure, ensuring that `cv2` integrity checks correctly catch corrupted or unreadable video chunks without catastrophic failure.
- Implemented fallback logic for missing data splits (so we can selectively download test/val without crashing).

### Potential Problems in Later Phases

1. **EPIC-KITCHENS-100 Video Access**:
   - **Problem**: EPIC-KITCHENS videos require downloading from the University of Bristol's servers. Some participants' videos may be hosted on different mirrors with varying download speeds.
   - **Solution**: Use the official `epic-kitchens-downloader` package which handles mirror selection and retry logic. Download only the validation split initially to conserve SSD space.

2. **Charades-Ego Video Format Variability**:
   - **Problem**: Charades videos may have inconsistent frame rates or resolution across different recording sessions.
   - **Solution**: The `verify_data_integrity()` function logs FPS and duration for each video. Normalize frame access using timestamp-based seeking (`CAP_PROP_POS_MSEC`) rather than frame-index seeking.

3. **EgoProceL Composite Dataset**:
   - **Problem**: EgoProceL aggregates videos from multiple source datasets (CMU-MMAC, EGTEA Gaze+, MECCANO, EPIC-Tents). Each source may have different download procedures and licensing.
   - **Solution**: Document the download procedure for each source dataset. Since we only need the videos and their key-step annotations, prioritize subsets where video access is straightforward.

4. **Storage and Compute Burn (Video Downloads)**:
   - **Problem**: Downloading all three datasets in full could be significant. EPIC-KITCHENS-100 alone can be hundreds of GB.
   - **Solution**: Start with a manageable subset (e.g., EPIC-KITCHENS validation split, a subset of Charades-Ego, full EgoProceL which is smaller). Scale up once the pipeline is validated end-to-end.

5. **Dependency Drift and Virtual Environment**:
   - **Problem**: The scripts currently execute through the global state or isolated python venv, but system packages like `ffmpeg` (which `cv2` and `librosa` might rely on under the hood for untrimmed multi-hour MP4 files) are assumed.
   - **Solution**: Integrate a fast `brew install ffmpeg` block if running on macOS into any subsequent updated implementation plan to guarantee structural pipeline decoding for video analysis.
