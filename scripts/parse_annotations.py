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
        print(f"\\n[WARNING] {len(missing)} videos missing from disk:")
        for uid in missing[:10]:
            print(f"  - {uid}")
    
    if corrupt:
        print(f"\\n[ERROR] {len(corrupt)} videos unreadable:")
        for uid in corrupt:
            print(f"  - {uid}")
    
    return len(missing) == 0 and len(corrupt) == 0

if __name__ == '__main__':
    # Parse all available splits
    for split in ["train", "val", "test_unannotated"]:
        try:
            videos = parse_goalstep_annotations(split)
            
            # Print a sample for sanity checking
            if videos:
                sample_vid = next(iter(videos.values()))
                if sample_vid.queries:
                    q = sample_vid.queries[0]
                    print(f"  Sample query: [{q.goal_category}] {q.step_description}")
                    if q.gt_start_time >= 0:
                        print(f"  Ground truth: [{q.gt_start_time:.1f}s - {q.gt_end_time:.1f}s]")
        except FileNotFoundError:
            print(f"  {split} annotations not downloaded yet -- skipping")
    
    # Wait to build the uid list until annotations are actually there
    if os.path.exists(os.path.join(SSD_BASE, "annotations/goalstep_val.json")):
        build_video_uid_list("val")
    if os.path.exists(os.path.join(SSD_BASE, "annotations/goalstep_train.json")):
        build_video_uid_list("train")
    
    print("Verifying data integrity for val split:")
    try:
        verify_data_integrity("val")
    except FileNotFoundError:
       print("  val annotations not downloaded yet -- skipping integrity check")
