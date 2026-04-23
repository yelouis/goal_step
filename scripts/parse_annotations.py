"""
Ego4D Goal-Step Annotation Parser

Handles all three splits:
- train/valid: Hierarchical goal → step → substep with ground-truth timestamps
- test: Flat step_segments list with no timestamps

Output formats:
1. Internal StepQuery schema for pipeline consumption
2. NaQ-compatible format for VSLNet baseline
"""
import json
import os
import collections
from dataclasses import dataclass, field
from typing import Optional

SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"


# ─── Internal Schema ─────────────────────────────────────────────────────────

@dataclass
class StepQuery:
    """A single grounding query extracted from the annotation hierarchy."""
    video_uid: str
    annotation_uid: str
    query_idx: int
    step_description: str
    # Context (available in train/val, not test)
    goal_description: str = ""
    goal_category: str = ""
    # Ground truth (only available in train/val)
    gt_start_time: float = -1.0
    gt_end_time: float = -1.0
    # Hierarchy level
    level: str = "step"  # "step" or "substep"


@dataclass
class VideoAnnotation:
    """All queries associated with a single video."""
    video_uid: str
    video_duration: float
    queries: list = field(default_factory=list)


# ─── Train/Val Parser (Hierarchical) ─────────────────────────────────────────

def parse_goalstep_annotations(split: str = "val") -> dict:
    """
    Parse hierarchical Goal-Step annotations into StepQuery objects.

    Train/Val format: video → segments (goals) → segments (steps) → segments (substeps)
    Each step AND substep becomes a separate StepQuery (matching official baseline behavior).

    Args:
        split: One of 'train', 'val' (maps to goalstep_valid.json)

    Returns:
        Dict mapping video_uid → VideoAnnotation
    """
    # Map split names to actual filenames
    filename_map = {
        "train": "goalstep_train.json",
        "val": "goalstep_valid.json",
        "valid": "goalstep_valid.json",
    }
    filename = filename_map.get(split)
    if not filename:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")

    ann_path = os.path.join(SSD_BASE, f"annotations/{filename}")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with open(ann_path, 'r') as f:
        raw = json.load(f)

    videos = {}
    stats = collections.Counter()

    for video_entry in raw.get("videos", raw):
        video_uid = video_entry["video_uid"]
        stats["videos"] += 1

        if video_uid not in videos:
            videos[video_uid] = VideoAnnotation(
                video_uid=video_uid,
                video_duration=video_entry.get("end_time", 0.0)
            )

        query_idx = 0

        # Walk goal → step → substep hierarchy
        for goal in video_entry.get("segments", []):
            goal_desc = goal.get("goal_description", "")
            goal_cat = goal.get("goal_category", "")

            # Steps
            for step in goal.get("segments", []):
                step_desc = step.get("step_description", "").strip()
                if not step_desc:
                    continue

                query = StepQuery(
                    video_uid=video_uid,
                    annotation_uid=video_uid,
                    query_idx=query_idx,
                    step_description=step_desc,
                    goal_description=goal_desc,
                    goal_category=goal_cat,
                    gt_start_time=float(step.get("start_time", -1.0)),
                    gt_end_time=float(step.get("end_time", -1.0)),
                    level="step",
                )
                videos[video_uid].queries.append(query)
                query_idx += 1
                stats["steps"] += 1

                # Substeps
                for substep in step.get("segments", []):
                    sub_desc = substep.get("step_description", "").strip()
                    if not sub_desc:
                        continue

                    query = StepQuery(
                        video_uid=video_uid,
                        annotation_uid=video_uid,
                        query_idx=query_idx,
                        step_description=sub_desc,
                        goal_description=goal_desc,
                        goal_category=goal_cat,
                        gt_start_time=float(substep.get("start_time", -1.0)),
                        gt_end_time=float(substep.get("end_time", -1.0)),
                        level="substep",
                    )
                    videos[video_uid].queries.append(query)
                    query_idx += 1
                    stats["substeps"] += 1

    stats["queries"] = stats["steps"] + stats["substeps"]
    print(f"Parsed {split}: {stats}")
    return videos


# ─── Test Parser (Flat step_segments) ─────────────────────────────────────────

def parse_goalstep_test(ann_path: Optional[str] = None) -> dict:
    """
    Parse the test split which uses a flat step_segments format.

    Test format: video → step_segments (list of {step_description})
    No ground-truth timestamps. No goal hierarchy.

    Returns:
        Dict mapping video_uid → VideoAnnotation (queries have no GT times)
    """
    if ann_path is None:
        ann_path = os.path.join(SSD_BASE, "annotations/goalstep_test.json")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Test annotation file not found: {ann_path}")

    with open(ann_path, 'r') as f:
        raw = json.load(f)

    videos = {}
    stats = collections.Counter()

    for video_entry in raw.get("videos", raw):
        video_uid = video_entry["video_uid"]
        stats["videos"] += 1

        videos[video_uid] = VideoAnnotation(
            video_uid=video_uid,
            video_duration=0.0  # Duration not in test annotations
        )

        for idx, segment in enumerate(video_entry.get("step_segments", [])):
            step_desc = segment.get("step_description", "").strip()
            if not step_desc:
                continue

            query = StepQuery(
                video_uid=video_uid,
                annotation_uid=video_uid,
                query_idx=idx,
                step_description=step_desc,
                # No ground truth in test
                gt_start_time=-1.0,
                gt_end_time=-1.0,
                level="step",
            )
            videos[video_uid].queries.append(query)
            stats["queries"] += 1

    print(f"Parsed test: {stats}")
    return videos


# ─── NaQ-Compatible Export ────────────────────────────────────────────────────

def export_to_naq_format(videos: dict, split: str, out_dir: str):
    """
    Export parsed annotations to NaQ-compatible JSON format.

    This produces the same output as the official parse_goalstep_jsons.py
    from facebookresearch/ego4d-goalstep.
    """
    os.makedirs(out_dir, exist_ok=True)

    video_annots = []
    for video_uid, ann in videos.items():
        start_time = 0.0
        end_time = ann.video_duration

        language_queries = []
        for q in ann.queries:
            query_entry = {"query": q.step_description}
            if q.gt_start_time >= 0:
                query_entry["clip_start_sec"] = q.gt_start_time
                query_entry["clip_end_sec"] = q.gt_end_time
            language_queries.append(query_entry)

        clip_uid = video_uid
        goal_clip_uid = f"{video_uid}_{start_time}_{end_time}"

        clip_annots = [{
            "clip_uid": clip_uid,
            "video_start_sec": start_time,
            "video_end_sec": end_time,
            "annotations": [{
                "language_queries": language_queries,
                "annotation_uid": goal_clip_uid if split != "test" else video_uid,
            }]
        }]

        video_annots.append({
            "video_uid": video_uid,
            "clips": clip_annots,
        })

    output = {
        "version": "v1",
        "date": "260422",
        "description": "Goal step annotations",
        "videos": video_annots,
    }

    out_path = os.path.join(out_dir, f"{split}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f)
    print(f"Exported NaQ format: {out_path} ({len(video_annots)} videos)")


# ─── Video UID List Export ────────────────────────────────────────────────────

def build_video_uid_list(videos: dict, split: str):
    """Export a text file of video UIDs for selective downloading."""
    out_path = os.path.join(SSD_BASE, f"annotations/goalstep_{split}_video_uids.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        for uid in sorted(videos.keys()):
            f.write(uid + "\n")

    print(f"Exported {len(videos)} video UIDs to {out_path}")


# ─── Data Integrity Check ────────────────────────────────────────────────────

def verify_data_integrity(videos: dict, check_features: bool = True):
    """Check that referenced videos and features exist on disk."""
    missing_videos = []
    missing_features = []

    for video_uid in videos:
        # Check video file
        video_path = os.path.join(SSD_BASE, f"v2/full_scale/{video_uid}.mp4")
        if not os.path.exists(video_path):
            missing_videos.append(video_uid)

        # Check feature file
        if check_features:
            feat_path = os.path.join(SSD_BASE, f"v2/omnivore_video_swinl/{video_uid}.pt")
            if not os.path.exists(feat_path):
                missing_features.append(video_uid)

    total = len(videos)
    print(f"\nIntegrity check: {total} videos")
    print(f"  Videos on disk: {total - len(missing_videos)}/{total}")
    if check_features:
        print(f"  Features on disk: {total - len(missing_features)}/{total}")

    if missing_videos:
        print(f"  [WARNING] {len(missing_videos)} videos missing (first 5):")
        for uid in missing_videos[:5]:
            print(f"    - {uid}")

    if missing_features:
        print(f"  [WARNING] {len(missing_features)} features missing (first 5):")
        for uid in missing_features[:5]:
            print(f"    - {uid}")

    return len(missing_videos) == 0


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Parse all available splits
    for split in ["train", "val"]:
        try:
            videos = parse_goalstep_annotations(split)
            build_video_uid_list(videos, split)

            # Print a sample
            if videos:
                sample_vid = next(iter(videos.values()))
                if sample_vid.queries:
                    q = sample_vid.queries[0]
                    print(f"  Sample [{q.level}]: {q.step_description}")
                    if q.gt_start_time >= 0:
                        print(f"  GT: [{q.gt_start_time:.1f}s - {q.gt_end_time:.1f}s]")
                    if q.goal_description:
                        print(f"  Goal: {q.goal_description} ({q.goal_category})")
        except FileNotFoundError as e:
            print(f"  {split}: {e}")

    # Parse test split (different format)
    try:
        test_videos = parse_goalstep_test()
        build_video_uid_list(test_videos, "test")

        if test_videos:
            sample_vid = next(iter(test_videos.values()))
            if sample_vid.queries:
                q = sample_vid.queries[0]
                print(f"  Test sample: [{q.query_idx}] {q.step_description}")
    except FileNotFoundError as e:
        print(f"  test: {e}")

    # Verify data integrity for val
    print("\n--- Data Integrity ---")
    try:
        val_videos = parse_goalstep_annotations("val")
        verify_data_integrity(val_videos, check_features=True)
    except FileNotFoundError:
        print("  val annotations not found, skipping integrity check")
