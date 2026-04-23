# Phase 1: Data Acquisition & Annotation Parsing

## Overview
Download Ego4D Goal-Step videos, annotations, and pre-extracted features. Parse all three splits into our internal schema, handling the different test-set format.

## Data Storage Layout

```
/Volumes/Extreme SSD/ego4d_data/
├── annotations/
│   ├── goalstep_train.json        # Hierarchical: video → goal → step → substep
│   ├── goalstep_valid.json        # Same format as train
│   ├── goalstep_test.json         # Flat: video → step_segments (no timestamps)
│   ├── ego4d.json                 # Video metadata (durations, etc.)
│   └── goalstep_video_groups.tsv  # Video grouping info for test set
├── v2/
│   └── full_scale/                # Raw video files ({video_uid}.mp4)
├── v2/
│   └── omnivore_video_swinl/      # Pre-extracted features ({video_uid}.pt)
├── models/
│   ├── huggingface/
│   └── vslnet/                    # VSLNet checkpoints
├── cache/
│   ├── phase1/                    # Unified query index
│   ├── phase2/                    # Event annotation logs
│   └── ...
└── results/
```

## Step 1.1: Install Ego4D CLI

```bash
pip install ego4d
```

## Step 1.2: Download Goal-Step Data

```bash
# Download goalstep videos & annotations
ego4d --datasets full_scale annotations --benchmark goalstep -o /Volumes/Extreme\ SSD/ego4d_data

# Download goalstep annotations only (faster, for initial setup)
ego4d --datasets annotations --benchmark goalstep -o /Volumes/Extreme\ SSD/ego4d_data
```

## Step 1.3: Download Pre-Extracted Features

The official VSLNet baseline requires Omnivore video features (SwinL backbone):

```bash
# Download omnivore features for goalstep videos
ego4d --datasets omnivore_video_swinl --benchmark goalstep -o /Volumes/Extreme\ SSD/ego4d_data
```

## Step 1.4: Download Video Metadata

The test set parser requires `ego4d.json` (video durations) and `goalstep_video_groups.tsv`:

```bash
# ego4d.json should come with the annotations download
# goalstep_video_groups.tsv is in the goalstep annotations
```

## Annotation Schema

### Train/Val Format (Hierarchical)
```json
{
  "videos": [{
    "video_uid": "9b58e3ab-...",
    "start_time": 0.0,
    "end_time": 510.18,
    "goal_category": "COOKING:MAKE_OMELET",
    "goal_description": "Make omelette",
    "segments": [{
      "start_time": 0, "end_time": 56.99,
      "step_description": "Toast bread",
      "segments": [{
        "start_time": 0, "end_time": 13.13,
        "step_description": "preheat the stove-top"
      }]
    }]
  }]
}
```

### Test Format (Flat, No Timestamps)
```json
{
  "videos": [{
    "video_uid": "...",
    "step_segments": [{
      "step_description": "preheat the stove-top"
    }]
  }]
}
```

> [!CAUTION]
> The test set has **no ground-truth timestamps** and uses `step_segments` (flat list) instead of the nested `segments → segments` hierarchy. The `query_idx` for submission is the index within this `step_segments` list.

## Step 1.5: Parse Annotations

The parsing script (`scripts/parse_annotations.py`) handles both formats:
- **Train/Val:** Walks the `goal → step → substep` hierarchy, flattening steps and substeps into `StepQuery` objects with ground-truth timestamps.
- **Test:** Reads the flat `step_segments` list — no timestamps, just step descriptions and query indices.

## Step 1.6: Export to NaQ Format

For the VSLNet baseline, annotations must be converted to NaQ format using the official `parse_goalstep_jsons.py` script from `facebookresearch/ego4d-goalstep`. This produces `train.json`, `val.json`, and `test.json` in the NaQ schema.

## Verification Strategy
- **Annotation Counts:** Verify train/val/test split sizes match expected numbers.
- **Schema Validation:** Assert train/val entries have `start_time`/`end_time`; test entries do not.
- **Feature Coverage:** Verify that every video_uid in the annotations has a corresponding `.pt` feature file.
