"""
CodaBench Submission Formatter for Ego4D Goal-Step Step Grounding

Takes model predictions and formats them into the exact JSON schema
required by the CodaBench competition, then packages as a .zip file.

Submission format:
{
  "version": "1.0",
  "challenge": "ego4d_goalstep_challenge",
  "results": [
    {
      "clip_uid": "<video_uid>",
      "annotation_uid": "<video_uid>",
      "query_idx": 0,
      "predicted_times": [
        [start_1, end_1],  # Most confident
        [start_2, end_2],
        [start_3, end_3],
        [start_4, end_4],
        [start_5, end_5]   # Least confident
      ]
    }
  ]
}

Each entry MUST have exactly 5 predicted_times, ranked by confidence.
"""
import json
import os
import zipfile
import sys

SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
NUM_PREDICTIONS = 5


def validate_predictions(predictions: list) -> list:
    """
    Validate and normalize predictions to meet CodaBench requirements.

    Each prediction must have:
    - clip_uid (str)
    - annotation_uid (str)
    - query_idx (int)
    - predicted_times (list of exactly 5 [start, end] pairs)

    If fewer than 5 predictions exist, pad with [0.0, 0.0].
    If more than 5, truncate to top 5.
    """
    errors = []
    validated = []

    for i, pred in enumerate(predictions):
        # Check required fields
        for field in ["clip_uid", "annotation_uid", "query_idx", "predicted_times"]:
            if field not in pred:
                errors.append(f"Entry {i}: missing required field '{field}'")
                continue

        times = pred["predicted_times"]

        # Validate each time window
        valid_times = []
        for t in times:
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                errors.append(f"Entry {i}: invalid time window {t}")
                continue
            start, end = float(t[0]), float(t[1])
            if end < start:
                errors.append(f"Entry {i}: end ({end}) < start ({start}), swapping")
                start, end = end, start
            valid_times.append([round(start, 4), round(end, 4)])

        # Pad or truncate to exactly 5
        while len(valid_times) < NUM_PREDICTIONS:
            valid_times.append([0.0, 0.0])
        valid_times = valid_times[:NUM_PREDICTIONS]

        validated.append({
            "clip_uid": str(pred["clip_uid"]),
            "annotation_uid": str(pred["annotation_uid"]),
            "query_idx": int(pred["query_idx"]),
            "predicted_times": valid_times,
        })

    if errors:
        print(f"[WARNING] {len(errors)} validation issues:")
        for err in errors[:10]:
            print(f"  - {err}")

    return validated


def format_submission(predictions: list, output_dir: str = None) -> str:
    """
    Format predictions into CodaBench submission JSON and package as .zip.

    Args:
        predictions: List of prediction dicts with clip_uid, annotation_uid,
                     query_idx, and predicted_times.
        output_dir: Directory to save the submission files.

    Returns:
        Path to the .zip file ready for upload.
    """
    if output_dir is None:
        output_dir = os.path.join(SSD_BASE, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Validate predictions
    validated = validate_predictions(predictions)
    print(f"Validated {len(validated)} predictions ({NUM_PREDICTIONS} windows each)")

    # Build submission JSON
    submission = {
        "version": "1.0",
        "challenge": "ego4d_goalstep_challenge",
        "results": validated,
    }

    # Write JSON
    json_path = os.path.join(output_dir, "submission.json")
    with open(json_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Submission JSON saved: {json_path}")

    # Package as .zip
    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, "submission.json")
    print(f"Submission ZIP saved: {zip_path}")

    # Summary stats
    unique_videos = len(set(p["clip_uid"] for p in validated))
    total_queries = len(validated)
    print(f"\nSubmission summary:")
    print(f"  Videos: {unique_videos}")
    print(f"  Queries: {total_queries}")
    print(f"  Predictions per query: {NUM_PREDICTIONS}")
    print(f"  Ready for upload: {zip_path}")

    return zip_path


def generate_dummy_submission(test_annotations_path: str = None) -> str:
    """
    Generate a dummy submission from test annotations for format validation.

    Each query gets 5 uniform predictions spanning the video duration.
    This is useful for testing the submission pipeline without a real model.
    """
    if test_annotations_path is None:
        test_annotations_path = os.path.join(SSD_BASE, "annotations/goalstep_test.json")

    with open(test_annotations_path, 'r') as f:
        raw = json.load(f)

    # We need video durations from ego4d.json
    metadata_path = os.path.join(SSD_BASE, "annotations/ego4d.json")
    video_durations = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        video_durations = {
            v["video_uid"]: v.get("duration_sec", 300.0)
            for v in metadata.get("videos", [])
        }

    predictions = []
    for video_entry in raw.get("videos", raw):
        video_uid = video_entry["video_uid"]
        duration = video_durations.get(video_uid, 300.0)

        for idx, segment in enumerate(video_entry.get("step_segments", [])):
            # Generate 5 evenly-spaced windows as dummy predictions
            window_size = duration / 10
            predicted_times = []
            for k in range(NUM_PREDICTIONS):
                offset = k * (duration / NUM_PREDICTIONS)
                start = round(offset, 4)
                end = round(min(offset + window_size, duration), 4)
                predicted_times.append([start, end])

            predictions.append({
                "clip_uid": video_uid,
                "annotation_uid": video_uid,
                "query_idx": idx,
                "predicted_times": predicted_times,
            })

    print(f"Generated {len(predictions)} dummy predictions")
    return format_submission(predictions)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--dummy":
        # Generate a dummy submission for format testing
        zip_path = generate_dummy_submission()
        print(f"\nDummy submission ready: {zip_path}")
    elif len(sys.argv) > 1:
        # Format a real predictions file
        predictions_path = sys.argv[1]
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        zip_path = format_submission(predictions)
        print(f"\nSubmission ready: {zip_path}")
    else:
        print("Usage:")
        print("  python format_submission.py --dummy           # Generate dummy submission")
        print("  python format_submission.py predictions.json  # Format real predictions")
