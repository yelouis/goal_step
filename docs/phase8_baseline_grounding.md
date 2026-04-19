# Phase 8: Full-Video Bayesian Baseline (Control Condition)

## Overview
To quantify the efficiency gain of the ToC pipeline, we must establish a **control condition**: running BayesianVSLNet on the *entire* video without any ToC-based narrowing. This phase produces the baseline predictions and logs the computational cost (wall-clock time, features processed, memory usage) for direct comparison with Phase 7's ToC-guided predictions.

> [!IMPORTANT]
> **This phase exists solely for comparison.** It demonstrates the cost of naively running BayesianVSLNet on full-length videos. The ToC pipeline (Phases 2–7) is the method we are evaluating; this phase provides the "what if we didn't do ToC" baseline.

## What This Phase Measures
1. **Accuracy baseline:** Does the ToC pipeline maintain accuracy compared to full-video inference?
2. **Efficiency gap:** How many fewer features does the ToC pipeline need to process?
3. **Latency difference:** How much faster is per-query inference with ToC narrowing?

## Pseudocode Implementation

```python
import json
import os
import time
import torch
import numpy as np

SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"
FEATURE_DIR = os.path.join(SSD_BASE, "features")

class FullVideoBaseline:
    """
    Run BayesianVSLNet on the entire video for a given text query.
    This is the control condition — no ToC narrowing is applied.
    """
    
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading BayesianVSLNet (baseline mode) onto {self.device}...")
        
        from bayesian_vslnet.model import BayesianVSLNet
        self.vsl_head = BayesianVSLNet(hidden_dim=1024).to(self.device).eval()
        
        ckpt_path = os.path.join(SSD_BASE, "models/bayesian_vslnet/best_model.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.vsl_head.load_state_dict(state_dict)
            print(f"Loaded BayesianVSLNet checkpoint from {ckpt_path}")
        else:
            print(f"[WARNING] No checkpoint found at {ckpt_path}")
    
    def predict_full_video(self, video_id: str, target_step: str, 
                            video_duration: float, feature_fps: float = 1.875) -> dict:
        """
        Run BayesianVSLNet on the FULL video (no windowing).
        
        Returns prediction + cost metrics for comparison with Phase 7.
        """
        start_wall = time.time()
        
        # 1. Load full video features
        omnivore_path = os.path.join(FEATURE_DIR, f"omnivore/{video_id}.npy")
        egovlp_path = os.path.join(FEATURE_DIR, f"egovlpv2/{video_id}.npy")
        
        omnivore_feats = np.load(omnivore_path)
        egovlp_feats = np.load(egovlp_path)
        
        combined = np.concatenate([omnivore_feats, egovlp_feats], axis=1)
        v_features = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        total_features = v_features.shape[1]
        
        # 2. Extract text features
        from phase7_bayesian_grounding import extract_text_features
        t_features, t_mask = extract_text_features(target_step, self.device)
        
        v_mask = torch.ones((1, total_features), dtype=torch.bool).to(self.device)
        
        # 3. Full-video forward pass
        with torch.no_grad():
            start_logits, end_logits = self.vsl_head(
                v_features=v_features,
                t_features=t_features,
                v_mask=v_mask,
                t_mask=t_mask
            )
        
        # 4. Bayesian prior + MAP estimation (same as Phase 7)
        p_start = torch.softmax(start_logits.squeeze(), dim=0)
        p_end = torch.softmax(end_logits.squeeze(), dim=0)
        
        p_joint = p_start.unsqueeze(1) * p_end.unsqueeze(0)
        upper_tri_mask = torch.triu(torch.ones(total_features, total_features)).to(self.device)
        p_joint = p_joint * upper_tri_mask
        
        max_idx = p_joint.argmax()
        start_idx = (max_idx // total_features).item()
        end_idx = (max_idx % total_features).item()
        
        predicted_start = start_idx / feature_fps
        predicted_end = end_idx / feature_fps
        
        wall_time = time.time() - start_wall
        
        # 5. Track peak memory (MPS doesn't have cuda memory tracking, approximate)
        peak_memory_mb = (v_features.element_size() * v_features.nelement()) / (1024 * 1024)
        
        return {
            "predicted_start": round(predicted_start, 2),
            "predicted_end": round(predicted_end, 2),
            "features_processed": total_features,
            "wall_clock_sec": round(wall_time, 4),
            "peak_feature_memory_mb": round(peak_memory_mb, 2),
            "video_duration_sec": video_duration,
            "method": "baseline_full_video"
        }


def run_baseline_for_dataset(dataset_queries: dict, output_dir: str):
    """
    Run the full-video baseline for all queries in a dataset split.
    
    Args:
        dataset_queries: Dict of video_id -> VideoAnnotation (from Phase 1)
        output_dir: Where to save baseline results
    """
    baseline = FullVideoBaseline()
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for video_id, annotation in dataset_queries.items():
        for query in annotation.queries:
            try:
                result = baseline.predict_full_video(
                    video_id=video_id,
                    target_step=query.step_description,
                    video_duration=annotation.video_duration
                )
                result["video_id"] = video_id
                result["query_idx"] = query.query_idx
                result["step_description"] = query.step_description
                result["gt_start"] = query.gt_start_time
                result["gt_end"] = query.gt_end_time
                
                all_results.append(result)
                
            except Exception as e:
                print(f"[ERROR] Baseline failed for {video_id} q{query.query_idx}: {e}")
    
    # Save all results
    out_path = os.path.join(output_dir, "baseline_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Baseline complete: {len(all_results)} predictions saved to {out_path}")
    return all_results


if __name__ == '__main__':
    # Example: run baseline on a single query
    baseline = FullVideoBaseline()
    result = baseline.predict_full_video(
        video_id="P01_101",
        target_step="wash the pan",
        video_duration=600.0  # 10 minutes
    )
    print(f"Baseline prediction: {result}")
    
    # Save
    out_path = os.path.join(SSD_BASE, "cache/phase8/sample_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
```

## Key Differences from Phase 7

| Aspect | Phase 7 (ToC Pipeline) | Phase 8 (Baseline) |
|---|---|---|
| **Feature window** | Librarian-identified chapter only | Entire video |
| **Features processed** | ~50–200 (typical chapter) | ~1,000–10,000+ (full video) |
| **Wall-clock time** | Milliseconds per query | Seconds per query |
| **Memory footprint** | Small feature tensor | Large feature tensor (may OOM on long videos) |
| **Accuracy** | TBD (should be competitive) | TBD (ground truth baseline) |

## Verification Strategy
- **Consistency Check:** For a known video, run both Phase 7 and Phase 8 with the same query. If the Librarian correctly identifies the chapter, both should produce similar `(start, end)` predictions (within $\pm$ a few seconds).
- **Cost Logging Validation:** Assert that `features_processed` in Phase 8 is always ≥ `features_processed` in Phase 7 for the same query.
- **Memory Safety:** For videos longer than 30 minutes, verify that the full-video feature tensor doesn't exceed the 24GB memory budget. If it does, implement feature chunking with sliding window inference.
