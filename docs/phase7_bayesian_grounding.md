# Phase 7: Focused Bayesian Grounding (High-Res Pass — ToC Pipeline)

## Overview
While the Librarian gives us a general "temporal neighborhood" or "chapter", precise step grounding requires highly accurate timestamps. Instead of running heavy feature extraction on the *entire* video, we only extract features for the specific chapter identified by the Librarian. This restricted matrix is fed into a `BayesianVSLNet` head to generate a precise posterior probability distribution for the `(start, end)` boundaries.

By doing this, we achieve massive speed-ups while maintaining state-of-the-art localization boundaries using Bayesian test-time priors.

> [!IMPORTANT]
> **Feature Backbone Alignment:** The original BayesianVSLNet (CVPR 2024 winner) was trained using **Omnivore-L + EgoVLPv2** concatenated visual features, with **EgoVLPv2 dual-encoder** text features. Using mismatched feature extractors will produce meaningless predictions. We must either:
> - (a) Use the exact same feature backbones, or
> - (b) Retrain the VSLNet head on our chosen features (not recommended without GPU cluster access)
>
> **Cross-Dataset Consideration:** Since we are evaluating on EPIC-KITCHENS-100, Charades-Ego, and EgoProceL (not Ego4D), pre-extracted Ego4D-native features won't be available. We will need to extract features from raw video using the same backbone models, or use a lightweight adaptation strategy (see Step 7.1).

## Theoretical Pipeline
1. Load the `start_time` and `end_time` bounds from the Librarian.
2. Pad the bounds by $\pm 10\%$ to ensure we don't clip the action.
3. Extract high-resolution visual features using **Omnivore-L + EgoVLPv2** (concatenated) mapped to `v_features`.
4. Tokenize the Target Step query into linguistic features `t_features` using **EgoVLPv2 text encoder**.
5. Pass `v_features`, `t_features`, and temporal masks into the `BayesianVSLNet` forward logic to get start/end frame logits.
6. Enforce a Bayesian Order Prior to ensure the predicted `start` strictly precedes the `end`.

## Feature Extraction Strategy

The BayesianVSLNet repository ([cplou99/BayesianVSLNet](https://github.com/cplou99/BayesianVSLNet)) expects:
- **Video features** pre-extracted and stored at `./data/features/` — concatenated Omnivore-L (768-d) + EgoVLPv2 dual-encoder (256-d) = **1024-d per frame**.
- **Text features** extracted using EgoVLPv2 weights stored at `./model/EgoVLP_weights/`.

Since our datasets (EPIC-KITCHENS-100, Charades-Ego, EgoProceL) don't come with Ego4D-format pre-extracted features, we have two options:
1. **On-the-fly extraction:** Load Omnivore-L and EgoVLPv2 models and extract features from raw video frames for the windowed chapter only. This is feasible because the ToC narrows the window significantly.
2. **Pre-extract and cache:** Run a one-time feature extraction pass over all videos and store on the SSD. More upfront cost but faster per-query inference.

> [!NOTE]
> For the initial proof-of-concept, option 1 (on-the-fly extraction within the ToC window) is preferred. It directly demonstrates the efficiency advantage: we only need to extract features for a small temporal slice, not the full video.

## Pseudocode Implementation

```python
import json
import os
import torch
import numpy as np

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"
FEATURE_DIR = os.path.join(SSD_BASE, "features")

# ---- Feature Loading (preferred: use pre-extracted features) ----

def load_preextracted_features(video_id: str, start_sec: float, end_sec: float, feature_fps: float = 1.875):
    """
    Load pre-extracted Omnivore-L + EgoVLPv2 features for a temporal window.
    
    Falls back gracefully if pre-extracted features don't exist for the
    target dataset (EPIC-KITCHENS, Charades-Ego, or EgoProceL).
    
    Args:
        video_id: Video identifier (dataset-prefixed)
        start_sec: Window start time in seconds
        end_sec: Window end time in seconds
        feature_fps: Feature sampling rate (features per second)
    
    Returns:
        Tensor of shape [1, num_frames, feature_dim]
    """
    # Omnivore features
    omnivore_path = os.path.join(FEATURE_DIR, f"omnivore/{video_id}.npy")
    # EgoVLPv2 features  
    egovlp_path = os.path.join(FEATURE_DIR, f"egovlpv2/{video_id}.npy")
    
    omnivore_feats = np.load(omnivore_path)  # shape: [total_frames, 768]
    egovlp_feats = np.load(egovlp_path)      # shape: [total_frames, 256]
    
    # Compute frame indices for the temporal window
    start_idx = max(0, int(start_sec * feature_fps))
    end_idx = min(len(omnivore_feats), int(end_sec * feature_fps) + 1)
    
    # Slice and concatenate
    omni_slice = omnivore_feats[start_idx:end_idx]   # [N, 768]
    ego_slice = egovlp_feats[start_idx:end_idx]      # [N, 256]
    
    combined = np.concatenate([omni_slice, ego_slice], axis=1)  # [N, 1024]
    
    return torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # [1, N, 1024]

def load_video_frames_fallback(video_path: str, start_sec: float, end_sec: float, target_fps: float = 1.875) -> torch.Tensor:
    """
    Fallback: extract raw frames and run through feature encoders.
    Only use this if pre-extracted features are unavailable.
    
    Returns frames as tensor [1, num_frames, 3, H, W] ready for encoding.
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if native_fps <= 0:
        raise ValueError(f"Cannot read FPS from {video_path}")
    
    # Calculate frame sampling interval to match target feature fps
    frame_interval = int(native_fps / target_fps)
    
    start_frame = int(start_sec * native_fps)
    end_frame = int(end_sec * native_fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if (current_frame - start_frame) % frame_interval == 0:
            # Resize to model input size and normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames.append(frame_tensor)
        
        current_frame += 1
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames extracted from {video_path} [{start_sec}s - {end_sec}s]")
    
    return torch.stack(frames).unsqueeze(0)  # [1, num_frames, 3, 224, 224]

# ---- Text Feature Extraction ----

def extract_text_features(target_step: str, device: torch.device):
    """
    Extract text features using EgoVLPv2 text encoder (matching BayesianVSLNet training).
    
    Falls back to a simpler text encoder if EgoVLPv2 weights are unavailable.
    """
    # Primary: EgoVLPv2 text encoder
    try:
        from egovlpv2.model import EgoVLPv2
        text_encoder = EgoVLPv2.load_text_encoder(
            os.path.join(SSD_BASE, "models/EgoVLP_weights")
        ).to(device).eval()
        
        with torch.no_grad():
            t_features = text_encoder.encode_text(target_step)
        t_mask = torch.ones((1, t_features.shape[1]), dtype=torch.bool).to(device)
        return t_features, t_mask
        
    except ImportError:
        print("[WARNING] EgoVLPv2 not available, falling back to RoBERTa (may degrade accuracy)")
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        text_encoder = RobertaModel.from_pretrained('roberta-base').to(device).eval()
        
        text_inputs = tokenizer(target_step, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            t_features = text_encoder(**text_inputs).last_hidden_state
        t_mask = text_inputs.attention_mask
        return t_features, t_mask

# ---- Bayesian Grounding ----

class BayesianGrounder:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading BayesianVSLNet onto {self.device}...")
        
        # Load Bayesian VSLNet Head (pretrained weights from cplou99/BayesianVSLNet)
        from bayesian_vslnet.model import BayesianVSLNet
        self.vsl_head = BayesianVSLNet(hidden_dim=1024).to(self.device).eval()
        
        # Load pretrained checkpoint
        ckpt_path = os.path.join(SSD_BASE, "models/bayesian_vslnet/best_model.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.vsl_head.load_state_dict(state_dict)
            print(f"Loaded BayesianVSLNet checkpoint from {ckpt_path}")
        else:
            print(f"[WARNING] No checkpoint found at {ckpt_path} — using random weights")
        
    def refine_timestamps(self, video_id: str, target_step: str, chapter: dict, 
                          video_path: str = None, feature_fps: float = 1.875) -> dict:
        start_t = chapter.get("start_time", 0)
        end_t = chapter.get("end_time", 0)
        
        # 1. Temporal Padding
        duration = end_t - start_t
        pad = max(duration * 0.1, 2.0)
        padded_start = max(0, start_t - pad)
        padded_end = end_t + pad
        
        # 2. Load Video Features (prefer pre-extracted)
        try:
            v_features = load_preextracted_features(
                video_id, padded_start, padded_end, feature_fps
            ).to(self.device)
        except FileNotFoundError:
            print(f"[FALLBACK] Pre-extracted features not found for {video_id}, extracting from video...")
            if video_path is None:
                raise ValueError(f"No video_path provided and pre-extracted features missing for {video_id}")
            raw_frames = load_video_frames_fallback(video_path, padded_start, padded_end, feature_fps)
            # Would need to run through Omnivore + EgoVLPv2 encoders here
            raise NotImplementedError("Live feature extraction requires Omnivore + EgoVLPv2 models")
        
        # 3. Extract Text Features
        t_features, t_mask = extract_text_features(target_step, self.device)
        
        num_frames = v_features.shape[1]
        v_mask = torch.ones((1, num_frames), dtype=torch.bool).to(self.device)
        
        # 4. Bayesian VSL Forward Pass 
        with torch.no_grad():
            start_logits, end_logits = self.vsl_head(
                v_features=v_features,
                t_features=t_features,
                v_mask=v_mask,
                t_mask=t_mask
            )
            
        # Convert logits to probability distributions
        p_start = torch.softmax(start_logits.squeeze(), dim=0)  # [num_frames]
        p_end = torch.softmax(end_logits.squeeze(), dim=0)      # [num_frames]
        
        # 5. Bayesian Test-Time Prior Application
        # Create 2D joint probability matrix P(start=i, end=j)
        p_joint = p_start.unsqueeze(1) * p_end.unsqueeze(0)  # [num_frames, num_frames]
        
        # Non-causal prior: zero out states where start > end
        upper_tri_mask = torch.triu(torch.ones(num_frames, num_frames)).to(self.device)
        p_joint = p_joint * upper_tri_mask
        
        # Maximum A Posteriori (MAP) estimation
        max_idx = p_joint.argmax()
        start_idx = (max_idx // num_frames).item()
        end_idx = (max_idx % num_frames).item()
        
        # 6. Transform back to global temporal space using feature_fps (not video fps)
        refined_start = padded_start + (start_idx / feature_fps)
        refined_end = padded_start + (end_idx / feature_fps)
        
        return {
            "refined_start": round(refined_start, 2),
            "refined_end": round(refined_end, 2),
            "features_processed": num_frames
        }

if __name__ == '__main__':
    grounder = BayesianGrounder()
    
    test_chapter = {"start_time": 10.0, "end_time": 25.0}
    result = grounder.refine_timestamps(
        "P01_101",
        "wash the pan",
        test_chapter
    )
    print(f"Refined: {result}")
    
    # Save
    out_path = os.path.join(SSD_BASE, "cache/phase7/sample_refined.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
```

## Verification Strategy
- **Prior Mask Check:** In a unit test, pass dummy logit vectors mapped to `[1, 0, 0]` and `[1, 0, 0]`. Ensure the Bayesian Prior matrix `p_joint` multiplication successfully nullifies any outcome where the start index is larger than the end index.
- **Feature Dimension Check:** Assert that `v_features.shape[-1] == 1024` after Omnivore-L (768) + EgoVLPv2 (256) concatenation.
- **Bounding Box Drift Check:** Run a validation sample where truth bounds are known. Compare the `refined_start` and `refined_end` against the ground truth label and assert IoU > 0.5.
- **FPS Consistency:** Assert that `feature_fps` matches the actual temporal stride of the pre-extracted feature files by checking `num_features * feature_fps ≈ video_duration`.
