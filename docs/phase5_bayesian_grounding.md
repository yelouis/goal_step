# Phase 5: Focused Bayesian Grounding (High-Res Pass)

## Overview
While the Librarian gives us a general "temporal neighborhood" or "chapter", the Ego4D challenge requires highly precise timestamps. 
Instead of running heavy feature extraction (OmniMAE) on the *entire* video, we only extract features for the specific chapter identified by the Librarian. This restricted matrix is fed into a `BayesianVSLNet` head to generate a precise posterior probability distribution for the `(start, end)` boundaries.

By doing this, we achieve massive speed-ups while maintaining state-of-the-art localization boundaries using Bayesian test-time priors.

## Theoretical Pipeline
1. Load the `start_time` and `end_time` bounds from the Librarian.
2. Pad the bounds by $\pm 10\%$ to ensure we don't clip the action.
3. Extract high-resolution OmniMAE features mapped to `v_features`.
4. Tokenize the Target Step query into linguistic features `t_features`.
5. Pass `v_features`, `t_features`, and temporal masks into the `BayesianVSLNet` forward logic to get start/end frame logits.
6. Enforce a Bayesian Order Prior to ensure the predicted `start` strictly precedes the `end`.

## Pseudocode Implementation

```python
import json
import os
import torch
import numpy as np

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"

# Assumed dependencies from typical ego4d backbones and HuggingFace
from transformers import RobertaTokenizer, RobertaModel
from omnimae.model import OmniMAE
from bayesian_vslnet.model import BayesianVSLNet

class BayesianGrounder:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading Models onto {self.device}...")
        
        # Load Video Backbone
        self.video_encoder = OmniMAE.from_pretrained("omnimae_ego4d_weights").to(self.device).eval()
        
        # Load Text Backbone (Typically RoBERTa for VSLNet architectures)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base').to(self.device).eval()
        
        # Load Bayesian VSLNet Head
        self.vsl_head = BayesianVSLNet(hidden_dim=768).to(self.device).eval()
        
    def refine_timestamps(self, video_id: str, target_step: str, chapter: dict) -> dict:
        start_t = chapter.get("start_time", 0)
        end_t = chapter.get("end_time", 0)
        
        # 1. Temporal Padding
        duration = end_t - start_t
        pad = max(duration * 0.1, 2.0)
        padded_start = max(0, start_t - pad)
        padded_end = end_t + pad
        video_path = os.path.join(SSD_BASE, f"videos/{video_id}.mp4")
        
        # 2. Extract Text Features (t_features)
        text_inputs = self.tokenizer(target_step, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            t_features = self.text_encoder(**text_inputs).last_hidden_state # shape: [1, num_tokens, feature_dim]
        t_mask = text_inputs.attention_mask # shape: [1, num_tokens]
        
        # 3. Extract Video Features (v_features)
        # Note: In practice, this requires a video loader chunking the frames from padded_start to padded_end
        video_frames = load_video_frames(video_path, padded_start, padded_end).to(self.device) 
        with torch.no_grad():
            v_features = self.video_encoder(video_frames) # shape: [1, num_frames, feature_dim]
            
        num_frames = v_features.shape[1]
        v_mask = torch.ones((1, num_frames), dtype=torch.bool).to(self.device) # Assume all frames are valid
        
        # 4. Bayesian VSL Forward Pass 
        # Output is logits for each frame being a start/end marker
        with torch.no_grad():
            start_logits, end_logits = self.vsl_head(
                v_features=v_features,
                t_features=t_features,
                v_mask=v_mask,
                t_mask=t_mask
            )
            
        # Convert logits to probability distributions representing P(start) and P(end)
        p_start = torch.softmax(start_logits.squeeze(), dim=0) # shape: [num_frames]
        p_end = torch.softmax(end_logits.squeeze(), dim=0)     # shape: [num_frames]
        
        # 5. Bayesian Test-Time Prior Application
        # Create a 2D joint probability matrix P(start=i, end=j)
        p_joint = p_start.unsqueeze(1) * p_end.unsqueeze(0) # shape [num_frames, num_frames]
        
        # The non-causal test time prior masks out states where start > end
        # Zero out the lower triangle
        upper_tri_mask = torch.triu(torch.ones(num_frames, num_frames)).to(self.device)
        p_joint = p_joint * upper_tri_mask
        
        # (Optional BayesianVSLNet refinement: apply Gaussian penalty based on average action length duration)
        
        # Maximum A Posteriori (MAP) estimation: Find the (i,j) coordinates of the highest probability
        max_idx = p_joint.argmax()
        start_idx = (max_idx // num_frames).item()
        end_idx = (max_idx % num_frames).item()
        
        # 6. Transform back to global temporal space
        fps = 30.0 # Extracted feature rate
        refined_start = padded_start + (start_idx / fps)
        refined_end = padded_start + (end_idx / fps)
        
        return {
            "refined_start": round(refined_start, 2),
            "refined_end": round(refined_end, 2)
        }

def load_video_frames(path: str, start: float, end: float) -> torch.Tensor:
    # Dummy video loader logic handling OpenCV/Decord extraction
    # Replaces mock torch.rand used in earlier planning
    pass
```

## Verification Strategy
- **Prior Mask Check:** In a unit test, pass dummy logit vectors mapped to `[1, 0, 0]` and `[1, 0, 0]`. Ensure the Bayesian Prior matrix `p_joint` multiplication successfully nullifies any outcome where the start index is larger than the end index.
- **Bounding Box Drift Check:** Run an Ego4D dataset sample where truth bounds are `[12.0s, 15.0s]`. Compare the `refined_start` and `refined_end` against this label and assert that the generated bounding box has an Intersection over Union (IoU) > 0.5.
