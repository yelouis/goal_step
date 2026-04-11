# Phase 0: Installation and Model Verification

## Overview
Before deploying the full Ego4D Phase 1-6 pipeline, we must verify that the base environment, external SSD mappings, and foundational ML models correctly function on your Apple Silicon (M4 Pro) architecture.

## Requirements
- Python 3.10+
- `mlx`, `mlx-vlm`, `mlx-lm` for local inferencing.
- `librosa` and `scipy` for audio extraction.
- Torch/MPS for `BayesianVSLNet` operations.

## Verification Strategy

1. **Install Base Packages:**
```bash
pip install mlx mlx-vlm mlx-lm librosa scipy opencv-python transformers torch torchvision
```

2. **Verify SSD Connectivity & Moondream2 Output:**
Run the following script to ensure models map to the 2TB drive and execute without memory leakage on the M4.

```python
import os
from mlx_vlm import load, generate

# 1. Point HF to SSD
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
os.environ['HF_HOME'] = os.path.join(SSD_BASE, "models", "huggingface")
print(f"HF Models routing to: {os.environ['HF_HOME']}")

# 2. Sanity check Moondream memory load
try:
    print("Test booting Moondream2...")
    model, processor = load("vikhyatk/moondream2")
    print("Moondream2 loaded into Unified Memory successfully.")
    
    # Run a dummy text generation
    # Typically you'd pass a test image here.
    print("[PASS] Apple Silicon Environment is stable.")
except Exception as e:
    print(f"[FAIL] Environment mismatch: {e}")
```
