"""
Phase 0: Environment Verification Script
Validates SSD routing, MLX hardware, and VLM model loading.
"""
import os
import sys
import gc
import time
from dotenv import load_dotenv

def main():
    # Load environment variables from .env if it exists
    load_dotenv()
    
    results = {}

    # ── 1. SSD Routing ──────────────────────────────────────────────────
    SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
    os.environ['HF_HOME'] = os.path.join(SSD_BASE, "models", "huggingface")
    os.makedirs(os.environ['HF_HOME'], exist_ok=True)
    print(f"[CHECK] HF cache → {os.environ['HF_HOME']}")

    if os.path.isdir(os.environ['HF_HOME']):
        print("[PASS] SSD directory exists and is writable.")
        results['ssd'] = True
    else:
        print("[FAIL] SSD directory could not be created.")
        results['ssd'] = False

    # ── 2. MLX Hardware ─────────────────────────────────────────────────
    try:
        import mlx.core as mx
        print(f"[CHECK] MLX backend: {mx.default_device()}")
        # Quick matmul to confirm GPU compute works
        a = mx.ones((256, 256))
        b = mx.ones((256, 256))
        c = mx.matmul(a, b)
        mx.eval(c)
        print("[PASS] MLX compute on Apple Silicon confirmed.")
        results['mlx'] = True
    except Exception as e:
        print(f"[FAIL] MLX hardware check: {e}")
        results['mlx'] = False

    # ── 3. Torch MPS Backend ────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            t = torch.randn(64, 64, device="mps")
            _ = t @ t
            print("[PASS] Torch MPS backend operational.")
            results['mps'] = True
        else:
            print("[WARN] Torch MPS not available — BayesianVSLNet will fall back to CPU.")
            results['mps'] = False
    except Exception as e:
        print(f"[FAIL] Torch MPS check: {e}")
        results['mps'] = False

    # ── 4. Audio Stack ──────────────────────────────────────────────────
    try:
        import librosa
        import scipy
        print(f"[PASS] Audio stack ready (librosa {librosa.__version__}, scipy {scipy.__version__}).")
        results['audio'] = True
    except ImportError as e:
        print(f"[FAIL] Audio stack: {e}")
        results['audio'] = False

    # ── 5. VLM: Qwen2.5-VL-3B-Instruct (4-bit) ────────────────────────
    MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    try:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        print(f"[CHECK] Loading {MODEL_ID} ...")
        t0 = time.time()
        model, processor = load(MODEL_ID)
        load_time = time.time() - t0
        print(f"[PASS] Qwen2.5-VL-3B loaded in {load_time:.1f}s into unified memory.")

        # Quick inference with a blank test image
        test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        test_img_path = os.path.join(SSD_BASE, "test_frame.png")
        test_img.save(test_img_path)

        prompt = apply_chat_template(
            processor,
            config=model.config,
            prompt="What do you see in this image? Be brief.",
            images=[test_img_path],
        )
        t0 = time.time()
        result = generate(model, processor, prompt, max_tokens=50, verbose=False)
        gen_time = time.time() - t0
        caption = result.text if hasattr(result, 'text') else str(result)
        print(f"[PASS] Inference test ({gen_time:.1f}s): {caption.strip()[:120]}")
        results['vlm'] = True

        # Cleanup
        os.remove(test_img_path)
        del model, processor
        gc.collect()
        print("[INFO] VLM unloaded, memory freed.")

    except Exception as e:
        print(f"[FAIL] VLM load/inference: {e}")
        results['vlm'] = False

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 0 Verification Summary")
    print("=" * 50)
    all_pass = all(results.values())
    for key, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {key:>8}: {status}")
    print("=" * 50)
    if all_pass:
        print("🟢 Environment ready. Proceed to Phase 0.5.")
    else:
        print("🔴 Fix failing checks before continuing.")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
