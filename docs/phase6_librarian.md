# Phase 6: The Librarian (Gemma 4 26B MoE Reasoning)

## Overview
The "Librarian" acts as the logical bridge between language (the target step query) and vision/audio (the Table of Contents generated in Phase 5). By feeding the entire ToC into an LLM, we can map a specific query (e.g., "Assemble wooden frame") to the most probable "Chapter" (a temporal slice) within the video. The ToC now includes both acoustic trigger captions and Phase 2 event annotations, giving the Librarian richer context.

The Librarian will handle situations where a step was captured perfectly via an action "Spike", and also complex interpolations where a step happened silently between two "Drop" triggers.

## LLM Strategy
Since this requires high logic, we utilize an advanced model (e.g., Gemma 4 26B MoE or Gemma 2 27B) running locally via `mlx-lm`. The MoE architecture activates only ~3.8B params per token, making inference fast on the M4 Pro despite the full 26B parameter count.

> [!IMPORTANT]
> **Memory Budget:** Gemma 4 26B at 4-bit quantization requires ~13-15GB of unified memory. Ensure Qwen2.5-VL-3B is unloaded before loading Gemma to stay within the 24GB budget. See the model lifecycle notes below.

## Model Lifecycle
To avoid OOM on the 24GB M4 Pro:
1. **Phase 5** loads Qwen2.5-VL-3B-Instruct-4bit (~2GB). After ToC generation for all videos, explicitly `del model` and `gc.collect()`.
2. **Phase 6** loads Gemma 4 26B (~14GB). Runs all Librarian queries, then unloads.
3. **Phase 7** loads BayesianVSLNet + text encoder (~4-8GB on MPS).

### Enhanced Prompt with Goal Context
When available, we pass contextual information (e.g., the goal description inferred from the dataset — EPIC-KITCHENS narrations, Charades scripts, or Ego4D task names) alongside the ToC to improve the Librarian's reasoning about step ordering and relevance.

## Pseudocode Implementation

```python
import json
import os
import re
import gc
from mlx_lm import load, generate

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
MODEL_CACHE = os.path.join(SSD_BASE, "models/gemma")

LIBRARIAN_PROMPT = """You are the "Procedural Video Librarian." Map the "Target Step" to the most likely chapters within a video's Table of Contents (ToC).

## Context
Goal Category: {goal_category}
Goal Description: {goal_description}

## Logical Rules
1. **Explicit Identification:** Prioritize chapters where the caption semantically matches the Target Step.
2. **Silent-Step Interpolation:** If the exact step is missing from captions, identify the **Pre-Condition** (what happens before) and **Post-Condition** (what happens after). The Target Step likely occurred in the temporal gap between them.
3. **Change Detection Logic:** Treat "drop" triggers as potential "Step Completion" markers and "spike" triggers as "Step Start" markers. A "spike" followed by a "drop" likely brackets one complete step.
4. **Event Sample Integration:** Entries with trigger_type "event_sample" are uniformly sampled observations from Phase 2. They provide continuous coverage of the video and can fill gaps between acoustic triggers.
5. **Temporal Ordering:** Steps in procedural activities follow a logical order. Use the goal description to reason about which steps must precede or follow the target step.

You MUST respond ONLY with valid JSON in the following format (no markdown, no explanation outside the JSON):
{{
    "top_chapters": [{{"start_time": X.X, "end_time": Y.Y}}],
    "confidence": 0.X,
    "reasoning": "brief explanation"
}}

Target Step: {target_step}

Table of Contents:
{toc_data}
"""

def extract_json_robust(text: str) -> dict:
    """
    Extract JSON from LLM output robustly, handling markdown backticks,
    preamble text, and nested structures.
    """
    # Strip markdown code fences if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try direct parse first (cleanest case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Find all candidate JSON objects by matching balanced braces
    depth = 0
    start_idx = None
    candidates = []
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(text[start_idx:i+1])
                start_idx = None
    
    # Try parsing each candidate, prefer the one with expected keys
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if "top_chapters" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Last resort: try the first candidate even without expected keys
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    raise ValueError(f"Could not extract valid JSON from LLM output: {text[:200]}...")

class VideoLibrarian:
    def __init__(self):
        print("Initializing Librarian Protocol (Gemma)...")
        os.environ['HF_HOME'] = MODEL_CACHE
        # Use Gemma 4 26B MoE (4-bit quantized for M4 Pro)
        # Adjust model ID to match the mlx-community quantized version
        self.model, self.tokenizer = load("mlx-community/gemma-4-27b-it-4bit") 
        
    def unload(self):
        """Explicitly free model memory for the next phase."""
        del self.model
        del self.tokenizer
        gc.collect()
        print("Librarian model unloaded from memory.")
        
    def find_step(self, video_uid: str, target_step: str, 
                  goal_description: str = "", goal_category: str = "",
                  max_retries: int = 3) -> dict:
        # Cache check — skip if this exact (video, query) was already processed
        import hashlib
        query_hash = hashlib.md5(target_step.encode()).hexdigest()[:8]
        cache_path = os.path.join(SSD_BASE, f"cache/phase6/{video_uid}_{query_hash}_hypothesis.json")
        if os.path.exists(cache_path):
            print(f"[CACHED] Librarian hypothesis for {video_uid} / '{target_step[:40]}...' already exists.")
            with open(cache_path, 'r') as f:
                return json.load(f)

        # Load the ToC
        toc_path = os.path.join(SSD_BASE, f"cache/phase5/{video_uid}_toc.json")
        with open(toc_path, 'r') as f:
            toc = json.load(f)
            
        # Format the Prompt with hierarchical context
        formatted_prompt = LIBRARIAN_PROMPT.format(
            target_step=target_step,
            goal_category=goal_category or "Unknown",
            goal_description=goal_description or "Unknown",
            toc_data=json.dumps(toc, indent=2)
        )
        
        # Apply chat template
        messages = [{"role": "user", "content": formatted_prompt}]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Retry loop for JSON compliance
        last_error = None
        for attempt in range(max_retries):
            print(f"Asking Librarian about: '{target_step}' (attempt {attempt + 1}/{max_retries})")
            
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=text_prompt, 
                max_tokens=512,
                temperature=0.1 + (attempt * 0.1)  # Increase temp on retries
            )
            
            try:
                structured_out = extract_json_robust(response)
                
                # Validate expected keys
                if "top_chapters" not in structured_out:
                    raise ValueError("Missing 'top_chapters' key")
                
                # Cache the successful result
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(structured_out, f, indent=4)
                    
                return structured_out
                
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                print(f"[Retry {attempt + 1}] JSON parse failed: {e}")
        
        # All retries exhausted
        print(f"[FAIL] Could not get valid JSON after {max_retries} attempts. Last error: {last_error}")
        print(f"Last raw output: {response[:300]}")
        return {
            "top_chapters": [],
            "confidence": 0.0,
            "reasoning": f"Parse failure after {max_retries} attempts."
        }

if __name__ == '__main__':
    librarian = VideoLibrarian()
    
    # Example with an EPIC-KITCHENS query
    query = "wash the pan"
    result = librarian.find_step(
        "video_uid_example", 
        query,
        goal_description="Kitchen activity by participant P01",
        goal_category="Cooking"
    )
    
    # Save the Librarian's hypothesis to the SSD
    out_path = os.path.join(SSD_BASE, "cache/phase6/sample_hypothesis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Hypothesis Generated with {result['confidence']*100}% confidence.")
    
    # Unload model before Phase 7
    librarian.unload()
```

## Session Resilience

| Item | Detail |
|---|---|
| **Input Dependencies** | `cache/phase5/{video_uid}_toc.json` (Phase 5), unified query index (Phase 1) |
| **Output Artifact** | `/Volumes/Extreme SSD/ego4d_data/cache/phase6/{video_uid}_{query_hash}_hypothesis.json` |
| **Cache Check** | On entry, `find_step()` checks if the (video_uid, query_hash) result already exists and returns early if so |
| **Verification Checkpoint** | After completing all queries, write `cache/phase6/_manifest.json` listing all processed (video_uid, query) pairs |
| **Resume Strategy** | On re-run, skip any (video, query) pair whose hypothesis JSON already exists on the SSD |

## Verification Strategy
- **JSON Schema Check:** Build a PyTest unit test that feeds the Librarian a mock `Target Step` and dummy `ToC array`. Assert that the resulting string successfully parses via `extract_json_robust()` and contains the keys `top_chapters`, `confidence`, and `reasoning`.
- **Interpolation Test:** Give the Librarian a ToC with "User grabs hammer" at 10.0s and "User puts hammer down" at 20.0s, and ask for the Target Step "User hits the nail". Assert that it identifies `[10.0, 20.0]` as the chapter.
- **Goal Context Test:** Run the same query with and without goal context. Compare the `confidence` and `reasoning` fields to verify the model uses the hierarchical information.
- **Retry Coverage:** Intentionally use a small `max_tokens=32` to force truncated JSON, and verify the retry mechanism kicks in and eventually succeeds.
- **Cache Consistency:** Run `find_step()` twice with the same inputs. Assert the second call returns cached results instantly.
