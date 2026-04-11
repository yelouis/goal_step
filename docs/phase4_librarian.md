# Phase 4: The Librarian (Gemma 4 26B MoE Reasoning)

## Overview
The "Librarian" acts as the logical bridge between language (the target step from the Ego4D challenge) and vision/audio (the Table of Contents generated in Phase 3). By feeding the entire ToC into an LLM, we can map a specific query (e.g., "Assemble wooden frame") to the most probable "Chapter" (a temporal slice) within the video.

The Librarian will handle situations where a step was captured perfectly via an action "Spike", and also complex interpolations where a step happened silently between two "Drop" triggers.

## LLM Strategy
Since this requires high logic, we utilize an advanced model (e.g., Gemma 4 26B or Gemma 2 27B) running locally via `mlx-lm`. 

## Pseudocode Implementation

```python
import json
import os
import re
from mlx_lm import load, generate

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
MODEL_CACHE = os.path.join(SSD_BASE, "models/gemma")

LIBRARIAN_PROMPT = """You are the "Procedural Video Librarian." Map the "Target Step" to the most likely chapters within a video's Table of Contents (ToC).

Logical Rules:
1. Explicit Identification: Prioritize chapters where the caption semantically matches the Target Step.
2. Silent-Step Interpolation: If the exact step is missing, identify the Pre-Condition and Post-Condition. The Target Step likely occurred in the temporal gap between them.
3. Change Detection Logic: Treat "drop" triggers as potential "Step Completion" markers and "spike" triggers as "Step Start" markers.

You MUST respond strictly in the following JSON format:
{
    "top_chapters": [{"start_time": X.X, "end_time": Y.Y}],
    "confidence": 0.X,
    "reasoning": "brief explanation"
}

Target Step: {target_step}
Table of Contents:
{toc_data}
"""

class VideoLibrarian:
    def __init__(self):
        print("Initializing Librarian Protocol (Gemma)...")
        os.environ['HF_HOME'] = MODEL_CACHE
        # Note: Replace with the exact quantization ID you pull for Gemma
        self.model, self.tokenizer = load("google/gemma-2-27b-it") 
        
    def find_step(self, video_id: str, target_step: str) -> dict:
        # Load the ToC
        toc_path = os.path.join(SSD_BASE, f"cache/phase3/{video_id}_toc.json")
        with open(toc_path, 'r') as f:
            toc = json.load(f)
            
        # Format the Prompt
        formatted_prompt = LIBRARIAN_PROMPT.format(
            target_step=target_step,
            toc_data=json.dumps(toc, indent=2)
        )
        
        # We use a chat template if the model requires it
        messages = [{"role": "user", "content": formatted_prompt}]
        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generation
        print(f"Asking Librarian about: '{target_step}'")
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=text_prompt, 
            max_tokens=512,
            temperature=0.1 # Keep variance low for JSON compliance
        )
        
        # Parse JSON from LLM Output
        try:
            # Extract JSON block using regex in case the LLM adds markdown backticks
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
            structured_out = json.loads(json_str)
        except Exception as e:
            print("Failed to parse Librarian Output into JSON. Falling back.")
            print(f"Raw Output: {response}")
            structured_out = {"top_chapters": [], "confidence": 0.0, "reasoning": "Parse failure."}
            
        return structured_out

if __name__ == '__main__':
    librarian = VideoLibrarian()
    # Assume Ego4D gives us a challenge query:
    query = "Tighten the screws on the metal bracket."
    result = librarian.find_step("sample_ego4d_vid_001", query)
    
    # Save the Librarian's hypothesis to the SSD
    out_path = os.path.join(SSD_BASE, "cache/phase4/sample_hypothesis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"Hypothesis Generated with {result['confidence']*100}% confidence.")
```

## Verification Strategy
- **JSON Scheme Check:** Build a small PyTest unit test that feeds the Librarian prompt a mock `Target Step` and dummy `ToC array`. Assert that the resulting string from `mlx-lm` successfully loads via `json.loads()` and contains the exact keys requested.
- **Interpolation Test:** Give the Librarian a ToC with "User grabs hammer" at 10.0s and "User puts hammer down" at 20.0s, and ask it for the Target Step "User hits the nail". Assert that it accurately identifies `[10.0, 20.0]` as the chapter.
