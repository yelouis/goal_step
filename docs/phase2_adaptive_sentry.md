# Phase 2: The Adaptive Sentry (Bidirectional Magnitude Triggers)

## Overview
Using the cleaned acoustic energy envelope generated in Phase 1, Phase 2 implements a sliding window algorithm to detect significant relative shifts in audio magnitude. This acts as an automated trigger directing the VLM (Visual Captioning) on exactly when to scan the video frame.

## Key Concepts
- **Short-Term Average Energy ($E_{sta}$):** Energy over a very short time (e.g., 0.5s - 1s). Represents the *current* state.
- **Long-Term Average Energy ($E_{lta}$):** Energy over a longer window (e.g., 5s - 10s). Represents the *contextual baseline*.
- **The Spike ($k_{up}$):** Triggered when a sudden action begins.
- **The Drop ($k_{down}$):** Triggered when an ongoing action ceases.
- **Semantic Rate Limiting:** Cooldowns and threshold saturation to prevent trigger spam.
- **The Silence Watcher:** A forced timeout metric ($T_{force}$) to ensure we don't miss long, silent steps.

## Pseudocode Implementation

```python
import numpy as np
import os
import json

# System Configuration
SSD_BASE = "/Volumes/Extreme SSD/ego4d_data"
FPS_AUDIO = 16000 / 512 # Approx 31.25 frames per second using Phase 1 standard
STA_WINDOW_FRAMES = int(0.5 * FPS_AUDIO) # 0.5 second window
LTA_WINDOW_FRAMES = int(5.0 * FPS_AUDIO) # 5.0 second window

class AdaptiveSentry:
    def __init__(self, video_id: str):
        self.video_id = video_id
        
        # Initial Thresholds
        self.k_up = 2.5
        self.k_down = 0.3
        
        # Rate Limiting & Cooldown Config
        self.cooldown_frames = int(3.0 * FPS_AUDIO) # 3s cooldown
        self.t_force_frames = int(45.0 * FPS_AUDIO) # 45s silence watcher
        self.saturation_window_frames = int(15.0 * FPS_AUDIO) # 15s saturation monitoring
        
        # State
        self.last_trigger_frame = 0
        self.trigger_history = [] # Stores (frame_idx, trigger_type)
        
    def check_saturation(self, current_frame: int):
        """ Exponentially backoff thresholds if triggered too often """
        recent_triggers = [t for t in self.trigger_history if (current_frame - t[0]) <= self.saturation_window_frames]
        
        if len(recent_triggers) >= 3:
            # Saturation occurring - increase thresholds
            self.k_up *= 1.5 
            self.k_down *= 0.5 # lower fraction means harder to drop
        else:
            # Decay back to normal
            self.k_up = max(2.5, self.k_up * 0.9)
            self.k_down = min(0.3, self.k_down * 1.1)

    def scan_track(self) -> list:
        # Load clean energy from Phase 1 cache
        energy_path = os.path.join(SSD_BASE, f"cache/phase1/{self.video_id}_clean_energy.npy")
        energy_envelope = np.load(energy_path)
        
        total_frames = len(energy_envelope)
        triggers = []
        
        # Start scanning after we have enough data to fill the LTA window
        for i in range(LTA_WINDOW_FRAMES, total_frames):
            # Check cooldown
            if (i - self.last_trigger_frame) < self.cooldown_frames:
                continue
                
            # Update saturation logic
            self.check_saturation(i)
            
            # Extract Windows
            lta_slice = energy_envelope[i - LTA_WINDOW_FRAMES : i]
            sta_slice = energy_envelope[i - STA_WINDOW_FRAMES : i]
            
            E_lta = np.mean(lta_slice) + 1e-6 # prevent div/0
            E_sta = np.mean(sta_slice)
            
            ratio = E_sta / E_lta
            
            trigger_type = None
            
            # Bidirectional Trigger Logic
            if ratio > self.k_up:
                trigger_type = "spike"
            elif ratio < self.k_down:
                trigger_type = "drop"
            elif (i - self.last_trigger_frame) > self.t_force_frames:
                trigger_type = "forced_silence"
                
            if trigger_type:
                timestamp_sec = i / FPS_AUDIO
                triggers.append({
                    "frame_idx": i,
                    "timestamp": round(timestamp_sec, 2),
                    "trigger_type": trigger_type
                })
                self.trigger_history.append((i, trigger_type))
                self.last_trigger_frame = i

        return triggers
        
if __name__ == '__main__':
    sentry = AdaptiveSentry("sample_ego4d_vid_001")
    event_triggers = sentry.scan_track()
    
    # Save triggers to SSD cache for Phase 3
    out_path = os.path.join(SSD_BASE, "cache/phase2/sample_ego4d_vid_001_triggers.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(event_triggers, f, indent=4)
    print(f"Sentry detected {len(event_triggers)} interest zones.")
```

## Verification Strategy
- **Alignment Test:** Pass an Ego4D dataset sample through the pipeline and print all `["spike", "drop"]` timestamps. Watch the raw video sequentially via VLC and assert that >80% of major tool actions/sudden clanging trigger a spike, and setting down the tool triggers a drop. 
- **Saturation Fallback Print:** Inject a `print("Saturation Triggered")` block inside `check_saturation` to ensure limits dynamically activate during sustained noise events (e.g., using a drill for 30s).
