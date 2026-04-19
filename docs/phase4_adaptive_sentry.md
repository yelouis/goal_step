# Phase 4: The Adaptive Sentry (Bidirectional Magnitude Triggers)

## Overview
Using the cleaned acoustic energy envelope generated in Phase 3, Phase 4 implements a sliding window algorithm to detect significant relative shifts in audio magnitude. This acts as an automated trigger directing the VLM (Visual Captioning) on exactly when to scan the video frame.

> [!WARNING]
> **Audio-Only Limitation:** Many egocentric procedural steps (measuring, aligning, inspecting) are visually obvious but acoustically silent. The `forced_silence` watcher handles some of these, but consider supplementing with a visual change-detection pass (e.g., frame-differencing or optical flow) as an additional trigger source in a future iteration.

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
SSD_BASE = "/Volumes/Extreme SSD/goal_step_data"
FPS_AUDIO = 16000 / 512 # Approx 31.25 frames per second using Phase 3 standard
STA_WINDOW_FRAMES = int(0.5 * FPS_AUDIO) # 0.5 second window
LTA_WINDOW_FRAMES = int(5.0 * FPS_AUDIO) # 5.0 second window

# Baseline thresholds (used for reset targets)
K_UP_DEFAULT = 2.5
K_DOWN_DEFAULT = 0.3

class AdaptiveSentry:
    def __init__(self, video_id: str):
        self.video_id = video_id
        
        # Initial Thresholds
        self.k_up = K_UP_DEFAULT
        self.k_down = K_DOWN_DEFAULT
        
        # Rate Limiting & Cooldown Config
        self.cooldown_frames = int(3.0 * FPS_AUDIO) # 3s cooldown
        self.t_force_frames = int(45.0 * FPS_AUDIO) # 45s silence watcher
        self.saturation_window_frames = int(15.0 * FPS_AUDIO) # 15s saturation monitoring
        
        # Decay control: how quickly thresholds relax back to default
        # Each trigger event applies one unit of decay (not per-frame)
        self.decay_rate = 0.85  # Per-trigger-event decay factor
        
        # State
        self.last_trigger_frame = 0
        self.last_saturation_check_frame = 0
        self.trigger_history = [] # Stores (frame_idx, trigger_type)
        
    def check_saturation(self, current_frame: int):
        """
        Exponentially backoff thresholds if triggered too often.
        
        Decay only runs once per trigger event (not per-frame) to 
        avoid snapping back to baseline in a handful of frames.
        """
        recent_triggers = [t for t in self.trigger_history 
                          if (current_frame - t[0]) <= self.saturation_window_frames]
        
        if len(recent_triggers) >= 3:
            # Saturation occurring - make triggers harder to fire
            self.k_up *= 1.5 
            self.k_down *= 0.5  # lower fraction means harder to trigger drop
            print(f"[Saturation] frame {current_frame}: k_up={self.k_up:.2f}, k_down={self.k_down:.4f}")
        # Note: decay is applied in _apply_decay_on_trigger() instead of here

    def _apply_decay_on_trigger(self):
        """
        Gradually relax thresholds back toward defaults.
        Called once per trigger event, not per frame.
        """
        if self.k_up > K_UP_DEFAULT:
            self.k_up = max(K_UP_DEFAULT, self.k_up * self.decay_rate)
        if self.k_down < K_DOWN_DEFAULT:
            self.k_down = min(K_DOWN_DEFAULT, self.k_down / self.decay_rate)

    def scan_track(self) -> list:
        # Load clean energy from Phase 3 cache
        energy_path = os.path.join(SSD_BASE, f"cache/phase3/{self.video_id}_clean_energy.npy")
        energy_envelope = np.load(energy_path)
        
        total_frames = len(energy_envelope)
        triggers = []
        
        # Start scanning after we have enough data to fill the LTA window
        for i in range(LTA_WINDOW_FRAMES, total_frames):
            # Check cooldown
            if (i - self.last_trigger_frame) < self.cooldown_frames:
                continue
                
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
                
                # Check for saturation after each trigger
                self.check_saturation(i)
                # Apply gentle decay toward defaults
                self._apply_decay_on_trigger()

        return triggers
        
if __name__ == '__main__':
    # Example: run on an EPIC-KITCHENS video
    sentry = AdaptiveSentry("P01_101")
    event_triggers = sentry.scan_track()
    
    # Save triggers to SSD cache for Phase 5
    out_path = os.path.join(SSD_BASE, "cache/phase4/P01_101_triggers.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(event_triggers, f, indent=4)
    print(f"Sentry detected {len(event_triggers)} interest zones.")
```

## Future Enhancement: Visual Change Detection
As a complementary trigger source (especially for silent procedural steps), a simple frame-differencing approach can detect visual state changes:

```python
def visual_change_score(prev_frame, curr_frame) -> float:
    """Compute pixel-level change between consecutive frames."""
    diff = np.abs(prev_frame.astype(float) - curr_frame.astype(float))
    return np.mean(diff) / 255.0  # Normalized 0-1
```

This would run in parallel with the acoustic sentry, producing a merged trigger list sorted by timestamp. Not yet implemented but noted as a priority improvement.

## Session Resilience

| Item | Detail |
|---|---|
| **Input Dependencies** | `cache/phase3/{video_id}_clean_energy.npy` (Phase 3) |
| **Output Artifact** | `/Volumes/Extreme SSD/goal_step_data/cache/phase4/{video_id}_triggers.json` |
| **Cache Check** | Before running `scan_track()`, check if `{video_id}_triggers.json` exists and load from cache if so |
| **Verification Checkpoint** | After completing all videos, write `cache/phase4/_manifest.json` listing all processed video IDs and trigger counts |
| **Resume Strategy** | On re-run, skip any video whose `_triggers.json` already exists on the SSD |

```python
def run_sentry_cached(video_id: str) -> list:
    """Run the Adaptive Sentry with cache check."""
    cache_path = os.path.join(SSD_BASE, f"cache/phase4/{video_id}_triggers.json")
    if os.path.exists(cache_path):
        print(f"[CACHED] Triggers for {video_id} already exist.")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    sentry = AdaptiveSentry(video_id)
    triggers = sentry.scan_track()
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(triggers, f, indent=4)
    print(f"Sentry detected {len(triggers)} interest zones for {video_id}.")
    return triggers
```

## Verification Strategy
- **Alignment Test:** Pass samples from each dataset through the pipeline and print all `["spike", "drop"]` timestamps. Watch the raw video sequentially via VLC and assert that >80% of major tool actions/sudden clanging trigger a spike, and setting down the tool triggers a drop. Test across all three datasets since audio profiles differ significantly (kitchen sounds vs. indoor activity vs. assembly). 
- **Saturation Fallback Print:** The `check_saturation` method now prints when saturation activates. Verify during sustained noise events (e.g., using a drill for 30s).
- **Decay Rate Test:** After saturation, run 5 more triggers and verify `k_up` gradually returns toward 2.5 (not instantly).
- **Cache Consistency:** Run `run_sentry_cached()` twice. Assert the second call returns cached results instantly.

