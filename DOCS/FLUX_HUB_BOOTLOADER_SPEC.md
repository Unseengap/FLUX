# FLUX Hub Bootloader UI Specification

**Memory Fabric Home/Office Device — Boot Interface**  
*Hardware UI Design Document*  
*Version 1.0 | April 2026*

---

## Overview

This document specifies the bootloader UI for the FLUX Memory Fabric Hub — a dedicated hardware device that runs the FLUX cognitive architecture locally. The UI displays during device startup, model loading, and runtime status.

---

## Hardware Context

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY FABRIC HUB                            │
├─────────────────────────────────────────────────────────────────┤
│  Display Output: HDMI (1080p minimum, 4K recommended)           │
│  Primary Display: Connected TV/Monitor                           │
│  Secondary: Front panel LED strip (status indicators)           │
│  Audio Output: 3.5mm / HDMI audio / Bluetooth                   │
│  Input: USB peripherals, network, voice                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Boot Sequence Phases

### Phase 0: Hardware Initialization (0-3 seconds)
**LED:** Pulsing blue  
**Display:** Black → FLUX logo fade-in

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                                                                  │
│                                                                  │
│                         ╔═══════════╗                           │
│                         ║   FLUX    ║                           │
│                         ║    ~~~    ║                           │
│                         ╚═══════════╝                           │
│                                                                  │
│                    MEMORY FABRIC HUB                            │
│                                                                  │
│                                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: System Check (3-8 seconds)
**LED:** Solid blue  
**Display:** Hardware validation

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ╔═══════════╗                                                  │
│  ║   FLUX    ║     SYSTEM CHECK                                 │
│  ╚═══════════╝                                                  │
│                                                                  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░  45%                │
│                                                                  │
│  ✓ GPU Detected: NVIDIA RTX 4090 (24GB VRAM)                   │
│  ✓ Memory: 64GB DDR5                                            │
│  ✓ Storage: 2TB NVMe (1.8TB free)                              │
│  ● Network: Connecting...                                       │
│  ○ Model: Pending                                               │
│                                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Model Discovery (8-15 seconds)
**LED:** Pulsing cyan  
**Display:** Locating .flx file

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ╔═══════════╗                                                  │
│  ║   FLUX    ║     LOADING MODEL                                │
│  ╚═══════════╝                                                  │
│                                                                  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  60%                │
│                                                                  │
│  ✓ GPU Detected: NVIDIA RTX 4090 (24GB VRAM)                   │
│  ✓ Memory: 64GB DDR5                                            │
│  ✓ Storage: 2TB NVMe (1.8TB free)                              │
│  ✓ Network: Connected (192.168.1.42)                           │
│  ● Model: Loading Flux-Apex-V1.flx...                          │
│                                                                  │
│    Found: /models/Flux-Apex-V1.flx (17.24 GB)                  │
│    Version: 8.3-autonomous                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Bootstrap Execution (15-45 seconds)
**LED:** Cycling cyan → purple  
**Display:** Module loading progress

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ╔═══════════╗     BOOTSTRAPPING                                │
│  ║   FLUX    ║     Flux-Apex-V1.flx                            │
│  ╚═══════════╝     v8.3-autonomous                              │
│                                                                  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░  78%                │
│                                                                  │
│  Loading embedded runtime...                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ✓ flux_model.py                                          │   │
│  │ ✓ phases/phase1/cse.py                                   │   │
│  │ ✓ phases/phase2/field.py                                 │   │
│  │ ✓ phases/phase_autonomous/tool_executor.py               │   │
│  │ ● phases/phase_autonomous/coder_pool.py                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Modules: 77/99                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Model Initialization (45-90 seconds)
**LED:** Solid purple  
**Display:** Component activation

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ╔═══════════╗     INITIALIZING COMPONENTS                     │
│  ║   FLUX    ║     8.34B parameters                             │
│  ╚═══════════╝                                                  │
│                                                                  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░  90%                │
│                                                                  │
│  Native FLUX                    Embedded Models                 │
│  ────────────                   ───────────────                 │
│  ✓ CSE (1.3M)                   ● instruct (1.54B) Loading...  │
│  ✓ Field (28.4M)                ○ vision (2.21B)               │
│  ✓ Memory (542M)                ○ coder (1.54B)                │
│  ✓ Gravity (75M)                ✓ clip (428M)                  │
│  ✓ Thermodynamic (135M)         ○ whisper (242M)               │
│  ✓ Causal (150M)                ○ tts (410M)                   │
│  ✓ Bridges (456M)               ✓ embedding (23M)              │
│                                                                  │
│  VRAM Usage: 4.2 / 24.0 GB                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 5: Ready State
**LED:** Breathing green  
**Display:** Main interface

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ╔═══════════╗     ┌────────────────────────────────────┐      │
│  ║   FLUX    ║     │ FLUX is ready                      │      │
│  ║    ~~~    ║     │                                     │      │
│  ╚═══════════╝     │ "Hello! How can I help you today?" │      │
│                    └────────────────────────────────────┘      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  Type or speak your message...                          │   │
│  │  ▌                                                       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │ 🎤   │ │ 📷   │ │ 📁   │ │ ⚙️   │ │ 💡   │ │ ❓   │       │
│  │Voice │ │Vision│ │Files │ │Config│ │Smart │ │Help  │       │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘       │
│                                                                  │
│  ──────────────────────────────────────────────────────────    │
│  v8.3-autonomous │ VRAM: 8.4GB │ 🟢 Online │ 🔒 Local Only    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## UI Component Specifications

### Color Palette

| Element | Hex | RGB | Usage |
|---------|-----|-----|-------|
| **Primary** | `#6366F1` | 99, 102, 241 | Accent, buttons |
| **Secondary** | `#8B5CF6` | 139, 92, 246 | Highlights |
| **Background** | `#0F172A` | 15, 23, 42 | Main background |
| **Surface** | `#1E293B` | 30, 41, 59 | Cards, panels |
| **Text Primary** | `#F1F5F9` | 241, 245, 249 | Main text |
| **Text Secondary** | `#94A3B8` | 148, 163, 184 | Muted text |
| **Success** | `#22C55E` | 34, 197, 94 | Ready, complete |
| **Warning** | `#F59E0B` | 245, 158, 11 | Loading, action |
| **Error** | `#EF4444` | 239, 68, 68 | Errors |

### Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Logo | Inter | 48px | 700 |
| H1 | Inter | 32px | 600 |
| H2 | Inter | 24px | 600 |
| Body | Inter | 16px | 400 |
| Caption | Inter | 14px | 400 |
| Monospace | JetBrains Mono | 14px | 400 |

### LED Strip Patterns

| State | Pattern | Color | Meaning |
|-------|---------|-------|---------|
| Boot | Pulse 1Hz | Blue `#3B82F6` | Hardware init |
| Loading | Pulse 2Hz | Cyan `#06B6D4` | Model loading |
| Processing | Chase | Purple `#8B5CF6` | Thinking |
| Ready | Breathe 0.5Hz | Green `#22C55E` | Ready for input |
| Speaking | VU meter | Purple `#A855F7` | Audio output |
| Listening | Pulse 3Hz | Blue `#3B82F6` | Voice input |
| Error | Solid | Red `#EF4444` | Error state |
| Update | Rainbow | Cycle | Updating |

---

## Technical Implementation

### Boot Script Entry Point

```python
#!/usr/bin/env python3
"""
FLUX Hub Boot Script
Runs on system startup via systemd service
"""

import sys
import os
from pathlib import Path

# Add FLUX to path
FLUX_ROOT = Path('/opt/flux')
sys.path.insert(0, str(FLUX_ROOT))

# Import UI framework
from flux_hub.ui import BootUI, MainUI
from flux_hub.hardware import LEDController, AudioController

def main():
    # Initialize hardware
    led = LEDController()
    audio = AudioController()
    
    # Phase 0: Hardware init
    led.set_pattern('pulse', color='blue')
    ui = BootUI()
    ui.show_logo()
    
    # Phase 1: System check
    ui.show_system_check()
    hw_info = check_hardware()
    ui.update_system_check(hw_info)
    
    # Phase 2: Model discovery
    led.set_pattern('pulse', color='cyan')
    model_path = find_model()
    ui.show_model_info(model_path)
    
    # Phase 3: Bootstrap
    led.set_pattern('cycle', colors=['cyan', 'purple'])
    from bootstrap import wake_up
    
    def on_module_loaded(name, idx, total):
        ui.update_bootstrap_progress(name, idx, total)
    
    result = wake_up(
        str(model_path),
        device='cuda',
        verbose=True,
        callback=on_module_loaded
    )
    
    # Phase 4: Initialize components
    led.set_pattern('solid', color='purple')
    ui.show_component_init()
    
    # Load eager models
    from flux_lazy_loader import FLUXLazyLoader
    loader = FLUXLazyLoader(result['flx'])
    
    for model_name in ['instruct', 'clip', 'embedding']:
        ui.update_component_status(model_name, 'loading')
        loader.load_model(model_name)
        ui.update_component_status(model_name, 'ready')
    
    # Phase 5: Ready
    led.set_pattern('breathe', color='green')
    audio.play_ready_chime()
    
    # Launch main UI
    main_ui = MainUI(result['flx'], loader)
    main_ui.run()


if __name__ == '__main__':
    main()
```

### Directory Structure

```
/opt/flux/
├── models/
│   └── Flux-Apex-V1.flx    # Main model file
├── flux_hub/               # Hub-specific code
│   ├── __init__.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── boot_ui.py      # Boot sequence UI
│   │   ├── main_ui.py      # Main interface
│   │   ├── components/     # Reusable UI components
│   │   └── styles/         # CSS/styling
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── led.py          # LED strip control
│   │   ├── audio.py        # Audio I/O
│   │   ├── midi.py         # MIDI control
│   │   └── gpio.py         # GPIO for buttons
│   └── services/
│       ├── __init__.py
│       ├── voice.py        # Voice assistant
│       ├── vision.py       # Camera processing
│       └── smart_home.py   # Home automation
├── config/
│   ├── hub.yaml            # Hub configuration
│   └── network.yaml        # Network settings
├── logs/
│   └── boot.log            # Boot logs
└── data/
    ├── memory/             # Episodic memory storage
    └── cache/              # Model cache
```

### Systemd Service

```ini
# /etc/systemd/system/flux-hub.service
[Unit]
Description=FLUX Memory Fabric Hub
After=network.target graphical.target

[Service]
Type=simple
User=flux
WorkingDirectory=/opt/flux
Environment=PYTHONPATH=/opt/flux
ExecStart=/opt/flux/.venv/bin/python /opt/flux/boot.py
Restart=on-failure
RestartSec=10

# GPU access
SupplementaryGroups=video render

# Resource limits
MemoryMax=48G
CPUQuota=90%

[Install]
WantedBy=multi-user.target
```

---

## UI Framework Options

### Selected: CustomTkinter (Python-native)

**Why CustomTkinter:**
- Pure Python — no Node.js/Electron overhead
- Modern, dark-themed UI out of the box
- GPU-friendly, lightweight
- Easy to bundle with FLUX
- Cross-platform (Linux, macOS, Windows)

**Stack:**
```
Framework: CustomTkinter 5.x
Audio: pygame (for brand sound)
Styling: Built-in dark theme + custom colors
Threading: asyncio for non-blocking model loading
```

**Dependencies:**
```
customtkinter>=5.2.0
pygame>=2.5.0
pillow>=10.0.0
```

### Brand Sound
- **File:** `assets/audio/Beyond_The_Flux.wav`
- **Plays:** During boot Phase 3 (Bootstrap) 
- **Duration:** Loops until ready state

---

## UI Screens Specification

### 1. Boot Screens (as shown above)

### 2. Main Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  FLUX                                          🔔  ⚙️  👤       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    CONVERSATION                           │   │
│  │ ─────────────────────────────────────────────────────────│   │
│  │                                                           │   │
│  │  👤  What's the weather forecast for tomorrow?           │   │
│  │                                                           │   │
│  │  🤖  Based on your location in San Francisco, tomorrow   │   │
│  │      will be partly cloudy with a high of 68°F (20°C)    │   │
│  │      and a low of 54°F (12°C). There's a 20% chance of   │   │
│  │      light rain in the evening.                          │   │
│  │                                                           │   │
│  │      Would you like me to set a reminder to bring an     │   │
│  │      umbrella tomorrow morning?                           │   │
│  │                                                           │   │
│  │  👤  Yes, please remind me at 8am                        │   │
│  │                                                           │   │
│  │  🤖  ✓ Reminder set for 8:00 AM tomorrow:               │   │
│  │      "Bring umbrella - 20% rain chance"                  │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Type a message...                              🎤 📎 ➤  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│   💬 Chat  │  📊 Insights  │  🏠 Home  │  📁 Files  │  ⚙️ Settings │
└─────────────────────────────────────────────────────────────────┘
```

### 3. System Status Panel

```
┌─────────────────────────────────────────────────────────────────┐
│  SYSTEM STATUS                                           ✕      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MODEL                                                          │
│  ──────                                                         │
│  Flux-Apex-V1.flx (v8.3-autonomous)                            │
│  Parameters: 8.34B │ Runtime: 99 modules                        │
│                                                                  │
│  RESOURCE USAGE                                                 │
│  ──────────────                                                 │
│  GPU:  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  45%  (10.8 / 24.0 GB)        │
│  RAM:  ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  32%  (20.5 / 64.0 GB)        │
│  CPU:  ▓▓▓▓░░░░░░░░░░░░░░░░░░░░  15%                           │
│  Disk: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  58%  (1.16 / 2.0 TB)         │
│                                                                  │
│  LOADED MODELS                                                  │
│  ─────────────                                                  │
│  ✓ instruct (Qwen2.5-1.5B)     ● Active   │ 3.7 GB            │
│  ✓ clip (ViT-L/14)             ● Active   │ 1.0 GB            │
│  ✓ embedding (MiniLM)          ● Active   │ 0.1 GB            │
│  ○ vision (Qwen2-VL-2B)        ○ Standby  │ —                  │
│  ○ coder (Qwen2.5-Coder)       ○ Standby  │ —                  │
│  ○ whisper (small)             ○ Standby  │ —                  │
│  ○ tts (bark-small)            ○ Standby  │ —                  │
│                                                                  │
│  MEMORY TIERS                                                   │
│  ────────────                                                   │
│  Working:  2,048 tokens │ Session context                      │
│  Episodic: 1,247 facts  │ 3.2 MB index                         │
│  Semantic: 110,592 cells │ Field active                         │
│                                                                  │
│  NETWORK                                                        │
│  ───────                                                        │
│  Local: 192.168.1.42 │ External: 🔒 Disabled                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Settings Panel

```
┌─────────────────────────────────────────────────────────────────┐
│  SETTINGS                                                ✕      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │ General         │   PRIVACY                                  │
│  │ ● Privacy       │   ────────                                 │
│  │ Models          │                                            │
│  │ Voice           │   ☑ Keep all data local                   │
│  │ Display         │     FLUX never sends data to external     │
│  │ Network         │     servers. All processing is on-device. │
│  │ Smart Home      │                                            │
│  │ Updates         │   ☑ Store conversation history            │
│  │ About           │     Save chats to local memory for        │
│  └─────────────────┘     context recall.                        │
│                                                                  │
│                         ☐ Allow network voice commands          │
│                           Enable voice control from other       │
│                           devices on your network.              │
│                                                                  │
│                         ☑ Auto-delete old memories              │
│                           Remove episodic memories older than:  │
│                           [ 90 days            ▼]               │
│                                                                  │
│                         ─────────────────────────────────────   │
│                                                                  │
│                         DATA MANAGEMENT                         │
│                                                                  │
│                         [Export All Data]  [Clear History]      │
│                                                                  │
│                         Memory Usage: 847 MB                    │
│                         Last Backup: April 3, 2026 2:30 PM     │
│                                                                  │
│                                            [Save]  [Cancel]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware Integration APIs

### LED Controller

```python
class LEDController:
    """Control the hub's LED strip."""
    
    def set_pattern(self, pattern: str, color: str = None, 
                    colors: list = None, speed: float = 1.0):
        """
        Set LED pattern.
        
        Patterns: 'solid', 'pulse', 'breathe', 'chase', 'rainbow', 'vu_meter'
        Colors: 'red', 'green', 'blue', 'cyan', 'purple', 'white', or hex
        """
        pass
    
    def set_brightness(self, level: float):
        """Set brightness (0.0 - 1.0)."""
        pass
    
    def off(self):
        """Turn off LEDs."""
        pass
```

### Audio Controller

```python
class AudioController:
    """Control audio input/output."""
    
    def play_chime(self, name: str):
        """Play system sound: 'ready', 'error', 'notification'."""
        pass
    
    def start_listening(self, callback: Callable):
        """Start voice input with wake word detection."""
        pass
    
    def speak(self, text: str, voice: str = 'default'):
        """Speak text using TTS model."""
        pass
    
    def set_volume(self, level: float):
        """Set volume (0.0 - 1.0)."""
        pass
```

### MIDI Controller

```python
class MIDIController:
    """Control MIDI input/output for music applications."""
    
    def list_devices(self) -> List[str]:
        """List connected MIDI devices."""
        pass
    
    def send_note(self, channel: int, note: int, velocity: int):
        """Send MIDI note."""
        pass
    
    def on_note(self, callback: Callable):
        """Register callback for incoming MIDI notes."""
        pass
```

---

## Performance Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Boot to ready | < 90 sec | 120 sec |
| First response | < 2 sec | 5 sec |
| UI frame rate | 60 fps | 30 fps |
| Voice wake latency | < 500ms | 1 sec |
| Model swap time | < 5 sec | 10 sec |
| Memory overhead | < 2 GB | 4 GB |

---

## Error Handling

### Boot Failures

| Error | LED | UI Message | Recovery |
|-------|-----|------------|----------|
| No GPU | Red solid | "GPU not detected" | Check hardware |
| Model missing | Red pulse | "Model not found" | Download prompt |
| OOM | Red/yellow | "Insufficient memory" | Reduce models |
| Network | Yellow | "Network unavailable" | Offline mode |
| Corrupt model | Red | "Model verification failed" | Re-download |

### Runtime Errors

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠️  FLUX encountered an issue                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  The vision model failed to load due to insufficient VRAM.      │
│                                                                  │
│  Current VRAM: 22.1 / 24.0 GB                                   │
│  Required: 5.3 GB additional                                    │
│                                                                  │
│  Options:                                                        │
│  ┌────────────────────┐  ┌────────────────────┐                │
│  │ Unload other models│  │ Continue without   │                │
│  │ to free VRAM       │  │ vision capability  │                │
│  └────────────────────┘  └────────────────────┘                │
│                                                                  │
│  [View Details]  [Report Issue]                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Accessibility

- **High contrast mode** — Toggle in settings
- **Font scaling** — 100% to 200% 
- **Screen reader support** — ARIA labels on all elements
- **Voice control** — Full UI navigation via voice
- **Keyboard shortcuts** — All actions accessible via keyboard

---

## Future Extensions

1. **Multi-user profiles** — Different memory spaces per user
2. **Remote access** — Secure web interface for mobile
3. **Plugin system** — Third-party skill modules
4. **Multi-hub sync** — Distributed FLUX across locations
5. **AR/VR interface** — Spatial computing UI

---

*FLUX Memory Fabric Hub — Your personal AI, running locally, always private.*
