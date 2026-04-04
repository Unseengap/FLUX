#!/usr/bin/env python3
"""
FLUX Hub Boot UI — CustomTkinter Implementation

A modern, dark-themed boot interface for the FLUX Memory Fabric Hub.
Plays brand sound during loading and displays boot progress.

Usage:
    python -m flux_hub.boot_ui
    
Or import:
    from flux_hub.boot_ui import FluxBootUI
    ui = FluxBootUI()
    ui.run()
"""

import customtkinter as ctk
import threading
import time
import sys
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

# Try to import pygame for audio
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠ pygame not installed - audio disabled. Install with: pip install pygame")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class BootConfig:
    """Boot UI configuration."""
    # Window
    width: int = 900
    height: int = 600
    title: str = "FLUX Memory Fabric"
    
    # Colors (FLUX brand palette)
    bg_dark: str = "#0F172A"
    bg_surface: str = "#1E293B"
    primary: str = "#49B0F0"
    secondary: str = "#07EDB4"
    success: str = "#22C55E"
    warning: str = "#F59E0B"
    error: str = "#EF4444"
    text_primary: str = "#F1F5F9"
    text_secondary: str = "#94A3B8"
    cyan: str = "#06B6D4"
    
    # Audio
    brand_sound: str = "flux_hub/assets/audio/Beyond_The_Flux.wav"
    
    # Fonts
    font_family: str = "Helvetica"
    

# ─────────────────────────────────────────────
# Audio Controller
# ─────────────────────────────────────────────

class AudioController:
    """Handle brand sound playback."""
    
    def __init__(self, config: BootConfig):
        self.config = config
        self.sound: Optional[pygame.mixer.Sound] = None
        self.channel: Optional[pygame.mixer.Channel] = None
        self._load_sound()
    
    def _load_sound(self):
        """Load the brand sound file."""
        if not AUDIO_AVAILABLE:
            return
            
        sound_path = Path(__file__).parent.parent / self.config.brand_sound
        if not sound_path.exists():
            # Try relative to cwd
            sound_path = Path(self.config.brand_sound)
        
        if sound_path.exists():
            try:
                self.sound = pygame.mixer.Sound(str(sound_path))
                print(f"✓ Loaded brand sound: {sound_path.name}")
            except Exception as e:
                print(f"⚠ Failed to load sound: {e}")
        else:
            print(f"⚠ Brand sound not found: {sound_path}")
    
    def play(self, loops: int = 0, fade_ms: int = 1000):
        """Play the brand sound.
        
        Args:
            loops: -1 for infinite loop, 0 for once
            fade_ms: Fade in duration in milliseconds
        """
        if self.sound:
            self.channel = self.sound.play(loops=loops, fade_ms=fade_ms)
    
    def stop(self, fade_ms: int = 2000):
        """Stop playback with fade out."""
        if self.channel and self.channel.get_busy():
            self.sound.fadeout(fade_ms)
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        if self.sound:
            self.sound.set_volume(volume)


# ─────────────────────────────────────────────
# Boot UI Components
# ─────────────────────────────────────────────

class WaveAnimation(ctk.CTkCanvas):
    """Animated wave visualization for loading."""
    
    def __init__(self, master, config: BootConfig, **kwargs):
        super().__init__(
            master, 
            bg=config.bg_dark, 
            highlightthickness=0,
            **kwargs
        )
        self.config = config
        self.phase = 0
        self.amplitude = 20
        self.frequency = 0.05
        self.speed = 0.1
        self.running = False
        self.wave_color = config.primary
        
    def start(self):
        """Start wave animation."""
        self.running = True
        self._animate()
        
    def stop(self):
        """Stop wave animation."""
        self.running = False
        
    def set_color(self, color: str):
        """Change wave color."""
        self.wave_color = color
        
    def _animate(self):
        """Animation loop."""
        if not self.running:
            return
            
        self.delete("wave")
        width = self.winfo_width()
        height = self.winfo_height()
        center_y = height // 2
        
        # Draw multiple wave layers
        for i, (amp_mult, alpha) in enumerate([(1.0, 1.0), (0.6, 0.6), (0.3, 0.3)]):
            points = []
            for x in range(0, width, 3):
                import math
                y = center_y + self.amplitude * amp_mult * math.sin(
                    self.frequency * x + self.phase + i * 0.5
                )
                points.extend([x, y])
            
            if len(points) >= 4:
                self.create_line(
                    *points, 
                    fill=self.wave_color, 
                    width=2, 
                    smooth=True, 
                    tags="wave"
                )
        
        self.phase += self.speed
        self.after(30, self._animate)


class ProgressBar(ctk.CTkFrame):
    """Custom progress bar with gradient effect."""
    
    def __init__(self, master, config: BootConfig, **kwargs):
        super().__init__(master, fg_color=config.bg_surface, **kwargs)
        self.config = config
        self.progress = 0
        
        # Background track
        self.track = ctk.CTkFrame(
            self, 
            fg_color=config.bg_dark,
            corner_radius=10,
            height=20
        )
        self.track.pack(fill="x", padx=2, pady=2)
        
        # Progress fill
        self.fill = ctk.CTkFrame(
            self.track,
            fg_color=config.primary,
            corner_radius=8,
            height=16,
            width=1  # Start with minimal width
        )
        self.fill.place(x=2, y=2)
        
        # Percentage label
        self.label = ctk.CTkLabel(
            self,
            text="0%",
            font=(config.font_family, 14, "bold"),
            text_color=config.text_primary
        )
        self.label.pack(pady=(5, 0))
        
    def set_progress(self, value: float):
        """Set progress value (0.0 to 1.0)."""
        self.progress = max(0, min(1, value))
        # Calculate width based on track width
        track_width = self.track.winfo_width()
        if track_width < 10:
            track_width = 500  # Default width estimate
        fill_width = int((track_width - 4) * self.progress * 0.98)
        self.fill.configure(width=max(1, fill_width))
        self.fill.place(x=2, y=2)
        self.label.configure(text=f"{int(self.progress * 100)}%")
        
    def set_color(self, color: str):
        """Change progress bar color."""
        self.fill.configure(fg_color=color)


class StatusItem(ctk.CTkFrame):
    """Individual status item with icon and text."""
    
    def __init__(self, master, config: BootConfig, text: str, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        
        # Status icon
        self.icon = ctk.CTkLabel(
            self,
            text="○",
            font=(config.font_family, 16),
            text_color=config.text_secondary,
            width=25
        )
        self.icon.pack(side="left")
        
        # Status text
        self.text_label = ctk.CTkLabel(
            self,
            text=text,
            font=(config.font_family, 14),
            text_color=config.text_secondary,
            anchor="w"
        )
        self.text_label.pack(side="left", fill="x", expand=True)
        
        # Detail text (right side)
        self.detail = ctk.CTkLabel(
            self,
            text="",
            font=(config.font_family, 12),
            text_color=config.text_secondary
        )
        self.detail.pack(side="right")
        
    def set_pending(self):
        """Set to pending state."""
        self.icon.configure(text="○", text_color=self.config.text_secondary)
        self.text_label.configure(text_color=self.config.text_secondary)
        
    def set_loading(self):
        """Set to loading state."""
        self.icon.configure(text="●", text_color=self.config.warning)
        self.text_label.configure(text_color=self.config.text_primary)
        
    def set_complete(self, detail: str = ""):
        """Set to complete state."""
        self.icon.configure(text="✓", text_color=self.config.success)
        self.text_label.configure(text_color=self.config.text_primary)
        if detail:
            self.detail.configure(text=detail)
            
    def set_error(self, detail: str = ""):
        """Set to error state."""
        self.icon.configure(text="✗", text_color=self.config.error)
        self.text_label.configure(text_color=self.config.error)
        if detail:
            self.detail.configure(text=detail, text_color=self.config.error)


# ─────────────────────────────────────────────
# Main Boot UI
# ─────────────────────────────────────────────

class FluxBootUI:
    """Main FLUX Boot UI using CustomTkinter."""
    
    def __init__(self, config: Optional[BootConfig] = None):
        self.config = config or BootConfig()
        self.audio = AudioController(self.config)
        
        # Setup CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create window
        self.root = ctk.CTk()
        self.root.title(self.config.title)
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        self.root.configure(fg_color=self.config.bg_dark)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.config.width) // 2
        y = (self.root.winfo_screenheight() - self.config.height) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # State
        self.current_phase = 0
        self.module_count = 0
        self.total_modules = 99
        self.boot_complete = False
        self.on_ready_callback: Optional[Callable] = None
        
        # Build UI
        self._build_ui()
        
    def _build_ui(self):
        """Build the boot UI components."""
        # Main container
        self.main_frame = ctk.CTkFrame(
            self.root,
            fg_color=self.config.bg_dark
        )
        self.main_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Logo section
        self.logo_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )
        self.logo_frame.pack(pady=(0, 20))
        
        # FLUX logo text
        self.logo_label = ctk.CTkLabel(
            self.logo_frame,
            text="FLUX",
            font=(self.config.font_family, 64, "bold"),
            text_color=self.config.primary
        )
        self.logo_label.pack()
        
        # Subtitle
        self.subtitle = ctk.CTkLabel(
            self.logo_frame,
            text="MEMORY FABRIC",
            font=(self.config.font_family, 18),
            text_color=self.config.text_secondary
        )
        self.subtitle.pack()
        
        # Wave animation
        self.wave = WaveAnimation(
            self.main_frame,
            self.config,
            width=self.config.width - 100,
            height=60
        )
        self.wave.pack(pady=20)
        
        # Status text
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Initializing...",
            font=(self.config.font_family, 16),
            text_color=self.config.text_primary
        )
        self.status_label.pack(pady=(10, 5))
        
        # Progress bar
        self.progress = ProgressBar(
            self.main_frame,
            self.config,
            width=self.config.width - 150
        )
        self.progress.pack(pady=10)
        
        # Status items container
        self.status_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=self.config.bg_surface,
            corner_radius=10
        )
        self.status_frame.pack(fill="x", pady=20, padx=20)
        
        # Create status items
        self.status_items: Dict[str, StatusItem] = {}
        items = [
            ("gpu", "GPU Detection"),
            ("memory", "System Memory"),
            ("storage", "Storage Check"),
            ("network", "Network"),
            ("model", "Model Loading"),
            ("bootstrap", "Bootstrap Runtime"),
        ]
        
        for key, text in items:
            item = StatusItem(self.status_frame, self.config, text)
            item.pack(fill="x", padx=15, pady=5)
            self.status_items[key] = item
            
        # Module loading detail
        self.module_label = ctk.CTkLabel(
            self.main_frame,
            text="",
            font=(self.config.font_family, 12),
            text_color=self.config.text_secondary
        )
        self.module_label.pack(pady=5)
        
        # Version info (bottom)
        self.version_label = ctk.CTkLabel(
            self.main_frame,
            text="v8.3-autonomous",
            font=(self.config.font_family, 12),
            text_color=self.config.text_secondary
        )
        self.version_label.pack(side="bottom", pady=10)
        
    def set_status(self, text: str):
        """Update main status text."""
        self.status_label.configure(text=text)
        
    def set_progress(self, value: float):
        """Update progress bar (0.0 to 1.0)."""
        self.progress.set_progress(value)
        
    def update_module(self, name: str, index: int, total: int):
        """Update module loading progress."""
        self.module_count = index
        self.total_modules = total
        short_name = name.split('/')[-1] if '/' in name else name
        self.module_label.configure(text=f"Loading: {short_name} ({index}/{total})")
        self.set_progress(0.3 + (index / total) * 0.5)  # 30-80% range
        
    def phase_hardware_init(self):
        """Phase 0: Hardware initialization."""
        self.current_phase = 0
        self.wave.start()
        self.wave.set_color(self.config.primary)
        self.set_status("Initializing hardware...")
        self.set_progress(0.05)
        
    def phase_system_check(self, gpu_info: str = "", ram_info: str = "", 
                          storage_info: str = ""):
        """Phase 1: System check."""
        self.current_phase = 1
        self.wave.set_color(self.config.cyan)
        self.set_status("Checking system requirements...")
        
        # Start brand sound
        self.audio.set_volume(0.5)
        self.audio.play(loops=-1, fade_ms=2000)
        
        self.status_items["gpu"].set_loading()
        self.root.after(300, lambda: self._complete_item("gpu", gpu_info or "Detected"))
        
        self.root.after(600, lambda: self.status_items["memory"].set_loading())
        self.root.after(900, lambda: self._complete_item("memory", ram_info or "OK"))
        
        self.root.after(1200, lambda: self.status_items["storage"].set_loading())
        self.root.after(1500, lambda: self._complete_item("storage", storage_info or "OK"))
        
        self.root.after(1800, lambda: self.status_items["network"].set_loading())
        self.root.after(2100, lambda: self._complete_item("network", "Local"))
        
        self.set_progress(0.2)
        
    def _complete_item(self, key: str, detail: str):
        """Mark a status item as complete."""
        self.status_items[key].set_complete(detail)
        
    def phase_model_loading(self, model_path: str = ""):
        """Phase 2: Model loading."""
        self.current_phase = 2
        self.wave.set_color(self.config.secondary)
        self.set_status("Loading FLUX model...")
        self.status_items["model"].set_loading()
        self.set_progress(0.3)
        
        if model_path:
            size = Path(model_path).stat().st_size / 1e9 if Path(model_path).exists() else 0
            self.module_label.configure(text=f"Found: {Path(model_path).name} ({size:.2f} GB)")
            
    def phase_bootstrap(self):
        """Phase 3: Bootstrap execution."""
        self.current_phase = 3
        self.wave.set_color(self.config.secondary)
        self.set_status("Bootstrapping embedded runtime...")
        self._complete_item("model", "Loaded")
        self.status_items["bootstrap"].set_loading()
        self.set_progress(0.4)
        
    def phase_ready(self):
        """Phase 4: Ready state."""
        self.current_phase = 4
        self.wave.set_color(self.config.success)
        self.set_status("FLUX is ready!")
        self._complete_item("bootstrap", f"{self.total_modules} modules")
        self.set_progress(1.0)
        self.progress.set_color(self.config.success)
        self.boot_complete = True
        
        # Fade out music
        self.audio.stop(fade_ms=3000)
        
        # Update logo color
        self.logo_label.configure(text_color=self.config.success)
        
        # Call ready callback if set
        if self.on_ready_callback:
            self.root.after(1000, self.on_ready_callback)
            
    def on_ready(self, callback: Callable):
        """Set callback for when boot is complete."""
        self.on_ready_callback = callback
        
    def show_error(self, message: str, detail: str = ""):
        """Show error state."""
        self.wave.set_color(self.config.error)
        self.set_status(f"Error: {message}")
        self.progress.set_color(self.config.error)
        self.audio.stop()
        
        if detail:
            self.module_label.configure(
                text=detail,
                text_color=self.config.error
            )
            
    def run(self):
        """Start the UI main loop."""
        self.root.mainloop()
        
    def close(self):
        """Close the UI."""
        self.audio.stop()
        self.wave.stop()
        self.root.destroy()


# ─────────────────────────────────────────────
# Demo / Test Mode
# ─────────────────────────────────────────────

def demo_boot_sequence():
    """Run a demo boot sequence."""
    ui = FluxBootUI()
    
    def run_sequence():
        # Phase 0
        ui.phase_hardware_init()
        
        # Phase 1 (after 1s)
        ui.root.after(1000, lambda: ui.phase_system_check(
            gpu_info="RTX 4090 (24GB)",
            ram_info="64GB DDR5",
            storage_info="2TB NVMe"
        ))
        
        # Phase 2 (after 4s)
        ui.root.after(4000, lambda: ui.phase_model_loading(
            "checkpoints/Flux-Apex-V1.flx"
        ))
        
        # Phase 3 (after 6s)
        ui.root.after(6000, ui.phase_bootstrap)
        
        # Simulate module loading
        def load_modules():
            modules = [
                "flux_model.py", "flux_utils.py", "bootstrap.py",
                "phases/phase1/cse.py", "phases/phase2/field.py",
                "phases/phase_autonomous/tool_executor.py",
                "phases/phase_autonomous/coder_pool.py",
            ]
            for i, mod in enumerate(modules):
                ui.root.after(6500 + i * 200, lambda m=mod, idx=i: ui.update_module(m, idx+1, len(modules)))
        
        ui.root.after(6500, load_modules)
        
        # Phase 4 - Ready (after 10s)
        ui.root.after(10000, ui.phase_ready)
    
    # Start sequence after window appears
    ui.root.after(500, run_sequence)
    
    # Handle ready state
    def on_ready():
        print("\n✓ FLUX Boot complete!")
        print("● Launching Dashboard...\n")
        # Close boot UI and launch dashboard
        ui.root.after(2000, lambda: launch_dashboard(ui))
    
    ui.on_ready(on_ready)
    ui.run()


def launch_dashboard(boot_ui: FluxBootUI = None):
    """Launch the main dashboard after boot."""
    if boot_ui:
        boot_ui.close()
    
    # Import and run dashboard
    from flux_hub.dashboard_ui import FluxDashboard
    dashboard = FluxDashboard()
    dashboard.run()


if __name__ == "__main__":
    # Check dependencies
    try:
        import customtkinter
    except ImportError:
        print("Installing customtkinter...")
        os.system(f"{sys.executable} -m pip install customtkinter")
        import customtkinter
    
    if not AUDIO_AVAILABLE:
        print("Installing pygame for audio...")
        os.system(f"{sys.executable} -m pip install pygame")
    
    print("Starting FLUX Boot UI demo...")
    demo_boot_sequence()
