#!/usr/bin/env python3
"""
FLUX Hub Dashboard UI — CustomTkinter Implementation

Main dashboard with sidebar navigation, setup wizards, and fabric connections.
Supports Personal, Family/Home, Teams, and Settings views.
"""

import customtkinter as ctk
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import json
import threading

# Try to import pygame for audio
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Try PIL for images
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class PersonProfile:
    """Individual person profile."""
    id: str
    first_name: str = ""
    last_name: str = ""
    birthday: str = ""  # YYYY-MM-DD
    photos: List[str] = field(default_factory=list)  # Paths to face photos
    voice_samples: List[str] = field(default_factory=list)  # Paths to voice recordings
    fabric_connected: bool = False
    avatar_color: str = "#6366F1"
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip() or "Unnamed"
    
    @property
    def initials(self) -> str:
        f = self.first_name[0] if self.first_name else ""
        l = self.last_name[0] if self.last_name else ""
        return (f + l).upper() or "?"


@dataclass
class FamilyHome:
    """Family/Home configuration."""
    name: str = "My Home"
    members: List[PersonProfile] = field(default_factory=list)


@dataclass 
class Team:
    """Team/Company configuration."""
    id: str
    name: str = ""
    company: str = ""
    members: List[PersonProfile] = field(default_factory=list)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    # Window
    width: int = 1200
    height: int = 800
    title: str = "FLUX Memory Fabric"
    
    # Colors
    bg_dark: str = "#0F172A"
    bg_surface: str = "#1E293B"
    bg_card: str = "#334155"
    primary: str = "#6366F1"
    secondary: str = "#8B5CF6"
    success: str = "#22C55E"
    warning: str = "#F59E0B"
    error: str = "#EF4444"
    text_primary: str = "#F1F5F9"
    text_secondary: str = "#94A3B8"
    cyan: str = "#06B6D4"
    pink: str = "#EC4899"
    
    # Sidebar
    sidebar_width: int = 280
    
    # Fonts
    font_family: str = "Helvetica"
    
    # Audio
    brand_sound: str = "flux_hub/assets/audio/Beyond_The_Flux.wav"


# ─────────────────────────────────────────────
# Audio Controller
# ─────────────────────────────────────────────

class AudioController:
    """Handle audio playback."""
    _instance = None
    
    def __new__(cls, config: DashboardConfig = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: DashboardConfig = None):
        if self._initialized:
            return
        self._initialized = True
        self.config = config or DashboardConfig()
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        self._load_sounds()
    
    def _load_sounds(self):
        if not AUDIO_AVAILABLE:
            return
        
        # Brand sound
        sound_path = Path(self.config.brand_sound)
        if sound_path.exists():
            try:
                self.sounds['brand'] = pygame.mixer.Sound(str(sound_path))
                print(f"✓ Audio loaded: {sound_path.name}")
            except:
                pass
    
    def play(self, name: str = 'brand', loops: int = 0, volume: float = 0.5):
        if name in self.sounds:
            self.sounds[name].set_volume(volume)
            self.sounds[name].play(loops=loops, fade_ms=1000)
    
    def stop(self, name: str = 'brand', fade_ms: int = 2000):
        if name in self.sounds:
            self.sounds[name].fadeout(fade_ms)


# ─────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────

class AvatarCircle(ctk.CTkFrame):
    """Circular avatar with initials or image."""
    
    def __init__(self, master, config: DashboardConfig, size: int = 50,
                 initials: str = "?", color: str = None, **kwargs):
        super().__init__(
            master,
            width=size,
            height=size,
            corner_radius=size // 2,
            fg_color=color or config.primary,
            **kwargs
        )
        self.config = config
        self.size = size
        
        self.label = ctk.CTkLabel(
            self,
            text=initials,
            font=(config.font_family, size // 3, "bold"),
            text_color=config.text_primary
        )
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Prevent resizing
        self.pack_propagate(False)
        self.grid_propagate(False)
    
    def set_initials(self, initials: str):
        self.label.configure(text=initials[:2].upper())
    
    def set_color(self, color: str):
        self.configure(fg_color=color)


class SidebarButton(ctk.CTkButton):
    """Styled sidebar navigation button."""
    
    def __init__(self, master, config: DashboardConfig, text: str, 
                 icon: str = "", active: bool = False, **kwargs):
        self.config = config
        self._active = active
        
        super().__init__(
            master,
            text=f"  {icon}  {text}" if icon else text,
            font=(config.font_family, 14),
            fg_color=config.primary if active else "transparent",
            hover_color=config.bg_card,
            text_color=config.text_primary,
            anchor="w",
            height=45,
            corner_radius=10,
            **kwargs
        )
    
    def set_active(self, active: bool):
        self._active = active
        self.configure(
            fg_color=self.config.primary if active else "transparent"
        )


class InfoCard(ctk.CTkFrame):
    """Information card with title and content."""
    
    def __init__(self, master, config: DashboardConfig, title: str = "",
                 **kwargs):
        super().__init__(
            master,
            fg_color=config.bg_surface,
            corner_radius=15,
            **kwargs
        )
        self.config = config
        
        if title:
            self.title = ctk.CTkLabel(
                self,
                text=title,
                font=(config.font_family, 16, "bold"),
                text_color=config.text_primary
            )
            self.title.pack(anchor="w", padx=20, pady=(15, 10))
        
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=20, pady=(0, 15))


class MemberCard(ctk.CTkFrame):
    """Card for displaying a person/member."""
    
    def __init__(self, master, config: DashboardConfig, person: PersonProfile,
                 on_click: Callable = None, **kwargs):
        super().__init__(
            master,
            fg_color=config.bg_card,
            corner_radius=12,
            **kwargs
        )
        self.config = config
        self.person = person
        self.on_click = on_click
        
        # Make clickable
        self.bind("<Button-1>", self._handle_click)
        
        # Avatar
        self.avatar = AvatarCircle(
            self, config, size=50,
            initials=person.initials,
            color=person.avatar_color
        )
        self.avatar.pack(side="left", padx=15, pady=15)
        self.avatar.bind("<Button-1>", self._handle_click)
        
        # Info
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, pady=15)
        info_frame.bind("<Button-1>", self._handle_click)
        
        name_label = ctk.CTkLabel(
            info_frame,
            text=person.full_name,
            font=(config.font_family, 14, "bold"),
            text_color=config.text_primary,
            anchor="w"
        )
        name_label.pack(anchor="w")
        name_label.bind("<Button-1>", self._handle_click)
        
        # Status
        status_text = "✓ Connected" if person.fabric_connected else "○ Not set up"
        status_color = config.success if person.fabric_connected else config.text_secondary
        status_label = ctk.CTkLabel(
            info_frame,
            text=status_text,
            font=(config.font_family, 12),
            text_color=status_color,
            anchor="w"
        )
        status_label.pack(anchor="w")
        status_label.bind("<Button-1>", self._handle_click)
        
        # Arrow
        arrow = ctk.CTkLabel(
            self,
            text="›",
            font=(config.font_family, 24),
            text_color=config.text_secondary
        )
        arrow.pack(side="right", padx=15)
        arrow.bind("<Button-1>", self._handle_click)
    
    def _handle_click(self, event=None):
        if self.on_click:
            self.on_click(self.person)


class WaveHeader(ctk.CTkCanvas):
    """Animated wave header."""
    
    def __init__(self, master, config: DashboardConfig, height: int = 100, **kwargs):
        super().__init__(
            master,
            bg=config.bg_dark,
            highlightthickness=0,
            height=height,
            **kwargs
        )
        self.config = config
        self.phase = 0
        self.running = False
        
    def start(self):
        self.running = True
        self._animate()
    
    def stop(self):
        self.running = False
    
    def _animate(self):
        if not self.running:
            return
        
        self.delete("wave")
        width = self.winfo_width() or 800
        height = self.winfo_height() or 100
        
        import math
        
        # Multiple wave layers
        colors = [self.config.primary, self.config.secondary, self.config.cyan]
        for i, color in enumerate(colors):
            points = []
            for x in range(0, width, 4):
                y = height // 2 + 20 * math.sin(0.02 * x + self.phase + i * 0.7)
                y += 10 * math.sin(0.01 * x + self.phase * 0.5)
                points.extend([x, y])
            
            if len(points) >= 4:
                self.create_line(
                    *points,
                    fill=color,
                    width=2,
                    smooth=True,
                    tags="wave"
                )
        
        self.phase += 0.05
        self.after(50, self._animate)


# ─────────────────────────────────────────────
# Setup Wizard
# ─────────────────────────────────────────────

class SetupWizard(ctk.CTkToplevel):
    """Setup wizard for new person."""
    
    def __init__(self, master, config: DashboardConfig, 
                 person: PersonProfile = None,
                 on_complete: Callable = None, **kwargs):
        super().__init__(master, **kwargs)
        self.config = config
        self.person = person or PersonProfile(id=f"person_{id(self)}")
        self.on_complete = on_complete
        self.step = 0
        
        # Window setup
        self.title("FLUX — Setup Person")
        self.geometry("600x700")
        self.configure(fg_color=config.bg_dark)
        
        # Center on parent
        self.transient(master)
        self.grab_set()
        
        # Build UI
        self._build_ui()
        self._show_step(0)
    
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=self.config.bg_surface, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        self.step_label = ctk.CTkLabel(
            header,
            text="Step 1 of 3",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        )
        self.step_label.pack(pady=(15, 5))
        
        self.title_label = ctk.CTkLabel(
            header,
            text="Basic Information",
            font=(self.config.font_family, 20, "bold"),
            text_color=self.config.text_primary
        )
        self.title_label.pack()
        
        # Progress dots
        self.dots_frame = ctk.CTkFrame(header, fg_color="transparent")
        self.dots_frame.pack(pady=10)
        self.dots = []
        for i in range(3):
            dot = ctk.CTkLabel(
                self.dots_frame,
                text="●" if i == 0 else "○",
                font=(self.config.font_family, 12),
                text_color=self.config.primary if i == 0 else self.config.text_secondary
            )
            dot.pack(side="left", padx=5)
            self.dots.append(dot)
        
        # Content area
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=40, pady=20)
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(self, fg_color="transparent", height=60)
        nav_frame.pack(fill="x", padx=40, pady=20)
        
        self.back_btn = ctk.CTkButton(
            nav_frame,
            text="← Back",
            font=(self.config.font_family, 14),
            fg_color="transparent",
            hover_color=self.config.bg_card,
            text_color=self.config.text_secondary,
            width=100,
            command=self._prev_step
        )
        self.back_btn.pack(side="left")
        
        self.next_btn = ctk.CTkButton(
            nav_frame,
            text="Next →",
            font=(self.config.font_family, 14, "bold"),
            fg_color=self.config.primary,
            hover_color=self.config.secondary,
            width=120,
            command=self._next_step
        )
        self.next_btn.pack(side="right")
    
    def _show_step(self, step: int):
        self.step = step
        
        # Update header
        titles = ["Basic Information", "Face Recognition", "Voice Setup"]
        self.step_label.configure(text=f"Step {step + 1} of 3")
        self.title_label.configure(text=titles[step])
        
        # Update dots
        for i, dot in enumerate(self.dots):
            dot.configure(
                text="●" if i <= step else "○",
                text_color=self.config.primary if i <= step else self.config.text_secondary
            )
        
        # Update buttons
        self.back_btn.configure(state="normal" if step > 0 else "disabled")
        self.next_btn.configure(text="Complete ✓" if step == 2 else "Next →")
        
        # Clear content
        for widget in self.content.winfo_children():
            widget.destroy()
        
        # Show step content
        if step == 0:
            self._step_basic_info()
        elif step == 1:
            self._step_face_setup()
        elif step == 2:
            self._step_voice_setup()
    
    def _step_basic_info(self):
        """Step 1: Basic information."""
        # Avatar preview
        avatar_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        avatar_frame.pack(pady=20)
        
        self.preview_avatar = AvatarCircle(
            avatar_frame, self.config, size=100,
            initials=self.person.initials,
            color=self.person.avatar_color
        )
        self.preview_avatar.pack()
        
        # Color picker
        colors_frame = ctk.CTkFrame(avatar_frame, fg_color="transparent")
        colors_frame.pack(pady=10)
        
        colors = ["#6366F1", "#8B5CF6", "#EC4899", "#EF4444", 
                  "#F59E0B", "#22C55E", "#06B6D4", "#3B82F6"]
        for color in colors:
            btn = ctk.CTkButton(
                colors_frame,
                text="",
                width=30,
                height=30,
                corner_radius=15,
                fg_color=color,
                hover_color=color,
                command=lambda c=color: self._set_avatar_color(c)
            )
            btn.pack(side="left", padx=3)
        
        # Form fields
        form = ctk.CTkFrame(self.content, fg_color="transparent")
        form.pack(fill="x", pady=20)
        
        # First name
        ctk.CTkLabel(
            form, text="First Name",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(anchor="w")
        
        self.first_name_entry = ctk.CTkEntry(
            form,
            placeholder_text="Enter first name",
            font=(self.config.font_family, 14),
            height=45,
            corner_radius=10
        )
        self.first_name_entry.pack(fill="x", pady=(5, 15))
        if self.person.first_name:
            self.first_name_entry.insert(0, self.person.first_name)
        self.first_name_entry.bind("<KeyRelease>", self._update_preview)
        
        # Last name
        ctk.CTkLabel(
            form, text="Last Name",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(anchor="w")
        
        self.last_name_entry = ctk.CTkEntry(
            form,
            placeholder_text="Enter last name",
            font=(self.config.font_family, 14),
            height=45,
            corner_radius=10
        )
        self.last_name_entry.pack(fill="x", pady=(5, 15))
        if self.person.last_name:
            self.last_name_entry.insert(0, self.person.last_name)
        self.last_name_entry.bind("<KeyRelease>", self._update_preview)
        
        # Birthday
        ctk.CTkLabel(
            form, text="Birthday",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(anchor="w")
        
        self.birthday_entry = ctk.CTkEntry(
            form,
            placeholder_text="YYYY-MM-DD",
            font=(self.config.font_family, 14),
            height=45,
            corner_radius=10
        )
        self.birthday_entry.pack(fill="x", pady=(5, 15))
        if self.person.birthday:
            self.birthday_entry.insert(0, self.person.birthday)
    
    def _step_face_setup(self):
        """Step 2: Face recognition photos."""
        # Instructions
        ctk.CTkLabel(
            self.content,
            text="Add photos for face recognition",
            font=(self.config.font_family, 16),
            text_color=self.config.text_primary
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            self.content,
            text="Upload multiple photos from different angles for better recognition",
            font=(self.config.font_family, 13),
            text_color=self.config.text_secondary
        ).pack(pady=(0, 20))
        
        # Photo grid
        photos_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        photos_frame.pack(fill="both", expand=True)
        
        # Photo slots (3x2 grid)
        for i in range(6):
            row = i // 3
            col = i % 3
            
            frame = ctk.CTkFrame(
                photos_frame,
                fg_color=self.config.bg_surface,
                corner_radius=10,
                width=150,
                height=150
            )
            frame.grid(row=row, column=col, padx=10, pady=10)
            frame.pack_propagate(False)
            
            if i < len(self.person.photos):
                # Show existing photo placeholder
                ctk.CTkLabel(
                    frame,
                    text="📷",
                    font=(self.config.font_family, 40)
                ).place(relx=0.5, rely=0.5, anchor="center")
            else:
                # Add button
                add_btn = ctk.CTkButton(
                    frame,
                    text="+",
                    font=(self.config.font_family, 30),
                    fg_color="transparent",
                    hover_color=self.config.bg_card,
                    text_color=self.config.text_secondary,
                    command=lambda: self._add_photo()
                )
                add_btn.place(relx=0.5, rely=0.5, anchor="center")
        
        # Demo note
        ctk.CTkLabel(
            self.content,
            text="📸 Demo mode: Click + to simulate adding photos",
            font=(self.config.font_family, 12),
            text_color=self.config.text_secondary
        ).pack(pady=20)
    
    def _step_voice_setup(self):
        """Step 3: Voice setup."""
        # Instructions
        ctk.CTkLabel(
            self.content,
            text="Record your voice for recognition",
            font=(self.config.font_family, 16),
            text_color=self.config.text_primary
        ).pack(pady=(20, 5))
        
        ctk.CTkLabel(
            self.content,
            text="Record a few samples so FLUX can recognize you",
            font=(self.config.font_family, 13),
            text_color=self.config.text_secondary
        ).pack(pady=(0, 30))
        
        # Voice visualizer placeholder
        viz_frame = ctk.CTkFrame(
            self.content,
            fg_color=self.config.bg_surface,
            corner_radius=15,
            height=150
        )
        viz_frame.pack(fill="x", pady=20)
        viz_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            viz_frame,
            text="🎤",
            font=(self.config.font_family, 50)
        ).place(relx=0.5, rely=0.5, anchor="center")
        
        # Record button
        self.record_btn = ctk.CTkButton(
            self.content,
            text="🔴  Start Recording",
            font=(self.config.font_family, 16, "bold"),
            fg_color=self.config.error,
            hover_color="#DC2626",
            height=50,
            corner_radius=25,
            command=self._toggle_recording
        )
        self.record_btn.pack(pady=20)
        
        # Sample phrases
        ctk.CTkLabel(
            self.content,
            text="Try saying these phrases:",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(pady=(20, 10))
        
        phrases = [
            '"Hey FLUX, what\'s on my calendar today?"',
            '"Play some relaxing music"',
            '"Turn off the lights in the living room"'
        ]
        for phrase in phrases:
            ctk.CTkLabel(
                self.content,
                text=phrase,
                font=(self.config.font_family, 13),
                text_color=self.config.text_primary
            ).pack(pady=3)
    
    def _set_avatar_color(self, color: str):
        self.person.avatar_color = color
        if hasattr(self, 'preview_avatar'):
            self.preview_avatar.set_color(color)
    
    def _update_preview(self, event=None):
        first = self.first_name_entry.get()
        last = self.last_name_entry.get()
        self.person.first_name = first
        self.person.last_name = last
        if hasattr(self, 'preview_avatar'):
            self.preview_avatar.set_initials(self.person.initials)
    
    def _add_photo(self):
        # Demo: simulate adding photo
        self.person.photos.append(f"photo_{len(self.person.photos)}.jpg")
        self._show_step(1)  # Refresh
    
    def _toggle_recording(self):
        # Demo: toggle recording state
        if "Start" in self.record_btn.cget("text"):
            self.record_btn.configure(text="⏹  Stop Recording")
        else:
            self.record_btn.configure(text="🔴  Start Recording")
            self.person.voice_samples.append(f"voice_{len(self.person.voice_samples)}.wav")
    
    def _prev_step(self):
        if self.step > 0:
            self._save_current_step()
            self._show_step(self.step - 1)
    
    def _next_step(self):
        self._save_current_step()
        if self.step < 2:
            self._show_step(self.step + 1)
        else:
            self._complete()
    
    def _save_current_step(self):
        if self.step == 0:
            self.person.first_name = self.first_name_entry.get()
            self.person.last_name = self.last_name_entry.get()
            self.person.birthday = self.birthday_entry.get()
    
    def _complete(self):
        self.person.fabric_connected = True
        if self.on_complete:
            self.on_complete(self.person)
        self.destroy()


# ─────────────────────────────────────────────
# Main Dashboard Views
# ─────────────────────────────────────────────

class PersonalView(ctk.CTkFrame):
    """Personal profile view."""
    
    def __init__(self, master, config: DashboardConfig, 
                 profile: PersonProfile = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        self.profile = profile or PersonProfile(id="personal")
        self._build_ui()
    
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            header,
            text="Personal",
            font=(self.config.font_family, 28, "bold"),
            text_color=self.config.text_primary
        ).pack(side="left")
        
        if not self.profile.fabric_connected:
            setup_btn = ctk.CTkButton(
                header,
                text="+ Set Up Profile",
                font=(self.config.font_family, 14),
                fg_color=self.config.primary,
                hover_color=self.config.secondary,
                command=self._open_setup
            )
            setup_btn.pack(side="right")
        
        # Content
        if self.profile.fabric_connected:
            self._show_connected_profile()
        else:
            self._show_setup_prompt()
    
    def _show_setup_prompt(self):
        # Welcome card
        card = InfoCard(self, self.config, title="Welcome to FLUX")
        card.pack(fill="x")
        
        ctk.CTkLabel(
            card.content,
            text="Set up your personal profile to get started.\n"
                 "FLUX will learn to recognize you and personalize your experience.",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary,
            justify="left"
        ).pack(anchor="w")
        
        ctk.CTkButton(
            card.content,
            text="Begin Setup →",
            font=(self.config.font_family, 14, "bold"),
            fg_color=self.config.primary,
            command=self._open_setup
        ).pack(anchor="w", pady=(15, 0))
    
    def _show_connected_profile(self):
        # Profile card
        card = InfoCard(self, self.config)
        card.pack(fill="x")
        
        profile_frame = ctk.CTkFrame(card.content, fg_color="transparent")
        profile_frame.pack(fill="x")
        
        avatar = AvatarCircle(
            profile_frame, self.config, size=80,
            initials=self.profile.initials,
            color=self.profile.avatar_color
        )
        avatar.pack(side="left")
        
        info = ctk.CTkFrame(profile_frame, fg_color="transparent")
        info.pack(side="left", padx=20, fill="x", expand=True)
        
        ctk.CTkLabel(
            info,
            text=self.profile.full_name,
            font=(self.config.font_family, 20, "bold"),
            text_color=self.config.text_primary,
            anchor="w"
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            info,
            text=f"✓ Fabric Connected",
            font=(self.config.font_family, 14),
            text_color=self.config.success,
            anchor="w"
        ).pack(anchor="w")
        
        # Stats
        stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        stats_frame.pack(fill="x", pady=20)
        
        stats = [
            ("📷", str(len(self.profile.photos)), "Photos"),
            ("🎤", str(len(self.profile.voice_samples)), "Voice Samples"),
            ("🧠", "Active", "AI Status"),
        ]
        
        for icon, value, label in stats:
            stat_card = ctk.CTkFrame(
                stats_frame,
                fg_color=self.config.bg_surface,
                corner_radius=12
            )
            stat_card.pack(side="left", fill="both", expand=True, padx=5)
            
            ctk.CTkLabel(
                stat_card, text=icon,
                font=(self.config.font_family, 24)
            ).pack(pady=(15, 5))
            
            ctk.CTkLabel(
                stat_card, text=value,
                font=(self.config.font_family, 18, "bold"),
                text_color=self.config.text_primary
            ).pack()
            
            ctk.CTkLabel(
                stat_card, text=label,
                font=(self.config.font_family, 12),
                text_color=self.config.text_secondary
            ).pack(pady=(0, 15))
    
    def _open_setup(self):
        wizard = SetupWizard(
            self.winfo_toplevel(), 
            self.config,
            person=self.profile,
            on_complete=self._on_setup_complete
        )
    
    def _on_setup_complete(self, person: PersonProfile):
        self.profile = person
        # Refresh view
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()


class FamilyView(ctk.CTkFrame):
    """Family/Home view."""
    
    def __init__(self, master, config: DashboardConfig,
                 family: FamilyHome = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        self.family = family or FamilyHome()
        self._build_ui()
    
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            header,
            text="🏠  " + self.family.name,
            font=(self.config.font_family, 28, "bold"),
            text_color=self.config.text_primary
        ).pack(side="left")
        
        add_btn = ctk.CTkButton(
            header,
            text="+ Add Member",
            font=(self.config.font_family, 14),
            fg_color=self.config.primary,
            hover_color=self.config.secondary,
            command=self._add_member
        )
        add_btn.pack(side="right")
        
        # Home name setting
        settings_card = InfoCard(self, self.config, title="Home Settings")
        settings_card.pack(fill="x", pady=(0, 20))
        
        name_frame = ctk.CTkFrame(settings_card.content, fg_color="transparent")
        name_frame.pack(fill="x")
        
        self.home_name_entry = ctk.CTkEntry(
            name_frame,
            placeholder_text="Home name",
            font=(self.config.font_family, 14),
            height=40,
            corner_radius=8
        )
        self.home_name_entry.pack(side="left", fill="x", expand=True)
        self.home_name_entry.insert(0, self.family.name)
        
        save_btn = ctk.CTkButton(
            name_frame,
            text="Save",
            font=(self.config.font_family, 14),
            fg_color=self.config.primary,
            width=80,
            command=self._save_home_name
        )
        save_btn.pack(side="right", padx=(10, 0))
        
        # Members list
        members_card = InfoCard(self, self.config, title=f"Members ({len(self.family.members)})")
        members_card.pack(fill="both", expand=True)
        
        if self.family.members:
            for person in self.family.members:
                card = MemberCard(
                    members_card.content, self.config, person,
                    on_click=self._edit_member
                )
                card.pack(fill="x", pady=5)
        else:
            ctk.CTkLabel(
                members_card.content,
                text="No family members yet.\nAdd someone to get started!",
                font=(self.config.font_family, 14),
                text_color=self.config.text_secondary
            ).pack(pady=40)
    
    def _save_home_name(self):
        self.family.name = self.home_name_entry.get()
        # Update header
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()
    
    def _add_member(self):
        person = PersonProfile(id=f"family_{len(self.family.members)}")
        wizard = SetupWizard(
            self.winfo_toplevel(),
            self.config,
            person=person,
            on_complete=self._on_member_added
        )
    
    def _on_member_added(self, person: PersonProfile):
        self.family.members.append(person)
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()
    
    def _edit_member(self, person: PersonProfile):
        wizard = SetupWizard(
            self.winfo_toplevel(),
            self.config,
            person=person,
            on_complete=lambda p: self._refresh()
        )
    
    def _refresh(self):
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()


class TeamsView(ctk.CTkFrame):
    """Teams/Company view."""
    
    def __init__(self, master, config: DashboardConfig,
                 teams: List[Team] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        self.teams = teams or []
        self._build_ui()
    
    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            header,
            text="Teams",
            font=(self.config.font_family, 28, "bold"),
            text_color=self.config.text_primary
        ).pack(side="left")
        
        add_btn = ctk.CTkButton(
            header,
            text="+ Create Team",
            font=(self.config.font_family, 14),
            fg_color=self.config.primary,
            hover_color=self.config.secondary,
            command=self._create_team
        )
        add_btn.pack(side="right")
        
        # Teams list
        if self.teams:
            for team in self.teams:
                self._render_team_card(team)
        else:
            # Empty state
            empty_card = InfoCard(self, self.config)
            empty_card.pack(fill="x")
            
            ctk.CTkLabel(
                empty_card.content,
                text="👥",
                font=(self.config.font_family, 48)
            ).pack(pady=(20, 10))
            
            ctk.CTkLabel(
                empty_card.content,
                text="No teams yet",
                font=(self.config.font_family, 18, "bold"),
                text_color=self.config.text_primary
            ).pack()
            
            ctk.CTkLabel(
                empty_card.content,
                text="Create a team to collaborate with your organization",
                font=(self.config.font_family, 14),
                text_color=self.config.text_secondary
            ).pack(pady=(5, 20))
            
            ctk.CTkButton(
                empty_card.content,
                text="Create Your First Team",
                font=(self.config.font_family, 14, "bold"),
                fg_color=self.config.primary,
                command=self._create_team
            ).pack(pady=(0, 20))
    
    def _render_team_card(self, team: Team):
        card = InfoCard(self, self.config, title=f"🏢  {team.name}")
        card.pack(fill="x", pady=(0, 15))
        
        if team.company:
            ctk.CTkLabel(
                card.content,
                text=team.company,
                font=(self.config.font_family, 14),
                text_color=self.config.text_secondary
            ).pack(anchor="w")
        
        ctk.CTkLabel(
            card.content,
            text=f"{len(team.members)} members",
            font=(self.config.font_family, 13),
            text_color=self.config.text_secondary
        ).pack(anchor="w", pady=(5, 10))
        
        # Member avatars row
        members_row = ctk.CTkFrame(card.content, fg_color="transparent")
        members_row.pack(fill="x")
        
        for i, person in enumerate(team.members[:5]):
            avatar = AvatarCircle(
                members_row, self.config, size=35,
                initials=person.initials,
                color=person.avatar_color
            )
            avatar.pack(side="left", padx=2)
        
        if len(team.members) > 5:
            ctk.CTkLabel(
                members_row,
                text=f"+{len(team.members) - 5}",
                font=(self.config.font_family, 12),
                text_color=self.config.text_secondary
            ).pack(side="left", padx=10)
        
        add_member_btn = ctk.CTkButton(
            members_row,
            text="+ Add",
            font=(self.config.font_family, 12),
            fg_color="transparent",
            hover_color=self.config.bg_card,
            text_color=self.config.primary,
            width=60,
            command=lambda t=team: self._add_team_member(t)
        )
        add_member_btn.pack(side="right")
    
    def _create_team(self):
        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Create Team")
        dialog.geometry("400x300")
        dialog.configure(fg_color=self.config.bg_dark)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        
        ctk.CTkLabel(
            dialog,
            text="Create New Team",
            font=(self.config.font_family, 20, "bold"),
            text_color=self.config.text_primary
        ).pack(pady=20)
        
        # Team name
        ctk.CTkLabel(
            dialog, text="Team Name",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(anchor="w", padx=30)
        
        name_entry = ctk.CTkEntry(
            dialog,
            placeholder_text="e.g. Engineering",
            font=(self.config.font_family, 14),
            height=40
        )
        name_entry.pack(fill="x", padx=30, pady=(5, 15))
        
        # Company
        ctk.CTkLabel(
            dialog, text="Company (optional)",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(anchor="w", padx=30)
        
        company_entry = ctk.CTkEntry(
            dialog,
            placeholder_text="e.g. Acme Corp",
            font=(self.config.font_family, 14),
            height=40
        )
        company_entry.pack(fill="x", padx=30, pady=(5, 20))
        
        def create():
            team = Team(
                id=f"team_{len(self.teams)}",
                name=name_entry.get() or "New Team",
                company=company_entry.get()
            )
            self.teams.append(team)
            dialog.destroy()
            self._refresh()
        
        ctk.CTkButton(
            dialog,
            text="Create Team",
            font=(self.config.font_family, 14, "bold"),
            fg_color=self.config.primary,
            command=create
        ).pack(pady=10)
    
    def _add_team_member(self, team: Team):
        person = PersonProfile(id=f"team_{team.id}_{len(team.members)}")
        wizard = SetupWizard(
            self.winfo_toplevel(),
            self.config,
            person=person,
            on_complete=lambda p: self._on_team_member_added(team, p)
        )
    
    def _on_team_member_added(self, team: Team, person: PersonProfile):
        team.members.append(person)
        self._refresh()
    
    def _refresh(self):
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()


class SettingsView(ctk.CTkFrame):
    """Settings view."""
    
    def __init__(self, master, config: DashboardConfig, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        self._build_ui()
    
    def _build_ui(self):
        # Header
        ctk.CTkLabel(
            self,
            text="Settings",
            font=(self.config.font_family, 28, "bold"),
            text_color=self.config.text_primary
        ).pack(anchor="w", pady=(0, 20))
        
        # Privacy
        privacy_card = InfoCard(self, self.config, title="🔒  Privacy")
        privacy_card.pack(fill="x", pady=(0, 15))
        
        self._add_toggle(privacy_card.content, "Keep all data local", True)
        self._add_toggle(privacy_card.content, "Store conversation history", True)
        self._add_toggle(privacy_card.content, "Enable voice commands", True)
        
        # Appearance
        appearance_card = InfoCard(self, self.config, title="🎨  Appearance")
        appearance_card.pack(fill="x", pady=(0, 15))
        
        self._add_toggle(appearance_card.content, "Dark mode", True)
        self._add_toggle(appearance_card.content, "Show wave animations", True)
        self._add_toggle(appearance_card.content, "Play sounds", True)
        
        # System
        system_card = InfoCard(self, self.config, title="⚙️  System")
        system_card.pack(fill="x", pady=(0, 15))
        
        # Model info
        model_frame = ctk.CTkFrame(system_card.content, fg_color="transparent")
        model_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            model_frame, text="Model",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(side="left")
        
        ctk.CTkLabel(
            model_frame, text="Flux-Apex-V1.flx (v8.3-autonomous)",
            font=(self.config.font_family, 14),
            text_color=self.config.text_primary
        ).pack(side="right")
        
        # Version
        version_frame = ctk.CTkFrame(system_card.content, fg_color="transparent")
        version_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            version_frame, text="FLUX Hub Version",
            font=(self.config.font_family, 14),
            text_color=self.config.text_secondary
        ).pack(side="left")
        
        ctk.CTkLabel(
            version_frame, text="1.0.0",
            font=(self.config.font_family, 14),
            text_color=self.config.text_primary
        ).pack(side="right")
        
        # Reset button
        ctk.CTkButton(
            system_card.content,
            text="Reset All Settings",
            font=(self.config.font_family, 14),
            fg_color="transparent",
            hover_color=self.config.bg_card,
            text_color=self.config.error,
            border_color=self.config.error,
            border_width=1
        ).pack(anchor="w", pady=(15, 0))
    
    def _add_toggle(self, parent, text: str, default: bool):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            frame, text=text,
            font=(self.config.font_family, 14),
            text_color=self.config.text_primary
        ).pack(side="left")
        
        switch = ctk.CTkSwitch(
            frame, text="",
            onvalue=True, offvalue=False,
            progress_color=self.config.primary
        )
        switch.pack(side="right")
        if default:
            switch.select()


# ─────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────

class FluxDashboard:
    """Main FLUX Dashboard UI."""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.audio = AudioController(self.config)
        
        # Data
        self.personal_profile = PersonProfile(id="personal")
        self.family = FamilyHome()
        self.teams: List[Team] = []
        
        # Current view
        self.current_view = "personal"
        
        # Setup window
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title(self.config.title)
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        self.root.configure(fg_color=self.config.bg_dark)
        
        # Center
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.config.width) // 2
        y = (self.root.winfo_screenheight() - self.config.height) // 2
        self.root.geometry(f"+{x}+{y}")
        
        self._build_ui()
    
    def _build_ui(self):
        # Main container
        self.main = ctk.CTkFrame(self.root, fg_color=self.config.bg_dark)
        self.main.pack(fill="both", expand=True)
        
        # Sidebar
        self.sidebar = ctk.CTkFrame(
            self.main,
            fg_color=self.config.bg_surface,
            width=self.config.sidebar_width,
            corner_radius=0
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        # Logo in sidebar
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=100)
        logo_frame.pack(fill="x")
        logo_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            logo_frame,
            text="FLUX",
            font=(self.config.font_family, 32, "bold"),
            text_color=self.config.primary
        ).pack(pady=(25, 0))
        
        ctk.CTkLabel(
            logo_frame,
            text="Memory Fabric",
            font=(self.config.font_family, 12),
            text_color=self.config.text_secondary
        ).pack()
        
        # Navigation
        nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_frame.pack(fill="x", padx=15, pady=20)
        
        self.nav_buttons: Dict[str, SidebarButton] = {}
        nav_items = [
            ("personal", "👤", "Personal"),
            ("family", "🏠", "Family / Home"),
            ("teams", "👥", "Teams"),
            ("settings", "⚙️", "Settings"),
        ]
        
        for key, icon, text in nav_items:
            btn = SidebarButton(
                nav_frame, self.config, text, icon,
                active=(key == self.current_view),
                command=lambda k=key: self._switch_view(k)
            )
            btn.pack(fill="x", pady=3)
            self.nav_buttons[key] = btn
        
        # Sidebar footer
        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=15, pady=20)
        
        ctk.CTkLabel(
            footer,
            text="v8.3-autonomous",
            font=(self.config.font_family, 11),
            text_color=self.config.text_secondary
        ).pack()
        
        ctk.CTkLabel(
            footer,
            text="🟢 All systems online",
            font=(self.config.font_family, 11),
            text_color=self.config.success
        ).pack()
        
        # Content area
        self.content_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.content_frame.pack(side="right", fill="both", expand=True, padx=30, pady=20)
        
        # Wave header
        self.wave_header = WaveHeader(self.content_frame, self.config, height=60)
        self.wave_header.pack(fill="x", pady=(0, 20))
        self.wave_header.start()
        
        # View container
        self.view_container = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.view_container.pack(fill="both", expand=True)
        
        # Show initial view
        self._switch_view("personal")
    
    def _switch_view(self, view_name: str):
        self.current_view = view_name
        
        # Update nav buttons
        for key, btn in self.nav_buttons.items():
            btn.set_active(key == view_name)
        
        # Clear view container
        for widget in self.view_container.winfo_children():
            widget.destroy()
        
        # Show new view
        if view_name == "personal":
            view = PersonalView(
                self.view_container, self.config, 
                profile=self.personal_profile
            )
        elif view_name == "family":
            view = FamilyView(
                self.view_container, self.config,
                family=self.family
            )
        elif view_name == "teams":
            view = TeamsView(
                self.view_container, self.config,
                teams=self.teams
            )
        elif view_name == "settings":
            view = SettingsView(self.view_container, self.config)
        
        view.pack(fill="both", expand=True)
    
    def play_intro_sound(self):
        """Play the brand intro sound."""
        self.audio.play(loops=0, volume=0.6)
    
    def run(self):
        """Start the dashboard."""
        # Play intro sound after window shows
        self.root.after(500, self.play_intro_sound)
        self.root.mainloop()
    
    def close(self):
        self.audio.stop()
        self.wave_header.stop()
        self.root.destroy()


# ─────────────────────────────────────────────
# Demo Entry
# ─────────────────────────────────────────────

def demo_dashboard():
    """Run dashboard demo."""
    print("Starting FLUX Dashboard...")
    dashboard = FluxDashboard()
    dashboard.run()


if __name__ == "__main__":
    demo_dashboard()
