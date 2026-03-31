# Memory Fabric: Personal AI Hardware Ecosystem

> **Your AI. Your Data. Runs Locally. Goes Everywhere.**

---

## The Big Picture: Personal Superintelligence

Memory Fabric isn't just an assistant — it's a **personal superintelligence layer** that spans your entire life.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    YOUR PERSONAL SUPERINTELLIGENCE                          │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                        FLUX ORCHESTRATOR                              ║ │
│  ║                                                                       ║ │
│  ║   • PERCEIVES through all your cameras, mics, sensors                ║ │
│  ║   • REMEMBERS your entire history (field + episodic memory)          ║ │
│  ║   • REASONS with causal tracking ("if X then Y")                     ║ │
│  ║   • BUILDS automations, code, integrations for you                   ║ │
│  ║   • CONTROLS your home, car, security, devices                       ║ │
│  ║   • LEARNS continuously from every interaction                       ║ │
│  ║   • KNOWS YOU across every app, every context, everywhere            ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                    │                                        │
│       ┌────────────────────────────┼────────────────────────────┐          │
│       ▼                            ▼                            ▼          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  HOME   │  │  CAR    │  │ OFFICE  │  │  PHONE  │  │ANY APP  │         │
│  │         │  │         │  │         │  │         │  │         │         │
│  │Cameras  │  │Cameras  │  │Cameras  │  │Location │  │Fabric   │         │
│  │Lights   │  │Sensors  │  │Access   │  │Calendar │  │Protocol │         │
│  │Security │  │Audio    │  │Systems  │  │Contacts │  │API      │         │
│  │Climate  │  │Nav      │  │Network  │  │Health   │  │         │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
│                                                                             │
│  SAME AI EVERYWHERE — knows you're the same person at home, car, work     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Makes This Different from Siri/Alexa/ChatGPT

| Feature | Cloud Assistants | Memory Fabric |
|---------|------------------|---------------|
| Memory | Forgets between sessions | **Remembers your entire life** |
| Identity | Different AI per app | **Same AI everywhere you go** |
| Privacy | Data sent to cloud | **100% local on your hardware** |
| Intelligence | Answers questions | **Reasons, plans, BUILDS for you** |
| Context | Single conversation | **Sees cameras, sensors, calendar, everything** |
| Learning | Static model | **Learns from every interaction** |
| Control | Voice commands only | **Autonomous actions + builds automations** |
| Access | One app at a time | **Any app via Fabric Protocol** |

---

## Builder Agent: Your AI That Creates

Memory Fabric doesn't just answer questions — it **BUILDS things for you**.

### What It Can Build

| Category | Examples |
|----------|----------|
| **Automations** | "When I get home, turn on lights, start coffee, play news" |
| **Code** | Python scripts, shell commands, integrations |
| **Rules** | "Alert me if unknown face at door for >2 minutes" |
| **Integrations** | Connect calendar → home system → car → phone |
| **Reports** | "Summarize my security footage from last week" |
| **Workflows** | "Every Monday morning, prep my weekly summary" |

### Real Example: Security Automation

```
You: "I want smart security that actually thinks"

Fabric: "I'll create an intelligent security system:

         ALWAYS RUNNING:
         ├── Face detection on all cameras
         ├── License plate recognition (driveway)
         ├── Motion pattern analysis
         └── Sound anomaly detection

         RULES I'LL CREATE:
         ├── Known faces → log arrival, no alert
         ├── Unknown face at door >30 sec → alert phone
         ├── Unknown car in driveway → check if matches 
         │   any scheduled service (plumber, delivery, etc.)
         ├── Motion at unusual hours → record + alert
         ├── Glass break sound → immediate alert + 911 ready
         └── Package delivered → notify + watch until retrieved

         INTELLIGENT RESPONSES:
         ├── If you're home → speak through intercom
         ├── If you're away → send video to phone
         ├── If dog at door → mute alert (I know your dog)
         └── If delivery expected → auto-acknowledge

         This integrates with:
         • Your calendar (knows when to expect people)
         • Your contacts (recognizes friends/family faces)
         • Car presence (knows if you're home)
         
         [Create This System]"
```

### Real Example: Cross-Device Intelligence

```
*Monday morning*

Fabric: "Good morning. Here's your day:

         PREPARED OVERNIGHT:
         ├── Car is charged (moved to off-peak rates)
         ├── Coffee started 10 min ago (you like it hot)
         ├── Route to work: 35 min (accident on I-95)
         ├── First meeting moved to 10am (client rescheduled)
         └── Security: quiet night, package delivered at 6am

         THINGS I NOTICED:
         ├── Your sleep was shorter (6h vs usual 7.5h)
         │   → Shifted meeting prep to lighter reading
         ├── Weather will be 75°F on your drive home
         │   → Garage door will pre-cool car at 4:45pm
         └── Sarah's birthday is in 3 days
             → Found that restaurant you mentioned

         NEED DECISIONS:
         ├── Accept lunch invite from Mike? (11:30am)
         └── Office says parking lot B closed — reroute?

         [Start Day] [Adjust Schedule]"
```

---

## Autonomous Control: Your AI Takes Action

Memory Fabric can **act on your behalf** — not just suggest, but DO.

### Control Surfaces

| System | What AI Controls | Example Actions |
|--------|------------------|-----------------|
| **Home** | Lights, HVAC, locks, blinds, appliances | "Lock up, I'm going to bed" |
| **Security** | Cameras, alarms, access, recording | Auto-record when stranger detected |
| **Car** | Climate, charging, location, pre-route | Pre-cool car before you leave work |
| **Office** | Desk booking, access, calendar rooms | Reserve conference room for your call |
| **Phone** | DND, notifications, calls, texts | Silence non-urgent during meetings |
| **Network** | Connected devices, IoT, smart home | "Block kids' devices after 9pm" |

### Autonomous Actions (Pre-Approved)

You pre-approve categories of actions the AI can take without asking:

```python
autonomous_permissions = {
    'energy_optimization': {
        'enabled': True,
        'actions': [
            'shift_ev_charging_to_off_peak',
            'adjust_thermostat_when_away',
            'turn_off_unused_lights',
        ],
    },
    'security': {
        'enabled': True,
        'actions': [
            'record_unknown_faces',
            'alert_on_anomalies',
            'lock_doors_at_night',
        ],
        'requires_confirmation': [
            'call_911',
            'unlock_for_visitor',
        ],
    },
    'scheduling': {
        'enabled': True,
        'actions': [
            'block_focus_time',
            'reschedule_conflicts',
            'send_running_late_messages',
        ],
    },
    'communication': {
        'enabled': False,  # Always ask first
        'actions': [],
    },
}
```

### Real-World Autonomy

```
SCENARIO: You're in a meeting, delivery arrives

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  AUTONOMOUS ACTION LOG (no interruption needed)                    │
│                                                                     │
│  10:32 AM │ Doorbell camera: FedEx driver detected                 │
│  10:32 AM │ Cross-ref: Package expected (Amazon order #7834)       │
│  10:32 AM │ Action: Unlocked package locker, spoke "Leave in box" │
│  10:33 AM │ Package placed in locker, locker locked               │
│  10:33 AM │ Photo logged, notification queued (low priority)      │
│  10:33 AM │ Car notified: package at home, no detour needed       │
│                                                                     │
│  [You see this after meeting, nothing interrupted your flow]       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Proactive Intelligence: AI That Anticipates

The FLUX orchestrator doesn't wait for commands — it **thinks ahead**.

### Pattern Learning

```
WHAT AI NOTICES:                    WHAT IT DOES:
───────────────────────────────────────────────────────────────────
Every Tuesday you have             → Pre-loads route, checks traffic
a 9am client meeting               → Warms up car 8:30am
                                   → Preps meeting notes

You always order Thai              → Suggests thai.place when
on Friday evenings                    Friday 5pm approaches
                                   → Knows your usual order

After big meetings you             → After detecting "meeting ended"
like 10 minutes alone              → Auto-sets DND for 10 min
                                   → Dims office lights slightly

When Sarah comes home              → Adjusts music to shared taste
the music should change            → Lowers volume for conversation
```

### Anticipatory Actions

```
*Thursday 4:30 PM*

Fabric: "You usually leave at 5. A few things:

         PROACTIVE PREP:
         ├── Car charging stopped at 80% (home charger waiting)
         ├── Route home: 40 min (unusual traffic on MLK Blvd)
         │   → Alternative via Oak St: 32 min
         ├── Dinner: You mentioned trying that new Thai place
         │   → Reservation available 7pm, want me to book?
         └── Home: Pre-cooling to 72° (you've been in AC all day)

         THINGS YOU MIGHT FORGET:
         ├── Dry cleaning pickup closes at 6pm (on route)
         ├── Sarah asked you to grab milk
         └── Tomorrow: Early meeting, shift wakeup 30 min?

         [Leave Now] [5 More Minutes]"
```

---

## Universal Access: The Same AI Everywhere

This is the key — Memory Fabric is **one AI identity** across every device, every app, everywhere you go.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       ONE AI. EVERYWHERE.                               │
│                                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │ AT HOME │  │ IN CAR  │  │AT OFFICE│  │ON PHONE │  │THIRD-   │      │
│  │         │  │         │  │         │  │         │  │PARTY APP│      │
│  │  "Hey   │  │  "Hey   │  │  "Hey   │  │  Tap    │  │  Fabric │      │
│  │ Fabric" │  │ Fabric" │  │ Fabric" │  │  icon   │  │ Protocol│      │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│       │            │            │            │            │            │
│       └────────────┴────────────┴────────────┴────────────┘            │
│                                 │                                       │
│                                 ▼                                       │
│                    ┌─────────────────────────┐                         │
│                    │      YOUR HOME HUB      │                         │
│                    │    (or nearest node)    │                         │
│                    │                         │                         │
│                    │   ┌─────────────────┐   │                         │
│                    │   │   YOUR .flx     │   │                         │
│                    │   │   ════════════  │   │                         │
│                    │   │ • 10 years of   │   │                         │
│                    │   │   memories      │   │                         │
│                    │   │ • Every prefer- │   │                         │
│                    │   │   ence learned  │   │                         │
│                    │   │ • Your style    │   │                         │
│                    │   │ • Your knowledge│   │                         │
│                    │   │ • Your skills   │   │                         │
│                    │   └─────────────────┘   │                         │
│                    └─────────────────────────┘                         │
│                                                                         │
│  RESULT: You never "meet a new AI"                                     │
│  Every interaction, everywhere = same AI that knows your entire life  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Scenario: Different Locations, Same AI

```
MORNING AT HOME: "Good morning! Coffee's ready. Traffic looks bad —
                  leave 10 minutes early for your 9am."

IN THE CAR:      "Continuing from home — I found 3 restaurants for
                  Sarah's birthday. Want me to read the reviews?"

AT OFFICE:       "Your 9am went well. I noticed you mentioned the
                  Q3 budget — want me to pull those numbers for 
                  the follow-up email?"

AT LUNCH:        *Phone* "The restaurant has your usual available.
                  Also, Sarah texted asking about tonight. Should I
                  suggest the birthday restaurant?"

DRIVING HOME:    "Picking up where we left off — Sarah said 7pm works.
                  I booked the table. Also grabbed that milk reminder."

AT HOME:         "Welcome back. Sarah's 5 minutes out. I set the table
                  lighting to 'romantic dinner' like you did last
                  anniversary. House is 72°. Music is your 'dinner'
                  playlist."
```

**That's not 6 different AIs — that's ONE AI following you through your day.**

---

## Vision

Memory Fabric is the hardware + software ecosystem that makes FLUX personal, portable, and private. One AI identity that knows you — running on your home hub, your laptop, your car, even a USB stick in your pocket.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEMORY FABRIC ECOSYSTEM                         │
│                                                                         │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐    │
│    │  HOME    │     │ PORTABLE │     │   CAR    │     │  OFFICE  │    │
│    │   HUB    │◄───►│   STICK  │◄───►│  SYSTEM  │◄───►│   NODE   │    │
│    │          │     │          │     │          │     │          │    │
│    │ 64GB RAM │     │  16GB    │     │  32GB    │     │ 128GB    │    │
│    │ RTX 4090 │     │ USB-C    │     │ In-dash  │     │ Multi-GPU│    │
│    └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘    │
│         │                │                │                │          │
│         └────────────────┴────────────────┴────────────────┘          │
│                                   │                                    │
│                          ┌────────▼────────┐                          │
│                          │   YOUR .flx     │                          │
│                          │   IDENTITY      │                          │
│                          │                 │                          │
│                          │ • Memories      │                          │
│                          │ • Knowledge     │                          │
│                          │ • Preferences   │                          │
│                          │ • Style         │                          │
│                          │ • Skills        │                          │
│                          └─────────────────┘                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Product Line

### 1. Memory Fabric Home Hub

**The brain of your home AI ecosystem.**

| Spec | Value |
|------|-------|
| **GPU** | RTX 4090 / 5090 (24GB VRAM) |
| **RAM** | 64GB DDR5 |
| **Storage** | 2TB NVMe (holds full FLUX + domains) |
| **Connectivity** | WiFi 6E, Bluetooth 5.3, Ethernet |
| **Form Factor** | Sleek living room appliance |

**Runs:**
- Full FLUX Lite Embedded (~15GB)
- All 6 models simultaneously
- Domain knowledge bases
- Real-time learning (updates .flx on the fly)

**Features:**
- Voice wake ("Hey Fabric")
- Connects to all home devices
- Syncs with portable stick
- Continuous learning from interactions

---

### 2. Memory Fabric Portable Stick

**Your AI in your pocket.**

| Spec | Value |
|------|-------|
| **Storage** | 256GB / 512GB / 1TB |
| **Interface** | USB-C / Thunderbolt 4 |
| **Compute** | Relies on host device |
| **Form Factor** | Elegant USB stick |

**Contains:**
- Your complete .flx file
- Episodic memories (everything it learned about you)
- Domain adapters (work, personal, creative modes)
- Encryption: AES-256 with biometric unlock

**Use Cases:**
- Plug into any PC → instant personal AI
- Plug into rental car → knows your preferences
- Plug into hotel TV → your entertainment AI
- Plug into friend's computer → your AI assists

---

### 3. Memory Fabric Car Module

**Your AI copilot.**

| Spec | Value |
|------|-------|
| **GPU** | Embedded (Jetson-class) |
| **RAM** | 32GB |
| **Storage** | 512GB |
| **Connectivity** | CAN bus, WiFi, LTE |
| **Form Factor** | In-dash / aftermarket |

**Runs:**
- FLUX Micro (~5GB, instruct + voice only)
- Voice interaction (Whisper + TTS)
- Calendar, navigation context
- Syncs when connected to home WiFi

**Features:**
- Natural conversation while driving
- Remembers where you parked
- Learns your routes, music, preferences
- Privacy: no cloud, all local

---

### 4. Memory Fabric Office Node

**Enterprise-grade personal AI.**

| Spec | Value |
|------|-------|
| **GPU** | Multi-GPU (A100 / H100) |
| **RAM** | 128GB+ |
| **Storage** | 8TB+ |
| **Form Factor** | Rack-mount / desktop tower |

**Runs:**
- Full FLUX with all domain models
- Medical, Legal, Finance adapters
- Multi-user (each user has their .flx)
- Enterprise knowledge bases

---

## FLUX Format Tiers

The .flx format scales across devices:

| Tier | Size | Models | Use Case |
|------|------|--------|----------|
| **FLUX Micro** | ~3GB | Instruct + Embedding | Portable stick, car, phone |
| **FLUX Lite** | ~8GB | + Vision + Coder | Laptop, tablet |
| **FLUX Standard** | ~15GB | All 6 models | Home hub |
| **FLUX Pro** | ~25GB | + Domain models | Office node |
| **FLUX Enterprise** | ~50GB+ | + Multi-agent | Data center |

### .flx Tier Structure

```python
# Shared core (all tiers)
flux_core = {
    'format': 'FLUX',
    'version': '6.0-fabric',
    
    # Your identity (always present)
    'identity': {
        'name': 'My AI',
        'voice_style': 'warm_professional',
        'preferences': {...},
        'created': '2026-03-30',
    },
    
    # Field + Memory (scales with tier)
    'field': {...},      # Micro: 24³, Lite: 48³, Standard: 96³
    'memory': {...},     # Your learned knowledge
    
    # Models (tier-dependent)
    'models': {
        'instruct': {...},   # All tiers
        'embedding': {...},  # All tiers
        'vision': {...},     # Lite+
        'coder': {...},      # Standard+
        'whisper': {...},    # Standard+
        'tts': {...},        # Standard+
    },
}
```

---

## Sync Protocol

### Continuous Sync

```
┌─────────────┐                           ┌─────────────┐
│  Home Hub   │                           │   Stick     │
│             │ ◄───── Δ memories ──────► │             │
│ Full .flx   │                           │ Full .flx   │
│  (master)   │ ◄───── Δ field ─────────► │   (copy)    │
│             │                           │             │
└─────────────┘                           └─────────────┘
        │
        │ WiFi sync when home
        ▼
┌─────────────┐
│    Car      │
│             │
│ Micro .flx  │
│  (subset)   │
└─────────────┘
```

### Sync Rules

1. **Home Hub is master** - Full model, continuous learning
2. **Portable syncs on connect** - Bi-directional memory merge
3. **Car syncs on WiFi** - Downloads new memories overnight
4. **Conflicts** - Timestamp-based merge, keep both if ambiguous

### Memory Merge Algorithm

```python
def merge_memories(master: dict, remote: dict) -> dict:
    """Merge episodic memories from two .flx files."""
    merged = {
        'vectors': [],
        'metadata': [],
    }
    
    # Deduplicate by content hash
    seen = set()
    
    for source in [master, remote]:
        for i, vec in enumerate(source.get('vectors', [])):
            content = source['metadata'][i]['content']
            hash_key = hash(content)
            
            if hash_key not in seen:
                seen.add(hash_key)
                merged['vectors'].append(vec)
                merged['metadata'].append(source['metadata'][i])
    
    # Sort by timestamp (most recent first)
    merged = sort_by_timestamp(merged)
    
    # Prune to memory limit
    return prune_to_limit(merged, max_entries=10000)
```

---

## Global Access: Cloud Relay

**Your hub runs locally. Access it from anywhere.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GLOBAL ACCESS ARCHITECTURE                       │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                         │
│   │  Phone   │    │   Car    │    │  Laptop  │                         │
│   │   App    │    │ Copilot  │    │   App    │                         │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                         │
│        │               │               │                                │
│        └───────────────┼───────────────┘                                │
│                        │                                                │
│                        ▼                                                │
│              ┌─────────────────┐                                        │
│              │   API Gateway   │  ◄── Cloud relay (Heroku/AWS)         │
│              │   $5-10/month   │      Encrypted tunnel only            │
│              └────────┬────────┘      No data stored                   │
│                       │                                                 │
│                       │ Secure tunnel to your home                      │
│                       ▼                                                 │
│              ┌─────────────────┐                                        │
│              │    HOME HUB     │  ◄── FLUX runs HERE                   │
│              │  (Your Device)  │      All inference local              │
│              │                 │      All memory local                 │
│              └─────────────────┘                                        │
│                                                                         │
│   ALTERNATIVE: Local Network (free, no relay)                          │
│   Phone ──► WiFi ──► Home Hub (direct connection)                      │
│                                                                         │
│   ALTERNATIVE: Offline Mode                                             │
│   Portable Stick ──► Runs FLUX on host device                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Home Hub runs 24/7** on your local network
2. **Cloud relay** is just a tunnel — no inference, no storage, no data
3. **Your phone** anywhere in the world → relay → your hub → response
4. **Car copilot** on the road → cellular → relay → your hub
5. **No hub available?** Portable stick runs locally on any device

### True Automation Across Your Life

```python
# Example: Task sent from your car while driving
request = {
    'source': 'car_copilot',
    'task': 'Research best Italian restaurants near the office for tomorrow lunch',
    'context': {
        'location': 'driving_home',
        'calendar': 'lunch_meeting_tomorrow_12pm',
    }
}

# → Sent via relay to Home Hub
# → FLUX processes with full memory (knows your food preferences, budget, past reviews)
# → Response ready when you get home
# → Or pushed to your phone immediately
```

---

## Fabric Protocol: Third-Party App Access

**The protocol that lets every app know who you are.**

### The Problem Today

| App | What It Knows | The Gap |
|-----|---------------|---------|
| ChatGPT | Your conversations with it | Doesn't know your schedule |
| Meal Planner | Your food preferences | Doesn't know your health conditions |
| Exercise App | Your workouts | Doesn't know your sleep schedule |
| Calendar | Your meetings | Doesn't know your energy levels |

**Result:** You re-explain yourself to every app. Every AI starts from zero.

### The Solution: Fabric Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FABRIC PROTOCOL                                  │
│                                                                         │
│   Third-Party Apps                          Your Memory Fabric          │
│   ┌──────────────┐                         ┌──────────────────┐        │
│   │ Meal Planner │ ──── REQUEST ────────► │                  │        │
│   │              │     "food_preferences"  │   YOUR HUB       │        │
│   │              │ ◄─── GRANT/DENY ─────── │                  │        │
│   └──────────────┘                         │  ┌────────────┐  │        │
│                                            │  │ Your .flx  │  │        │
│   ┌──────────────┐                         │  │            │  │        │
│   │ Fitness App  │ ──── REQUEST ────────► │  │ • Prefs    │  │        │
│   │              │     "health_data"       │  │ • Health   │  │        │
│   │              │ ◄─── GRANT/DENY ─────── │  │ • Schedule │  │        │
│   └──────────────┘                         │  │ • Style    │  │        │
│                                            │  │ • History  │  │        │
│   ┌──────────────┐                         │  └────────────┘  │        │
│   │ Any LLM App  │ ──── REQUEST ────────► │                  │        │
│   │              │     "user_context"      │                  │        │
│   │              │ ◄─── GRANT/DENY ─────── │                  │        │
│   └──────────────┘                         └──────────────────┘        │
│                                                                         │
│   YOU CONTROL:                                                          │
│   ✓ Which apps can access                                               │
│   ✓ What data categories they see                                       │
│   ✓ Read-only vs read-write                                             │
│   ✓ Revoke anytime                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Protocol Permissions

```python
# Fabric Protocol permission schema
permissions = {
    'meal_planner_app': {
        'granted': ['food_preferences', 'allergies', 'dietary_restrictions'],
        'denied': ['work_schedule', 'finances', 'health_diagnoses'],
        'access_level': 'read_only',
        'expires': '2027-03-30',
    },
    'fitness_app': {
        'granted': ['exercise_history', 'sleep_data', 'health_goals'],
        'denied': ['work_data', 'personal_relationships'],
        'access_level': 'read_write',  # Can add workout memories
        'expires': None,  # Until revoked
    },
    'chatgpt_wrapper': {
        'granted': ['writing_style', 'communication_preferences'],
        'denied': ['everything_else'],
        'access_level': 'read_only',
        'expires': 'session',  # Per-session only
    },
}
```

### API Endpoint (Local)

```python
# Third-party app requests user context
POST http://localhost:8080/fabric/v1/context
Authorization: Bearer <app_token>
Content-Type: application/json

{
    "app_id": "meal_planner_pro",
    "requested_scopes": ["food_preferences", "health_data"],
    "purpose": "Generate personalized meal plan"
}

# Response (if granted)
{
    "granted_scopes": ["food_preferences"],
    "denied_scopes": ["health_data"],
    "context": {
        "food_preferences": {
            "likes": ["thai", "mexican", "mediterranean"],
            "dislikes": ["oysters", "blue_cheese"],
            "allergies": ["shellfish"],
            "dietary": "pescatarian"
        }
    }
}
```

### What This Enables

| Scenario | Without Fabric | With Fabric |
|----------|---------------|-------------|
| Meal planning | "What foods do you like?" every time | Knows instantly, plans around your allergies |
| Fitness coaching | Generic workout advice | Knows your sleep, stress, schedule |
| Work assistant | Re-explain every project | Knows your codebase, deadlines, style |
| Travel booking | "Where have you been?" | Knows your preferences, past trips |
| Healthcare | Fill out forms every visit | Doctor app sees your full context |

**The AI that knows you across your entire life — because YOU let it.**

---

## Privacy Architecture

**Core Principle: Your AI never phones home.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRIVACY BY DESIGN                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    YOUR DATA STAYS LOCAL                        │   │
│  │                                                                 │   │
│  │  • All inference runs on device                                 │   │
│  │  • No cloud API calls                                           │   │
│  │  • No telemetry                                                 │   │
│  │  • No model weights sent anywhere                               │   │
│  │  • Encrypted storage (AES-256)                                  │   │
│  │  • Biometric unlock for portable stick                          │   │
│  │  • Secure enclave for sensitive memories                        │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Optional Cloud Features (user opt-in only):                           │
│  • Backup encrypted .flx to personal cloud                             │
│  • Sync between your own devices                                       │
│  • Never shared with third parties                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## User Experience

### First Run

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    Welcome to Memory Fabric                             │
│                                                                         │
│        ┌─────────────────────────────────────────────────┐             │
│        │                                                 │             │
│        │  "Hi! I'm your personal AI. I run entirely     │             │
│        │   on this device — no cloud, no tracking.      │             │
│        │                                                 │             │
│        │   Let's set up your profile. What should       │             │
│        │   I call you?"                                 │             │
│        │                                                 │             │
│        │   [    Your name: _______________    ]         │             │
│        │                                                 │             │
│        │   "Great! I'll learn your preferences as       │             │
│        │    we interact. Everything stays private."     │             │
│        │                                                 │             │
│        └─────────────────────────────────────────────────┘             │
│                                                                         │
│                         [ Get Started ]                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Daily Use

```
You: "What's on my calendar today?"

Fabric: "You have 3 meetings:
  • 9am Team standup
  • 2pm Client call with Acme Corp — I remember last 
    time they asked about the Q3 projections
  • 5pm Coffee with Sarah — she mentioned wanting to 
    try that new place on Main Street"

You: "Remind me about the Acme projections"

Fabric: "From your last call on March 15th:
  • They wanted 15% growth target
  • CFO was concerned about marketing spend
  • You promised updated deck by end of month
  
  Want me to pull up your notes?"
```

### Portable Mode

```
*Plugs stick into hotel room TV*

Fabric: "Welcome back. I see we're in Chicago — 
         different timezone, adjusting your 
         schedule display.
         
         Want me to find dinner options nearby? 
         I know you prefer Thai and avoid places 
         that are too loud for calls."
```

---

## Builder Mode

**For power users who want to customize their AI:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MEMORY FABRIC BUILDER                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  KNOWLEDGE      │  │    SKILLS       │  │   STYLE         │        │
│  │                 │  │                 │  │                 │        │
│  │ [+] Add domain  │  │ [+] Add model   │  │ Voice: Warm     │        │
│  │                 │  │                 │  │ Verbosity: Med  │        │
│  │ ☑ Medical      │  │ ☑ Coder        │  │ Formality: Low  │        │
│  │ ☐ Legal        │  │ ☐ Art Gen      │  │ Humor: High     │        │
│  │ ☑ Personal     │  │ ☑ Vision       │  │                 │        │
│  │                 │  │                 │  │                 │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MEMORY BROWSER                                                 │   │
│  │                                                                 │   │
│  │  Search: [________________________] [🔍]                        │   │
│  │                                                                 │   │
│  │  • Mar 30, 2026 — Learned: User prefers morning meetings       │   │
│  │  • Mar 29, 2026 — Remembered: Project deadline April 15        │   │
│  │  • Mar 28, 2026 — Noted: Allergic to shellfish                 │   │
│  │                                                                 │   │
│  │  [Edit] [Delete] [Export]                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Model Size: 12.3 GB          Memory: 2,847 entries                    │
│  Last Sync: 5 minutes ago     Stick Connected: ✓                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### Home Hub Software Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MEMORY FABRIC OS                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FLUX Runtime                                                   │   │
│  │  ├── flux_model.py (FLUXModel class)                           │   │
│  │  ├── flux_utils.py (utilities)                                 │   │
│  │  └── EmbeddedLazyModel (lazy loading)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Services                                                       │   │
│  │  ├── voice_service (Whisper + TTS)                             │   │
│  │  ├── sync_service (device sync)                                │   │
│  │  ├── learning_service (continuous field updates)              │   │
│  │  └── api_service (local REST + WebSocket)                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  UI                                                             │   │
│  │  ├── Web dashboard (localhost:8080)                            │   │
│  │  ├── Mobile app (local network)                                │   │
│  │  └── Voice interface (always listening, local wake word)       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Storage                                                        │   │
│  │  └── /fabric/                                                   │   │
│  │      ├── identity.flx (your personal AI)                       │   │
│  │      ├── domains/                                               │   │
│  │      │   ├── medical.lora                                      │   │
│  │      │   └── work.lora                                         │   │
│  │      └── backups/                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Portable Stick File Layout

```
/MEMORY_FABRIC/
├── identity.flx          # Your AI (8-15GB)
├── domains/
│   ├── work.lora         # Work adapter (~100MB)
│   └── personal.lora     # Personal adapter (~100MB)
├── config.json           # Device preferences
├── sync_log.json         # Last sync state
└── .encrypted            # Marker for encryption

Total: 16-20GB typical
```

---

## Roadmap

### Phase 1: Software (Q2 2026) ✓
- [x] FLUX Lite Embedded specification
- [x] Multi-model orchestration
- [ ] Tier system (Micro/Lite/Standard)
- [ ] Sync protocol implementation
- [ ] Builder UI

### Phase 2: Home Hub (Q3 2026)
- [ ] Hardware design finalization
- [ ] Custom Linux distro (Fabric OS)
- [ ] Voice wake word ("Hey Fabric")
- [ ] Mobile app for local control
- [ ] Beta program

### Phase 3: Portable Devices (Q4 2026)
- [ ] Portable Stick design
- [ ] Encryption + biometric
- [ ] Plug-and-play drivers
- [ ] Cross-platform support

### Phase 4: Automotive (Q1 2027)
- [ ] Car module hardware
- [ ] CAN bus integration
- [ ] Voice-first interface
- [ ] OEM partnerships

---

## Pricing Concept

### Hardware (One-Time Purchase)

| Product | Target Price | What You Get |
|---------|-------------|--------------|
| Portable Stick (256GB) | $199 | Your AI anywhere, plug into any device |
| Portable Stick (1TB) | $349 | Full model + domains |
| Home Hub (16GB VRAM) | $1,499 | Always-on local AI brain |
| Home Hub (24GB VRAM) | $2,499 | Full FLUX + all models |
| Car Module | $799 | Voice copilot, syncs with hub |
| Office Node | $4,999+ | Enterprise-grade, multi-user |

### Software & Services

| Service | Price | What You Get |
|---------|-------|--------------|
| FLUX Software | **Free** | Open source, runs on your hardware |
| Local-Only Mode | **$0/month** | Hub works on local network only |
| Cloud Relay | **$5-10/month** | Access hub from anywhere globally |
| Fabric Protocol | **Included** | Third-party apps can request access |
| Premium Support | $20/month | Priority support, early features |

### Business Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│  REVENUE STREAMS                                                        │
│                                                                         │
│  PRIMARY:                                                               │
│  ├── Hardware sales (Hub, Stick, Car Module)                           │
│  └── Cloud relay subscriptions ($5-10/month)                           │
│                                                                         │
│  SECONDARY:                                                             │
│  ├── Premium support subscriptions                                      │
│  ├── Enterprise licensing (Office Node)                                │
│  └── Domain adapter marketplace (community + paid adapters)            │
│                                                                         │
│  FREE:                                                                  │
│  ├── FLUX software (open source)                                       │
│  ├── Fabric Protocol SDK (open standard)                               │
│  └── Local-only mode (no subscription required)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Users pay for hardware + optional convenience (cloud relay). The AI itself is free and runs locally. This aligns incentives — we make money when you own more devices, not when we harvest your data.

---

## Why This Works

1. **Privacy is the product** — No cloud dependency, no data harvesting
2. **Portability is freedom** — Your AI goes where you go
3. **Local is fast** — No latency, no API limits
4. **Personal is powerful** — AI that actually knows you
5. **Open format** — .flx is documented, users own their data

---

## The Tagline

> **Memory Fabric: Your Personal AI Ecosystem. Runs Locally.**

---

## References

- [FLUX_LITE_EMBEDDED_MODELS.md](FLUX_LITE_EMBEDDED_MODELS.md) — Technical implementation
- [FLUX_FILE_FORMAT.md](FLUX_FILE_FORMAT.md) — .flx specification
- [PHASE_ORCHESTRATOR_SPEC.md](PHASE_ORCHESTRATOR_SPEC.md) — Multi-model orchestration
- [PHASE_FABRIC.md](PHASE_FABRIC.md) — Software fabric architecture
- [flux_model.py](../flux_model.py) — Runtime API
