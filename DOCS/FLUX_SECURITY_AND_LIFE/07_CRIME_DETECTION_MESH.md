# Crime Detection Mesh Specification

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

The Crime Detection Mesh is an **opt-in distributed forensic synchronization network** that allows FLUX cameras to coordinate during active incidents, creating a real-time witness network.

### Core Concept

When a crime is detected, nearby FLUX cameras are alerted to **watch for the suspect** and **timestamp sightings**. This creates a synchronized evidence chain across multiple independent nodes.

---

## How It Works

### Step 1: Trigger Event

Crime committed at Location A:
- Captured by Victim's FLUX or Witness FLUX
- Police FLUX generates "BOLO package":
  - Face embedding
  - Vehicle description
  - Clothing
  - Direction of travel
  - Temporal window

### Step 2: Opt-In Broadcast

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUX CRIME ALERT                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   FLUX CRIME ALERT (Opt-In Recipients Only)                             │
│   ═══════════════════════════════════════════                           │
│   Type: ARMED ROBBERY                                                   │
│   Location: 0.3 miles from you (Oak & 5th)                              │
│   Time: 14:32:15 (3 minutes ago)                                        │
│   Suspect: Male, 5'10", black hoodie, red backpack                      │
│   Vehicle: Blue Honda Civic, partial plate "7X3"                        │
│   Direction: Heading east on Pine Street                                │
│                                                                          │
│   YOUR FLUX CAMERA IS REQUESTED TO:                                     │
│   → Monitor for suspect description                                     │
│   → Timestamp any sightings                                             │
│   → Preserve 10-minute window if detected                               │
│   → Auto-submit if match confidence >85%                                │
│                                                                          │
│   [OPT-IN] [IGNORE] [VIEW MAP]                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 3: Synchronized Monitoring

Each participating FLUX node:
1. Receives BOLO package with suspect embeddings
2. Runs local detection against camera feeds
3. Timestamps any matches
4. Preserves relevant footage
5. Submits according to opt-in level

### Step 4: Timestamp Fusion

When multiple cameras catch the suspect:

```
Camera 1 (Grocery store): 14:35:22 - Suspect enters
Camera 2 (Gas station 2 blocks east): 14:37:15 - Suspect buys cigarettes
Camera 3 (Residential doorbell): 14:38:45 - Suspect passes on foot

FLUX Fusion creates:
├── Precise travel speed (1.2 mph = on foot, not driving)
├── Direction consistency (eastbound)
├── Behavior pattern (avoiding main streets, sticking to alleys)
└── Chain of custody: Each timestamp cryptographically signed
```

---

## The "Witness Mesh" Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WITNESS MESH ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   CRIME SCENE (Location A)                                              │
│           ↓                                                              │
│   Police FLUX creates Evidence Package                                  │
│           ↓                                                              │
│   FLUX Protocol broadcasts to OPT-IN nodes within radius                │
│           ↓                                                              │
│   ┌─────────────────┬─────────────────┬─────────────────┐               │
│   │  FLUX Home #1   │  FLUX Home #2   │  FLUX Business  │               │
│   │  (0.2 mi away)  │  (0.5 mi away)  │  (0.3 mi away)  │               │
│   ├─────────────────┼─────────────────┼─────────────────┤               │
│   │ Face detection  │ No match        │ Vehicle spotted │               │
│   │ active for BOLO │ (obstructed     │ at 14:36:12     │               │
│   │                 │  view)          │                 │               │
│   └────────┬────────┴─────────────────┴────────┬────────┘               │
│            ↓                                    ↓                        │
│       Timestamp: 14:35:45                 Timestamp: 14:36:12           │
│       Face match: 94%                     Plate partial: "7X3"          │
│       Direction: East                     Direction: East               │
│            ↓                                    ↓                        │
│            └──────────────┬─────────────────────┘                       │
│                           ↓                                              │
│                 Police FLUX Fusion Engine                               │
│                           ↓                                              │
│            Travel time between points: 27 seconds                       │
│            Distance: 0.2 miles                                          │
│            Speed: 26 mph (running? vehicle?)                            │
│            Next predicted location: Pine & 8th                          │
│            Deploy units to intercept                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Principles

### What Is Shared

| Data | Purpose | Privacy Impact |
|------|---------|----------------|
| BOLO embeddings | Enable local matching | Suspect-specific only |
| Sighting timestamps | Build timeline | Time + location of suspect |
| Vehicle confirmations | Track movement | License plate if captured |
| Direction of travel | Predict route | General direction |

### What Is NOT Shared

| Data | Why Not |
|------|---------|
| Historical footage | Only from alert time forward |
| Non-suspect individuals | Face blurring for bystanders |
| Audio | Unless specifically voice-ID needed |
| Interior footage | Opt-in can restrict to exterior only |
| Submitter identity | Can be anonymous to police |

### Auto-Expiry

| Condition | Expiry |
|-----------|--------|
| Suspect apprehended | Alert expires immediately |
| 24 hours elapsed | Alert expires automatically |
| Participating cameras | Delete BOLO data after expiry |
| Evidence submissions | Retained by Police FLUX only |

---

## Opt-In Tiers

### Tier 1: Passive Opt-In (Default for Participants)

```
Your FLUX receives alerts
│
└── If suspect passes your camera:
    └── You get notification: "Suspect detected, submit evidence?"
        └── You manually approve or decline submission
```

- **Control**: Maximum user control
- **Delay**: ~30 seconds for human approval
- **Privacy**: You see everything before it goes anywhere

### Tier 2: Active Opt-In (Pre-Authorized)

```
Your FLUX auto-detects and auto-submits
│
└── If confidence >90%:
    └── Evidence submitted automatically
        └── You receive notification of submission
            └── You can revoke within 1 hour
```

- **Control**: You set the threshold
- **Delay**: Near real-time submission
- **Privacy**: You can review and revoke after the fact

### Tier 3: Neighborhood Watch Mode

```
Your FLUX actively scans for all alerts
│
└── Immediate submission for high-confidence matches
    └── No human approval delay
        └── Maximum contribution to community safety
```

- **Control**: Pre-authorized maximum participation
- **Delay**: Zero (within seconds)
- **Privacy**: Trade-off for community benefit
- **Requirement**: Established FLUX node (6+ months) for eligibility

---

## The Timestamp Evidence Value

### Why Synchronized Timestamps Matter

In court, synchronized timestamps from multiple independent FLUX nodes creates **irrefutable timeline reconstruction**:

```
"Your Honor, the defense claims my client was at work. But:

FLUX Camera #472 (bodega, 14:35:22) captured defendant's face
FLUX Camera #891 (residential, 14:36:45) captured defendant walking

Both timestamps are cryptographically signed by their respective 
hardware security modules. The travel time between points is 73 seconds, 
consistent with walking speed. 

The defense's alibi is physically impossible."
```

### Evidence Strength Comparison

| Evidence Type | Strength | Weakness |
|---------------|----------|----------|
| Single camera footage | Medium | Could be doctored, timing uncertain |
| Multiple unsynced cameras | Medium-High | Timing disputes possible |
| FLUX Mesh timestamps | Very High | Cryptographic proof, multi-source |
| FLUX Mesh + GPS | Maximum | Spatiotemporal proof beyond dispute |

---

## The "Fast Watch" Concept

Traditional investigation:
```
Crime happens → Days pass → Detective reviews footage → 
Manually finds timestamps → Maybe finds something → Months to trial
```

FLUX Mesh investigation:
```
Crime happens at 14:32:00
    ↓ +1 minute
Nearby FLUX cameras alerted
    ↓ +3 minutes  
3 cameras have logged sightings
    ↓ +4 minutes
Police FLUX has reconstructed path
    ↓ +8 minutes
Suspect surrounded
    ↓ +15 minutes
Arrest with complete evidence chain
```

---

## City-Wide Manhunt: Full Scenario

### From Crime to Capture: 23 Minutes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FULL TIMELINE: MURDER TO ARREST                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   14:32:12 — CRIME: John shoots Sarah                                   │
│              Home FLUX documents everything                             │
│                                                                          │
│   14:32:16 — HOME FLUX: Sends evidence to Police FLUX                   │
│              • Suspect ID: John Chen                                    │
│              • Vehicle: Silver Toyota Camry ABC-1234                    │
│              • Direction: East from 123 Oak Street                      │
│                                                                          │
│   14:32:18 — John flees in vehicle                                      │
│                                                                          │
│   14:32:30 — POLICE FLUX: Receives, verifies, dispatches                │
│              • 3 units en route to scene                                │
│              • BOLO issued for vehicle                                  │
│              • City FLUX Network activated                              │
│                                                                          │
│   14:33:12 — CITY CAMERA #47: Vehicle spotted                           │
│              • Location: Oak St, heading south on Main                  │
│              • Speed: 45mph in 25mph zone (flagged)                     │
│              • Plate confirmed: ABC-1234                                │
│                                                                          │
│   14:35:47 — CITY CAMERA #89: Vehicle continues south                   │
│              • Turned onto Highway 7                                    │
│              • Prediction: Heading toward I-95                          │
│                                                                          │
│   14:37:22 — HIGHWAY CAMERA: I-95 on-ramp                               │
│              • Vehicle entered I-95 South                               │
│              • Traveling at 75mph                                       │
│                                                                          │
│   14:37:30 — POLICE FLUX: Route prediction                              │
│              • Likely destinations: Airport (45 min), Border (2 hr)     │
│              • Intercept points identified                              │
│              • 2 units repositioning                                    │
│                                                                          │
│   14:41:55 — GAS STATION CAMERA: Rest stop                              │
│              • Vehicle stopped for gas                                  │
│              • Driver face confirmed: John Chen                         │
│              • Currently pumping gas                                    │
│                                                                          │
│   14:42:30 — POLICE: Units converging                                   │
│              • 3 units from different directions                        │
│              • ETA: 4 minutes                                           │
│                                                                          │
│   14:45:00 — VEHICLE BLOCKED                                            │
│              • John returns to car                                      │
│              • Sees police vehicles                                     │
│              • Attempts to flee on foot                                 │
│                                                                          │
│   14:45:30 — FOOT PURSUIT                                               │
│              • Duration: 2 minutes                                      │
│              • John apprehended in parking lot                          │
│                                                                          │
│   14:47:30 — ARREST                                                     │
│              • John Chen in custody                                     │
│              • Weapon recovered from vehicle                            │
│              • Evidence chain complete                                  │
│                                                                          │
│   ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│   TOTAL TIME: 15 minutes 18 seconds (crime to custody)                  │
│                                                                          │
│   TRADITIONAL TIMELINE: 48+ hours to identify suspect                   │
│                         Weeks/months for arrest                         │
│                         Often: Cold case (never solved)                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Legal Safeguards

### Chain of Custody

Every piece of evidence has cryptographic provenance:

```json
{
  "evidence_id": "FLX-EVD-2026-04-02-143312-001",
  "source_camera": "FLUX Home #472",
  "capture_time": "2026-04-02T14:35:22Z",
  "capture_timestamp_authority": "RFC 3161",
  
  "integrity": {
    "hash": "sha256:a7f3b2c1d4e5f6...",
    "signed_by": "TPM-472-HARDWARE-KEY",
    "signature": "RSA-4096:..."
  },
  
  "submission": {
    "submitted_to": "Police FLUX Central",
    "submitted_at": "2026-04-02T14:35:45Z",
    "authorization": "user_opted_in_tier_2"
  },
  
  "chain": [
    {"event": "captured", "time": "14:35:22", "actor": "FLUX-472"},
    {"event": "processed", "time": "14:35:23", "actor": "FLUX-472"},
    {"event": "submitted", "time": "14:35:45", "actor": "FLUX-472"},
    {"event": "received", "time": "14:35:46", "actor": "Police-Central"},
    {"event": "verified", "time": "14:35:47", "actor": "Police-Central"}
  ]
}
```

### Warrant Requirements

| Scenario | Warrant Needed? |
|----------|-----------------|
| Citizen voluntarily submits | No |
| Police queries opt-in network | No (consent-based) |
| Police wants specific camera | Yes (unless owner consents) |
| Cross-jurisdiction tracking | Reciprocal warrant |
| Historical footage request | Yes |

---

## Concern: Pre-Crime Surveillance Infrastructure

### The Problem

> You're describing the ability to track any vehicle through a city in real-time. Today this requires police resources and warrants. Your system automates it.

### The Response

**Architectural limits prevent misuse:**

1. **No mass surveillance queries**
   - Cannot query "show me all vehicles"
   - Must specify: incident type + location + time window
   - "Fishing expeditions" blocked at protocol level

2. **Active incident only**
   - Mesh only activates during declared emergencies
   - Cannot track someone "just because"
   - Auto-expires when incident closes

3. **Audit everything**
   - Every query logged with officer ID, justification
   - Logs transmitted to independent oversight
   - Abuse is detectable and prosecutable

4. **Citizen opt-in**
   - Private FLUX cameras only participate voluntarily
   - Government cameras still require warrants for historical access
   - The mesh is citizen-controlled, not state-controlled

### The Alternative Is Worse

Traditional surveillance:
- Centralized in government hands
- Opaque (FOIA required to see anything)
- No citizen control
- No expiry

FLUX Mesh:
- Distributed across citizen nodes
- Transparent (open format, auditable)
- Citizen-controlled participation
- Auto-expiry built in

---

## Technical Implementation

### BOLO Package Format

```json
{
  "bolo_id": "FLX-BOLO-2026-04-02-143216",
  "crime_type": "MURDER",
  "urgency": "IMMEDIATE",
  "generated_by": "Police FLUX Central",
  "timestamp": "2026-04-02T14:32:16Z",
  
  "suspect": {
    "face_embedding": "base64_encoded_512d_vector",
    "face_images": ["multiple_angles.jpg"],
    "height_cm": 178,
    "build": "average",
    "clothing": "dark jacket, jeans",
    "distinctive": "red backpack"
  },
  
  "vehicle": {
    "type": "sedan",
    "make": "Toyota",
    "model": "Camry",
    "color": "silver",
    "plate": "ABC-1234",
    "direction_last_seen": "east"
  },
  
  "search_parameters": {
    "origin": {"lat": 40.7128, "lng": -74.0060},
    "radius_m": 5000,
    "time_window_start": "2026-04-02T14:32:00Z",
    "time_window_end": "2026-04-02T15:32:00Z"  
  },
  
  "match_thresholds": {
    "face": 0.85,
    "vehicle": 0.80,
    "plate": 0.95
  }
}
```

### Sighting Report Format

```json
{
  "sighting_id": "FLX-SIGHT-472-2026-04-02-143522",
  "bolo_id": "FLX-BOLO-2026-04-02-143216",
  "source_camera": "FLUX-HOME-472",
  "timestamp": "2026-04-02T14:35:22Z",
  
  "detections": {
    "face_match": {
      "confidence": 0.94,
      "embedding_distance": 0.12
    },
    "vehicle_match": null,
    "plate_match": null
  },
  
  "location": {
    "lat": 40.7145,
    "lng": -74.0032,
    "address_approximate": "Oak St & 5th Ave"
  },
  
  "direction": "east",
  "speed_estimate": "walking",
  
  "evidence": {
    "video_segment": "encrypted_bytes...",
    "duration_s": 8
  },
  
  "cryptographic_seal": {
    "hash": "sha256:...",
    "signature": "RSA-4096:...",
    "timestamp_authority": "RFC 3161"
  }
}
```

---

## Related Documents

- [06_FLUX_AMBER_ALERTS.md](06_FLUX_AMBER_ALERTS.md) — Emergency alert system
- [01_NETWORK_ARCHITECTURE.md](01_NETWORK_ARCHITECTURE.md) — Three-tier network model
- [13_POLICE_INTEGRATION.md](13_POLICE_INTEGRATION.md) — Law enforcement coordination
- [09_PRIVACY_OPT_IN_MODEL.md](09_PRIVACY_OPT_IN_MODEL.md) — Complete privacy specification
