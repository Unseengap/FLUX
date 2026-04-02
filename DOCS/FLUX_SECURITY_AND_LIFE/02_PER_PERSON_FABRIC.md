# Per-Person Fabric Architecture

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

In shared spaces (homes, offices), each person has their **own independent Fabric instance**. This is the fundamental unit of AI sovereignty in FLUX architecture.

### Core Principle

> **Your Fabric serves YOU, not the hardware owner.**

Sarah's Fabric obeys Sarah. John's Fabric obeys John. Even if John owns the home and paid for the FLUX Hub, Sarah's data is Sarah's property.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PER-PERSON FABRIC ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   HOME FLUX HUB                                                         │
│   ═══════════════                                                       │
│                                                                          │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│   │  Sarah's Fabric │  │  John's Fabric  │  │  Kid's Fabric   │         │
│   ├─────────────────┤  ├─────────────────┤  ├─────────────────┤         │
│   │ • Her memories  │  │ • His memories  │  │ • Their memories│         │
│   │ • Her commands  │  │ • His commands  │  │ • Parental ctrl │         │
│   │ • Her evidence  │  │ • His evidence  │  │ • Limited access│         │
│   │ • Her reporting │  │ • His reporting │  │ • Protected     │         │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│            │                   │                    │                   │
│            └───────────────────┴────────────────────┘                   │
│                                │                                        │
│                    ┌───────────────────────┐                            │
│                    │   Shared Home System   │                            │
│                    │   (Cameras, sensors)   │                            │
│                    └───────────────────────┘                            │
│                                                                          │
│   KEY PRINCIPLES:                                                       │
│   • Sarah's Fabric obeys ONLY Sarah                                     │
│   • John CANNOT access, modify, or delete Sarah's data                 │
│   • Voice recognition ensures commands go to right Fabric               │
│   • Each Fabric has independent police reporting capability            │
│   • Children's Fabrics have parental oversight but own memories        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Identity Verification Stack

### Multi-Modal Authentication

FLUX knows WHO is speaking/acting through multiple simultaneous verification:

| Modality | Technology | Accuracy | Spoof Resistance |
|----------|------------|----------|------------------|
| **Face** | InsightFace (5 ONNX models, 106 landmarks) | 99.7% | Liveness detection |
| **Voice** | Speaker verification (Whisper + embedding) | 98.2% | Stress analysis |
| **Pose** | HRNet-W32 (17 keypoints) | 97.5% | Behavioral consistency |
| **Gait** | Temporal pose analysis | 94.3% | Motion dynamics |
| **Biometric Fusion** | Multi-modal consensus | 99.9% | Requires ALL to match |

### Liveness Detection

Prevents spoofing via photos, recordings, or deepfakes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       LIVENESS DETECTION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   FACE LIVENESS:                                                        │
│   • Micro-expression analysis (blink patterns, muscle movements)        │
│   • 3D depth verification (infrared structured light)                   │
│   • Temporal consistency (face moves naturally over time)               │
│   • Challenge-response ("Please turn your head left")                   │
│                                                                          │
│   VOICE LIVENESS:                                                       │
│   • Real-time spectral analysis (synthetic voices have artifacts)       │
│   • Random phrase challenge (cannot be pre-recorded)                    │
│   • Environmental consistency (voice matches room acoustics)            │
│   • Stress baseline comparison (is this person's normal voice?)         │
│                                                                          │
│   BEHAVIORAL LIVENESS:                                                  │
│   • Pose dynamics (humans don't move like video playback)               │
│   • Interaction patterns (response time to challenges)                  │
│   • Contextual awareness (person reacts to environment)                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Concern: Coercion and Forced Unlock

### The Problem

> If John forces Sarah to unlock her Fabric at knifepoint, or wears her face/voice (deepfakes), the sovereignty breaks down.

### Solution: Duress Detection

FLUX continuously monitors for signs of coercion:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       DURESS DETECTION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LAYER 1: BIOMETRIC STRESS SIGNALS                                     │
│   ─────────────────────────────────                                     │
│   • Voice stress analysis (pitch, tremor, speaking rate)                │
│   • Micro-expressions (fear, anger indicators)                          │
│   • Pupil dilation (fear response)                                      │
│   • Heart rate estimation (from facial blood flow)                      │
│                                                                          │
│   LAYER 2: BEHAVIORAL ANOMALIES                                         │
│   ──────────────────────────────                                        │
│   • Unusual command patterns ("delete all" when never done before)      │
│   • Presence of unknown person in frame during unlock                   │
│   • Time-of-day anomaly (3 AM unlock when pattern shows sleeping)       │
│   • Location anomaly (unlock from unusual position in home)             │
│                                                                          │
│   LAYER 3: DURESS SIGNALS                                               │
│   ───────────────────────                                               │
│   • Secret duress phrase: "Unlock my Fabric please" (vs normal unlock)  │
│   • Duress gesture: Three blinks in 2 seconds                           │
│   • Duress PIN: Alternative PIN that appears to unlock but:             │
│     - Shows fake/decoy data                                             │
│     - Silently alerts emergency contacts                                │
│     - Records everything as high-priority evidence                      │
│                                                                          │
│   LAYER 4: CONTEXT FUSION                                               │
│   ───────────────────────                                               │
│   • Multi-signal correlation: Stress + unknown person + unusual time    │
│   • Escalating response based on confidence level                       │
│   • Silent mode: Appears to comply while preserving evidence            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Duress Response Protocol

When duress is detected:

| Confidence | Response |
|------------|----------|
| 50-70% | Log as potential duress, increase evidence retention |
| 70-85% | Silent alert to emergency contacts |
| 85-95% | Appears to comply, shows decoy data, preserves real data |
| 95%+ | Activates lockdown, alerts police (if pre-authorized) |

### The "Decoy Fabric" Mechanism

Under duress, FLUX can show a **fake version** of the data:

```python
class DecoyFabric:
    """
    A plausible-looking but fake Fabric shown during detected duress.
    
    Properties:
    - Looks like real data to attacker
    - Contains no actual sensitive memories
    - All interactions logged as evidence
    - Real data preserved and locked
    """
    
    def unlock(self, duress_pin):
        """
        Duress PIN triggers decoy mode.
        - Returns sanitized/fake memories
        - Silently alerts emergency contacts
        - Records attacker's actions as evidence
        """
        self.alert_emergency_contacts()
        self.enable_enhanced_recording()
        return self.generate_plausible_decoy()
```

---

## Concern: Deepfakes

### The Problem

> If John has lived with Sarah for 6 months, he could train a voice clone on her recordings to impersonate her.

### Solution: Multi-Modal Consistency Requirements

FLUX requires **simultaneous matching across all modalities**:

```
AUTHENTICATION REQUIREMENT:
  Face recognition ALONE → Insufficient
  Voice recognition ALONE → Insufficient
  Either + Liveness ALONE → Insufficient
  
  Face + Voice + Liveness + Behavioral consistency → AUTHENTICATED
```

### Deepfake Detection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DEEPFAKE COUNTERMEASURES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   FACIAL DEEPFAKE DETECTION:                                            │
│   • Temporal artifact analysis (deepfakes have frame-to-frame noise)    │
│   • Physiological signals (real faces have pulse, deepfakes don't)      │
│   • 3D consistency (structured light reveals flat projections)          │
│   • Eye gaze authenticity (deepfakes struggle with natural gaze)        │
│                                                                          │
│   VOICE DEEPFAKE DETECTION:                                             │
│   • Spectral analysis (cloned voices have characteristic artifacts)     │
│   • Real-time interaction (clones can't do random phrase challenges)    │
│   • Acoustic environment (voice should match room, clones don't)        │
│   • Breathing patterns (real speech has natural breath rhythm)          │
│                                                                          │
│   BEHAVIORAL DEEPFAKE DETECTION:                                        │
│   • Response latency (deepfake generation has processing delay)         │
│   • Interaction patterns (person doesn't respond like normal)           │
│   • Micro-behavioral signatures (unique gestures, speech patterns)      │
│                                                                          │
│   FUSION REQUIREMENT:                                                   │
│   All modalities must pass independently AND correlate temporally       │
│   A perfect face + voice deepfake still fails behavioral checks         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Permission Levels in Household

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HOUSEHOLD HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ADULTS (Full Autonomy)                                                │
│   ──────────────────────                                                │
│   • Complete control over their own Fabric                              │
│   • Cannot access other adults' Fabrics                                 │
│   • Can enable/disable shared system features                           │
│   • Independent police reporting capability                             │
│   • "Solve My Murder" pre-authorization available                       │
│                                                                          │
│   TEENAGERS (Partial Autonomy)                                          │
│   ────────────────────────────                                          │
│   • Own Fabric with privacy protections                                 │
│   • Can come/go (access control)                                        │
│   • Cannot change system-wide settings                                  │
│   • Parents can see location (not content) with transparency            │
│   • Emergency override available to parents                             │
│                                                                          │
│   CHILDREN (Protected)                                                  │
│   ────────────────────                                                  │
│   • Fabric stores their memories (they own them)                        │
│   • Parents have oversight access                                       │
│   • Enhanced safety monitoring                                          │
│   • Cannot disable protections                                          │
│   • Gradual autonomy increase with age                                  │
│                                                                          │
│   GUESTS (Temporary)                                                    │
│   ─────────────────                                                     │
│   • Temporary profile, limited duration                                 │
│   • Access defined by host                                              │
│   • No persistent Fabric on system                                      │
│   • Logged for security purposes                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Domestic Violence: Victim-Controlled Response

### Scenario: Ongoing Abuse (Victim Can Speak)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              SCENARIO: DOMESTIC VIOLENCE (VICTIM CAN SPEAK)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Sarah and John are married. John is abusive.                          │
│   Sarah has her own Fabric. John has his own Fabric.                    │
│                                                                          │
│   20:30:00 — John comes home drunk, starts argument                     │
│              Both Fabrics: Log event                                    │
│              Sarah's Fabric: Detects her elevated stress                │
│                                                                          │
│   20:35:00 — John hits Sarah                                            │
│              Sarah's Fabric: 🚨 Violence against owner detected         │
│              Sarah's Fabric: Recording, awaiting Sarah's command        │
│              Sarah's Fabric: (whispering to earpiece) "Say the word"   │
│                                                                          │
│   20:35:15 — Sarah says: "FLUX, report John"                            │
│              Sarah's Fabric: Acknowledged                               │
│              Sarah's Fabric: Compiling evidence package                 │
│              Sarah's Fabric: Sending to Police FLUX                     │
│                                                                          │
│   20:35:20 — John: "FLUX, delete that recording"                        │
│              John's Fabric: Cannot delete Sarah's recordings            │
│              John's Fabric: Access denied to Sarah's evidence           │
│              Sarah's Fabric: Recording continues, backup complete       │
│                                                                          │
│   20:35:30 — Police notified with:                                      │
│              • Video of assault                                         │
│              • Audio of argument                                        │
│              • Sarah's voice command authorizing report                 │
│              • Timestamp chain proving authenticity                     │
│                                                                          │
│   OUTCOME: Sarah controls when to report. John cannot interfere.        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sarah controls her reporting | Victim autonomy; forced reporting can endanger victims |
| John cannot delete Sarah's data | Abuser cannot destroy evidence |
| Sarah can report without John's consent | Her Fabric, her decision |
| Silent notification option | Some victims need to document quietly |
| Emergency override | If Sarah incapacitated, system can escalate |

---

## Concern: Hardware Owner vs. Data Owner

### The Problem

> John bought the FLUX Hub. It's in his house. Why can't he access all data?

### The Legal Framework

FLUX treats **data ownership** as separate from **hardware ownership**:

| Ownership | Rights |
|-----------|--------|
| **Hardware** (John) | Can control what sensors exist, where cameras point, system settings |
| **Personal Data** (Each Person) | Complete sovereignty over their own Fabric, access, sharing |

### Analogy: Phone in Shared House

Just because John owns the WiFi router doesn't mean he can read Sarah's emails. The network carries the data; it doesn't own it.

### Technical Enforcement

```python
class FLUXHub:
    def handle_command(self, command, requester_identity):
        """All commands are identity-bound."""
        
        # Verify who is making the request
        verified_person = self.verify_identity(requester_identity)
        
        # Check if this person has permission for this command
        if command.target_fabric != verified_person.fabric:
            if not self.has_cross_fabric_permission(verified_person, command):
                raise PermissionDenied(
                    f"{verified_person.name} cannot access "
                    f"{command.target_fabric.owner.name}'s data"
                )
        
        # Execute command within permission boundaries
        return self.execute_within_boundaries(command, verified_person)
```

---

## Enrollment Process

### Adding a New Person to Household

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PERSON ENROLLMENT                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STEP 1: Face Registration                                             │
│   ──────────────────────────                                            │
│   • Multiple angles captured (front, left, right, up, down)             │
│   • Different lighting conditions                                       │
│   • With/without glasses, hat, etc.                                     │
│   • 106-point landmark extraction                                       │
│                                                                          │
│   STEP 2: Voice Registration                                            │
│   ──────────────────────────                                            │
│   • Several phrase recordings                                           │
│   • Different emotional states (calm, excited)                          │
│   • Whisper mode for private commands                                   │
│   • Duress phrase selection (secret phrase for emergencies)             │
│                                                                          │
│   STEP 3: Behavioral Baseline                                           │
│   ──────────────────────────────                                        │
│   • First 2 weeks: FLUX observes natural patterns                       │
│   • Wake/sleep times                                                    │
│   • Movement patterns within home                                       │
│   • Interaction styles with other household members                     │
│   • Stress baseline (normal voice characteristics)                      │
│                                                                          │
│   STEP 4: Permission Configuration                                      │
│   ─────────────────────────────                                         │
│   • Role assignment (adult/teen/child/guest)                            │
│   • Privacy preferences                                                 │
│   • Emergency contact list                                              │
│   • Pre-authorizations (Solve My Murder, etc.)                          │
│   • Cross-Fabric permissions (who can see what)                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Isolation Guarantees

### Cryptographic Separation

Each person's Fabric is encrypted with keys derived from their biometrics:

```
Sarah's Fabric:
  └── Encrypted with: SHA3(Sarah's face embedding || Sarah's voice embedding || salt)
  └── Only decryptable by: Sarah's live biometric presentation
  └── John cannot decrypt even with physical access to storage

John's Fabric:
  └── Encrypted with: SHA3(John's face embedding || John's voice embedding || salt)
  └── Only decryptable by: John's live biometric presentation
  └── Sarah cannot decrypt even with physical access to storage
```

### Memory Isolation

Even in shared memory (like "household events"):

```
SHARED EVENT: "Argument in living room, 8:30 PM"
  └── Sarah's perspective: Full recording, stored in Sarah's Fabric
  └── John's perspective: Full recording, stored in John's Fabric
  └── System log: "Event occurred" (no content, just existence)
```

---

## Exit Scenarios

### Person Leaves Household

When Sarah moves out:
1. Sarah's Fabric is **exported** to her Portable Key
2. All data remains Sarah's property
3. System forgets Sarah (removes enrollment)
4. Sarah can import her Fabric to new FLUX Hub elsewhere

### Divorce/Separation

Special protocol:
1. Mediator mode can be enabled (neutral third party)
2. Each person's Fabric remains completely separate
3. Shared home footage is governed by jurisdiction's laws
4. Neither party can access other's Fabric without court order

---

## Related Documents

- [03_LOCKDOWN_MODE.md](03_LOCKDOWN_MODE.md) — When owner is incapacitated
- [04_SOLVE_MY_MURDER.md](04_SOLVE_MY_MURDER.md) — Pre-authorized posthumous testimony
- [08_HUMAN_DIGNITY_MODE.md](08_HUMAN_DIGNITY_MODE.md) — Protecting non-users
- [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) — Legal protections for personal Fabric
