# Lockdown Mode Specification

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

Lockdown Mode is the evidence preservation state that activates when FLUX detects a serious crime where the victim **cannot speak for themselves**. It transforms the system from "owned property" to "scene custodian."

### Core Principle

> **FLUX cannot be silenced. Not by the perpetrator. Not by the owner. Not even by destroying the hardware.**

---

## When Lockdown Activates

### Trigger Conditions (Any of These)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       LOCKDOWN TRIGGER CONDITIONS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   HIGH-CONFIDENCE TRIGGERS (Automatic):                                 │
│   ─────────────────────────────────────                                 │
│   • Owner killed (pose → fall + no vitals + no response for 60s)        │
│   • Owner unconscious after violence detected                           │
│   • Owner kidnapped (forced removal from home under duress)             │
│   • Lethal weapon discharged + owner down                               │
│                                                                          │
│   PRE-AUTHORIZED TRIGGERS ("Solve My Murder" enabled):                  │
│   ────────────────────────────────────────────────────                  │
│   • Owner enables feature knowing their situation                       │
│   • Can specify additional trigger conditions                           │
│   • Can designate specific suspects for enhanced monitoring             │
│                                                                          │
│   REQUIRED CONDITIONS FOR ALL:                                          │
│   ────────────────────────────                                          │
│   • Violence OR threat detected (not random medical emergency)          │
│   • Owner unable to respond/consent (unconscious, dead, taken)          │
│   • Confidence threshold met (prevents false triggers)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Concern: False Triggers

### The Problem

> If the AI misclassifies a consensual BDSM scene as "violence against owner," or a movie playing on TV as "gunshot," the system locks the owner out and transmits intimate data to law enforcement.

### Solution: Multi-Signal Verification

Lockdown requires **convergent evidence** from multiple independent detection systems:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FALSE TRIGGER PREVENTION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SINGLE SIGNAL → NOT SUFFICIENT FOR LOCKDOWN                           │
│   ═══════════════════════════════════════════                           │
│                                                                          │
│   Examples of non-triggers:                                             │
│   • Gunshot audio alone (could be TV/game/backfire)                     │
│   • Violent pose alone (could be exercise/play/consensual)              │
│   • Person down alone (could be sleeping/tired/yoga)                    │
│   • Scream alone (could be excitement/sports/movie)                     │
│                                                                          │
│   MULTIPLE CONVERGING SIGNALS → LOCKDOWN EVALUATION                     │
│   ═══════════════════════════════════════════════════                   │
│                                                                          │
│   Required convergence:                                                 │
│   • Violence detection (pose dynamics showing assault pattern)          │
│   • Audio corroboration (impact sounds, screaming, gunshot)            │
│   • Victim response (or lack thereof - no movement for 60s)             │
│   • Temporal sequence (weapon → shot → fall within 10s)                │
│   • Perpetrator behavior (fleeing, cleaning, hiding)                   │
│                                                                          │
│   CONTEXT AWARENESS                                                     │
│   ─────────────────                                                     │
│   • Is TV/stereo playing? (audio source localization)                   │
│   • Is this a known activity pattern? (couples wrestling → baseline)   │
│   • Time of day matches normal behavior?                                │
│   • Other household members behaving normally?                          │
│                                                                          │
│   CHALLENGE BEFORE LOCKDOWN                                             │
│   ─────────────────────────                                             │
│   • FLUX: "Are you okay? Please respond verbally."                      │
│   • Wait 30-60 seconds for response                                     │
│   • No response + violence indicators → Lockdown                       │
│   • Normal response → Stand down, log event                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Confidence Thresholds

| Scenario | Confidence Required | Challenge? |
|----------|---------------------|------------|
| Movie/TV violence | N/A (audio source detection) | No |
| Argument (no physical contact) | N/A | No |
| Physical contact (consensual baseline) | N/A | No |
| Physical contact (outside baseline) | 70% | Yes |
| Weapon visible + aggressive posture | 85% | Yes |
| Weapon discharged + person down | 95% | 30s wait |
| Gunshot + person down + no response | 98% | Lockdown after 60s |

### Example: The False Positive Avoided

```
SCENARIO: Couple watching action movie

21:30:00 — Loud gunshot audio detected
           FLUX: Audio source → Living room TV (not room)
           FLUX: Both residents on couch, relaxed poses
           FLUX: Heart rates normal (facial estimation)
           Result: NO TRIGGER (audio from media source)

21:45:00 — Couple play-wrestling (part of their baseline)
           FLUX: Physical contact detected
           FLUX: Checking against behavioral baseline...
           FLUX: Pattern matches "couple intimacy" profile
           FLUX: Both participants showing positive engagement
           Result: NO TRIGGER (baseline activity)

22:00:00 — Movie has dramatic scream
           FLUX: Scream audio detected
           FLUX: Audio source → Living room TV
           FLUX: Residents laughing afterwards
           Result: NO TRIGGER (media + positive response)
```

### Example: The True Positive Caught

```
SCENARIO: Actual home invasion

03:15:00 — Glass break detected
           FLUX: Audio source → Kitchen window (not TV)
           FLUX: Time anomaly (no one usually up at 3 AM)
           FLUX: Alert mode activated, not lockdown yet

03:15:30 — Unknown person enters kitchen
           FLUX: Face → No match (not household member)
           FLUX: Object detection → Knife in hand
           FLUX: Elevated alert, owner notified

03:16:00 — Confrontation in hallway
           FLUX: Owner woken, confronts intruder
           FLUX: Audio: shouting, threats
           FLUX: Pose: defensive posture (owner), aggressive (intruder)

03:16:15 — Physical attack
           FLUX: Violence detected (stabbing motion)
           FLUX: Owner: Impact detected → falls
           FLUX: Audio: Scream, then silence
           
03:16:20 — Status check
           FLUX: "Owner, are you okay? Please respond."
           FLUX: No response...
           
03:16:50 — Still no response
           FLUX: Owner pose → prone, not moving
           FLUX: Breathing check → inconclusive (too dark)
           
03:17:00 — Intruder searching house
           FLUX: Perpetrator behavior (searching, not helping)
           
03:17:20 — LOCKDOWN ACTIVATED
           Confidence: 97%
           Signals: Unknown intruder + weapon + attack + owner down + no response + fleeing behavior
           Action: Evidence package transmitted to Police FLUX
```

---

## What Happens in Lockdown

### Immediate Actions (Within 5 Seconds)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOCKDOWN SEQUENCE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   T+0.0s  LOCKDOWN TRIGGERED                                            │
│           ├── All recordings marked as locked evidence                  │
│           ├── Owner credentials suspended                               │
│           └── Evidence encryption keys rotated                          │
│                                                                          │
│   T+0.5s  EVIDENCE BACKUP INITIATED                                     │
│           ├── Critical window (T-5min to T+0) prioritized               │
│           ├── Encrypted transmission to Police FLUX begins             │
│           └── Local backup to tamper-evident storage                    │
│                                                                          │
│   T+2.0s  EVIDENCE BACKUP COMPLETE                                      │
│           ├── All relevant footage now exists in 3 locations:          │
│           │   • Local (encrypted, owner locked out)                     │
│           │   • Police FLUX (chain of custody begins)                   │
│           │   • Cloud escrow (if configured)                            │
│           └── Destruction of local device now irrelevant               │
│                                                                          │
│   T+5.0s  CONTINUED MONITORING                                          │
│           ├── System continues recording                                │
│           ├── Perpetrator actions logged (cleaning, fleeing)           │
│           ├── All new data transmitted to Police FLUX                   │
│           └── External monitoring activated (if neighbor FLUX exists)   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Owner Lockout

During lockdown, the owner (or perpetrator if owner is perpetrator) **cannot**:

| Action | Status | Reason |
|--------|--------|--------|
| Delete recordings | **BLOCKED** | Evidence preservation |
| Modify timeline | **BLOCKED** | Chain of custody integrity |
| Access evidence | **BLOCKED** | Prevent spoliation |
| Factory reset | **BLOCKED** | Hardware safeguard |
| Power off | **MITIGATED** | Battery backup + already transmitted |
| Destroy device | **MITIGATED** | Evidence already at Police FLUX |

### What Owner CAN Do

- Use other devices normally (lockdown doesn't affect phones, etc.)
- Speak to FLUX (responses are recorded)
- Exit the premises (tracked if they leave)
- Call for help (encouraged, logged)

---

## Tamper Response Escalation

Every tamper attempt becomes **additional evidence**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TAMPER RESPONSE ESCALATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ATTEMPT                        FLUX RESPONSE                          │
│   ═══════════════════════════════════════════════════════════════════   │
│                                                                          │
│   "Delete all recordings"                                               │
│   → Access denied. System in lockdown.                                  │
│   → Deletion attempt logged as evidence                                 │
│   → Timestamp: perpetrator tried to destroy evidence at T+3min         │
│                                                                          │
│   Unplugs FLUX hub                                                      │
│   → Evidence already backed up (completed at T+2s)                     │
│   → Battery keeps logging for 4+ hours                                  │
│   → Police FLUX notified of power loss                                  │
│   → Power disconnection logged as tampering evidence                    │
│                                                                          │
│   Physically destroys FLUX hub                                          │
│   → Evidence was backed up in <5 seconds                               │
│   → Destruction observed by neighbor FLUX (if exists)                  │
│   → Physical destruction logged as cover-up attempt                     │
│   → Additional evidence of consciousness of guilt                       │
│                                                                          │
│   Takes hub and drives away                                             │
│   → Evidence already transmitted                                        │
│   → Hub GPS tracked (if equipped)                                       │
│   → Movement logged as evidence of tampering                            │
│   → Neighbor FLUX captures vehicle departure                            │
│                                                                          │
│   Factory reset attempt                                                 │
│   → Lockdown mode prevents factory reset                               │
│   → Reset attempt counter incremented                                   │
│   → Biometric unlock required (victim's, not perpetrator's)            │
│   → Multiple failed attempts = additional evidence                      │
│                                                                          │
│   KEY INSIGHT: Every tampering attempt becomes ADDITIONAL EVIDENCE      │
│               demonstrating consciousness of guilt                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Lockdown Release Conditions

### Who Can Release

| Authorized Release | Requirements |
|-------------------|--------------|
| **Police FLUX** | Investigation formally closed |
| **Victim (if survives)** | Physical + biometric verification at police station |
| **Next-of-kin (if deceased)** | Death certificate + police authorization + identity verification |
| **Court order** | Judicial determination |

### Minimum Lockdown Period

- **72 hours minimum** — prevents premature tampering
- **Extended** if investigation ongoing
- **Permanent** if case goes to trial (evidence preserved until verdict)

### Release Process

```
1. Authorized party requests release
2. Police FLUX verifies authorization
3. Evidence package transferred to court records (if applicable)
4. Local FLUX receives unlock command with cryptographic signature
5. Owner credentials restored gradually
6. System returns to normal operation
7. Lockdown log preserved permanently
```

---

## Concern: What if Owner IS the Victim AND the Trigger is Wrong?

### The Scenario

Sarah falls and hits her head. She's unconscious but not from violence. FLUX triggers lockdown and sends her intimate footage to police.

### Safeguards

1. **Challenge System**: FLUX asks "Are you okay?" before lockdown
2. **Violence Requirement**: Pure medical emergency without violence detected → Emergency services called, NOT lockdown
3. **Context Analysis**: Fall without preceding aggression = medical, not crime
4. **Quick Release**: If Sarah wakes up and says "I'm fine, I fell," FLUX verifies identity and can release if no actual crime occurred

### The Medical vs. Violence Distinction

```
MEDICAL EMERGENCY (No Lockdown):
  • Person collapses
  • No violence detected beforehand
  • No weapon
  • No intruder
  • No threat audio
  → Call emergency services, grant access to EMTs, NO lockdown

VIOLENCE (Lockdown):
  • Person collapses
  • Preceded by: assault, weapon, intruder, threats
  • Evidence of external harm
  → Lockdown activated
```

---

## Fifth Amendment Considerations

**Critical**: Lockdown ONLY affects **victim's FLUX** and **shared spaces**.

If John kills Sarah:
- **Sarah's FLUX**: Locks John out (not his property)
- **John's FLUX**: John retains full access (his Fifth Amendment right)

John cannot be compelled to unlock his own FLUX. But Sarah's FLUX has independent evidence that John cannot suppress.

See [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) for detailed legal analysis.

---

## Technical Implementation

### Hardware Requirements for Tamper Resistance

| Component | Purpose |
|-----------|---------|
| TPM 2.0 | Secure key storage, tamper detection |
| Battery | 4+ hours operation after power loss |
| Cellular | Backup transmission if WiFi cut |
| Tamper switch | Detects physical enclosure breach |
| GPS | Optional location tracking if device moved |

### Cryptographic Architecture

```
Evidence Encryption:
  └── Encrypted with: Police FLUX public key + Escrow public key
  └── Owner private key REMOVED from local storage during lockdown
  └── Decryption requires: Police FLUX private key OR Court order

Transmission Security:
  └── TLS 1.3 to Police FLUX
  └── Certificate pinning (prevents MITM)
  └── Signed by local HSM (proves origin)
  └── Timestamped by Police FLUX (proves receipt time)
```

---

## Evidence Package Format

What Police FLUX receives:

```json
{
  "lockdown_id": "FLX-2026-04-02-031720-001",
  "source_device": "Sarah Chen Home FLUX",
  "trigger_time": "2026-04-02T03:17:20Z",
  "trigger_reason": "Owner incapacitated + violence + no response",
  "trigger_confidence": 0.97,
  
  "evidence": {
    "video": [
      {"file": "incident_03-15-00_to_03-18-00.mp4", "size_mb": 245}
    ],
    "audio": [
      {"file": "enhanced_audio.wav", "size_mb": 12}
    ],
    "detections": {
      "intruder_face": {"embedding": "...", "confidence": 0.94},
      "weapon": {"type": "knife", "confidence": 0.91},
      "victim_status": {"state": "prone", "response": "none"}
    }
  },
  
  "causal_chain": [
    {"event": "glass_break", "time": "03:15:00", "confidence": 0.99},
    {"event": "intruder_entry", "time": "03:15:30", "confidence": 0.96},
    {"event": "confrontation", "time": "03:16:00", "confidence": 0.94},
    {"event": "attack", "time": "03:16:15", "confidence": 0.97},
    {"event": "victim_down", "time": "03:16:18", "confidence": 0.98},
    {"event": "no_response", "time": "03:17:20", "confidence": 1.0}
  ],
  
  "authorization": "owner_pre_authorized_solve_my_murder",
  "authorization_date": "2025-11-15T10:30:00Z",
  
  "cryptographic_seal": {
    "hash": "sha256:a7f3b2c1d4e5f6...",
    "signature": "RSA-4096:...",
    "timestamp_authority": "Police FLUX Timestamp Service"
  }
}
```

---

## Related Documents

- [04_SOLVE_MY_MURDER.md](04_SOLVE_MY_MURDER.md) — Pre-authorization feature
- [08_HUMAN_DIGNITY_MODE.md](08_HUMAN_DIGNITY_MODE.md) — Non-user victim protection
- [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) — Constitutional considerations
- [11_EVIDENCE_TESTIMONY.md](11_EVIDENCE_TESTIMONY.md) — Court admissibility
