# Solve My Murder Feature Specification

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

"Solve My Murder" is a pre-authorization feature that allows users to consent **before** a potential crime occurs to have their FLUX testify on their behalf if they are killed or incapacitated.

### Core Principle

> **If you're harmed and cannot speak, your AI speaks for you.**

This is not just about the moment of violence—it's about the **longitudinal pattern** of behavior that led to it.

---

## What Pre-Authorization Includes

When a user enables "Solve My Murder," they authorize:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOLVE MY MURDER AUTHORIZATION                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   IMMEDIATE EVIDENCE (Incident)                                         │
│   ─────────────────────────────                                         │
│   ✓ Video/audio of the attack                                          │
│   ✓ Perpetrator identification (face, voice, body)                     │
│   ✓ Weapon detection                                                    │
│   ✓ Causal sequence reconstruction                                      │
│   ✓ Cryptographic timestamps                                            │
│                                                                          │
│   BEHAVIORAL HISTORY (Context)                                          │
│   ─────────────────────────────                                         │
│   ✓ Pattern of relationship dynamics                                    │
│   ✓ History of conflicts/arguments                                      │
│   ✓ Prior threats (recorded and timestamped)                           │
│   ✓ Behavioral anomalies leading up to incident                        │
│   ✓ Suspect's statements that may contradict alibis                    │
│                                                                          │
│   PRIME SUSPECT IDENTIFICATION                                          │
│   ─────────────────────────────                                         │
│   ✓ Relationship deterioration indicators                               │
│   ✓ Financial motives (life insurance queries, etc.)                   │
│   ✓ Infidelity detection                                                │
│   ✓ Specific threats against the owner                                  │
│   ✓ Behavioral changes suggesting premeditation                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Longitudinal Memory Advantage

### Traditional Forensics vs. FLUX Forensics

| Traditional | FLUX |
|-------------|------|
| "We have a body, no witnesses, find the killer" | "We have a body AND 18 months of behavioral data" |
| Rely on human witness memory (unreliable) | Timestamped, multi-modal evidence |
| Alibis hard to verify | Alibis cross-referenced against recorded timeline |
| "He said / she said" | "He said X, but recording shows Y" |

### Example: The Pattern Evidence

```
┌─────────────────────────────────────────────────────────────────────────┐
│              PATTERN OF LIFE AS FORENSIC EVIDENCE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   WHAT FLUX KNOWS OVER TIME:                                            │
│                                                                          │
│   Month 1-6: Normal relationship                                        │
│   • Sleep in same bed                                                   │
│   • Regular intimacy patterns                                           │
│   • Occasional minor arguments (baseline)                               │
│                                                                          │
│   Month 7: Change detected                                              │
│   • John starts coming home late (pattern shift)                        │
│   • Fewer shared meals                                                  │
│   • Sarah shows stress indicators during interactions                   │
│                                                                          │
│   Month 8: Infidelity indicators                                        │
│   • John's phone calls in private (audio: female voice)                │
│   • Sarah confronts John (argument audio captured)                      │
│   • Sleeping in separate rooms begins                                   │
│                                                                          │
│   Month 9-10: Escalation                                                │
│   • Arguments increase (47 detected vs 3/month baseline)               │
│   • Physical aggression detected (John pushes Sarah)                   │
│   • Sarah enables "Solve My Murder" feature (significant)              │
│                                                                          │
│   Month 11: Threats                                                     │
│   • Audio captured: "I'm going to kill you" (John, timestamped)        │
│   • Sarah's stress levels consistently elevated                         │
│   • John researches "untraceable poisons" (if integrated)              │
│                                                                          │
│   Month 12: Incident                                                    │
│   • Sarah found dead                                                    │
│   • FLUX provides complete 12-month relationship trajectory            │
│   • John's claim: "We never fought, perfect marriage"                  │
│   • FLUX: 47 arguments, sleeping separately, specific death threat     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prime Suspect Identification

### How FLUX Identifies Suspects

FLUX doesn't just record—it **understands significance**:

```
SUSPECT SCORING ALGORITHM:

For each person in owner's life:
  
  MOTIVE INDICATORS:
  + Has made threats against owner
  + Benefits financially from owner's death
  + Has documented grievances
  + Relationship deteriorating over time
  
  OPPORTUNITY INDICATORS:
  + Has access to owner's home
  + Present during suspicious incidents
  + Behavior changes around critical dates
  + Alibi inconsistencies in past
  
  BEHAVIORAL INDICATORS:
  + Elevated aggression baseline
  + Deception indicators in conversations
  + Secretive behavior increase
  + Research into violence/death/poisons

PRIME SUSPECT = Highest composite score
```

### What FLUX Provides to Police

Even if FLUX didn't witness the murder directly:

```json
{
  "victim": "Sarah Chen",
  "prime_suspect": {
    "name": "John Chen",
    "relationship": "Husband",
    "confidence": 0.94
  },
  
  "motive_evidence": [
    {
      "type": "threat",
      "timestamp": "2026-03-15T20:45:00Z",
      "content": "Audio of John saying 'I'm going to kill you'",
      "verified_speaker": true
    },
    {
      "type": "infidelity_discovered",
      "timestamp": "2026-02-20T19:30:00Z",
      "summary": "Sarah discovered John's affair, major argument"
    },
    {
      "type": "financial_motive",
      "indicator": "John queried life insurance policy value 3 days before death"
    }
  ],
  
  "alibi_vulnerabilities": [
    {
      "claim": "John claims they never fought",
      "contradiction": "47 arguments documented over 6 months"
    },
    {
      "claim": "John claims he was at work",
      "contradiction": "No work badge-in on date of death, FLUX shows him home"
    }
  ],
  
  "behavioral_timeline": {
    "relationship_sentiment": "deteriorating_over_9_months",
    "violence_escalation": "3_physical_incidents_documented",
    "threat_history": "2_explicit_death_threats_recorded"
  }
}
```

---

## Concern: Pre-Death Consent to Post-Death Surveillance

### The Legal Question

> If someone enables "Solve My Murder" but then commits suicide, and the AI interprets it as "violence against owner," does lockdown trigger?

### The Safeguard

FLUX requires **evidence of external harm**:

```
SUICIDE DETECTION:
  • Self-inflicted injuries (pose analysis)
  • No intruder present
  • No preceding assault
  • Self-harm indicators in behavioral history
  
  → Medical emergency response, NOT lockdown
  → Mental health resources activated
  → Next-of-kin notified
  → No police evidence package (not a crime)

MURDER:
  • Harm from external source
  • Intruder OR known assailant
  • Violence precedes incapacitation
  • Perpetrator behavior (fleeing, cleaning)
  
  → Lockdown activated
  → Evidence preserved
  → Police notified
```

### Staged Suicide Detection

If a murderer stages a suicide:
- FLUX detects inconsistencies (staging behavior)
- Pose analysis shows forced positioning
- Timeline shows perpetrator manipulation of scene
- Pre-death behavioral context shows murder motive

---

## Enabling "Solve My Murder"

### User Interface Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│               SOLVE MY MURDER CONFIGURATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ⚠️  This feature enables pre-authorized evidence sharing with        │
│      law enforcement if you are killed or seriously incapacated.       │
│                                                                          │
│   WHAT THIS MEANS:                                                      │
│   ─────────────────                                                     │
│   If violence is detected AND you cannot respond:                       │
│   • FLUX will send evidence to police automatically                     │
│   • Evidence includes video, audio, and behavioral history             │
│   • Your data will be used to identify your attacker                   │
│   • This cannot be reversed after activation unless you disable it     │
│                                                                          │
│   UNDERSTANDING CHECK:                                                  │
│   ────────────────────                                                  │
│   □ I understand this shares my FLUX data with police                  │
│   □ I understand this includes relationship history                    │
│   □ I understand I can disable this anytime while alive               │
│   □ I am enabling this of my own free will                            │
│                                                                          │
│   AUTHORIZATION LEVEL:                                                  │
│   ○ Basic: Incident only (attack footage, perpetrator ID)              │
│   ○ Standard: Incident + 30 days prior context                         │
│   ● Enhanced: Incident + full relationship history (recommended)       │
│                                                                          │
│   DIGITAL SIGNATURE:                                                    │
│   ─────────────────                                                     │
│   [ Face scan + Voice confirmation + Liveness check ]                  │
│                                                                          │
│   Enabled by: Sarah Chen                                                │
│   Date: 2025-11-15 10:30:00 UTC                                        │
│   Signature: sha256:a7f3...                                            │
│                                                                          │
│   [ ENABLE "SOLVE MY MURDER" ]                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Verification Requirements

To enable this feature:
1. **Identity verification**: Multi-modal biometric confirmation
2. **Liveness check**: Confirms not under duress
3. **Understanding quiz**: User demonstrates comprehension
4. **Waiting period**: 24-hour cooling off before activation
5. **Re-confirmation**: Monthly reminder that feature is active

### Disabling the Feature

User can disable at any time while alive:
1. Voice command: "FLUX, disable Solve My Murder"
2. Biometric verification
3. Confirmation: "Are you sure? This removes pre-authorization."
4. Immediate deactivation

---

## Concern: AI "Testimony" in Court

### Current Legal Status

> The Sixth Amendment requires confrontation of witnesses. An AI recording is evidence, not testimony.

### FLUX's Role

FLUX doesn't "testify" in the legal sense—it provides **structured forensic analysis**:

| Traditional Evidence | FLUX Evidence |
|---------------------|---------------|
| Unprocessed video (hours of footage) | Relevant clips with timestamps |
| Human analyst interpretation | AI-verified causal chains |
| "We think this is what happened" | "Here is what happened, here is the confidence, here is the reasoning" |
| Chain of custody challenges | Cryptographic proof of integrity |

### Court Demonstrations

FLUX evidence can be presented with:
1. **Raw footage** (the actual recordings)
2. **AI analysis** (what the system detected and why)
3. **Confidence scores** (transparency about certainty)
4. **Causal reasoning** (the CGN graph showing how conclusions were reached)

The defense can challenge:
- The accuracy of detection models
- The interpretation of behavior
- The calibration of confidence scores
- The validity of causal chains

But they cannot claim tampering because:
- Cryptographic seals prove integrity
- Timestamps are verified by third-party authority
- The `.flx` format is fully inspectable

---

## Hearsay and Evidence Rules

### Why FLUX Evidence Isn't Hearsay

Hearsay: "Out-of-court statement offered for truth of the matter asserted"

FLUX recordings are:
- **Not statements**: Video/audio is a recording of events, not a statement about them
- **Business records exception**: FLUX logs created in regular course of operation
- **Present sense impression**: Real-time recording of events as they occur
- **Res gestae**: Part of the transaction itself

### The AI Analysis Layer

FLUX's interpretation (causal chains, confidence scores) is:
- **Expert evidence**: Admissible when methodology is sound (Daubert standard)
- **Subject to cross-examination**: Defense can challenge the model's reasoning
- **Transparent**: The `.flx` format allows full inspection of how conclusions were reached

---

## Evidence Package: The "Solve My Murder" Report

When triggered, FLUX generates a comprehensive report:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              FLUX EVIDENCE PACKAGE — SOLVE MY MURDER                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Case ID: FLX-2026-04-02-143212-001                                     │
│  Generated: 2026-04-02 14:32:30 UTC                                     │
│  Source: Sarah Chen's Home FLUX (authorized)                            │
│  Authorization: Solve My Murder (enabled 2025-11-15)                    │
│                                                                          │
│  ════════════════════════════════════════════════════════════           │
│                                                                          │
│  INCIDENT SUMMARY                                                       │
│  ─────────────────                                                      │
│  Date: 2026-04-02                                                       │
│  Time: 14:32:12 UTC                                                     │
│  Location: 123 Oak Street, Living Room                                  │
│  Victim: Sarah Chen (homeowner)                                         │
│  Status: Deceased (high confidence)                                     │
│                                                                          │
│  PRIME SUSPECT                                                          │
│  ─────────────                                                          │
│  Name: John Chen                                                        │
│  Relationship: Husband                                                  │
│  Confidence: 98.7%                                                      │
│                                                                          │
│  INCIDENT TIMELINE                                                      │
│  ─────────────────                                                      │
│  14:31:15 — John arrives home (unexpected, usually at work)             │
│  14:31:45 — Argument begins (audio captured)                            │
│  14:32:07 — Weapon detected (handgun)                                   │
│  14:32:12 — Gunshot detected, Sarah falls                               │
│  14:32:18 — John exits, enters vehicle                                  │
│                                                                          │
│  ATTACHMENTS (INCIDENT)                                                 │
│  ──────────────────────                                                 │
│  [1] video_14-31-00_to_14-33-00.mp4 (120 seconds, 4K)                  │
│  [2] audio_argument_enhanced.wav                                        │
│  [3] audio_gunshot.wav                                                  │
│  [4] face_john_chen_incident.jpg                                        │
│  [5] weapon_detected.jpg                                                │
│                                                                          │
│  ════════════════════════════════════════════════════════════           │
│                                                                          │
│  RELATIONSHIP CONTEXT (12 Months)                                       │
│  ─────────────────────────────────                                      │
│                                                                          │
│  RELATIONSHIP TRAJECTORY:                                               │
│  Month 1-6:  ████████████████░░░░ Stable                               │
│  Month 7-8:  ██████████░░░░░░░░░░ Declining                            │
│  Month 9-10: █████░░░░░░░░░░░░░░░ Hostile                              │
│  Month 11-12:██░░░░░░░░░░░░░░░░░░ Critical                             │
│                                                                          │
│  KEY EVENTS:                                                            │
│  • 2025-10-20: Infidelity confrontation (John's affair discovered)     │
│  • 2025-11-02: Physical altercation (John pushed Sarah)                │
│  • 2025-11-15: Sarah enabled "Solve My Murder" feature                 │
│  • 2025-12-03: Threat: "I'm going to kill you" (audio captured)        │
│  • 2026-01-15: Sleeping in separate rooms began                        │
│  • 2026-03-28: John queried life insurance policy value                │
│                                                                          │
│  ARGUMENT FREQUENCY:                                                    │
│  Baseline (Month 1-6): 3 arguments/month                               │
│  Month 9-12: 47 arguments total (15x increase)                         │
│                                                                          │
│  ATTACHMENTS (CONTEXT)                                                  │
│  ─────────────────────                                                  │
│  [6] threats_compilation.mp3 (3 recorded threats)                      │
│  [7] argument_summary_report.pdf                                       │
│  [8] behavioral_timeline.json                                          │
│                                                                          │
│  ════════════════════════════════════════════════════════════           │
│                                                                          │
│  ALIBI CONTRADICTION ANALYSIS                                           │
│  ─────────────────────────────                                          │
│                                                                          │
│  If suspect claims:                 FLUX can provide:                   │
│  "We never fought"              →  47 arguments documented              │
│  "Perfect marriage"             →  Separate bedrooms for 3 months       │
│  "I was at work"               →  FLUX shows him home at 14:31         │
│  "She was depressed, suicide"  →  Audio of HIS threats against HER    │
│  "I loved her"                 →  Affair evidence + threats            │
│                                                                          │
│  ════════════════════════════════════════════════════════════           │
│                                                                          │
│  CRYPTOGRAPHIC VERIFICATION                                             │
│  ──────────────────────────                                             │
│  Evidence hash: sha256:a7f3b2c1d4e5f6789...                            │
│  Signature: RSA-4096 (FLUX HSM)                                        │
│  Timestamp: RFC 3161 (third-party authority)                           │
│  Chain of custody: Established at 14:32:16 UTC                         │
│                                                                          │
│  Authorization verification:                                             │
│  "Solve My Murder" enabled: 2025-11-15 10:30:00 UTC                    │
│  User signature: Valid                                                  │
│  Liveness at enable: Confirmed                                          │
│  No duress indicators at enable: Confirmed                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Privacy Considerations

### What Is Shared

- Evidence directly relevant to the crime
- Behavioral history with suspects
- Threats, motives, opportunity indicators

### What Is NOT Shared

- Unrelated personal content
- Third-party data (neighbors, guests) unless relevant
- Content before "Solve My Murder" was enabled (unless re-authorized)

### Data Minimization

FLUX applies relevance filters:
- Only interactions with identified suspects
- Only behavioral patterns indicating motive/opportunity
- Not entire life history—just forensically relevant portions

---

## Related Documents

- [03_LOCKDOWN_MODE.md](03_LOCKDOWN_MODE.md) — Evidence preservation mechanism
- [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) — Constitutional considerations
- [11_EVIDENCE_TESTIMONY.md](11_EVIDENCE_TESTIMONY.md) — Court admissibility
- [14_CONCERNS_JUSTIFICATIONS.md](14_CONCERNS_JUSTIFICATIONS.md) — Objections and responses
