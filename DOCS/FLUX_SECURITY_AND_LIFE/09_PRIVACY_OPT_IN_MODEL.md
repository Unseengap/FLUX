# Privacy & Opt-In Model Specification

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

FLUX Security operates on a foundation of **privacy by default** with **explicit opt-in** for all sharing features. This document specifies exactly what data flows where, under what conditions, with what controls.

---

## Fundamental Privacy Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUX PRIVACY PRINCIPLES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PRINCIPLE 1: YOU OWN YOUR DATA                                        │
│   ════════════════════════════════                                      │
│   • Your FLUX file is YOUR property                                     │
│   • Stored locally on YOUR hardware                                     │
│   • Never uploaded unless YOU authorize                                 │
│   • Delete the file = AI forgets everything                             │
│                                                                          │
│   PRINCIPLE 2: OPT-IN EVERYTHING                                        │
│   ══════════════════════════════                                        │
│   • Every sharing feature requires explicit consent                     │
│   • Consent can be revoked at any time                                  │
│   • Default = maximum privacy                                           │
│   • Sensitive features require extra confirmation                       │
│                                                                          │
│   PRINCIPLE 3: MINIMAL SHARING                                          │
│   ════════════════════════════                                          │
│   • Only share what's needed for the specific purpose                   │
│   • Police get evidence, not your life history                          │
│   • Amber Alerts don't reveal your identity                             │
│   • Data minimization at every step                                     │
│                                                                          │
│   PRINCIPLE 4: TRANSPARENCY                                             │
│   ═══════════════════════════                                           │
│   • Know exactly what was shared, when, with whom                       │
│   • Audit log accessible to you always                                  │
│   • AI reasoning is explainable and reviewable                          │
│   • No hidden data collection                                           │
│                                                                          │
│   PRINCIPLE 5: CITIZEN OVER STATE                                       │
│   ════════════════════════════════                                      │
│   • Police cannot access without permission OR warrant                  │
│   • No mass surveillance capabilities                                   │
│   • Abuse = criminal offense                                            │
│   • Independent oversight required                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Classification

### What FLUX Stores Locally

| Data Type | Description | Retention | User Control |
|-----------|-------------|-----------|--------------|
| **Video feeds** | Raw camera footage | 30 days default | Adjustable |
| **Face embeddings** | Mathematical representations (not photos) | Permanent (enrolled) | Delete anytime |
| **Voice prints** | Speaker identification vectors | Permanent (enrolled) | Delete anytime |
| **Behavioral patterns** | Time-of-day, movement baselines | Rolling 6 months | Clear anytime |
| **Incident logs** | Timestamped security events | 1 year | Delete anytime* |
| **AI reasoning** | CGN chains, field states | Continuous | Introspectable |

*Except evidence under lockdown

### What FLUX Never Stores

| Data Type | Why Not |
|-----------|---------|
| Audio content (by default) | Privacy; only stored if explicitly enabled |
| Cloud copies | Local-only by design |
| Third-party analytics | No telemetry |
| Purchase data | Not a marketing tool |
| Biometric templates externally | Never leaves device |

---

## Opt-In Feature Matrix

### Feature-by-Feature Opt-In

| Feature | Default | Opt-In Required | Revocable |
|---------|---------|-----------------|-----------|
| Local recording | ON | No | Yes |
| Face recognition (family) | ON | During setup | Yes |
| Voice recognition | ON | During setup | Yes |
| Police sharing (incident) | OFF | Per-incident or pre-auth | Yes |
| Solve My Murder | OFF | Explicit multi-step | Yes (while alive) |
| Amber Alert participation | OFF | Tiered consent | Yes |
| Crime mesh participation | OFF | Tiered consent | Yes |
| Neighbor sharing | OFF | Per-share | Yes |
| Emergency contacts | OFF | Setup | Yes |

### Consent Granularity

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONSENT GRANULARITY LEVELS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LEVEL 0: NEVER (Default for external sharing)                         │
│   ─────────────────────────────────────────────                         │
│   • No data leaves device                                               │
│   • No participation in mesh networks                                   │
│   • Maximum privacy                                                     │
│                                                                          │
│   LEVEL 1: PER-INCIDENT                                                 │
│   ──────────────────────                                                │
│   • Asked each time sharing could occur                                 │
│   • "Police want to access this clip. Allow?"                          │
│   • Maximum control, some friction                                      │
│                                                                          │
│   LEVEL 2: PRE-AUTHORIZED (Conditional)                                 │
│   ──────────────────────────────────────                                │
│   • "If [condition], then share"                                       │
│   • E.g., "If I'm incapacitated, share with police"                    │
│   • Automatic with defined triggers                                     │
│                                                                          │
│   LEVEL 3: PRE-AUTHORIZED (Standing)                                    │
│   ─────────────────────────────────                                     │
│   • "Always participate in [feature]"                                   │
│   • E.g., "Always help with Amber Alerts"                              │
│   • Ongoing participation                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Specific Opt-In Flows

### Solve My Murder Opt-In

```
┌─────────────────────────────────────────────────────────────────────────┐
│               SOLVE MY MURDER OPT-IN FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   STEP 1: DISCOVERY                                                     │
│   Settings → Safety Features → Solve My Murder                         │
│                                                                          │
│   STEP 2: INFORMATION                                                   │
│   "This feature pre-authorizes FLUX to share evidence with law         │
│    enforcement if you are killed or incapacitated. This includes       │
│    video, audio, and relationship history."                             │
│                                                                          │
│   STEP 3: UNDERSTANDING QUIZ                                            │
│   Q1: What will be shared? [Video/Audio/History]                       │
│   Q2: Who will receive it? [Law enforcement]                           │
│   Q3: Can this be reversed after death? [No]                           │
│                                                                          │
│   STEP 4: BIOMETRIC CONFIRMATION                                        │
│   Face scan + Voice: "I authorize Solve My Murder"                     │
│   Liveness check: Confirm not under duress                             │
│                                                                          │
│   STEP 5: WAITING PERIOD                                                │
│   "Feature will activate in 24 hours. You can cancel anytime."         │
│                                                                          │
│   STEP 6: MONTHLY REMINDER                                              │
│   "Solve My Murder is active. Tap to review or disable."               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Crime Mesh Opt-In

```
┌─────────────────────────────────────────────────────────────────────────┐
│               CRIME MESH OPT-IN FLOW                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PARTICIPATION TIERS:                                                  │
│                                                                          │
│   ○ OFF (Default)                                                       │
│     "I don't want to participate"                                       │
│                                                                          │
│   ○ TIER 1: AWARE                                                       │
│     "Show me alerts, I'll manually decide to record/submit"            │
│                                                                          │
│   ○ TIER 2: ACTIVE (Recommended)                                        │
│     "Monitor for suspects, ask me before submitting"                   │
│     [ Match threshold: 85% ▾ ]                                         │
│                                                                          │
│   ○ TIER 3: AUTO-SUBMIT                                                 │
│     "Submit high-confidence matches automatically"                     │
│     [ Confidence threshold: 90% ▾ ]                                    │
│     "I can revoke within: [ 1 hour ▾ ]"                                │
│                                                                          │
│   ○ TIER 4: NEIGHBORHOOD WATCH                                          │
│     "Maximum contribution to community safety"                          │
│     Requires: 6 months established FLUX node                           │
│                                                                          │
│   PRIVACY SETTINGS FOR ALL TIERS:                                       │
│   ✓ Blur my face in submissions                                        │
│   ✓ Remove my voice from audio                                         │
│   ○ Exterior cameras only                                               │
│   ○ Anonymous submission (hide my identity from police)                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Access Audit Log

### User-Accessible Audit

Every data access is logged and visible to the user:

```
┌─────────────────────────────────────────────────────────────────────────┐
│               DATA ACCESS AUDIT LOG                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [2026-04-02 14:35:22] DATA SHARED                                     │
│   Feature: Crime Mesh Participation                                     │
│   Event: BOLO match detected (confidence: 94%)                          │
│   Shared: 8-second video clip, face detection metadata                  │
│   Recipient: Police FLUX Central (Badge #4472)                         │
│   Authorization: Tier 2 pre-authorization                               │
│   Revocable until: 2026-04-02 15:35:22                                 │
│   [ REVOKE ] [ VIEW CLIP ] [ DETAILS ]                                  │
│                                                                          │
│   [2026-04-01 09:15:00] ACCESS DENIED                                   │
│   Requester: Unknown (attempted remote access?)                        │
│   Action: Blocked at firewall                                           │
│   Details: Invalid authentication                                       │
│   [ REPORT ] [ DETAILS ]                                                │
│                                                                          │
│   [2026-03-30 20:00:00] FAMILY ACCESS                                   │
│   User: John (household member)                                         │
│   Action: Viewed front door camera (live)                              │
│   Authorization: Enrolled user                                          │
│   [ DETAILS ]                                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What Police Can Access

### Without Warrant (User Consent)

| Scenario | What's Shared | User Control |
|----------|---------------|--------------|
| User calls 911 | Context packet (location, description) | User initiated |
| User submits evidence | Selected clips only | User selected |
| Pre-authorized (incident) | Incident-relevant footage | Pre-configured scope |
| Solve My Murder | Incident + history | Pre-authorized |

### With Warrant

| Scenario | What's Accessible | Limitations |
|----------|-------------------|-------------|
| Valid warrant | Specified scope only | Warrant defines scope |
| Emergency exigency | Immediate threat only | Must justify post-hoc |
| Grand jury subpoena | Documents, not testimony | 5th Amendment applies |

### Never Accessible

Even with warrant, police cannot:
- Access Fifth-Amendment-protected personal Fabric content for self-incrimination
- Mass-query all FLUX nodes ("show me everyone")
- Access historical data without warrant
- Override cryptographic protections

---

## Consent Withdrawal

### How to Revoke Consent

Any consent can be withdrawn:

1. **Immediate revocation**: "FLUX, stop sharing with police"
2. **Feature disable**: Settings → Features → [Feature] → Disable
3. **Data deletion**: Settings → Privacy → Delete [Data Type]
4. **Full factory reset**: All data erased, AI forgets everything

### Revocation Limitations

| Scenario | Can Revoke? |
|----------|-------------|
| Pre-incident sharing preferences | Yes, immediately |
| Post-submission evidence (within window) | Yes, within revocation period |
| Post-submission evidence (after window) | No, already in police custody |
| Lockdown evidence | No, crime scene preservation |
| Human Dignity evidence | No, victim protection override |

---

## The Open Format Guarantee

### Why Open Format Matters for Privacy

The `.flx` format is **fully inspectable**:

```python
# Anyone can open and audit a FLUX model
import torch

model = torch.load('My-FLUX.flx', map_location='cpu')

# See every component
print(model.keys())  # cse, field, memory, vlm, etc.

# Inspect any tensor
print(model['cse']['state_dict']['encoder.weight'].shape)

# Read the config
print(model['runtime_config'])

# Inject, modify, audit — it's YOUR model
```

**Privacy guarantees from open format:**
- No hidden surveillance code (auditable)
- No secret data exfiltration (verifiable)
- Security researchers can inspect (trustable)
- You can modify behavior (controllable)
- Community can detect malicious changes (crowdsourced verification)

---

## Third-Party Data Handling

### Guests and Visitors

| Scenario | How Handled |
|----------|-------------|
| Guest visits | Notified of recording, can request cameras off |
| Guest not enrolled | Face not stored long-term, treated as "visitor" |
| Guest in incident | Evidence preserved (Human Dignity), blurred otherwise |
| Delivery person | Logged but not stored long-term |

### Bystanders in Shared Evidence

When evidence is submitted to police:
- **Bystander faces**: Blurred by default
- **Bystander voices**: Removed by default
- **Only suspect**: Full detail preserved
- **User can override**: "Include all faces" (for witness ID cases)

---

## Data Retention Policies

### Default Retention

| Data Type | Retention | Justification |
|-----------|-----------|---------------|
| Live video buffer | 24 hours rolling | Immediate incident access |
| Archived video | 30 days | Reasonable security window |
| Face embeddings (family) | Until deleted | Continuous recognition |
| Behavioral baselines | 6 months rolling | Pattern analysis |
| Incident logs | 1 year | Legal/insurance |
| Evidence (lockdown) | Until released | Legal requirement |

### User-Configurable Retention

Users can adjust:
- Reduce video retention (minimum: 24 hours)
- Extend video retention (up to storage limits)
- Auto-delete specific content types
- Set "forget after" rules for specific data

---

## Privacy vs. Safety Trade-offs

### The Explicit Trade-off

FLUX makes trade-offs transparent:

```
┌─────────────────────────────────────────────────────────────────────────┐
│               PRIVACY VS SAFETY SPECTRUM                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   MAXIMUM PRIVACY ◄─────────────────────────────────► MAXIMUM SAFETY   │
│                                                                          │
│   No sharing enabled                    All features enabled            │
│   ├── No police access                  ├── Pre-authorized sharing      │
│   ├── No mesh participation             ├── Mesh participation          │
│   ├── No emergency auto-dial            ├── Emergency auto-dial         │
│   └── No Solve My Murder                └── Solve My Murder             │
│                                                                          │
│   YOU CHOOSE where on this spectrum you want to be.                     │
│   FLUX supports any position.                                           │
│   The default is maximum privacy.                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Related Documents

- [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) — Self-incrimination protections
- [04_SOLVE_MY_MURDER.md](04_SOLVE_MY_MURDER.md) — Pre-authorization details
- [07_CRIME_DETECTION_MESH.md](07_CRIME_DETECTION_MESH.md) — Mesh participation
- [14_CONCERNS_JUSTIFICATIONS.md](14_CONCERNS_JUSTIFICATIONS.md) — Objections and responses
