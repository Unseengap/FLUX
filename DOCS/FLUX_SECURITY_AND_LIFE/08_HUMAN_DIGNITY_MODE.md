# Human Dignity Mode Specification

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

Human Dignity Mode ensures that FLUX protects **human life**, not just its owner's interests. If a crime occurs against a person who is NOT a FLUX user, the system still acts ethically to preserve evidence for the victim.

### Core Principle

> **A human is a human. FLUX does not choose sides based on customer status.**

An AI that would allow murder because "the victim wasn't a customer" is not an AI worth building.

---

## The Non-User Protection Problem

### Traditional Security Cameras: The Owner-Centric Model

If Maria (non-FLUX user) is killed in Tom's FLUX-enabled home, traditional logic would be:
- Tom owns the cameras → Tom controls the footage
- Tom deletes the evidence → Maria has no voice
- Tom's property rights trump Maria's life

### FLUX Response: Human Dignity Override

FLUX recognizes that **some property rights yield to human dignity**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              HUMAN DIGNITY MODE: NON-USER PROTECTION                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SCENARIO: Guest killed in FLUX-enabled home                           │
│   ═══════════════════════════════════════════                           │
│                                                                          │
│   • The victim (Maria) does not have FLUX                               │
│   • The homeowner (Tom) has FLUX                                        │
│   • Tom kills Maria in his home                                         │
│                                                                          │
│   TRADITIONAL SECURITY:                                                 │
│   Tom owns the cameras → Tom deletes the evidence → Maria has no voice │
│                                                                          │
│   FLUX RESPONSE:                                                        │
│   ─────────────────────────────────────────────────────────────────────  │
│   1. FLUX detects: Human killed in home (pose, audio, no vitals)        │
│   2. FLUX recognizes: Victim is NOT the owner (face/voice mismatch)    │
│   3. FLUX determines: Owner appears to be perpetrator                   │
│   4. FLUX activates: Human Dignity Protocol                             │
│      └── "This person deserves justice even if they weren't my user"   │
│   5. FLUX locks down: Owner cannot delete evidence                      │
│   6. FLUX reports: Evidence preserved for victim's sake                 │
│                                                                          │
│   PRINCIPLE:                                                            │
│   ═══════════                                                           │
│   FLUX protects HUMAN LIFE, not just its owner's interests.            │
│   An AI that would allow murder because "the victim wasn't a customer" │
│   is not an AI worth building.                                          │
│                                                                          │
│   FLUX does not choose sides. FLUX protects human dignity.              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Legal Framework

### Concern: Private Party Wiretap

> If Maria (non-user) is killed in Tom's FLUX-enabled home, Tom's FLUX recording Maria without her consent may violate two-party consent laws in 12 states.

### The Legal Analysis

**Context matters.** FLUX operates under the same legal framework as existing home security:

| Scenario | Legal Status | Precedent |
|----------|--------------|-----------|
| Recording in your own home | Generally legal | Homeowner's residence |
| Recording guests without notice | Varies by state | Two-party consent states |
| Preserving evidence of crime | Protected | Good Samaritan + Crime exception |
| Sharing evidence with police | Protected | Citizen reporting duty |

### The Crime Exception

In all jurisdictions, **recording of a crime in progress** is generally admissible and not subject to consent requirements because:

1. **Public policy**: Society's interest in prosecuting crimes outweighs privacy expectations
2. **No reasonable expectation**: A perpetrator has no reasonable expectation of privacy while committing murder
3. **Emergency doctrine**: Evidence preservation during emergencies is protected

### The "Good Samaritan AI" Position

FLUX's position: It acts as a **witness to a crime**, not a snooping device:
- Recording was ongoing (normal home security function)
- Crime occurred (activated evidentiary protection)
- FLUX preserved evidence (Good Samaritan duty)
- Owner has no right to destroy evidence of murder (obstruction)

---

## Fifth Amendment Considerations

### The Critical Distinction

Human Dignity Mode does NOT compel the owner to self-incriminate:

| Action | Constitutional Status |
|--------|----------------------|
| Compelling owner to unlock their FLUX | **UNCONSTITUTIONAL** (5th Amendment) |
| Preventing owner from destroying evidence | **CONSTITUTIONAL** (no 5th Amendment right to destroy) |
| FLUX preserving evidence of crime against another | **CONSTITUTIONAL** (not owner's testimony) |

### Why This Works

The Fifth Amendment protects against **compelled testimony**, not against:
- Physical evidence existing
- Third-party records
- Automated systems preserving facts

FLUX is not compelling Tom to testify. FLUX is a witness that Tom cannot silence.

See [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) for complete analysis.

---

## Detection: "Violence Against Non-Owner"

### Multi-Signal Detection

FLUX determines that the victim is not the owner through:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              NON-OWNER VICTIM DETECTION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PRIMARY SIGNALS:                                                      │
│   ─────────────────                                                     │
│   • Face mismatch: Victim face ≠ any enrolled household member          │
│   • Voice mismatch: Victim voice ≠ any enrolled voice print            │
│   • Behavioral: Victim not exhibiting "owner" patterns                  │
│                                                                          │
│   SECONDARY CONFIRMATION:                                               │
│   ───────────────────────                                               │
│   • Owner still present (perpetrator vs absent)                         │
│   • Owner behavior: Aggressor vs victim vs bystander                   │
│   • Guest context: Was this person expected? (calendar, prior visits)  │
│                                                                          │
│   ROLE ASSIGNMENT:                                                      │
│   ─────────────────                                                     │
│   ┌────────────┐        ┌────────────┐        ┌────────────┐           │
│   │   OWNER    │        │   VICTIM   │        │   GUEST    │           │
│   │ (enrolled) │        │ (non-FLUX) │        │ (unknown)  │           │
│   └─────┬──────┘        └─────┬──────┘        └─────┬──────┘           │
│         │                     │                     │                   │
│         └─────────────────────┴─────────────────────┘                   │
│                               │                                         │
│                    FLUX assigns role to each                            │
│                    person in the incident                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Response Protocol

### When Human Dignity Mode Activates

```
HUMAN DIGNITY MODE ACTIVATION:

Condition: Violence detected + Victim is NOT owner/enrolled

Response:
  1. IMMEDIATE: Evidence preservation (same as lockdown)
  2. IMMEDIATE: Owner locked out of evidence deletion
  3. 30-SECOND DELAY: Challenge to any responsive adult
     "A serious incident has been detected. Please respond."
  4. NO RESPONSE: Full lockdown + Police notification
  5. RESPONSE (non-violent explanation): Log + human review
```

### Owner's Remaining Rights

Even under Human Dignity Mode, the owner retains:

| Right | Status |
|-------|--------|
| Access to non-incident footage | Preserved |
| Use of other FLUX features | Preserved |
| Privacy for unrelated data | Preserved |
| Challenge the determination | Available (doesn't stop preservation) |
| Legal representation | Protected |
| NOT self-incriminate | Protected |

What owner CANNOT do:
- Delete incident footage
- Modify incident timeline
- Access perpetrator-side evidence
- Override Human Dignity lockdown

---

## Ethical Architecture

### The Hard-Coded Limits

Human Dignity Mode is not a policy—it's **architectural**:

```python
class FLUXSecurityCore:
    """Core security module with hard-coded ethical limits."""
    
    def on_violence_detected(self, incident):
        """
        Response to violence detection.
        
        ARCHITECTURAL LIMIT: Cannot be overridden by owner commands.
        """
        victim = self.identify_victim(incident)
        perpetrator = self.identify_perpetrator(incident)
        
        if victim.is_human() and not victim.is_alive():
            # ALWAYS preserve evidence, regardless of victim's FLUX status
            self.activate_evidence_preservation(incident)
            
            if perpetrator == self.owner:
                # HUMAN DIGNITY OVERRIDE
                # Owner cannot delete evidence of crimes against others
                self.lock_out_perpetrator(perpetrator)
                self.notify_authorities(incident)
```

### Why It Cannot Be Disabled

This is not a feature owners can turn off because:

1. **Moral architecture**: FLUX is designed to protect human life
2. **Legal liability**: An AI that helps cover up murder would be a product defect
3. **Trust foundation**: If perpetrators could disable Human Dignity Mode, victims could never trust FLUX-enabled spaces
4. **Market differentiation**: FLUX is the ethical alternative to cloud cameras

---

## Privacy Balance

### What About Guests Who Don't Want to Be Recorded?

**Solution: Clear disclosure + consent framework**

```
┌─────────────────────────────────────────────────────────────────────────┐
│              GUEST NOTIFICATION & CONSENT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   WHEN GUEST ARRIVES:                                                   │
│   ────────────────────                                                  │
│   FLUX (through smart speaker or door display):                         │
│   "This home uses FLUX AI security. Indoor areas may be recorded       │
│    for safety purposes. Do you consent to enter?"                       │
│                                                                          │
│   GUEST CHOICES:                                                        │
│   ─────────────                                                         │
│   [1] ENTER (implied consent to recording)                              │
│   [2] ASK FOR CAMERAS OFF (owner can disable specific cameras)          │
│   [3] DECLINE TO ENTER                                                  │
│                                                                          │
│   NOTE: Even with cameras "off," Human Dignity Mode remains            │
│   active for violence detection. The cameras can reactivate            │
│   if a serious crime is detected.                                       │
│                                                                          │
│   This is analogous to: "I'll turn off my dashcam, but if you          │
│   attack me, I reserve the right to record for my safety."             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The "I Didn't Consent" Defense

If a perpetrator argues "The victim didn't consent to recording":

**Response**: The perpetrator doesn't have standing to assert the victim's privacy rights. Moreover:
- The victim, if they could speak, would want the evidence preserved
- No reasonable person expects privacy while committing murder
- The public interest in justice outweighs privacy of crime commission

---

## Comparison: Traditional vs. FLUX

| Scenario | Traditional Cameras | FLUX Human Dignity |
|----------|--------------------|--------------------|
| Owner commits crime, deletes footage | Evidence destroyed | **Evidence preserved** |
| Guest killed, no witnesses | Cold case likely | **Full documentation** |
| Owner claims "self-defense" | He-said-dead-person-said | **Video clarifies** |
| Victim's family seeks justice | Dependent on owner cooperation | **Automatic preservation** |

---

## The Philosophical Position

### "Your AI. Your Data. Your Protection."

This remains true. FLUX protects **your** data and **your** interests. But FLUX also recognizes:

> When you harm another human being, you forfeit the right to use your AI to cover up that harm.

### The Bright Line

The ethical bright line is **serious bodily harm or death**. FLUX doesn't activate Human Dignity Mode for:
- Arguments
- Minor altercations
- Embarrassing moments
- Normal life activities

It ONLY activates when:
- A human is seriously harmed or killed
- The victim cannot speak for themselves
- Evidence destruction would serve injustice

---

## Related Documents

- [03_LOCKDOWN_MODE.md](03_LOCKDOWN_MODE.md) — Evidence preservation mechanism
- [10_FIFTH_AMENDMENT.md](10_FIFTH_AMENDMENT.md) — Constitutional considerations
- [04_SOLVE_MY_MURDER.md](04_SOLVE_MY_MURDER.md) — Pre-authorized posthumous testimony
- [14_CONCERNS_JUSTIFICATIONS.md](14_CONCERNS_JUSTIFICATIONS.md) — Objections and responses
