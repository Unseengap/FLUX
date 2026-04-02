# Enterprise & Business Applications

**Version:** 1.0  
**Last Updated:** April 2, 2026  
**Status:** Specification

---

## Overview

The same FLUX architecture that powers home security scales to enterprise applications. This document covers corporate security, insider threat detection, access control, and compliance use cases.

---

## FLUX Enterprise: Core Capabilities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FLUX ENTERPRISE CAPABILITIES                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   INSIDER THREAT DETECTION                                              │
│   ─────────────────────────                                             │
│   • CSE encodes: Code commits, file access patterns, USB insertions     │
│   • Field accumulates: "Normal employee behavior" attractors            │
│   • CGN detects causal chains:                                          │
│     Access confidential file → Copy to USB → Resigned next day          │
│   • 6-month behavioral trail for anomaly detection                      │
│                                                                          │
│   ACCESS CONTROL INTEGRATION                                            │
│   ──────────────────────────                                            │
│   • Badge verification + face match + behavioral consistency            │
│   • Tailgating detection (two faces, one badge)                        │
│   • Time-based anomaly detection (2 AM access to secure areas)          │
│   • Duress detection (forced entry under coercion)                      │
│                                                                          │
│   COMPLIANCE FORENSICS                                                  │
│   ─────────────────────                                                 │
│   • Automatic evidence packaging for audits                             │
│   • Cryptographic chain of custody                                      │
│   • Retention policies enforced at storage layer                        │
│   • Export formats for legal proceedings                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Models

### Single-Site Enterprise

| Component | Specification |
|-----------|--------------|
| **Hub** | Server-grade GPU cluster |
| **Cameras** | 50-500+ simultaneous feeds |
| **Storage** | NAS with RAID, 2-year retention typical |
| **Integration** | Badge readers, door controllers, workstations |

### Multi-Site Federation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-SITE FEDERATION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SITE A (HQ)              SITE B (Datacenter)       SITE C (Branch)   │
│   ┌───────────┐            ┌───────────┐            ┌───────────┐      │
│   │ FLUX Hub  │            │ FLUX Hub  │            │ FLUX Hub  │      │
│   │ Primary   │◄──────────►│ Secondary │◄──────────►│ Secondary │      │
│   └───────────┘            └───────────┘            └───────────┘      │
│        │                        │                        │              │
│        ▼                        ▼                        ▼              │
│   200 cameras              50 cameras               75 cameras         │
│   500 employees            25 employees             100 employees      │
│   HR Primary               IT Primary               Sales Primary      │
│                                                                          │
│   FEDERATION FEATURES:                                                  │
│   • Employee badge works at all sites                                   │
│   • Behavioral baseline follows employee                                │
│   • Incident at Site B → Alert at Site A                               │
│   • Unified admin console                                               │
│   • Cross-site reporting                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Insider Threat Detection

### Behavioral Baseline

FLUX learns "normal" for each employee:

```
EMPLOYEE BASELINE PROFILE:

Employee: John Smith (Engineering)

ACCESS PATTERNS:
  Typical hours: 09:00-18:00 Mon-Fri
  Typical locations: Building A, Engineering floor
  Typical badge-ins: 2-3 per day
  After-hours access: Rare (2x/month average)

FILE ACCESS:
  Repositories: frontend, api-service (his projects)
  Outside repos: Rare (3x/month for code review)
  Sensitive areas: Never accessed

DEVICE BEHAVIOR:
  USB insertions: Never
  External emails: 5/day average (work-related)
  Downloads: Normal (libraries, docs)

PHYSICAL BEHAVIOR:
  Tailgating: Never
  Escort of visitors: Occasionally
  Unusual movements: None detected
```

### Anomaly Detection

When behavior deviates from baseline:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ANOMALY DETECTION ALERT                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   EMPLOYEE: John Smith                                                  │
│   RISK SCORE: 0.78 (elevated)                                           │
│   PERIOD: Last 14 days                                                  │
│                                                                          │
│   ANOMALIES DETECTED:                                                   │
│                                                                          │
│   [1] ACCESS PATTERN CHANGE                                             │
│       Normal: 09:00-18:00                                               │
│       Recent: 22:00-01:00 (5 instances)                                │
│       Anomaly score: 0.85                                               │
│                                                                          │
│   [2] FILE ACCESS DEVIATION                                             │
│       Normal: frontend, api-service repos                               │
│       Recent: payment-processing, customer-data repos                  │
│       Outside normal scope: YES                                         │
│       Anomaly score: 0.91                                               │
│                                                                          │
│   [3] USB DEVICE INSERTION                                              │
│       Baseline: Never                                                   │
│       Recent: 3 insertions (all after hours)                           │
│       Anomaly score: 0.95                                               │
│                                                                          │
│   [4] RESIGNATION INDICATOR                                             │
│       LinkedIn activity spike: Yes                                      │
│       External recruiter emails: 2 detected                             │
│       Calendar: "Dentist appointment" (offsite interview?)              │
│                                                                          │
│   RECOMMENDATION: Security team review                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Causal Chain Construction

FLUX builds causal chains showing **why** behavior is suspicious:

```
CGN CAUSAL CHAIN:

[Resignation indicators detected]
        │
        ▼
[Accessed payment-processing repo (outside normal scope)]
        │
        ▼
[USB device inserted (first ever)]
        │
        ▼
[Downloaded 15GB over 3 weeks (unusual)]
        │
        ▼
[After-hours access pattern (new behavior)]
        │
        ▼
[CONCLUSION: Possible data exfiltration prior to departure]
Confidence: 0.87

RECOMMENDED ACTIONS:
1. Review USB device contents if possible
2. Audit file access logs
3. Consider exit interview focus areas
4. Preserve evidence for potential litigation
```

---

## Access Control Integration

### Smart Access Points

FLUX integrates with physical access control:

| Capability | Description |
|------------|-------------|
| **Badge + Face** | Badge alone insufficient; face must match |
| **Tailgating detection** | Two people, one badge = alert |
| **Time-based rules** | After-hours access requires approval |
| **Location consistency** | Can't badge in Building A if badged in Building B 5 minutes ago |
| **Visitor escort** | Visitors tracked, must be escorted |

### Duress Detection at Access Points

```
SCENARIO: Employee forced to badge in under coercion

Detection signals:
  • Stress indicators (face, posture)
  • Unknown person behind employee
  • Unusual entry time
  • Employee not responding to greeting

Response:
  • Door opens (to avoid escalation)
  • Silent alert to security
  • Enhanced recording begins
  • Security team dispatched
```

---

## Compliance Use Cases

### GDPR Compliance

| Requirement | FLUX Implementation |
|-------------|---------------------|
| Data minimization | Only process necessary data |
| Purpose limitation | Security purposes only |
| Storage limitation | Configurable retention policies |
| Right to access | Employee can request their data |
| Right to erasure | Erasure supported (except legal holds) |
| Lawful basis | Legitimate interest (security) + employee notice |

### HIPAA Compliance (Healthcare)

| Requirement | FLUX Implementation |
|-------------|---------------------|
| Access controls | Badge + face + authorization |
| Audit logs | All access logged, timestamped |
| Minimum necessary | Role-based access to patient areas |
| Breach notification | Automatic detection of unauthorized access |
| BAA support | Can sign Business Associate Agreements |

### SOX Compliance (Financial)

| Requirement | FLUX Implementation |
|-------------|---------------------|
| Access to financial systems | Logged and auditable |
| Segregation of duties | Enforced at physical and logical layer |
| Change management | All changes logged with approval chain |
| Audit trails | Immutable, cryptographically signed |

---

## The Portable Key in Enterprise

### Employee Portable Keys

Employees can carry personal FLUX Portable Keys:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PORTABLE KEY IN ENTERPRISE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   EMPLOYEE'S PORTABLE KEY:                                              │
│   ─────────────────────────                                             │
│   • Contains personal Fabric (their AI preferences)                     │
│   • Authenticates to company FLUX Enterprise                            │
│   • Personal data stays personal                                        │
│   • Work data stays on company systems                                  │
│                                                                          │
│   WORKFLOW:                                                             │
│   ───────────                                                           │
│   Morning: Employee plugs in Portable Key                               │
│   ↓                                                                      │
│   Key authenticates to FLUX Enterprise                                  │
│   ↓                                                                      │
│   Workstation recognizes employee preferences                           │
│   ↓                                                                      │
│   Work continues with personalized AI assistance                        │
│   ↓                                                                      │
│   Evening: Employee takes key home                                      │
│   ↓                                                                      │
│   Personal data never stored on company servers                         │
│                                                                          │
│   SEPARATION:                                                           │
│   ────────────                                                          │
│   Personal Fabric: Employee owns, company cannot access                 │
│   Work Fabric: Company owns, employee can access their portion          │
│   Boundary: Clear, enforced at cryptographic level                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Industry-Specific Applications

### Financial Services

| Use Case | FLUX Capability |
|----------|-----------------|
| Trading floor surveillance | Real-time compliance monitoring |
| Vault access | Multi-factor (badge + face + iris) |
| Insider trading detection | Communication pattern analysis |
| Audit response | Instant evidence packaging |

### Healthcare

| Use Case | FLUX Capability |
|----------|-----------------|
| Patient wandering | Pose + location tracking |
| Medication cabinet | Access logging with face verification |
| Visitor management | Escort tracking, unauthorized area alerts |
| Emergency response | Automatic 911 integration |

### Manufacturing

| Use Case | FLUX Capability |
|----------|-----------------|
| Safety compliance | PPE detection (helmets, vests) |
| Restricted areas | Access control + behavioral monitoring |
| Equipment monitoring | Anomaly detection for maintenance |
| Theft prevention | Object tracking, removal alerts |

### Education

| Use Case | FLUX Capability |
|----------|-----------------|
| Campus safety | Intruder detection, lockdown support |
| After-hours access | Student vs. staff rules |
| Visitor management | Background check integration |
| Emergency coordination | FLUX Amber Alert participation |

---

## Privacy Considerations for Employees

### What Employees Should Know

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMPLOYEE PRIVACY NOTICE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   This organization uses FLUX Enterprise security.                      │
│                                                                          │
│   WHAT IS MONITORED:                                                    │
│   ─────────────────                                                     │
│   ✓ Building access (who enters, when, where)                          │
│   ✓ Camera-covered areas (common areas, hallways)                      │
│   ✓ Workstation activity (for security purposes)                       │
│   ✓ File access patterns (for insider threat detection)                │
│                                                                          │
│   WHAT IS NOT MONITORED:                                                │
│   ──────────────────────                                                │
│   ✗ Personal device content (unless connected to company network)      │
│   ✗ Off-site activities                                                 │
│   ✗ Private conversations (audio not recorded in most areas)           │
│   ✗ Personal Portable Key contents                                      │
│                                                                          │
│   YOUR RIGHTS:                                                          │
│   ────────────                                                          │
│   • Request access to your monitoring data                              │
│   • Request correction of inaccurate data                               │
│   • Understand how data is used                                         │
│   • Report concerns to [Privacy Officer]                                │
│                                                                          │
│   DATA RETENTION: [X] days/months unless legal hold                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Per-Person Fabric in Enterprise

Even in enterprise, individuals have some data sovereignty:

| Data Type | Ownership |
|-----------|-----------|
| Personal preferences | Employee (via Portable Key) |
| Work product | Company |
| Access logs | Company |
| Behavioral baseline | Company (shared with employee on request) |
| Personal communications | Employee (if on personal device) |

---

## Pricing Model

| Tier | Scale | Features |
|------|-------|----------|
| **FLUX Business** | <50 employees | Basic security, access control |
| **FLUX Enterprise** | 50-500 employees | Full insider threat, compliance |
| **FLUX Campus** | 500+ employees | Multi-site, advanced analytics |
| **FLUX Government** | Regulated | Air-gapped, highest security |

---

## Related Documents

- [01_NETWORK_ARCHITECTURE.md](01_NETWORK_ARCHITECTURE.md) — Three-tier architecture
- [13_POLICE_INTEGRATION.md](13_POLICE_INTEGRATION.md) — Law enforcement coordination
- [09_PRIVACY_OPT_IN_MODEL.md](09_PRIVACY_OPT_IN_MODEL.md) — Privacy framework
