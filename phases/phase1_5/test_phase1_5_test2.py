"""
PHASE 1.5 TEST 2: Contradicting pairs produce higher tension than neutral pairs.

Procedure:
    For all 50 contradiction pairs:
        tension_contra  = tension_score(statement + contradiction)
        tension_neutral = tension_score(statement + non_contradiction)
        Assert: tension_contra > tension_neutral

Pass criteria:
    - 45/50 contradiction pairs correctly detected (higher tension)
    - Mean tension gap (contra - neutral) > 0.2
    - No false positives: neutral pairs have tension < 0.3 on average
    - Runs in < 30 seconds
"""
# TODO: Copilot — Implement test script as per PHASE_1.5_SPEC.md
