"""
PHASE 1.5 DEMO 2: Tension detection on contradicting statements.

Run: python demo_phase1_5_demo2.py

Shows:
    - Tension scores for neutral sequences (near zero, blue)
    - Tension scores for contradicting sequences (elevated, red)
    - The exact position in the sequence where tension spikes

Expected output:
    "the sky is blue. birds can fly."
    Tension: [0.04, 0.06, 0.05, 0.07]  ← low, no contradiction

    "the sky is blue. the sky is green."
    Tension: [0.08, 0.09, 0.71, 0.68]  ← spikes at contradiction point

    "water is liquid. water is solid."
    Tension: [0.06, 0.07, 0.74, 0.72]  ← same pattern

    CATASTROPHIC FORGETTING SCORE: 0.00 (field unchanged by tension)
    Saves: demo1_5_contradiction_tension.png
"""
# TODO: Copilot — Implement demo script as per PHASE_1.5_SPEC.md
