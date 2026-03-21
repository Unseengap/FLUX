"""
PHASE 1.5 DEMO 1: Causal coherence of ordered vs shuffled sequences.

Run: python demo_phase1_5_demo1.py

Shows:
    - A heatmap of causal coherence scores along a sentence
    - Green = high coherence (meaning flows naturally)
    - Red = low coherence (abrupt shift or reversal)
    - Compare original sentence against word-shuffled version

Expected output:
    Original:  "the dog chased the cat across the yard"
    Coherence: [0.82, 0.79, 0.84, 0.81, 0.77, 0.83, 0.80]  (high throughout)
    Shuffled:  "yard the across dog cat the chased"
    Coherence: [0.21, 0.18, 0.34, 0.15, 0.29, 0.19, 0.22]  (low throughout)
    Order sensitivity delta: 0.60+ (original >> shuffled)

    Saves: demo1_5_order_sensitivity.png
"""
# TODO: Copilot — Implement demo script as per PHASE_1.5_SPEC.md
