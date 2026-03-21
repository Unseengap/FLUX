"""
PHASE 1.5 DEMO 3: Forward propagation through implication chains.

Run: python demo_phase1_5_demo3.py

Shows:
    Starting from a seed concept, trace what the model implies.

Expected output:
    Seed: "it started raining"

    Step 1 implications (direct):
        → "people got wet"           strength: 0.81
        → "umbrellas were opened"    strength: 0.74
        → "the ground became wet"    strength: 0.79

    Step 2 implications (transitive):
        → "people sought shelter"    strength: 0.61
        → "visibility decreased"     strength: 0.58

    Step 3 implications (deep):
        → "travel slowed down"       strength: 0.42

    Chain depth reached: 3
    Total unique concepts reachable: 6
    Deepest chain strength: 0.42

    This is REASONING — not retrieval. The model never saw
    "rain → travel slowed" directly. It inferred it via the chain.

    Saves: demo1_5_implication_chains.png
"""
# TODO: Copilot — Implement demo script as per PHASE_1_5_SPEC.md
