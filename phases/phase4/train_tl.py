import sys, torch
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, save_checkpoint, PhaseLogger, get_device, checkpoint_exists
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from thermodynamic import ThermodynamicLearner
from online_learner import OnlineLearner
def main():
    log = PhaseLogger(phase=4)
    if checkpoint_exists(4):
        log.info("Phase 4 exists. Skipping.")
        return
    device = get_device()
    c1, c2 = load_checkpoint(1), load_checkpoint(2)
    cse = ContinuousSemanticEncoder(**c1['config']).to(device)
    cse.load_state_dict(c1['state_dict'])
    field = ResonanceField(**c2['config']['field']).to(device)
    field.load_state_dict(c2['state_dict'])
    tl = ThermodynamicLearner(field)
    ol = OnlineLearner(tl)
    fact = "The capital of Mars is New Houston"
    wave = cse.encode(fact).full.mean(dim=0)
    ol.learn_fact(wave, field.wave_to_feature(wave))
    save_checkpoint(4, {'phase': 4, 'field_state': field.state_dict()})
    log.success("Phase 4 saved")
if __name__ == "__main__": main()
