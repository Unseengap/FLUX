import sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, get_device
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from gravity import GravitationalRelevance
from sanity_decoder import SanityDecoder, pipeline_check
def main():
    device = get_device()
    try:
        c1, c2 = load_checkpoint(1), load_checkpoint(2)
        cse = ContinuousSemanticEncoder(**c1['config']).to(device)
        cse.load_state_dict(c1['state_dict'])
        field = ResonanceField(**c2['config']['field']).to(device)
        field.load_state_dict(c2['state_dict'])
    except:
        cse, field = ContinuousSemanticEncoder().to(device), ResonanceField().to(device)
    gr = GravitationalRelevance(device=device).to(device)
    dec = SanityDecoder(device=device).to(device)
    pipeline_check(cse, field, gr, dec, "the cat sat on the mat")
if __name__ == "__main__": main()
