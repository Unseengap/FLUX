import sys
from pathlib import Path
import torch
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, save_checkpoint, PhaseLogger, get_device, checkpoint_exists
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from gravity import GravitationalRelevance
from sanity_decoder import SanityDecoder, pipeline_check
from benchmark_attention import run_benchmark

def main():
    log = PhaseLogger(phase=3)
    if checkpoint_exists(3):
        log.info("Phase 3 checkpoint exists. Skipping.")
        return
    device = get_device()
    ckpt1 = load_checkpoint(1)
    cse = ContinuousSemanticEncoder(**ckpt1['config']).to(device)
    cse.load_state_dict(ckpt1['state_dict'])
    ckpt2 = load_checkpoint(2)
    field = ResonanceField(**ckpt2['config']['field']).to(device)
    field.load_state_dict(ckpt2['state_dict'])
    gr = GravitationalRelevance(feature_dim=512, device=device).to(device)
    decoder = SanityDecoder(feature_dim=512, device=device).to(device)
    run_benchmark(gr, device=device)
    pipeline_check(cse, field, gr, decoder, "hello world")
    save_checkpoint(3, {'phase': 3, 'gr_state': gr.save_state(), 'decoder_state': decoder.state_dict()})
if __name__ == "__main__": main()
