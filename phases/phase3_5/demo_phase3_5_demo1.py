import sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'phase1'))
sys.path.append(str(Path(__file__).parent.parent / 'phase2'))
from flux_utils import load_checkpoint, get_device, save_checkpoint, checkpoint_exists
from cse import ContinuousSemanticEncoder
from field import ResonanceField
from personal_fabric import PersonalFabric
from fabric_identity import FabricIdentity
from assistant_traits import AssistantTraitsEncoder
def main():
    if checkpoint_exists(3.5):
        print("Phase 3.5 checkpoint exists. Skipping.")
        return
    print("Initializing fabric...")
    device = get_device()
    try:
        c1, c2 = load_checkpoint(1), load_checkpoint(2)
        cse = ContinuousSemanticEncoder(**c1['config']).to(device)
        cse.load_state_dict(c1['state_dict'])
        field = ResonanceField(**c2['config']['field']).to(device)
        field.load_state_dict(c2['state_dict'])
    except:
        cse, field = ContinuousSemanticEncoder().to(device), ResonanceField().to(device)
    fid = FabricIdentity(cse)
    id_wave = fid.derive_identity_wave(["coffee", "sea"])
    fab_id = fid.derive_fabric_id(id_wave)
    fabric = PersonalFabric(fab_id, field, cse, None, device=device)
    fabric.initialize_identity(id_wave)
    traits = AssistantTraitsEncoder(cse, None, field, device=device)
    traits.seed_all_traits()
    fabric.perturb_personal("I am an engineer")
    save_checkpoint(3.5, {'fabric_id': fab_id})
    print(f"Fabric initialized: {fab_id}")
if __name__ == "__main__": main()
