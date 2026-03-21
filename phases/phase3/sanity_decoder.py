import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

class SanityDecoder(nn.Module):
    def __init__(self, feature_dim: int = 512, device: str = 'cuda'):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.decode_net = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256),
        )
    def forward(self, features: Tensor) -> Tensor:
        return self.decode_net(features)
    def decode(self, features: Tensor) -> str:
        with torch.no_grad():
            logits = self.forward(features)
            byte_ids = logits.argmax(dim=-1)
            byte_list = byte_ids.cpu().tolist()
            cleaned = [b for b in byte_list if 32 <= b <= 126]
            try: return bytes(cleaned).decode('ascii', errors='replace')
            except Exception: return '[decode error]'
    def reconstruct_and_compare(self, original_text: str, features: Tensor) -> dict:
        reconstructed = self.decode(features)
        min_len = min(len(original_text), len(reconstructed))
        char_acc = sum(a == b for a, b in zip(original_text[:min_len], reconstructed[:min_len])) / max(len(original_text), 1)
        original_chars = set(original_text.lower())
        output_chars = set(reconstructed.lower())
        overlap = len(original_chars & output_chars) / max(len(original_chars), 1)
        return {'original': original_text, 'reconstructed': reconstructed, 'char_accuracy': char_acc, 'char_overlap': overlap, 'recognizable': overlap > 0.2}

def pipeline_check(cse, field, gr, decoder, test_input, verbose=True):
    wave = cse.encode(test_input)
    wave_full_mean = wave.full.mean(dim=0)
    field_features_query = field.wave_to_feature(wave_full_mean)
    field_out, _, _ = field.query(wave_full_mean, k=8)
    gr_output = gr(field_out.unsqueeze(0)).squeeze(0)
    result = decoder.reconstruct_and_compare(test_input, gr_output)
    if verbose:
        print(f"\nInput: '{test_input}'\nReconstructed: '{result['reconstructed']}'\nRecognizable: {'YES' if result['recognizable'] else 'NO'}")
    return result
