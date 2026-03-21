import sys
sys.path.insert(0, 'phases/phase3')
sys.path.insert(0, 'phases/phase2')
sys.path.insert(0, 'phases/phase1')
from sanity_decoder import SanityDecoder, MAX_DECODE_LEN
import torch

dec = SanityDecoder(feature_dim=512, device='cpu')
print(f'SanityDecoder created: max_len={dec.max_len}')

fake_input = torch.randn(8, 512)
logits = dec(fake_input)
print(f'Forward: input={fake_input.shape} -> logits={logits.shape}')

text = dec.decode(fake_input)
print(f'Decode (untrained): len={len(text)}')

loss = dec.compute_loss(fake_input, 'hello world')
print(f'compute_loss: {loss.item():.4f}')

target = SanityDecoder.text_to_target('hello world')
print(f'text_to_target: shape={target.shape}, first 11={target[:11].tolist()}')

opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
for i in range(100):
    loss = dec.compute_loss(fake_input, 'hello world')
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (i+1) % 25 == 0:
        text_now = dec.decode(fake_input)
        print(f'  Step {i+1}: loss={loss.item():.4f}, output="{text_now[:30]}"')

text_after = dec.decode(fake_input)
print(f'Final decode: "{text_after[:60]}" (len={len(text_after)})')
print('All SanityDecoder tests passed!')
