"""
Phase 7: Baseline LSTM for Quality Comparison

A small byte-level LSTM model that serves as the comparison baseline
for FLUX text generation quality.

Architecture:
  - Byte-level input (256 vocab, no tokenizer — fair comparison to CSE)
  - 2-layer LSTM, hidden_dim=256
  - Standard backpropagation training

This is intentionally simple — it represents the "traditional" approach
that FLUX is designed to surpass.
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class LSTMTrainResult:
    """Training result for the baseline LSTM."""
    total_epochs: int = 0
    final_loss: float = 0.0
    final_perplexity: float = 0.0
    avg_loss: float = 0.0
    total_time_seconds: float = 0.0
    loss_history: List[float] = field(default_factory=list)


class BaselineLSTM(nn.Module):
    """
    Small byte-level LSTM for text generation comparison.

    Uses the same byte-level input as FLUX (no tokenizer, 256 vocab)
    to ensure a fair comparison. Standard neural network architecture
    with backpropagation training.

    Args:
        vocab_size: Byte vocabulary size (256 for byte-level)
        embed_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate between layers
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: [batch, seq_len] byte indices (long)
            hidden: Optional (h, c) tuple

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: Updated (h, c) tuple
        """
        embedded = self.dropout(self.embedding(x))     # [B, S, E]
        lstm_out, hidden = self.lstm(embedded, hidden)  # [B, S, H]
        logits = self.fc(self.dropout(lstm_out))        # [B, S, V]
        return logits, hidden

    def init_hidden(self, batch_size: int = 1,
                    device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.8, device: str = 'cpu') -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Starting text
            max_length: Maximum bytes to generate
            temperature: Sampling temperature
            device: Device

        Returns:
            Generated text (prompt + continuation)
        """
        self.eval()
        prompt_bytes = list(prompt.encode('utf-8'))
        generated = list(prompt_bytes)

        with torch.no_grad():
            hidden = self.init_hidden(1, device)

            # Feed prompt through LSTM
            if len(prompt_bytes) > 0:
                x = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
                logits, hidden = self(x, hidden)

            # Generate new bytes
            last_byte = prompt_bytes[-1] if prompt_bytes else ord(' ')

            for _ in range(max_length):
                x = torch.tensor([[last_byte]], dtype=torch.long, device=device)
                logits, hidden = self(x, hidden)
                logits = logits[0, -1] / max(temperature, 1e-8)
                probs = F.softmax(logits, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()

                if next_byte == 0:
                    break

                generated.append(next_byte)
                last_byte = next_byte

        try:
            return bytes(generated).decode('utf-8', errors='replace')
        except Exception:
            return bytes(generated).decode('latin-1', errors='replace')

    def compute_perplexity(self, text: str, device: str = 'cpu') -> float:
        """
        Compute byte-level perplexity of a text.

        Args:
            text: Text to evaluate
            device: Device

        Returns:
            Perplexity score
        """
        self.eval()
        text_bytes = list(text.encode('utf-8'))

        if len(text_bytes) < 2:
            return float('inf')

        with torch.no_grad():
            x = torch.tensor([text_bytes[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([text_bytes[1:]], dtype=torch.long, device=device)

            hidden = self.init_hidden(1, device)
            logits, _ = self(x, hidden)

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
            )

        return math.exp(min(loss.item(), 20.0))


def train_baseline_lstm(texts: List[str], epochs: int = 20,
                        lr: float = 1e-3, hidden_dim: int = 256,
                        device: str = 'cpu',
                        verbose: bool = True) -> Tuple[BaselineLSTM, LSTMTrainResult]:
    """
    Train a baseline LSTM on text data.

    Uses standard epoch-based backpropagation training.

    Args:
        texts: Training texts
        epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: LSTM hidden dimension
        device: Device
        verbose: Print progress

    Returns:
        (trained_model, training_result)
    """
    model = BaselineLSTM(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert texts to byte sequences
    sequences = []
    for text in texts:
        text_bytes = list(text.encode('utf-8'))
        if len(text_bytes) >= 2:
            sequences.append(text_bytes)

    t0 = time.time()
    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for seq in sequences:
            x = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([seq[1:]], dtype=torch.long, device=device)

            hidden = model.init_hidden(1, device)
            logits, _ = model(x, hidden)

            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                targets.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        loss_history.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            ppl = math.exp(min(avg_loss, 20.0))
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  ppl={ppl:.2f}")

    elapsed = time.time() - t0
    final_loss = loss_history[-1] if loss_history else 0.0

    result = LSTMTrainResult(
        total_epochs=epochs,
        final_loss=final_loss,
        final_perplexity=math.exp(min(final_loss, 20.0)),
        avg_loss=sum(loss_history) / max(len(loss_history), 1),
        total_time_seconds=elapsed,
        loss_history=loss_history,
    )

    return model, result
