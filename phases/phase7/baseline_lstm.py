import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    """
    Small LSTM for quality comparison.
    """
    def __init__(self, vocab_size=256, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)
