"""
LSTM Music Model — note-by-note prediction with an embedding layer + LSTM + classifier.

Architecture:
  Embedding → LSTM → Dropout → Linear → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMMusicModel(nn.Module):
    """LSTM-based language model for MIDI token sequences."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len) token indices
            hidden: Optional (h, c) tuple for stateful generation

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden: (h, c) for stateful continuation
        """
        emb = self.embedding(x)  # (B, S, E)

        lstm_out, hidden = self.lstm(emb, hidden)  # (B, S, H)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # (B, S, V)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state (zeros)."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
