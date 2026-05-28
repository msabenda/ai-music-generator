"""
Transformer Music Model — causal decoder-only transformer for note sequences.

Architecture:
  Embedding + PositionalEncoding → TransformerDecoder → Linear → Softmax
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


class TransformerMusicModel(nn.Module):
    """Causal decoder-only transformer for music token sequences."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token indices

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Embed + positional encode
        emb = self.token_embedding(x)  # (B, S, E)
        emb = self.pos_encoding(emb)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        causal_mask = causal_mask.float().masked_fill(causal_mask == float("-inf"), float("-inf"))

        # Decode (tgt is same as memory for decoder-only)
        out = self.decoder(
            tgt=emb,
            memory=emb,  # decoder-only: tgt = memory
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            memory_mask=causal_mask,
            memory_is_causal=True,
        )
        out = self.norm(out)
        logits = self.fc_out(out)  # (B, S, V)

        return logits
