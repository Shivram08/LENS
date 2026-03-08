# src/encoders/text_encoder.py
#
# MiniLM-L6-v2 text encoder.
# Frozen — weights never updated during LENS training.
#
# Input  : list of strings
# Output : L2-normalized embeddings of shape (N, 384)

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """
    Frozen all-MiniLM-L6-v2 sentence encoder.

    Architecture:
      - Backbone  : 6-layer BERT-style transformer (22M params, d=384)
      - Pooling   : mean pooling with attention mask
      - Output    : L2-normalized embedding in R^384

    sentence-transformers handles tokenization, forward pass,
    mean pooling, and normalization internally.
    The model is always in eval() mode with no gradients.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        device: str = "cuda",
    ):
        super().__init__()

        self.device        = torch.device(device)
        self.embedding_dim = embedding_dim

        # ── load MiniLM via sentence-transformers ──────────────────────────
        # sentence-transformers handles:
        #   tokenization → transformer forward → mean pooling → normalization
        print(f"  Loading {model_name}...")
        self.model = SentenceTransformer(model_name, device=str(self.device))

        # ── freeze all parameters ──────────────────────────────────────────
        for param in self.model.parameters():
            param.requires_grad = False

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  MiniLM loaded: {n_params/1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Encodes a list of text strings in batches.

        Internally sentence-transformers:
          1. Tokenizes each string (WordPiece, max 512 tokens, truncated)
          2. Runs 6-layer transformer forward pass
          3. Mean pools over non-padding token hidden states
          4. L2 normalizes the result

        Args:
            texts         : list of N strings
            batch_size    : number of texts per batch
            show_progress : whether to show tqdm progress bar

        Returns:
            embeddings: tensor of shape (N, 384) on CPU, L2-normalized
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            normalize_embeddings=True,   # L2 normalize — matches FAISS IndexFlatIP
            device=str(self.device),
        )

        return embeddings.cpu()          # (N, 384)