# src/encoders/image_encoder.py
#
# DINOv2 ViT-S/14 image encoder.
# Frozen — weights never updated during LENS training.
#
# Input  : batch of PIL Images or file paths
# Output : L2-normalized embeddings of shape (B, 384)

import torch
import torch.nn as nn
import timm
from PIL import Image
from pathlib import Path
from torchvision import transforms

# ── ImageNet normalization constants ───────────────────────────────────────────
# DINOv2 was pretrained on ImageNet-scale data.
# These are the per-channel mean and std of ImageNet pixel values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 518) -> transforms.Compose:
    """
    Builds the preprocessing pipeline for DINOv2.

    Steps:
      1. Resize to image_size x image_size  (already done, but kept for safety)
      2. Convert PIL Image to float tensor in [0, 1]
      3. Normalize by ImageNet mean and std per channel

    Args:
        image_size: target spatial resolution (default 224)

    Returns:
        torchvision transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),                        # PIL → tensor [0,1], shape (3,H,W)
        transforms.Normalize(                         # per-channel: (x - mu) / sigma
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


class ImageEncoder(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 encoder.

    Architecture:
      - Backbone : DINOv2 ViT-S/14 (21M params, d=384)
      - Pooling  : CLS token (index 0 of last layer output)
      - Output   : L2-normalized embedding in R^384

    The model is always in eval() mode and requires no gradients.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        embedding_dim: int = 384,
        device: str = "cuda",
    ):
        super().__init__()

        self.device        = torch.device(device)
        self.embedding_dim = embedding_dim
        self.transform     = build_transform(image_size=518)

        # ── load DINOv2 from timm ──────────────────────────────────────────
        # num_classes=0 removes the classification head →
        # forward() returns the CLS token embedding directly
        print(f"  Loading {model_name}...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,      # remove classifier head
        )

        # ── freeze all parameters ──────────────────────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()
        self.backbone.to(self.device)

        n_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"  DINOv2 loaded: {n_params/1e6:.1f}M parameters (frozen)")

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of preprocessed image tensors.

        Args:
            images: tensor of shape (B, 3, 224, 224), already normalized

        Returns:
            embeddings: L2-normalized tensor of shape (B, 384)
        """
        # forward pass through ViT → CLS token output
        # backbone returns shape (B, 384) when num_classes=0
        embeddings = self.backbone(images)              # (B, 384)

        # L2 normalize: each row divided by its L2 norm
        embeddings = torch.nn.functional.normalize(
            embeddings, p=2, dim=1
        )                                               # (B, 384), unit vectors

        return embeddings

    @torch.no_grad()
    def encode_paths(
        self,
        image_paths: list[Path],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Encodes a list of image file paths in batches.

        Args:
            image_paths : list of Path objects pointing to .jpg files
            batch_size  : number of images per GPU batch
            show_progress: whether to show tqdm progress bar

        Returns:
            all_embeddings: tensor of shape (N, 384) on CPU
        """
        from tqdm import tqdm

        all_embeddings = []
        n = len(image_paths)

        iterator = range(0, n, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="  Encoding images", unit="batch")

        for start in iterator:
            batch_paths = image_paths[start : start + batch_size]

            # ── load and preprocess batch ──────────────────────────────────
            tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    t   = self.transform(img)           # (3, 224, 224)
                    tensors.append(t)
                except Exception:
                    # corrupted image → zero tensor (will be zero embedding)
                    tensors.append(torch.zeros(3, 224, 224))

            batch = torch.stack(tensors).to(self.device)  # (B, 3, 224, 224)

            # ── encode ────────────────────────────────────────────────────
            embs = self.forward(batch)                    # (B, 384)
            all_embeddings.append(embs.cpu())

        return torch.cat(all_embeddings, dim=0)           # (N, 384)