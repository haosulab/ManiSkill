import torch
import torch.nn as nn
from torchvision import transforms

class DINO2DFeatureEncoder(nn.Module):
    """
    Thin wrapper around DINOv2 ViT-S/14 to produce dense 2D feature maps.

    Inputs are expected in shape (B, C, H, W) with values in [0, 1].
    Normalization is applied internally.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load backbone lazily via torch.hub to avoid extra dependencies
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = getattr(self.backbone, "embed_dim", 384)
        self.dino_proj = nn.Conv2d(384, 384, kernel_size=1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # DINOv2 normalization
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    @torch.no_grad()
    def _forward_dino_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-patch token embeddings without the [CLS] token.
        Shape: (B, N, C), where N = (H/14)*(W/14), C = embed_dim.
        """
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)  # (B, 1 + N, C)
        x = x[:, 1:, :]            # drop CLS → (B, N, C)
        return x

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_bchw: Float tensor in [0, 1], shape (B, 3, H, W)

        Returns:
            fmap: (B, C, Hf, Wf) where C = embed_dim and Hf = H//14, Wf = W//14
        """
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        # Normalize per DINOv2 recipe
        images_bchw = self.normalize(images_bchw)

        B, _, H, W = images_bchw.shape
        tokens = self._forward_dino_tokens(images_bchw)  # (B, N, C)

        C = self.embed_dim
        Hf, Wf = H // 14, W // 14
        fmap = tokens.permute(0, 2, 1).reshape(B, C, Hf, Wf).contiguous()
        fmap = self.dino_proj(fmap)
        return fmap


class PlainCNNFeatureEncoder(nn.Module):
    """
    Lightweight CNN that produces a dense 2D feature map.

    - Inputs are expected in shape (B, C, H, W) with values in [0, 1].
    - Outputs a feature map of fixed spatial resolution via adaptive pooling,
      making it compatible with the rest of the pipeline (e.g., local fusion).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        target_spatial_size: tuple[int, int] = (6, 6),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._target_spatial_size = target_spatial_size

        # Convolutional trunk roughly inspired by the provided PlainConv
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, embed_dim, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        # Always adapt spatial size to match downstream assumptions (e.g., 6x6)
        self.pool_to_fixed = nn.AdaptiveAvgPool2d(self._target_spatial_size)

        # Zero-initialize biases for stability, mirroring the provided snippet
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images_bchw: torch.Tensor) -> torch.Tensor:
        if images_bchw.dtype != torch.float32:
            images_bchw = images_bchw.float()
        x = self.cnn(images_bchw)
        x = self.pool_to_fixed(x)
        return x