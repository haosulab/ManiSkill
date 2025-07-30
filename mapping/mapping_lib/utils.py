import torch
import torch.nn.functional as F
from torchvision import transforms

__all__ = [
    "get_visual_features",
    "positional_encoding",
    "get_3d_coordinates",
    "transform",
]

# --------------------------------------------------------------------------- #
#  Visual feature extraction (dense CLIP)                                     #
# --------------------------------------------------------------------------- #

def get_visual_features(model, x: torch.Tensor) -> torch.Tensor:
    """Extracts **normalized** dense CLIP features.

    Args:
        model : open_clip model returned by ``open_clip.create_model_and_transforms``.
        x     : (B, 3, H, W) image batch **already normalized** to [0,1] and
                 pre-scaled to 224×224.

    Returns:
        (B, C, h, w) dense feature map where ``C`` is the CLIP embedding dim and
        ``h==w`` (typically 16).
    """
    vision_model = model.visual.trunk
    x = vision_model.forward_features(x)
    x = vision_model.fc_norm(x)
    x = vision_model.head_drop(x)
    x = vision_model.head(x)

    dense_features = x[:, 1:, :]
    dense_features = F.normalize(dense_features, dim=-1)
    num_patches = dense_features.shape[1]
    grid_size = int(num_patches ** 0.5)
    dense_features = dense_features.permute(0, 2, 1)
    dense_features = dense_features.reshape(x.shape[0], -1, grid_size, grid_size)
    return dense_features

# --------------------------------------------------------------------------- #
#  Positional encoding                                                        #
# --------------------------------------------------------------------------- #

def positional_encoding(x: torch.Tensor, L: int = 10) -> torch.Tensor:
    pe = []
    for i in range(L):
        freq = 2 ** i
        pe.append(torch.sin(x * freq * torch.pi))
        pe.append(torch.cos(x * freq * torch.pi))
    return torch.cat(pe, dim=-1)

# --------------------------------------------------------------------------- #
#  3-D coordinates from depth map                                             #
# --------------------------------------------------------------------------- #

def get_3d_coordinates(
    depth: torch.Tensor,
    camera_extrinsic: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    original_size: int = 224,
):
    device = depth.device

    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)

    B, H_feat, W_feat = depth.shape
    scale_x = W_feat / float(original_size)
    scale_y = H_feat / float(original_size)
    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    u = torch.arange(W_feat, device=device).view(1, -1).expand(H_feat, W_feat) + 0.5
    v = torch.arange(H_feat, device=device).view(-1, 1).expand(H_feat, W_feat) + 0.5
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)

    x_cam = (u - cx_new) * depth / fx_new
    y_cam = (v - cy_new) * depth / fy_new
    z_cam = depth
    coords_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)

    camera_extrinsic = camera_extrinsic.squeeze(1)
    ones_row = torch.tensor([0, 0, 0, 1], device=device, dtype=camera_extrinsic.dtype).view(1, 1, 4)
    ones_row = ones_row.expand(B, 1, 4)
    extrinsic_4x4 = torch.cat([camera_extrinsic, ones_row], dim=1)
    extrinsic_inv = torch.inverse(extrinsic_4x4)

    _, _, Hf, Wf = coords_cam.shape
    ones_map = torch.ones(B, 1, Hf, Wf, device=device)
    coords_hom = torch.cat([coords_cam, ones_map], dim=1)
    coords_hom_flat = coords_hom.view(B, 4, -1)
    world_coords_hom = torch.bmm(extrinsic_inv, coords_hom_flat)
    world_coords_hom = world_coords_hom.view(B, 4, Hf, Wf)
    coords_world = world_coords_hom[:, :3, :, :]
    return coords_world, coords_cam

# --------------------------------------------------------------------------- #
#  Image transform (OpenCLIP style)                                           #
# --------------------------------------------------------------------------- #

transform = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
) 