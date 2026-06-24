"""
Phase I: Occlusion-Robust Road Segmentation using Swin-UNet Transformer
Handles shadows, vehicles, and partial occlusions via attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Tuple, List, Optional
import math


# ─── Swin Transformer Blocks ──────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed them."""
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)                        # B, E, H/P, W/P
        x = rearrange(x, 'b e h w -> b (h w) e')
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        flat = torch.flatten(grid, 1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = rearrange(relative, 'c h w -> h w c')
        relative[..., 0] += window_size - 1
        relative[..., 1] += window_size - 1
        relative[..., 0] *= 2 * window_size - 1
        self.register_buffer('relative_position_index', relative.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        idx = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[idx].view(
            self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(self.proj_drop(x))
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with shifted window attention."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, dropout=dropout)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
        self.drop_path = nn.Identity()

    def _window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    def _window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_win = self._window_partition(x, self.window_size)
        x_win = x_win.view(-1, self.window_size ** 2, C)
        attn_out = self.attn(x_win)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = self._window_reverse(attn_out, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch merging for downsampling in encoder."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class PatchExpanding(nn.Module):
    """Patch expanding for upsampling in decoder."""
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        return self.norm(x)


# ─── Swin-UNet Architecture ───────────────────────────────────────────────────

class SwinUNet(nn.Module):
    """
    Swin-UNet: U-Net with Swin Transformer encoder/decoder.
    Robust to occlusions via global attention context.
    """
    def __init__(self, img_size=512, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = True
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i),
                    num_heads=h,
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ) for j in range(d)
            ])
            self.encoder_layers.append(layer)
            if i < self.num_layers - 1:
                self.downsample_layers.append(PatchMerging(int(embed_dim * 2 ** i)))

        # Bottleneck norm
        self.norm = nn.LayerNorm(self.num_features)

        # Decoder (mirror of encoder)
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_norms = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            dim = int(embed_dim * 2 ** i)
            self.upsample_layers.append(PatchExpanding(dim))
            self.skip_norms.append(nn.LayerNorm(dim // 2))
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim // 2,
                    num_heads=num_heads[i - 1],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ) for j in range(depths[i - 1])
            ])
            self.decoder_layers.append(layer)

        # Final upsampling & head
        self.final_expand = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * patch_size ** 2),
            nn.GELU()
        )
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)          # B, N, E
        x = self.pos_drop(x)

        patch_H = H // 4   # after patch embedding with patch_size=4
        patch_W = W // 4

        # Encoder pass with skip connections
        encoder_features = []
        cur_H, cur_W = patch_H, patch_W
        for i, layer_blocks in enumerate(self.encoder_layers):
            for blk in layer_blocks:
                x = blk(x, cur_H, cur_W)
            encoder_features.append((x, cur_H, cur_W))
            if i < self.num_layers - 1:
                x = self.downsample_layers[i](x, cur_H, cur_W)
                cur_H, cur_W = cur_H // 2, cur_W // 2

        x = self.norm(x)

        # Decoder pass with skip connections
        for i, (layer_blocks, up) in enumerate(zip(self.decoder_layers, self.upsample_layers)):
            x = up(x, cur_H, cur_W)
            cur_H, cur_W = cur_H * 2, cur_W * 2
            skip, s_H, s_W = encoder_features[-(i + 2)]
            skip = self.skip_norms[i](skip)
            x = x + skip   # additive skip connection
            for blk in layer_blocks:
                x = blk(x, cur_H, cur_W)

        # Final reconstruction
        x = self.final_expand(x)   # B, N, E*P^2
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                       h=patch_H, w=patch_W, p1=4, p2=4)
        return self.head(x)


# ─── Inference Engine ─────────────────────────────────────────────────────────

class RoadSegmentor:
    """
    High-level inference wrapper for road segmentation.
    Handles tiled inference, occlusion-aware post-processing.
    """
    def __init__(self, config: dict, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.model = self._build_model().to(self.device)
        self.model.eval()

    def _build_model(self) -> SwinUNet:
        cfg = self.config
        return SwinUNet(
            img_size=cfg.get('input_size', [512, 512])[0],
            patch_size=cfg.get('patch_size', 4),
            embed_dim=cfg.get('embed_dim', 96),
            depths=cfg.get('depths', [2, 2, 6, 2]),
            num_heads=cfg.get('num_heads', [3, 6, 12, 24]),
            window_size=cfg.get('window_size', 7),
            num_classes=cfg.get('num_classes', 2),
            dropout=cfg.get('dropout', 0.1),
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state)
        print(f"[Phase I] Loaded checkpoint: {path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Normalize and tensorize. image: H,W,3 uint8."""
        img = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    def tiled_predict(self, image: np.ndarray, tile_size: int = 512,
                       overlap: int = 64) -> np.ndarray:
        """
        Predict on large satellite images via overlapping tiles.
        Returns probability map of shape H x W.
        """
        H, W = image.shape[:2]
        stride = tile_size - overlap
        prob_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        ys = list(range(0, H - tile_size + 1, stride)) + [max(0, H - tile_size)]
        xs = list(range(0, W - tile_size + 1, stride)) + [max(0, W - tile_size)]

        with torch.no_grad():
            for y in set(ys):
                for x in set(xs):
                    tile = image[y:y + tile_size, x:x + tile_size]
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        pad_h = tile_size - tile.shape[0]
                        pad_w = tile_size - tile.shape[1]
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)))
                    tensor = self.preprocess(tile).to(self.device)
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
                    th, tw = min(tile_size, H - y), min(tile_size, W - x)
                    prob_map[y:y + th, x:x + tw] += probs[:th, :tw]
                    count_map[y:y + th, x:x + tw] += 1.0

        return prob_map / np.maximum(count_map, 1.0)

    def segment(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full segmentation pipeline.
        Returns: (binary_mask, probability_map)
        """
        prob_map = self.tiled_predict(image)
        binary = (prob_map > threshold).astype(np.uint8)
        return binary, prob_map

    def generate_synthetic_demo(self, size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a realistic synthetic road network for demo purposes
        when no real checkpoint/image is available.
        """
        import cv2
        mask = np.zeros((size, size), dtype=np.uint8)

        # Main arterials
        road_specs = [
            # (pt1, pt2, thickness)
            ((0, size // 2), (size, size // 2), 12),
            ((size // 2, 0), (size // 2, size), 12),
            ((0, size // 4), (size, size // 4), 8),
            ((0, 3 * size // 4), (size, 3 * size // 4), 8),
            ((size // 4, 0), (size // 4, size), 8),
            ((3 * size // 4, 0), (3 * size // 4, size), 8),
            ((0, 0), (size, size), 6),
            ((size, 0), (0, size), 6),
        ]
        for p1, p2, t in road_specs:
            cv2.line(mask, p1, p2, 255, t)

        # Curved connector roads
        for i in range(6):
            pts = np.array([
                [np.random.randint(0, size), np.random.randint(0, size)],
                [np.random.randint(0, size), np.random.randint(0, size)],
                [np.random.randint(0, size), np.random.randint(0, size)],
            ], dtype=np.int32)
            cv2.polylines(mask, [pts], False, 255, 5)

        # Simulate occlusions (tree canopy, shadows)
        for _ in range(8):
            cx, cy = np.random.randint(50, size - 50, 2)
            r = np.random.randint(15, 40)
            cv2.circle(mask, (cx, cy), r, 0, -1)   # black out a patch

        mask_bin = (mask > 127).astype(np.uint8)
        prob_map = mask.astype(np.float32) / 255.0
        return mask_bin, prob_map
