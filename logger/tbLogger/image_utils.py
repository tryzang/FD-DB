from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def denormalize(img: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """Inverse of Normalize((0.5),(0.5)) used in data pipeline."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    mean_t = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=img.device).view(1, -1, 1, 1)
    out = img * std_t + mean_t
    return out.clamp(0.0, 1.0)


def resize_to_max_side(img: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return img
    _, _, h, w = img.shape
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    new_h = max(int(h * scale), 1)
    new_w = max(int(w * scale), 1)
    return F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _mask_to_outline(mask: torch.Tensor, thickness: int = 1) -> torch.Tensor:
    """
    Convert a binary mask to an outline-only mask.
    A light outline keeps the panel colors visible while preserving localization.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask_1c = (mask[:1] > 0).float()  # force single-channel for edge detection
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype)
    neighbors = torch.nn.functional.conv2d(mask_1c.unsqueeze(0), kernel, padding=1)
    edges = (mask_1c > 0) & (neighbors < 9)
    edges = edges.float().squeeze(0)
    if thickness > 1:
        # Dilate edges slightly to make them visible at TensorBoard resolution.
        pad = thickness - 1
        edges = torch.nn.functional.max_pool2d(
            edges.unsqueeze(0), kernel_size=thickness * 2 - 1, stride=1, padding=pad
        ).squeeze(0)
    return edges


def _apply_mask_overlay(
    img: torch.Tensor,
    mask: torch.Tensor,
    color: Sequence[float] = (1.0, 0.0, 0.0),
    alpha: float = 0.4,
    mode: str = "fill",
) -> torch.Tensor:
    """Overlay a binary mask onto an image (fill/edge/off)."""
    if mode == "off":
        return img
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mode == "edge":
        mask = _mask_to_outline(mask, thickness=2)
    if mask.shape[0] == 1:
        mask = mask.repeat(3, 1, 1)
    mask = (mask > 0).float()
    color_t = torch.tensor(color, device=img.device, dtype=img.dtype).view(3, 1, 1)
    return img * (1 - alpha * mask) + color_t * (alpha * mask)


def _apply_bboxes_overlay(img: torch.Tensor, bboxes: Sequence[Sequence[int]], color=(1.0, 0.6, 0.0), thickness: int = 2) -> torch.Tensor:
    """Draw simple rectangle outlines on a CHW image tensor."""
    if not bboxes:
        return img
    out = img.clone()
    c, h, w = out.shape
    color_t = torch.tensor(color, device=img.device, dtype=img.dtype).view(c, 1, 1)
    for bbox in bboxes:
        if len(bbox) < 4:
            continue
        x, y, bw, bh = [int(v) for v in bbox[:4]]
        x2 = min(x + bw, w)
        y2 = min(y + bh, h)
        x = max(x, 0)
        y = max(y, 0)
        for t in range(thickness):
            xs = slice(max(x - t, 0), min(x + t + 1, w))
            xe = slice(max(x2 - t - 1, 0), min(x2 + t, w))
            ys = slice(max(y - t, 0), min(y + t + 1, h))
            ye = slice(max(y2 - t - 1, 0), min(y2 + t, h))
            out[:, ys, xs] = color_t
            out[:, ye, xs] = color_t
            out[:, ys, xe] = color_t
            out[:, ye, xe] = color_t
    return out


def make_panel(
    source: torch.Tensor,
    fake: torch.Tensor,
    real: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bboxes: Optional[Sequence[Sequence[int]]] = None,
    layout: str = "2x2",
    mask_overlay_mode: str = "fill",
    mask_overlay_alpha: float = 0.4,
    mask_overlay_color: Sequence[float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """
    Build a small panel grid: source | fake(+overlay) | real | diff.
    """
    src = source
    tgt = real
    gen = fake

    if mask is not None and mask.numel() > 0:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[1:] != gen.shape[1:]:
            mask = F.interpolate(
                mask.unsqueeze(0), size=gen.shape[1:], mode="nearest"
            ).squeeze(0)
        gen_overlay = _apply_mask_overlay(
            gen,
            mask,
            color=mask_overlay_color,
            alpha=mask_overlay_alpha,
            mode=str(mask_overlay_mode).lower(),
        )
    elif bboxes:
        gen_overlay = _apply_bboxes_overlay(gen, bboxes)
    else:
        gen_overlay = gen

    diff = torch.abs(gen - src).mean(dim=0, keepdim=True).repeat(3, 1, 1)
    imgs: List[torch.Tensor] = [src, gen_overlay, tgt, diff]
    nrow = 2 if layout == "2x2" else len(imgs)
    grid = make_grid(imgs, nrow=nrow)
    return grid


def make_edit_panel(
    y_edit: torch.Tensor,
    r_free_abs: torch.Tensor,
    delta: torch.Tensor,
    lowfreq_diff: Optional[torch.Tensor] = None,
    spatial_weight: Optional[torch.Tensor] = None,
    r_free_modulated: Optional[torch.Tensor] = None,
    edge_map: Optional[torch.Tensor] = None,
    texture_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build an edit-focused panel grid with spatial guidance visualization.
    Row 1: y_edit | r_free_abs | delta
    Row 2: lowfreq_diff | spatial_weight | r_modulated
    Row 3: edge_map | texture_map | (empty)
    """
    imgs = [y_edit, r_free_abs, delta]
    
    if lowfreq_diff is not None:
        imgs.append(lowfreq_diff)
    else:
        imgs.append(None)
    
    if spatial_weight is not None:
        imgs.append(spatial_weight)
    else:
        imgs.append(None)
        
    if r_free_modulated is not None:
        imgs.append(r_free_modulated)
    else:
        imgs.append(None)
    
    if edge_map is not None:
        imgs.append(edge_map)
    else:
        imgs.append(None)
    
    if texture_map is not None:
        imgs.append(texture_map)
    else:
        imgs.append(None)

    processed = []
    for img in imgs:
        if img is None:
            h, w = 64, 64
            if len(processed) > 0:
                h, w = processed[0].shape[1], processed[0].shape[2]
            blank = torch.zeros(3, h, w)
            processed.append(blank)
            continue
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        processed.append(img)

    return make_grid(processed, nrow=3)


def to_heatmap(img: torch.Tensor, colormap: str = "hot") -> torch.Tensor:
    """
    Convert a single-channel tensor into a heatmap.
    Args:
        img: (1, H, W) or (B, 1, H, W)
        colormap: heatmap style ("hot", "jet", "viridis", ...)
    Returns:
        (3, H, W) or (B, 3, H, W) RGB heatmap
    """
    import matplotlib.cm as cm
    import numpy as np
    
    if img.ndim == 4:
        img = img.squeeze(1)  # (B, H, W)
        batch_mode = True
    elif img.ndim == 3:
        img = img.squeeze(0)  # (H, W)
        batch_mode = False
    else:
        raise ValueError(f"Expected 3 or 4D tensor, got {img.ndim}D")
    
    img_np = img.detach().cpu().numpy()
    if img_np.max() > img_np.min():
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    else:
        img_np = np.zeros_like(img_np)
    
    cmap = cm.get_cmap(colormap)
    
    if batch_mode:
        # (B, H, W) -> list of (B, 3, H, W)
        heatmaps = []
        for b in range(img_np.shape[0]):
            hm = cmap(img_np[b])  # (H, W, 4) RGBA
            hm = torch.from_numpy(hm[:, :, :3].transpose(2, 0, 1)).float()  # (3, H, W)
            heatmaps.append(hm)
        return torch.stack(heatmaps, dim=0)  # (B, 3, H, W)
    else:
        # (H, W) -> (3, H, W)
        hm = cmap(img_np)  # (H, W, 4)
        hm = torch.from_numpy(hm[:, :, :3].transpose(2, 0, 1)).float()  # (3, H, W)
        return hm


def add_text_label(img: torch.Tensor, text: str, pos: tuple = (5, 5), 
                   font_scale: float = 0.5, thickness: int = 1,
                   text_color: tuple = (1.0, 1.0, 1.0),
                   bg_color: tuple = (0.0, 0.0, 0.0), bg_alpha: float = 0.7) -> torch.Tensor:
    """
    Add a text label to the top-left corner using PIL.
    Args:
        img: (3, H, W) CHW RGB image in [0, 1]
        text: label content
        pos: text position (left, top)
        font_scale: font scale factor
        thickness: text thickness
        text_color: (R, G, B) in [0, 1]
        bg_color: (R, G, B) in [0, 1]
        bg_alpha: background alpha
    Returns:
        labeled image (3, H, W)
    """
    from PIL import Image, ImageDraw, ImageFont
    
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='RGB')
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                  int(14 * font_scale))
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox(pos, text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    bg_pos = (pos[0] - 2, pos[1] - 2, pos[0] + text_width + 2, pos[1] + text_height + 2)
    bg_color_rgba = tuple(int(c * 255) for c in bg_color) + (int(255 * bg_alpha),)
    draw.rectangle(bg_pos, fill=bg_color_rgba)
    
    text_color_rgb = tuple(int(c * 255) for c in text_color)
    draw.text(pos, text, font=font, fill=text_color_rgb)
    
    img_with_label = torch.from_numpy(np.array(pil_img)).float() / 255.0
    return img_with_label.permute(2, 0, 1)  # (3, H, W)


def make_comprehensive_panel(
    source: torch.Tensor,
    diff: torch.Tensor,
    final: torch.Tensor,
    y_edit: torch.Tensor,
    r_free_abs: torch.Tensor,
    real: torch.Tensor,
    grain_vis: Optional[torch.Tensor] = None,
    spatial_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build a 3x3 summary panel:
    Row 1: [Source] [Diff] [Final]
    Row 2: [EditG] [FreeG] [Real]
    Row 3: [Grain] [Guide Weight] [Empty]
    
    Args:
        source: (1, 3, H, W) source image
        diff: (1, 3, H, W) Final - Source
        final: (1, 3, H, W) final output
        y_edit: (1, 3, H, W) EditG output
        r_free_abs: (1, 1, H, W) or (1, 3, H, W) absolute FreeG residual
        real: (1, 3, H, W) real image
        grain_vis: (1, 3, H, W) optional grain visualization
        spatial_weight: (1, 1, H, W) or (1, H, W), optional guidance weights
    
    Returns:
        grid image of shape (3, 3*H + 2*padding, H + 2*padding)
    """
    import numpy as np
    
    labels = [
        "Source", "Diff", "Final",
        "Edit", "Free(res)", "Real",
        "", "", ""
    ]
    
    def normalize_img(img):
        if img is None:
            return None
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img.clamp(0.0, 1.0)
    
    imgs = [
        normalize_img(source),
        normalize_img(diff),
        normalize_img(final),
        normalize_img(y_edit),
        normalize_img(r_free_abs),
        normalize_img(real),
        normalize_img(grain_vis) if grain_vis is not None else torch.zeros(3, source.shape[-2], source.shape[-1]),
        None,  # spatial_weight must be converted to a heatmap
        None,
    ]
    
    if spatial_weight is not None:
        sm = spatial_weight
        if sm.ndim == 4:
            sm = sm.squeeze(0)
        if sm.ndim == 3:
            sm = sm.unsqueeze(1)  # (1, H, W) -> (1, 1, H, W)
        sm_heatmap = to_heatmap(sm, colormap="hot")
        if sm_heatmap.ndim == 4:
            sm_heatmap = sm_heatmap.squeeze(0)
        imgs[7] = sm_heatmap
    
    target_size = source.shape[-2:]  # (H, W)
    for i in range(len(imgs)):
        if imgs[i] is None:
            imgs[i] = torch.zeros(3, *target_size)
        else:
            if imgs[i].shape[-2:] != target_size:
                imgs[i] = F.interpolate(
                    imgs[i].unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
    
    for i, img in enumerate(imgs):
        imgs[i] = add_text_label(img, labels[i], pos=(5, 5), font_scale=1.8)
    
    return make_grid(imgs, nrow=3, padding=2, pad_value=0.0)
