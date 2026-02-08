import torch
import torch.nn.functional as F


def discriminator_loss(logits_real, logits_fake, mode="hinge"):
    if mode == "hinge":
        loss_real = F.relu(1.0 - logits_real).mean()
        loss_fake = F.relu(1.0 + logits_fake).mean()
        return loss_real + loss_fake
    target_real = torch.ones_like(logits_real)
    target_fake = torch.zeros_like(logits_fake)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, target_real)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, target_fake)
    return loss_real + loss_fake


def generator_loss(logits_fake, attn_map=None, mode="hinge"):
    if mode == "hinge":
        loss = -logits_fake
        if attn_map is not None:
            weight = F.interpolate(
                attn_map.unsqueeze(1),
                size=logits_fake.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            weight = weight / (weight.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
            loss = loss * weight
        return loss.mean()
    target = torch.ones_like(logits_fake)
    loss = F.binary_cross_entropy_with_logits(logits_fake, target)
    if attn_map is not None:
        weight = F.interpolate(
            attn_map.unsqueeze(1),
            size=logits_fake.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        weight = weight / (weight.mean(dim=[1, 2, 3], keepdim=True) + 1e-6)
        loss = (loss * weight.squeeze(1)).mean()
    return loss
