import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_features,
        key_features,
        indices,
        weights=None,
        return_debug: bool = False,
    ):
        """
        query_features: decoder-side features (B, C, H, W)
        key_features: encoder-side features (B, C, H, W)
        indices: sampled patch indices (B, num_patches)
        """
        b, c, h, w = query_features.shape
        query_tokens = (
            query_features.permute(0, 2, 3, 1).contiguous().view(b, -1, c).float()
        )
        key_tokens = key_features.permute(0, 2, 3, 1).contiguous().view(b, -1, c).float()

        # Normalize tokens to unit sphere to remove the incentive to grow norms.
        query_tokens = F.normalize(query_tokens, dim=-1, eps=1e-6)
        key_tokens = F.normalize(key_tokens, dim=-1, eps=1e-6)

        indices = indices.to(query_features.device)
        expanded_idx = indices.unsqueeze(-1).expand(-1, -1, c)
        q = torch.gather(query_tokens, dim=1, index=expanded_idx)
        k_pos = torch.gather(key_tokens, dim=1, index=expanded_idx)

        logits = torch.matmul(q, key_tokens.transpose(1, 2)) / self.temperature
        labels = indices.view(-1)
        loss_per = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels, reduction="none"
        ).view(b, indices.size(1))

        if weights is not None:
            w = weights.to(query_features.device)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
            loss = (loss_per * w).sum(dim=1).mean()
        else:
            loss = loss_per.mean()
        positive_sim = F.cosine_similarity(q, k_pos, dim=-1).mean().detach()

        if not return_debug:
            return loss, positive_sim

        with torch.no_grad():
            pos_logits = logits.gather(
                dim=2, index=indices.unsqueeze(-1)
            ).squeeze(-1)
            neg_mean = (logits.sum(dim=2) - pos_logits) / (
                logits.size(2) - 1 + 1e-8
            )
            debug = {
                "pos_logit_mean": pos_logits.mean().item(),
                "neg_logit_mean": neg_mean.mean().item(),
                "margin": (pos_logits.mean() - neg_mean.mean()).item(),
                "pos_sim": positive_sim.item(),
            }

        return loss, positive_sim, debug
