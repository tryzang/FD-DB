import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .gan import generator_loss
from .patchnce import PatchNCELoss


class LossManager:
    def __init__(self, config, generator):
        self.lambda_gan = config.get("lambda_gan", 1.0)
        self.lambda_nce = config.get("lambda_nce_attn", config.get("lambda_nce", 1.0))
        self.lambda_identity = config.get("lambda_identity", 0.0)
        self.gan_mode = config.get("gan_mode", "hinge")

        sampler_cfg = config.get("patch_sampler", {}) or {}
        self.num_patches = int(sampler_cfg.get("num_patches", 256))
        self.sampling_enabled = bool(sampler_cfg.get("enabled", True))
        if not self.sampling_enabled:
            self.num_patches = 0

        nce_cfg = config.get("patchnce", {})
        layer_cfg = nce_cfg.get("layers", nce_cfg.get("layer", -1))
        if isinstance(layer_cfg, (list, tuple)):
            self.nce_layers = [int(x) for x in layer_cfg]
        else:
            self.nce_layers = [int(layer_cfg)]
        self.nce_mlp_dim = int(nce_cfg.get("mlp_dim", 256))
        self.nce_detach_key = bool(nce_cfg.get("detach_key", True))
        self.nce_identity = bool(nce_cfg.get("identity", True))

        self.generator = generator
        self.patch_nce = PatchNCELoss(temperature=nce_cfg.get("temperature", 0.07))
        self.identity_criterion = nn.L1Loss()

        self.nce_mlps = nn.ModuleDict()
        for idx, ch in enumerate(generator.enco_channels):
            mlp = nn.Sequential(
                nn.Conv2d(ch, self.nce_mlp_dim, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nce_mlp_dim, self.nce_mlp_dim, kernel_size=1, bias=True),
            )
            self.nce_mlps[str(idx)] = mlp
        try:
            gen_device = next(generator.parameters()).device
            self.patch_nce.to(gen_device)
            self.nce_mlps.to(gen_device)
        except StopIteration:
            pass

    def get_nce_parameters(self):
        """
        Return an iterator over trainable PatchNCE projector parameters.
        """
        return self.nce_mlps.parameters()

    def set_nce_requires_grad(self, requires_grad: bool):
        """
        Toggle gradients for PatchNCE projectors for phase-wise freeze/unfreeze.
        """
        for param in self.nce_mlps.parameters():
            param.requires_grad_(requires_grad)

    @staticmethod
    def _sample_uniform_indices(feat: torch.Tensor, num_patches: int) -> torch.Tensor:
        b, _, h, w = feat.shape
        total = h * w
        return torch.randint(0, total, (b, num_patches), device=feat.device)

    def _project_feat(self, feat: torch.Tensor, layer_idx: int, detach: bool = False) -> torch.Tensor:
        """
        Apply CUT-style MLP projection (1x1 conv stack), optionally detaching key features.
        """
        layer_key = str(layer_idx)
        if layer_key in self.nce_mlps:
            proj = self.nce_mlps[layer_key](feat)
        else:
            proj = feat
        if detach:
            proj = proj.detach()
        return proj

    def compute_generator_losses(
        self,
        logits_fake,
        enc_feats,
        dec_feats,
        target_enc_feats=None,
        target_dec_feats=None,
        identity_input=None,
        return_debug: bool = False,
        lambda_multipliers: Optional[Dict] = None,
        generator_kwargs: Optional[Dict] = None,
        gan_mode: Optional[str] = None,
    ):
        losses = {}
        total_loss = 0.0
        debug_info = {}
        lambda_multipliers = lambda_multipliers or {}
        lambda_gan = self.lambda_gan * float(lambda_multipliers.get("gan", 1.0))
        lambda_nce = self.lambda_nce * float(lambda_multipliers.get("nce", 1.0))
        lambda_identity = self.lambda_identity * float(
            lambda_multipliers.get("identity", 1.0)
        )
        generator_kwargs = generator_kwargs or {}
        gan_mode = gan_mode or self.gan_mode

        if lambda_gan > 0:
            losses["gan"] = generator_loss(logits_fake, None, mode=gan_mode)
            weighted_gan = lambda_gan * losses["gan"]
            total_loss = total_loss + weighted_gan
            losses["weighted_gan"] = weighted_gan

        if (
            lambda_nce > 0
            and self.num_patches > 0
            and self.sampling_enabled
            and enc_feats
            and dec_feats
        ):
            nce_losses = []
            nce_pos = []
            nce_debugs = []

            layers = self.nce_layers or [-1]
            for layer_idx in layers:
                idx = min(layer_idx if layer_idx >= 0 else len(dec_feats) - 1, len(dec_feats) - 1)
                q_raw = dec_feats[idx]
                k_raw = enc_feats[idx]
                q_feat = self._project_feat(q_raw, idx, detach=False)
                k_feat = self._project_feat(k_raw, idx, detach=self.nce_detach_key)
                patch_indices = self._sample_uniform_indices(q_feat, self.num_patches)
                if return_debug:
                    nce_loss, pos_sim, nce_dbg = self.patch_nce(
                        q_feat,
                        k_feat,
                        patch_indices,
                        return_debug=True,
                    )
                    nce_debugs.append(nce_dbg or {})
                else:
                    nce_loss, pos_sim = self.patch_nce(
                        q_feat,
                        k_feat,
                        patch_indices,
                    )
                nce_losses.append(nce_loss)
                nce_pos.append(pos_sim)

            if (
                self.nce_identity
                and target_enc_feats
                and target_dec_feats
            ):
                for layer_idx in layers:
                    idx = min(layer_idx if layer_idx >= 0 else len(target_dec_feats) - 1, len(target_dec_feats) - 1)
                    q_raw = target_dec_feats[idx]
                    k_raw = target_enc_feats[idx]
                    q_feat = self._project_feat(q_raw, idx, detach=False)
                    k_feat = self._project_feat(k_raw, idx, detach=self.nce_detach_key)
                    patch_indices = self._sample_uniform_indices(q_feat, self.num_patches)
                    if return_debug:
                        nce_loss, pos_sim, nce_dbg = self.patch_nce(
                            q_feat,
                            k_feat,
                            patch_indices,
                            return_debug=True,
                        )
                        nce_debugs.append(nce_dbg or {})
                    else:
                        nce_loss, pos_sim = self.patch_nce(
                            q_feat,
                            k_feat,
                            patch_indices,
                        )
                    nce_losses.append(nce_loss)
                    nce_pos.append(pos_sim)

            if nce_losses:
                losses["nce"] = torch.stack(nce_losses).mean()
                losses["nce_pos_sim"] = torch.stack(nce_pos).mean().detach()
                weighted_nce = lambda_nce * losses["nce"]
                total_loss = total_loss + weighted_nce
                losses["weighted_nce"] = weighted_nce
                if return_debug and nce_debugs:
                    debug_info["nce"] = nce_debugs[-1]

        if (
            lambda_identity > 0
            and identity_input is not None
            and identity_input.shape[1]
            == self.generator.config.get("in_channels", identity_input.shape[1])
        ):
            with torch.no_grad():
                identity_target = identity_input
            identity_output, *_ = self.generator.free_branch(identity_input, out_mode="full")
            losses["identity"] = self.identity_criterion(identity_output, identity_target)
            weighted_identity = lambda_identity * losses["identity"]
            total_loss = total_loss + weighted_identity
            losses["weighted_identity"] = weighted_identity

        losses["total"] = total_loss
        if debug_info:
            losses["nce_debug"] = debug_info.get("nce")
        return total_loss, losses
