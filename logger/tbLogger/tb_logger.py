import json
import os
from datetime import datetime
from typing import Dict, Optional, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

from .config import LoggerConfig
from .image_utils import add_text_label, denormalize, resize_to_max_side, make_comprehensive_panel
from models.ops.frequency import LowPassFilter, SigmaScaler, gaussian_blur


CORE_LOSS_TAGS: Sequence[str] = (
    "loss/D_total",
    "loss/G_total",
    "loss/total",
    "loss/weighted_gan",
    "loss/weighted_nce",
    "loss/weighted_lowfreq_anchor",
    "loss/weighted_edit_reg",
)


class TBLogger:
    """
    TensorBoard logger with minimal/default signals plus debug switches.
    """

    def __init__(self, cfg: Dict, device: torch.device):
        self.config = LoggerConfig.from_dict(cfg)
        self.device = device
        self.enabled = bool(self.config.enabled)
        self.writer: Optional[SummaryWriter] = None
        self.fixed_samples: Optional[Dict] = None
        self.fixed_noise: Optional[torch.Tensor] = None
        self.fixed_edit_noise: Optional[torch.Tensor] = None
        self._frequency_cfg: Optional[Dict] = None
        self._lowfreq_cfg: Optional[Dict] = None
        self._lowpass: Optional[LowPassFilter] = None
        self._sigma_scaler: Optional[SigmaScaler] = None

    def start_run(self, hparams: Optional[Dict] = None, meta: Optional[Dict] = None):
        if not self.enabled:
            return
        os.makedirs(self.config.log_dir, exist_ok=True)
        run_name = self.config.run_name
        if not run_name:
            run_name = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
            self.config.run_name = run_name
        run_dir = os.path.join(self.config.log_dir, run_name)
        purge_step = self.config.purge_step
        if purge_step is not None:
            try:
                purge_step = int(purge_step)
            except (TypeError, ValueError):
                purge_step = None
        if purge_step is not None and purge_step < 0:
            purge_step = None
        self.writer = SummaryWriter(run_dir, purge_step=purge_step)
        if hparams and self.config.log_hparams:
            try:
                self.writer.add_text("hparams/json", json.dumps(hparams, indent=2))
            except Exception:
                pass
        if meta and self.config.log_meta:
            try:
                self.writer.add_text("meta", json.dumps(meta, indent=2))
            except Exception:
                pass

    def set_fixed_samples(self, fixed_samples: Optional[Dict]):
        self.fixed_samples = fixed_samples
        # Reset cached noise so panels stay paired with the current fixed samples.
        self.fixed_noise = None
        self.fixed_edit_noise = None

    def should_log_images(self, step: int) -> bool:
        return self.enabled and (step % self.config.images_every_steps == 0)

    def log_train(
        self,
        step: int,
        epoch: int,
        scalars: Dict[str, float],
        logits_fake: Optional[torch.Tensor] = None,
        logits_real: Optional[torch.Tensor] = None,
        models: Optional[Dict[str, torch.nn.Module]] = None,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        generator: Optional[torch.nn.Module] = None,
        nce_debug: Optional[Dict] = None,
        gates: Optional[Dict[str, float]] = None,
        lambdas: Optional[Dict[str, float]] = None,
        op_flags: Optional[Dict[str, float]] = None,
        theta: Optional[Dict[str, torch.Tensor]] = None,
        theta_stats: Optional[Dict[str, float]] = None,
        frequency_cfg: Optional[Dict] = None,
        lowfreq_cfg: Optional[Dict] = None,
        phase_status: Optional[Dict[str, float]] = None,
        switch_event: bool = False,
    ):
        if not self.enabled or self.writer is None:
            return

        cfg = self.config
        writer = self.writer
        if frequency_cfg is not None:
            self._frequency_cfg = frequency_cfg
            padding = str((frequency_cfg or {}).get("padding", "reflect")).lower()
            self._lowpass = LowPassFilter(padding=padding)
            self._sigma_scaler = SigmaScaler.from_config(frequency_cfg)
        if lowfreq_cfg is not None:
            self._lowfreq_cfg = lowfreq_cfg

        if cfg.log_losses and step % cfg.scalars_every_steps == 0:
            for name, val in scalars.items():
                if name in CORE_LOSS_TAGS or name.startswith("loss/weighted_"):
                    writer.add_scalar(name, float(val), step)

            if logits_real is not None:
                writer.add_scalar("D/logit_real_mean", float(logits_real.mean()), step)
            if logits_fake is not None:
                writer.add_scalar("D/logit_fake_mean", float(logits_fake.mean()), step)
            if cfg.log_d_hist and step % cfg.debug_every_steps == 0:
                if logits_fake is not None:
                    writer.add_histogram("D/hist/logits_fake", logits_fake.detach().cpu(), step)
                if logits_real is not None:
                    writer.add_histogram("D/hist/logits_real", logits_real.detach().cpu(), step)

            if cfg.log_schedule:
                self._log_phase_schedule(
                    writer=writer,
                    step=step,
                    phase_status=phase_status,
                    switch_event=switch_event,
                )

            if cfg.log_theta_stats and theta_stats:
                for k, v in theta_stats.items():
                    writer.add_scalar(f"edit_G/{k}", float(v), step)
                edit_values = {
                    k[len("theta/") : -len("_mean")]: float(v)
                    for k, v in theta_stats.items()
                    if k.startswith("theta/") and k.endswith("_mean")
                }
                if edit_values:
                    writer.add_scalars("edit_G/values", edit_values, step)

        if cfg.log_theta_hist and theta is not None and step % cfg.debug_every_steps == 0:
            for name, value in theta.items():
                if value is None:
                    continue
                writer.add_histogram(
                    f"edit_G/hist/theta_{name}", value.detach().cpu(), step
                )


        if cfg.log_images_panel and self.should_log_images(step):
            self._log_panels(
                step, 
                generator, 
                gates=gates, 
                op_flags=op_flags,
            )

        if cfg.flush_every_steps > 0 and step % cfg.flush_every_steps == 0:
            writer.flush()

    def _log_phase_schedule(
        self,
        writer: SummaryWriter,
        step: int,
        phase_status: Optional[Dict[str, float]],
        switch_event: bool,
    ):
        if not phase_status:
            return

        def _cleanup(values: Dict[str, Optional[float]]) -> Dict[str, float]:
            return {k: float(v) for k, v in values.items() if v is not None}

        stability = _cleanup(
            {
                "theta_ok": phase_status.get("theta_ok"),
                "loss_ok": phase_status.get("loss_ok"),
                "window_ready": phase_status.get("window_ready"),
                "window_ok": phase_status.get("window_ok"),
                "min_steps_ok": phase_status.get("min_steps_ok"),
                "len_fill": phase_status.get("len_fill"),
                "cov_target": phase_status.get("cov_target"),
                "theta_cov": phase_status.get("theta_cov"),
                "loss_cov": phase_status.get("loss_cov"),
                "streak_ratio": phase_status.get("streak_ratio"),
            }
        )
        if stability:
            writer.add_scalars("schedule/stability", stability, step)

        mode = _cleanup(
            {
                "mode_edit": phase_status.get("phase_is_edit"),
                "switch_event": 1.0 if switch_event else 0.0,
                "window_ok": phase_status.get("window_ok"),
                "window_ready": phase_status.get("window_ready"),
                "streak": phase_status.get("streak"),
                "patience": phase_status.get("patience"),
                "len_fill": phase_status.get("len_fill"),
            }
        )
        if mode:
            writer.add_scalars("schedule/mode_switch", mode, step)

    def _get_fixed_noise(self, a: torch.Tensor, generator: torch.nn.Module) -> Optional[torch.Tensor]:
        if self.fixed_noise is not None:
            return self.fixed_noise

        # Stick to a deterministic noise tensor for fixed panels to avoid resampling artifacts.
        noise_channels = getattr(generator, "noise_channels", None)
        if noise_channels is None or int(noise_channels) <= 0:
            return None
        noise_scale = float(getattr(generator, "noise_scale", 1.0))
        g = torch.Generator(device=a.device)
        g.manual_seed(int(getattr(self.config, "seed", 42)))
        noise_shape = (a.shape[0], int(noise_channels), a.shape[2], a.shape[3])
        self.fixed_noise = noise_scale * torch.randn(
            noise_shape, device=a.device, dtype=a.dtype, generator=g
        )
        return self.fixed_noise

    def _get_fixed_edit_noise(self, a: torch.Tensor) -> torch.Tensor:
        if self.fixed_edit_noise is not None:
            return self.fixed_edit_noise
        g = torch.Generator(device=a.device)
        seed = int(getattr(self.config, "seed", 42)) + 1
        g.manual_seed(seed)
        # Generate 3 channels so it can support both RGB and grayscale grain
        noise_shape = (a.shape[0], 3, a.shape[2], a.shape[3])
        self.fixed_edit_noise = torch.randn(
            noise_shape, device=a.device, dtype=a.dtype, generator=g
        )
        return self.fixed_edit_noise

    def _log_panels(
        self,
        step: int,
        generator: Optional[torch.nn.Module],
        gates: Optional[Dict[str, float]] = None,
        op_flags: Optional[Dict[str, float]] = None,
    ):
        if generator is None or self.fixed_samples is None:
            return
        writer = self.writer
        if writer is None:
            return

        a = self.fixed_samples["A"].to(self.device)
        b = self.fixed_samples["B"].to(self.device)
        a_ann = self.fixed_samples.get("A_ann") or []

        freq_cfg = self._frequency_cfg or getattr(generator, "config", {}).get("frequency", {})
        lowfreq_cfg = self._lowfreq_cfg or {}
        padding = str((freq_cfg or {}).get("padding", "reflect")).lower()
        self._lowpass = self._lowpass or LowPassFilter(padding=padding)
        self._sigma_scaler = self._sigma_scaler or SigmaScaler.from_config(freq_cfg)
        lowpass = self._lowpass
        sigma_scaler = self._sigma_scaler

        sigmas = []
        y01 = None
        y_edit01 = None
        r_free_hp = None
        r_free_fused = None
        final = None
        with torch.no_grad():
            edit_noise = self._get_fixed_edit_noise(a)
            out = generator(a, gates=gates, op_flags=op_flags, edit_noise=edit_noise)
            fake = out.y if hasattr(out, "y") else out[0]
            final = getattr(out, "y_final", fake)
            y_edit = getattr(out, "y_edit", None)
            r_free = getattr(out, "r_free", None)
            theta = getattr(out, "theta", None)
            if r_free is None:
                r_free = fake - a
            if r_free is not None:
                sigma_override = getattr(generator, "sigma_fuse", None)
                sigma_fuse = sigma_scaler.resolve(r_free, sigma=sigma_override)
                if sigma_fuse > 0:
                    r_free_hp = r_free - lowpass(r_free, sigma_fuse)
                else:
                    r_free_hp = r_free
                g_res_val = getattr(generator, "g_res_default", 1.0)
                if gates is not None and gates.get("g_res") is not None:
                    g_res_val = float(gates["g_res"])
                g_res_val = max(0.0, min(1.0, g_res_val))
                r_free_fused = r_free_hp * g_res_val
            if self.config.log_edit_images and y_edit is not None:
                sigmas = lowfreq_cfg.get("sigmas") or [
                    sigma_scaler.sigma_base if sigma_scaler.use_scale else sigma_scaler.sigma
                ]
                y01 = (fake + 1.0) * 0.5
                y_edit01 = (y_edit + 1.0) * 0.5
                blur_sigma = None
                if theta is not None and "blur_sigma" in theta:
                    blur_sigma = theta["blur_sigma"]

        for idx in range(min(len(a), self.config.num_fixed_samples)):
            src = denormalize(a[idx: idx + 1]).cpu()
            tgt = denormalize(b[idx: idx + 1]).cpu()
            gen_norm = final[idx: idx + 1] if final is not None else fake[idx: idx + 1]
            gen = denormalize(gen_norm).cpu()

            src = resize_to_max_side(src, self.config.image_max_side)
            tgt = resize_to_max_side(tgt, self.config.image_max_side)
            gen = resize_to_max_side(gen, self.config.image_max_side)

            diff_val = (gen_norm - a[idx: idx + 1]).clamp(-1.0, 1.0)
            diff_img = denormalize(diff_val).cpu()
            diff_img = resize_to_max_side(diff_img, self.config.image_max_side)
            
            r_free_panel = None
            if r_free_fused is not None and idx < r_free_fused.shape[0]:
                r_free_panel = r_free_fused[idx: idx + 1].clamp(-1.0, 1.0).cpu()
                r_free_panel = resize_to_max_side(r_free_panel, self.config.image_max_side)
            
            y_edit_img = None
            if y_edit is not None and idx < y_edit.shape[0]:
                y_edit_img = denormalize(y_edit[idx: idx + 1]).cpu()
                y_edit_img = resize_to_max_side(y_edit_img, self.config.image_max_side)
            
            grain_vis = None
            if theta is not None and "grain_amp" in theta and self.config.log_edit_images:
                amp = out.theta["grain_amp"][idx: idx + 1]
                size_tensor = out.theta.get("grain_size")
                grain_size_val = float(
                    size_tensor[idx].item()
                ) if size_tensor is not None else float(
                    getattr(generator.edit_branch.edit_ops, "grain_size_default", 1.0)
                )
                sigma_scale = float(getattr(generator.edit_branch.edit_ops, "grain_sigma_scale", 1.0))
                sigma_val = max(grain_size_val * sigma_scale, float(getattr(generator.edit_branch.edit_ops, "grain_sigma_eps", 1e-3)))

                raw_noise = edit_noise[idx: idx + 1]
                rgb = bool(getattr(generator.edit_branch.edit_ops, "grain_rgb", False))
                c = 3 if rgb else 1
                if raw_noise.shape[1] != c:
                    if raw_noise.shape[1] == 1:
                        raw_noise = raw_noise.repeat(1, c, 1, 1)
                    else:
                        raw_noise = raw_noise[:, :c]

                vis_noise = gaussian_blur(raw_noise, sigma_val, padding=str((self._frequency_cfg or {}).get("padding", "reflect")))
                vis_amp = amp.view(-1, 1, 1, 1)
                grain_vis = 0.8 + vis_noise * vis_amp
                grain_vis = grain_vis.clamp(0.0, 1.0).cpu()
                grain_vis = resize_to_max_side(grain_vis, self.config.image_max_side)

            blur_vis = None
            if (
                self.config.log_edit_panel
                and self.config.log_edit_images
                and y_edit01 is not None
                and lowpass is not None
                and blur_sigma is not None
                and idx < blur_sigma.shape[0]
            ):
                sigma_val = float(blur_sigma[idx].item())
                sigma_eps = float(
                    getattr(getattr(generator, "edit_branch", None), "edit_ops", None).blur_sigma_eps
                    if getattr(getattr(generator, "edit_branch", None), "edit_ops", None) is not None
                    else 1e-3
                )
                if sigma_val > sigma_eps:
                    blur_img = lowpass(y_edit01[idx: idx + 1], sigma_val).clamp(0.0, 1.0).cpu()
                    blur_vis = resize_to_max_side(blur_img, self.config.image_max_side)
            
            comprehensive_panel = make_comprehensive_panel(
                source=src,
                diff=diff_img,
                final=gen,
                y_edit=y_edit_img,
                r_free_abs=r_free_panel,
                real=tgt,
                grain_vis=None,
                spatial_weight=None,
            )
            writer.add_image(f"image_panel/panel_{idx}", comprehensive_panel, step)
            free_tiles = []
            free_base = add_text_label(
                gen.squeeze(0),
                "Free output",
                pos=(5, 5),
                font_scale=1.4,
                thickness=2,
            )
            free_tiles.append(free_base.clamp(0.0, 1.0))
            if y_edit is not None and idx < y_edit.shape[0]:
                free_minus_edit = (gen_norm - y_edit[idx: idx + 1]).clamp(-1.0, 1.0)
                free_diff = denormalize(free_minus_edit).cpu()
                free_diff = resize_to_max_side(free_diff, self.config.image_max_side)
                diff_tile = add_text_label(
                    free_diff.squeeze(0),
                    "Free - Edit",
                    pos=(5, 5),
                    font_scale=1.4,
                    thickness=2,
                )
                free_tiles.append(diff_tile.clamp(0.0, 1.0))

            if len(free_tiles) == 1:
                free_panel = free_tiles[0]
            else:
                free_panel = torch.cat(free_tiles, dim=2)

            writer.add_image(f"free_G/residual_{idx}", free_panel, step)
            self._log_edit_panel(
                writer=writer,
                step=step,
                idx=idx,
                grain_vis=grain_vis,
                blur_vis=blur_vis,
            )

    def _log_edit_panel(
        self,
        writer: SummaryWriter,
        step: int,
        idx: int,
        grain_vis: Optional[torch.Tensor],
        blur_vis: Optional[torch.Tensor],
    ):
        if not self.config.log_edit_panel:
            return

        tiles = []
        for name, img in (("Grain", grain_vis), ("Blur", blur_vis)):
            if img is None:
                continue
            if img.dim() == 4:
                img = img.squeeze(0)
            labeled = add_text_label(
                img.clamp(0.0, 1.0).cpu(),
                name,
                pos=(5, 5),
                font_scale=1.4,
                thickness=2,
            )
            tiles.append(labeled)

        if not tiles:
            return

        # Stack horizontally: (3, H, W_total)
        panel = torch.cat(tiles, dim=2)
        writer.add_image(f"edit_G/panel_{idx}", panel, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def flush(self):
        if self.writer is not None:
            self.writer.flush()
