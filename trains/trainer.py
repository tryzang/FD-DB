import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.bop.translation import get_bop_translation_dataloaders  # noqa: E402
from logger import InfoLogger  # noqa: E402
from logger.tbLogger import PanelDataProvider, TBLogger  # noqa: E402
from models.discriminator.discriminator import PatchDiscriminator  # noqa: E402
from models.generator.generator import Generator  # noqa: E402
from models.losses.edit_regularizers import (
    compute_edit_regularizers,
    compute_theta_stats,
)  # noqa: E402
from models.losses.gan import discriminator_loss  # noqa: E402
from models.losses.loss_manager import LossManager  # noqa: E402
from models.losses.lowfreq_anchor import LowFreqAnchorLoss  # noqa: E402
from models.losses.two_phase_scheduler import PhaseControls, TwoPhaseScheduler  # noqa: E402


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config["training"].get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = InfoLogger(name="Trainer")
        opt_cfg = self.config["training"]["optimizer"]
        self.opt_cfg = opt_cfg
        self.opt_free_cfg = opt_cfg.get("free", {}) or {}
        self.d_step_interval_edit = max(int(opt_cfg.get("d_step_interval", 1)), 1)
        self.d_step_interval_free = max(
            int(self.opt_free_cfg.get("d_step_interval", self.d_step_interval_edit)), 1
        )
        self.d_step_interval = self.d_step_interval_edit
        self._nce_params_added = False
        self._init_amp()
        self._build_models()
        self._build_scheduler()
        self._build_optimizers()
        self._apply_start_phase()
        self._build_data()
        self._build_tb_logger()
        self.current_phase = "edit"

    def _init_amp(self):
        amp_cfg = self.config["training"].get("amp", {})
        self.use_amp = bool(amp_cfg.get("enabled", self.device.type == "cuda"))
        dtype_str = str(amp_cfg.get("dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        # Use a no-op context when AMP is disabled or when running on non-CUDA devices.
        self.autocast_ctx = (
            (lambda: autocast(dtype=self.amp_dtype))
            if (self.use_amp and self.device.type == "cuda")
            else (lambda: nullcontext())
        )

    def _build_models(self):
        gen_cfg = dict(self.config["generator"])
        if "frequency" in self.config:
            gen_cfg["frequency"] = self.config.get("frequency", {}) or {}
        self.generator = Generator(gen_cfg).to(self.device)

        self.disc_cfg = dict(self.config["discriminator"])
        self._build_discriminator(phase="edit")
        self.loss_manager = LossManager(
            self.config["training"]["loss"], self.generator
        )
        self.loss_manager.set_nce_requires_grad(False)
        loss_cfg = self.config["training"].get("loss", {}) or {}
        freq_cfg = self.config.get("frequency", {}) or {}
        self.lowfreq_anchor = LowFreqAnchorLoss(
            loss_cfg.get("lowfreq_anchor", {}), freq_cfg
        ).to(self.device)

    def _build_optimizers(self):
        opt_cfg = self.opt_cfg
        betas = tuple(opt_cfg.get("betas", [0.5, 0.999]))
        params_g = list(self.generator.parameters())
        self.optimizer_g = optim.Adam(
            params_g, lr=opt_cfg.get("lr_g", 2e-4), betas=betas
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=opt_cfg.get("lr_d", 2e-4),
            betas=betas,
        )
        self.last_d_loss = torch.tensor(0.0, device=self.device)
        self._phase_last = "edit"

    def _reset_free_branch(self):
        """
        Reset free-branch weights and clear related optimizer state without touching the edit branch.
        """

        def _reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.generator.free_branch.apply(_reset_module)

        if hasattr(self, "optimizer_g") and self.optimizer_g is not None:
            free_params = {p for p in self.generator.free_branch.parameters()}
            state = self.optimizer_g.state
            for param in list(state.keys()):
                if param in free_params:
                    state.pop(param, None)

    def _build_discriminator(self, phase: str):
        """
        Build discriminator by phase (edit: default affine; free: optional affine_free).
        Rebuild the discriminator optimizer together with the discriminator.
        """
        if phase == "free":
            affine_free = self.disc_cfg.get(
                "affine_free", self.disc_cfg.get("affine", True)
            )
            disc_cfg = dict(self.disc_cfg)
            disc_cfg["affine"] = affine_free
            lr_d = self.opt_free_cfg.get("lr_d", self.opt_cfg.get("lr_d", 2e-4))
            betas_d = tuple(self.opt_free_cfg.get("betas", self.opt_cfg.get("betas", [0.5, 0.999])))
            self.d_step_interval = self.d_step_interval_free
        else:
            disc_cfg = dict(self.disc_cfg)
            lr_d = self.opt_cfg.get("lr_d", 2e-4)
            betas_d = tuple(self.opt_cfg.get("betas", [0.5, 0.999]))
            self.d_step_interval = self.d_step_interval_edit

        self.discriminator = PatchDiscriminator(disc_cfg).to(self.device)
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=betas_d,
        )
        self.last_d_loss = torch.tensor(0.0, device=self.device)
        self._phase_last = phase

    def _apply_phase_optimizer_settings(self, phase: str):
        """
        Switch hyperparameters and update interval by phase; rebuild D by phase and update G lr/betas.
        """
        if phase == "free":
            if self._phase_last != "free":
                self._reset_free_branch()
            lr_g = self.opt_free_cfg.get("lr_g", self.opt_cfg.get("lr_g", 2e-4))
            betas_g = tuple(self.opt_free_cfg.get("betas", self.opt_cfg.get("betas", [0.5, 0.999])))
            if self._phase_last != "free":
                self._build_discriminator(phase="free")
            self.loss_manager.set_nce_requires_grad(True)
            if not getattr(self, "_nce_params_added", False):
                nce_params = list(self.loss_manager.get_nce_parameters())
                if nce_params:
                    self.optimizer_g.add_param_group(
                        {"params": nce_params, "lr": lr_g, "betas": betas_g}
                    )
                self._nce_params_added = True
        else:
            lr_g = self.opt_cfg.get("lr_g", 2e-4)
            betas_g = tuple(self.opt_cfg.get("betas", [0.5, 0.999]))
            if self._phase_last != "edit":
                self._build_discriminator(phase="edit")
            self.loss_manager.set_nce_requires_grad(False)

        for param_group in self.optimizer_g.param_groups:
            param_group["lr"] = lr_g
            param_group["betas"] = betas_g

    def _apply_start_phase(self):
        """
        Set the initial phase from config (supports starting from free) and ensure NCE MLP is in the optimizer.
        """
        train_cfg = self.config.get("training", {}) or {}
        start_phase = str(train_cfg.get("start_phase", "edit")).lower()
        if start_phase not in ("edit", "free"):
            start_phase = "edit"
        if getattr(self, "phase_scheduler", None) is not None and getattr(
            self.phase_scheduler, "enabled", False
        ):
            self.phase_scheduler.phase = start_phase
        self._apply_phase_optimizer_settings(start_phase)

    def _build_scheduler(self):
        phase_cfg = self.config.get("phase_schedule", {}) or {}
        self.phase_scheduler = TwoPhaseScheduler(phase_cfg)

    def _build_data(self):
        data_cfg = self.config["data"]
        self.data_mode = "bop"
        (
            self.train_loader,
            self.val_loader,
            self.syn_adapter,
            self.real_adapter,
            self.bop_transform,
        ) = get_bop_translation_dataloaders(data_cfg, return_adapters=True)

    def _build_tb_logger(self):
        tb_cfg = self.config.get("tb_logger", {})
        self.tb_logger = TBLogger(tb_cfg, device=self.device)
        if not self.tb_logger.enabled:
            return

        opt_cfg = self.config["training"].get("optimizer", {})
        loss_cfg = self.config["training"].get("loss", {})
        lowfreq_cfg = loss_cfg.get("lowfreq_anchor", {}) or {}
        hparams = {
            "lr_g": opt_cfg.get("lr_g"),
            "lr_d": opt_cfg.get("lr_d"),
            "lambda_gan": loss_cfg.get("lambda_gan"),
            "lambda_nce_attn": loss_cfg.get("lambda_nce_attn"),
            "lambda_identity": loss_cfg.get("lambda_identity"),
            "lambda_lowfreq_anchor": lowfreq_cfg.get("lambda_base"),
            "lowfreq_sigmas": lowfreq_cfg.get("sigmas"),
        }
        self.tb_logger.start_run(
            hparams=hparams,
            meta={"device": str(self.device), "data_mode": getattr(self, "data_mode", "unknown")},
        )

        if getattr(self, "data_mode", "") == "bop":
            provider = PanelDataProvider(
                syn_adapter=self.syn_adapter,
                real_adapter=self.real_adapter,
                transform=self.bop_transform,
                num_samples=self.tb_logger.config.num_fixed_samples,
                seed=tb_cfg.get("seed", 42),
            )
            provider.set_fixed_samples()
            self.panel_provider = provider
            self.tb_logger.set_fixed_samples(provider.get_fixed_samples())

    def _maybe_save_checkpoint(self, global_step: int, epoch: int) -> None:
        ckpt_cfg = self.config["training"].get("checkpoint", {}) or {}
        if not ckpt_cfg.get("enabled", False):
            return
        save_every = int(ckpt_cfg.get("save_every_steps", 0))
        if save_every <= 0 or (global_step + 1) % save_every != 0:
            return
        ckpt_dir = Path(ckpt_cfg.get("dir", "run/checkpoint"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"step{global_step + 1:07d}_epoch{epoch + 1:04d}.pt"
        payload = {
            "step": global_step,
            "epoch": epoch,
            "config": self.config,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "loss_manager_nce_mlps": self.loss_manager.nce_mlps.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(payload, ckpt_path)
        self.logger.info(f"Checkpoint saved to {ckpt_path}")

    @staticmethod
    def _set_requires_grad(module, requires_grad: bool):
        for param in module.parameters():
            param.requires_grad_(requires_grad)

    def train(self):
        epochs = self.config["training"].get("epochs", 1)
        log_every = self.config["training"].get("log_every", 10)
        max_steps = self.config["training"].get("max_steps")
        steps_per_epoch = len(self.train_loader)
        if max_steps is not None:
            steps_per_epoch = min(int(max_steps), steps_per_epoch)
        total_steps = max(1, epochs * steps_per_epoch)
        try:
            for epoch in range(epochs):
                data_iterable = self.train_loader
                loader_desc = "bop"

                for step, batches in enumerate(data_iterable):
                    if max_steps is not None and step >= max_steps:
                        break
                    global_step = epoch * steps_per_epoch + step
                    if loader_desc == "bop":
                        source_images = batches["A"].to(self.device)
                        target_images = batches["B"].to(self.device)

                    current_phase = None
                    phase_controls: Optional[PhaseControls] = None
                    use_edit_as_source = False
                    gates = None
                    op_flags = None
                    lambda_multipliers = None
                    gan_mode_override = None
                    if getattr(self.phase_scheduler, "enabled", False):
                        phase_controls = self.phase_scheduler.get_controls()
                        current_phase = phase_controls.phase
                        gates = phase_controls.gates
                        op_flags = phase_controls.op_flags
                        lambda_multipliers = phase_controls.lambdas
                        gan_mode_override = phase_controls.gan_mode
                        use_edit_as_source = (
                            phase_controls.use_edit_as_source if current_phase == "free" else False
                        )
                        if current_phase != self._phase_last:
                            self._apply_phase_optimizer_settings(current_phase)
                            self._phase_last = current_phase

                    with self.autocast_ctx():
                        gen_out = self.generator(
                            source_images,
                            gates=gates,
                            op_flags=op_flags,
                            spatial_weight=None,
                            use_edit_as_source=use_edit_as_source,
                        )
                        fake = gen_out.y
                        enc_feats = gen_out.enc_feats
                        dec_feats = gen_out.dec_feats
                        target_enc_feats = None
                        target_dec_feats = None
                        lambda_nce_eff = self.loss_manager.lambda_nce * float(
                            (lambda_multipliers or {}).get("nce", 1.0)
                        )
                        need_target_nce = (
                            lambda_nce_eff > 0
                            and self.loss_manager.num_patches > 0
                            and self.loss_manager.sampling_enabled
                        )
                        if need_target_nce:
                            tgt_out, tgt_enc, tgt_dec = self.generator.free_branch(
                                target_images, out_mode="full"
                            )
                            target_enc_feats = tgt_enc
                            target_dec_feats = tgt_dec

                    # Train D
                    update_d = (step % self.d_step_interval == 0)
                    logits_fake = None
                    logits_real = None
                    if update_d:
                        self._set_requires_grad(self.discriminator, True)
                        self.optimizer_d.zero_grad(set_to_none=True)
                        with self.autocast_ctx():
                            logits_fake = self.discriminator(fake.detach())
                            logits_real = self.discriminator(target_images)
                            loss_d = discriminator_loss(
                                logits_real,
                                logits_fake,
                                mode=gan_mode_override or self.loss_manager.gan_mode,
                            )
                        self.scaler.scale(loss_d).backward()
                        self.scaler.step(self.optimizer_d)
                        self.last_d_loss = loss_d.detach()
                    else:
                        loss_d = self.last_d_loss

                    # Train G
                    # Freeze D so its params/grads remain from the D step only; prevents D gradients
                    # from mixing into grad_norm_D when logging after the G backward pass.
                    self._set_requires_grad(self.discriminator, False)
                    self.optimizer_g.zero_grad(set_to_none=True)
                    with self.autocast_ctx():
                        logits_fake_g = self.discriminator(fake)
                        total_g, loss_dict = self.loss_manager.compute_generator_losses(
                            logits_fake=logits_fake_g,
                            enc_feats=enc_feats,
                            dec_feats=dec_feats,
                            target_enc_feats=target_enc_feats,
                            target_dec_feats=target_dec_feats,
                            identity_input=target_images
                            if (
                                self.loss_manager.lambda_identity > 0
                                and (
                                    lambda_multipliers is None
                                    or lambda_multipliers.get("identity", 1.0) > 0
                                )
                            )
                            else None,
                            return_debug=False,
                            lambda_multipliers=lambda_multipliers,
                            generator_kwargs={"gates": gates, "op_flags": op_flags},
                            gan_mode=gan_mode_override,
                        )
                        edit_losses = {}
                        edit_total = None
                        loss_cfg = self.config["training"].get("loss", {})
                        lowfreq_cfg = loss_cfg.get("lowfreq_anchor", {}) or {}
                        if (
                            lowfreq_cfg.get("enable", False)
                            and getattr(self.generator, "param_edit_enabled", False)
                            and gen_out.y_edit is not None
                        ):
                            lowfreq_raw, lowfreq_details = self.lowfreq_anchor(
                                fake, gen_out.y_edit, return_details=True
                            )
                            lambda_low = float(lowfreq_cfg.get("lambda_base", 1.0))
                            multiplier = None
                            if lambda_multipliers is not None:
                                multiplier = lambda_multipliers.get("lowfreq_anchor")
                            if multiplier is None and lowfreq_cfg.get(
                                "use_gate_coupling", True
                            ):
                                alpha = float(lowfreq_cfg.get("alpha", 1.0))
                                g_res_val = 0.0
                                if gen_out.gates is not None:
                                    g_res_val = float(gen_out.gates.get("g_res", 0.0))
                                multiplier = max(g_res_val, 0.0) ** alpha
                            if multiplier is None:
                                multiplier = 1.0
                            lambda_low *= float(multiplier)
                            weighted_low = lambda_low * lowfreq_raw
                            total_g = total_g + weighted_low
                            loss_dict["lowfreq_anchor_raw"] = lowfreq_raw
                            loss_dict["lowfreq_anchor_weighted"] = weighted_low
                            if lowfreq_details:
                                for name, val in lowfreq_details.items():
                                    loss_dict[f"lowfreq_anchor_{name}"] = lowfreq_raw.new_tensor(
                                        val
                                    )

                        edit_cfg = loss_cfg.get("edit_reg", {}) or {}
                        if (
                            getattr(self.generator, "param_edit_enabled", False)
                            and edit_cfg.get("enable", False)
                            and gen_out.theta is not None
                        ):
                            edit_total, edit_losses = compute_edit_regularizers(
                                gen_out.theta,
                                self.generator.param_specs,
                                weights=edit_cfg.get("weights", {}),
                                op_flags=op_flags,
                            )
                            if edit_total is not None:
                                lambda_edit = float(loss_cfg.get("lambda_edit_reg", 1.0))
                                if lambda_multipliers is not None:
                                    lambda_edit *= float(lambda_multipliers.get("edit_reg", 1.0))
                                weighted_edit = lambda_edit * edit_total
                                total_g = total_g + weighted_edit
                                loss_dict["edit_reg"] = edit_total
                                loss_dict["weighted_edit_reg"] = weighted_edit
                                for name, val in edit_losses.items():
                                    if name == "total":
                                        continue
                                    loss_dict[f"edit_reg_{name}"] = val
                        loss_dict["total"] = total_g
                    self.scaler.scale(total_g).backward()
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()
                    self._set_requires_grad(self.discriminator, True)

                    if step % log_every == 0:
                        metrics = {
                            "D": loss_d,
                            "G": total_g,
                        }
                        for name, val in loss_dict.items():
                            if hasattr(val, "item"):
                                metrics[name] = val
                        self.logger.progress(epoch, step, **metrics)

                    switch_event = False
                    phase_status = None
                    if getattr(self.phase_scheduler, "enabled", False):
                        loss_val = None
                        if "total" in loss_dict and hasattr(loss_dict["total"], "item"):
                            loss_val = float(loss_dict["total"].item())
                        switch_event = self.phase_scheduler.update(
                            global_step,
                            theta=gen_out.theta,
                            loss_value=loss_val,
                        )
                        if switch_event and self.phase_scheduler.get_controls().freeze_edit:
                            if getattr(self.generator, "edit_branch", None) is not None:
                                self._set_requires_grad(self.generator.edit_branch, False)
                        phase_status = self.phase_scheduler.get_status(global_step)

                    if hasattr(self, "tb_logger") and self.tb_logger.enabled:
                        scalars = {
                            "loss/D_total": loss_d.item(),
                            "loss/G_total": total_g.item(),
                        }
                        if current_phase is not None:
                            scalars["phase/edit_active"] = 1.0 if current_phase == "edit" else 0.0
                        for name, val in loss_dict.items():
                            if name == "nce_debug":
                                continue
                            if hasattr(val, "item"):
                                scalars[f"loss/{name}"] = val.item()
                        theta_stats = None
                        if gen_out.theta is not None and getattr(self.tb_logger.config, "log_theta_stats", False):
                            theta_stats = compute_theta_stats(gen_out.theta)

                        logits_fake_log = logits_fake if logits_fake is not None else logits_fake_g
                        if logits_fake_log is not None:
                            logits_fake_log = logits_fake_log.detach()
                        logits_real_log = logits_real
                        if logits_real_log is not None:
                            logits_real_log = logits_real_log.detach()

                        self.tb_logger.log_train(
                            step=step,
                            epoch=epoch,
                            scalars=scalars,
                            logits_fake=logits_fake_log,
                            logits_real=logits_real_log,
                            models={"G": self.generator, "D": self.discriminator},
                            optimizers={"G": self.optimizer_g, "D": self.optimizer_d},
                            generator=self.generator,
                            nce_debug=loss_dict.get("nce_debug"),
                            gates=gates,
                            lambdas=lambda_multipliers,
                            op_flags=op_flags,
                            theta=gen_out.theta if gen_out.theta is not None else None,
                            theta_stats=theta_stats,
                            frequency_cfg=self.config.get("frequency", {}) or {},
                            lowfreq_cfg=lowfreq_cfg,
                            phase_status=phase_status,
                            switch_event=switch_event,
                        )

                    self._maybe_save_checkpoint(global_step, epoch)
        finally:
            if hasattr(self, "tb_logger"):
                self.tb_logger.flush()
                self.tb_logger.close()

    @staticmethod
    def from_yaml(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return Trainer(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Attention-guided domain adaptation trainer"
    )
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "default.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()
    trainer = Trainer.from_yaml(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
