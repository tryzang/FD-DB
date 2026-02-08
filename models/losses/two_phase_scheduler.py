from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


def _median(values):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return 0.5 * (float(sorted_vals[mid - 1]) + float(sorted_vals[mid]))


class StabilityTracker:
    """
    Track per-parameter stability over a sliding window.
    Consider stable when most samples fall inside median +/- tol.
    """

    def __init__(
        self,
        window_size: int = 50,
        min_coverage: float = 0.9,
        rel_tol: float = 0.05,
        abs_tol: float = 0.01,
        min_window: Optional[int] = None,
    ):
        self.window_size = max(1, int(window_size))
        self.min_coverage = float(min_coverage)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.min_window = max(1, int(min_window)) if min_window else None
        self.buffers: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, values: Dict[str, float]):
        for name, val in values.items():
            self.buffers[name].append(float(val))

    def window_status(self) -> Dict[str, float]:
        """Summarize coverage/length and stability for tracked buffers."""

        needed = self.min_window if self.min_window is not None else self.window_size
        if not self.buffers:
            return {
                "stable": False,
                "coverage_min": 0.0,
                "coverage_mean": 0.0,
                "len_min": 0,
                "len_max": 0,
                "len_needed": needed,
                "tol_mean": max(self.abs_tol, self.rel_tol),
                "median_mean": 0.0,
            }

        coverages = []
        lengths = []
        medians = []
        tolerances = []
        stable_flags = []
        for buf in self.buffers.values():
            length = len(buf)
            lengths.append(length)
            if not buf:
                coverages.append(0.0)
                medians.append(0.0)
                tolerances.append(max(self.abs_tol, self.rel_tol))
                stable_flags.append(False)
                continue
            med = _median(buf)
            tol = max(self.abs_tol, self.rel_tol * max(abs(med), 1e-6))
            within = sum(1 for v in buf if abs(v - med) <= tol)
            coverage = within / float(length)
            coverages.append(coverage)
            medians.append(med)
            tolerances.append(tol)
            stable_flags.append(length >= needed and coverage >= self.min_coverage)

        return {
            "stable": bool(stable_flags and all(stable_flags)),
            "coverage_min": min(coverages),
            "coverage_mean": sum(coverages) / len(coverages),
            "len_min": min(lengths),
            "len_max": max(lengths),
            "len_needed": needed,
            "tol_mean": sum(tolerances) / len(tolerances),
            "median_mean": sum(medians) / len(medians) if medians else 0.0,
        }

    def is_stable(self) -> bool:
        return bool(self.window_status().get("stable", False))


@dataclass
class PhaseControls:
    phase: str
    gates: Dict[str, float]
    lambdas: Dict[str, float]
    op_flags: Dict[str, float]
    gan_mode: Optional[str] = None
    use_edit_as_source: bool = True
    freeze_edit: bool = False


class TwoPhaseScheduler:
    """
    Hard two-phase schedule:
      - Phase edit: only editG active, D is trained; free branch is disabled.
      - When edit theta + loss stay stable over a window, switch to phase free.
    """

    def __init__(self, cfg: Optional[Dict]):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enable", False))
        self.phase = "edit"
        self.switch_step: Optional[int] = None

        crit = cfg.get("criteria", {}) or {}
        self.min_steps = int(crit.get("min_steps", 200))
        self.patience = int(crit.get("patience", 50))
        self.check_every = int(crit.get("check_every", 1))

        window = int(crit.get("window", 50))
        min_cov = float(crit.get("min_coverage", 0.9))
        rel_tol = float(crit.get("rel_tol", 0.05))
        abs_tol = float(crit.get("abs_tol", 0.01))
        min_window = crit.get("min_window")
        loss_rel_tol = float(crit.get("loss_rel_tol", rel_tol))
        loss_abs_tol = float(crit.get("loss_abs_tol", abs_tol))

        self.theta_tracker = StabilityTracker(
            window_size=window,
            min_coverage=min_cov,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            min_window=min_window,
        )
        self.loss_tracker = StabilityTracker(
            window_size=window,
            min_coverage=min_cov,
            rel_tol=loss_rel_tol,
            abs_tol=loss_abs_tol,
            min_window=min_window,
        )
        self._stable_steps = 0

        edit_cfg = cfg.get("edit_phase", {}) or {}
        free_cfg = cfg.get("free_phase", {}) or {}
        self.edit_controls = self._build_controls(edit_cfg, phase="edit")
        self.free_controls = self._build_controls(free_cfg, phase="free")

    def get_status(self, global_step: Optional[int] = None) -> Dict[str, float]:
        """Return stability window state for logging/diagnostics."""

        theta_status = self.theta_tracker.window_status()
        loss_status = self.loss_tracker.window_status()

        def _fill_ratio(status: Dict[str, float]) -> float:
            needed = float(status.get("len_needed", 1.0))
            if needed <= 0:
                return 0.0
            length = float(status.get("len_min", 0.0))
            return max(0.0, min(1.0, length / needed))

        theta_ok = bool(theta_status.get("stable", False))
        loss_ok = bool(loss_status.get("stable", False))
        min_steps_ok = False
        if global_step is not None:
            min_steps_ok = global_step >= self.min_steps

        window_ready = theta_ok and loss_ok
        window_ok = window_ready and min_steps_ok
        patience = max(1, self.patience)
        streak_ratio = min(1.0, float(self._stable_steps) / float(patience))

        return {
            "theta_ok": 1.0 if theta_ok else 0.0,
            "theta_cov": float(theta_status.get("coverage_min", 0.0)),
            "loss_ok": 1.0 if loss_ok else 0.0,
            "loss_cov": float(loss_status.get("coverage_min", 0.0)),
            "cov_target": float(self.theta_tracker.min_coverage),
            "len_fill": min(_fill_ratio(theta_status), _fill_ratio(loss_status)),
            "min_steps_ok": 1.0 if min_steps_ok else 0.0,
            "window_ready": 1.0 if window_ready else 0.0,
            "window_ok": 1.0 if window_ok else 0.0,
            "streak": float(self._stable_steps),
            "streak_ratio": streak_ratio,
            "patience": float(patience),
            "last_switch_step": float(self.switch_step) if self.switch_step is not None else -1.0,
            "phase_is_edit": 1.0 if self.phase == "edit" else 0.0,
        }

    @staticmethod
    def _build_controls(cfg: Dict, phase: str) -> PhaseControls:
        g_res = float(cfg.get("g_res", 0.0 if phase == "edit" else 1.0))
        op_all = float(cfg.get("op_all", 1.0 if phase == "edit" else 0.0))
        lambdas = {k: float(v) for k, v in (cfg.get("lambdas", {}) or {}).items()}
        gan_mode = cfg.get("gan_mode")
        use_edit_as_source = bool(cfg.get("use_edit_as_source", phase == "free"))
        freeze_edit = bool(cfg.get("freeze_edit", phase == "free"))
        return PhaseControls(
            phase=phase,
            gates={"g_res": g_res},
            lambdas=lambdas,
            op_flags={"all": op_all},
            gan_mode=gan_mode,
            use_edit_as_source=use_edit_as_source,
            freeze_edit=freeze_edit,
        )

    def get_controls(self) -> PhaseControls:
        return self.edit_controls if self.phase == "edit" else self.free_controls

    def _update_trackers(self, theta: Optional[Dict], loss_value: Optional[float]):
        if theta:
            means = {}
            for name, val in theta.items():
                if val is None:
                    continue
                try:
                    means[name] = float(val.mean().detach().item())
                except Exception:
                    continue
            if means:
                self.theta_tracker.update(means)
        if loss_value is not None:
            self.loss_tracker.update({"total": float(loss_value)})

    def _should_switch(self, global_step: int) -> bool:
        if global_step < self.min_steps:
            self._stable_steps = 0
            return False
        if not self.theta_tracker.is_stable():
            self._stable_steps = 0
            return False
        if not self.loss_tracker.is_stable():
            self._stable_steps = 0
            return False
        self._stable_steps += 1
        return self._stable_steps >= self.patience

    def update(self, global_step: int, theta: Optional[Dict], loss_value: Optional[float]) -> bool:
        """
        Returns True if phase switched on this call.
        """
        if not self.enabled or self.phase == "free":
            return False

        if self.check_every > 1 and (global_step % self.check_every != 0):
            self._update_trackers(theta, loss_value)
            return False

        self._update_trackers(theta, loss_value)
        if self._should_switch(global_step):
            self.phase = "free"
            self.switch_step = global_step
            return True
        return False
