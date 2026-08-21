"""Environment layer that applies Dynamic Line Rating to EVERY arm.

Deliberately sits below PACT-1 in the class hierarchy:

    PZMAEnvRecoDNLimit          (stock task)
        PZMAEnvDLR              (+ dynamic ratings)   <- MAPPO / MASAC use this
            PACT1Env            (+ estimator/compensator)

so a given severity produces identical physics for every algorithm.  A dial only
the method's own arm experienced would be worthless as evidence.
"""
import numpy as np

from ..PZMAEnvWithHeuristics import PZMAEnvRecoDNLimit
from . import dlr


class PZMAEnvDLR(PZMAEnvRecoDNLimit):
    """Stock task + ambient-driven thermal ratings.

    severity == 0 short-circuits every code path below, so the environment is
    byte-identical to the published task -- not "approximately", literally the
    same calls in the same order.
    """

    def __init__(self, severity=0.0, dlr_update_every=1, **kwargs):
        super().__init__(**kwargs)
        self.severity = float(severity)
        self._dlr_every = max(1, int(dlr_update_every))
        self._base_limits = None
        self._dlr_step = 0
        self._last_ratio = 1.0
        self._last_A = 0.0

        if self.severity > 0.0:
            try:
                self._base_limits = np.array(
                    self.env_g2op.get_thermal_limit(), dtype=np.float64, copy=True)
            except Exception as exc:                      # noqa: BLE001
                raise RuntimeError(
                    "severity > 0 needs the environment's static thermal "
                    f"limits and they could not be read: {exc}") from exc
            print("=" * 72)
            print("  DYNAMIC LINE RATING ACTIVE".ljust(72))
            print(f"  {dlr.describe(self.severity)}")
            print(f"  applied to ALL arms; n_line={len(self._base_limits)}")
            print("=" * 72)

    # ------------------------------------------------------------------
    def _apply_dlr(self, g2op_obs):
        """Rescale every line's limit by the current ampacity ratio."""
        if self.severity <= 0.0 or self._base_limits is None or g2op_obs is None:
            return
        self._dlr_step += 1
        if (self._dlr_step % self._dlr_every) != 0:
            return
        month = float(getattr(g2op_obs, "month", 1))
        hour = float(getattr(g2op_obs, "hour_of_day", 0))
        ratio = dlr.ampacity_ratio(month, hour, self.severity)
        self._last_ratio = ratio
        self._last_A = dlr.driver_level(month, hour, self.severity)
        # One scalar ratio across all lines: ambient is a regional quantity, and
        # a per-line ratio would need per-line weather this dataset does not
        # have.  Uniform scaling also keeps the dial from re-shaping WHICH line
        # binds, so sigma changes severity without changing the task's identity.
        self.env_g2op.set_thermal_limit(self._base_limits * ratio)

    def step(self, gym_action):
        out = super().step(gym_action)
        self._apply_dlr(getattr(self, "_previous_act", None))
        return out

    def reset(self, *, seed=None, options=None):
        # Restore static ratings before reset so chronic selection and the
        # heuristic warmup see the same grid every episode.
        if self.severity > 0.0 and self._base_limits is not None:
            self.env_g2op.set_thermal_limit(self._base_limits)
        obs, info = super().reset(seed=seed, options=options)
        self._apply_dlr(getattr(self, "_previous_act", None))
        return obs, info
