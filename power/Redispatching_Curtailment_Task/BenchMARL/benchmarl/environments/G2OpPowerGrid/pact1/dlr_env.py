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
        # Counters, so "the dial silently stopped applying" is visible rather
        # than invisible.  A high skip fraction means the limits are not
        # reaching the physics and the severity arm is really a sigma=0 arm.
        self._dlr_applied = 0
        self._dlr_skipped = 0

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
    def _set_limits(self, limits):
        """set_thermal_limit, guarded.

        grid2op refuses the call on an environment that is not initialised --
        freshly forked, or sitting on a game over -- and raises.  Inside a
        collector worker that exception kills the child and the parent sees only
        `EOFError` from the pipe, with no traceback pointing here.  Failing soft
        is right: the limits are re-applied on the very next step anyway.
        """
        try:
            self.env_g2op.set_thermal_limit(limits)
            return True
        except Exception:                                     # noqa: BLE001
            self._dlr_skipped += 1
            return False

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
        if self._set_limits(self._base_limits * ratio):
            self._dlr_applied += 1

    def step(self, gym_action):
        out = super().step(gym_action)
        # out is (obs, rew, done, truncated, info); done is a per-agent dict.
        done = out[2]
        finished = any(done.values()) if isinstance(done, dict) else bool(done)
        if not finished:
            self._apply_dlr(getattr(self, "_previous_act", None))
        return out

    def reset(self, *, seed=None, options=None):
        # Reset FIRST, then touch the limits.  Doing it the other way round
        # calls set_thermal_limit on an uninitialised env, which is exactly how
        # every collector worker died at startup.  Restoring the static ratings
        # here still gives each episode the same starting grid, because the
        # ambient ratio is re-applied immediately afterwards.
        obs, info = super().reset(seed=seed, options=options)
        if self.severity > 0.0 and self._base_limits is not None:
            self._set_limits(self._base_limits)
            self._apply_dlr(getattr(self, "_previous_act", None))
        return obs, info
