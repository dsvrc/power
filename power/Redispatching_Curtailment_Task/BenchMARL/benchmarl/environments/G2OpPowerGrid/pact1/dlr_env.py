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

    def __init__(self, severity=0.0, dlr_update_every=6, dlr_spatial=True,
                 dlr_weather="clock", dlr_geographic=False, dlr_points=8,
                 dlr_weather_seed=0, **kwargs):
        super().__init__(**kwargs)
        self.dlr_spatial = bool(dlr_spatial)
        self.severity = float(severity)
        # Weather obstacle. "clock" reproduces the shipped deterministic
        # climatology byte for byte and is the CONTROL condition; the others
        # remove the simplification that makes the disturbance a lookup table
        # over (month, hour). See pact1/weather.py.
        self.dlr_weather = str(dlr_weather)
        self.dlr_geographic = bool(dlr_geographic)
        self._weather = None
        self._episode_idx = -1
        self._dlr_weather_seed = int(dlr_weather_seed)
        self._dlr_points = int(dlr_points)
        # Ambient temperature moves on the scale of hours; grid2op steps are 5
        # minutes, so re-rating every step costs 6x more set_thermal_limit
        # calls than the physics justifies.  Measured in July at sigma=1, the
        # ratio moves at most 1.69% per hour (0.8335 at 13:00 -> 0.8247 at
        # 15:00), so a 30-minute interval costs under ~0.9% of the ratio
        # against an 18% derating being studied.  6 steps = 30 minutes.
        self._dlr_every = max(1, int(dlr_update_every))
        self._base_limits = None
        self._dlr_step = 0
        self._last_ratio = 1.0
        self._last_spread = 1.0
        self._last_A = 0.0
        # Counters, so "the dial silently stopped applying" is visible rather
        # than invisible.  A high skip fraction means the limits are not
        # reaching the physics and the severity arm is really a sigma=0 arm.
        self._dlr_applied = 0
        self._dlr_skipped = 0

        # Map each line to the zone whose local weather rates it.  Lines not
        # owned by any zone keep the regional (zone-averaged) ratio.
        self._line_zone = None
        if self.severity > 0.0 and self.dlr_spatial:
            try:
                from ..utils import ZONES_DICT
                names = sorted(ZONES_DICT.keys())
                n_line = len(self.env_g2op.get_thermal_limit())
                lz = np.full(n_line, -1, dtype=int)
                for zi, z in enumerate(names):
                    for l in ZONES_DICT[z]["line_in_zone_idx"]:
                        if 0 <= int(l) < n_line:
                            lz[int(l)] = zi
                self._line_zone = lz
                self._n_zones_dlr = len(names)
            except Exception:                                # noqa: BLE001
                self._line_zone = None

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
            if self._line_zone is not None:
                sp = dlr.spatial_spread(7, 15, self._n_zones_dlr, self.severity)
                print(f"  SPATIAL: per-zone ratings, {self._n_zones_dlr} zones, "
                      f"max/min spread at the summer peak = {sp:.3f}x")
                print(f"  (uniform = 1.000x; spread is what creates the "
                      f"coordination gap)")
            else:
                print("  UNIFORM ratings (spatial disabled)")

            # Weather obstacle. Built only at severity > 0; at severity == 0
            # nothing here runs and the task is stock grid2op byte for byte.
            if self.dlr_weather != "clock" or self.dlr_geographic:
                from . import weather as _wx
                layout = None
                try:
                    gl = self.env_g2op.grid_layout
                    layout = {i: gl[n] for i, n
                              in enumerate(self.env_g2op.name_sub) if n in gl}
                except Exception:                            # noqa: BLE001
                    layout = None
                n_line = len(self._base_limits)
                if self.dlr_geographic:
                    lp, npts = _wx.geographic_points(
                        n_line, self.env_g2op.line_or_to_subid,
                        self.env_g2op.line_ex_to_subid, layout=layout,
                        n_points=self._dlr_points)
                else:
                    lp, npts = np.zeros(n_line, dtype=int), 1
                self._weather = _wx.WeatherProcess(
                    n_line, lp, sigma=self.severity, mode=self.dlr_weather,
                    geographic=self.dlr_geographic, n_points=npts,
                    seed=self._dlr_weather_seed)
                print(f"  WEATHER OBSTACLE: {self._weather.describe()}")
                print(f"  geo layout: "
                      f"{'grid_layout coordinates' if layout else 'substation-id fallback'}")
                if self.dlr_weather == "wind":
                    print("  NOTE: wind mode has NO winter placebo -- a still "
                          "cold night derates.")
                print("  the realised ratio is NOT a function of the clock, so "
                      "it cannot be\n  memorised from (month, hour); it is "
                      "published to every arm alike.")
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
        self._last_A = dlr.driver_level(month, hour, self.severity)

        if self._weather is not None:
            # Stochastic and/or geographic weather. The process advances once
            # per DLR update -- this point is already gated to that cadence --
            # so its AR(1) retention is in units of dlr_update_every steps.
            self._weather.step()
            ratios = self._weather.line_ratios(month, hour)
            self._last_ratio = float(ratios.mean())
            self._last_spread = float(ratios.max() / max(ratios.min(), 1e-9))
            if self._set_limits(self._base_limits * ratios):
                self._dlr_applied += 1
            return

        if self._line_zone is not None:
            # PER-ZONE ratings.  Weather is not uniform over a 100 km region,
            # and this is what creates coordination pressure: when one zone is
            # derated and its neighbour is not, the efficient lever for the hot
            # zone's overload sits in the cool zone, and no agent can find it
            # from its own loading alone.
            n = self._n_zones_dlr
            per_zone = np.array(
                [dlr.ampacity_ratio_zone(month, hour, zi, n, self.severity)
                 for zi in range(n)], dtype=np.float64)
            regional = float(per_zone.mean())
            ratios = np.where(self._line_zone >= 0,
                              per_zone[np.clip(self._line_zone, 0, n - 1)],
                              regional)
            self._last_ratio = float(ratios.mean())
            self._last_spread = float(per_zone.max() / max(per_zone.min(), 1e-9))
            ok = self._set_limits(self._base_limits * ratios)
        else:
            ratio = dlr.ampacity_ratio(month, hour, self.severity)
            self._last_ratio = ratio
            self._last_spread = 1.0
            ok = self._set_limits(self._base_limits * ratio)
        if ok:
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
        if self._weather is not None:
            # A new episode is new weather. Keyed on the chronic actually
            # loaded, not on a running counter, so replaying scenario k gives
            # the weather it had last time -- which is what lets a severity or
            # method sweep pin scenarios and still have the weather be the same
            # (NS_FORM_SPEC E.2.5: an extra reset silently advancing the
            # scenario made every sigma run on different episodes).
            ep = getattr(self.env_g2op.chronics_handler, "get_id", lambda: None)()
            if not isinstance(ep, (int, np.integer)):
                self._episode_idx += 1
                ep = self._episode_idx
            self._weather.reset(int(ep) if isinstance(ep, (int, np.integer))
                                else abs(hash(str(ep))) % (2 ** 31 - 1))
        if self.severity > 0.0 and self._base_limits is not None:
            self._set_limits(self._base_limits)
            self._apply_dlr(getattr(self, "_previous_act", None))
        return obs, info
