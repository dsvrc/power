"""Obstacles: weather that a policy cannot memorise from the clock.

WHY THIS MODULE EXISTS
----------------------
dlr.ambient_temp() is documented as "Ambient temperature from the calendar
alone. Deterministic in (month, hour) ... Both fields are already in the
observation."  That is climatology -- the EXPECTED temperature for a month and
hour -- and it makes the whole disturbance a closed-form function of two numbers
every agent observes directly.  A policy does not need to coordinate, estimate,
or watch its peers; it needs a lookup table over 24 hours x 12 five-minute
slots, and 2M frames is far more than enough to build one.

That is not a fair test of a coordination method, and it is also not the
physics.  Real ambient deviates from climatology by fronts, cloud cover and
wind.  The giveaway is that Dynamic Line Rating exists as an engineering
practice at all: if ampacity were predictable from the calendar, operators
would publish a seasonal rating table -- which is exactly the STATIC rating
grid2op already ships.  dlr.py says it "restores physics the simulator
abstracts away" and then abstracts away the part that makes DLR necessary.

So every obstacle here is a removed simplification, defensible without
reference to any method's score:

  clock       the shipped model.  Deterministic climatology.  Kept as the
              CONTROL: it is the condition under which a blind learner should
              be able to match a compensator, and showing that it does is
              worth more than hiding it.
  stochastic  climatology + an AR(1) deviation.  Weather is not its own
              monthly mean.
  wind        stochastic, plus wind-driven convection.  IEEE 738 ampacity
              depends more strongly on wind speed than on ambient temperature,
              and wind has almost no diurnal cycle to memorise.  The shipped
              model uses only the most predictable input.

              COST, and it must be stated in any table this mode appears in:
              THIS MODE HAS NO PLACEBO REGIME.  Winter survives as a placebo
              under `clock` and `stochastic` because the temperature term
              clips at 1.0 (January pre-dawn is ~2.6 degC, giving a pre-clip
              ratio of 1.147, which no plausible temperature excursion
              overturns).  Wind does overturn it: a still cold night genuinely
              derates, and measured here the January ratio reaches 0.863.
              That is the correct physics -- ampacity depends on convection,
              not only on ambient -- but it costs the "the dial provably does
              nothing in February" defence.  NS_FORM_SPEC Part E's rule
              applies: where a row is missing, be explicit rather than quiet.
              Use `stochastic` when the placebo matters more than the extra
              unpredictability.

and orthogonally:

  geographic  the deviation is a spatial field over the substations' own
              coordinates, not a deterministic wave indexed by zone NUMBER.
              Two consequences: it cannot be memorised, and it is independent
              of N -- dlr.zone_phase(zone_index, n_zones) makes the weather a
              function of how finely you partitioned, which silently couples an
              N sweep to the physics.

INVARIANTS, ASSERTED IN selfcheck()
-----------------------------------
  * sigma = 0 returns EXACTLY 1.0 in every mode, at every time, for every line.
    Short-circuited before any RNG is touched, so the stock task is recovered
    byte for byte and no obstacle can leak into the sigma=0 arm.
  * ratios are clipped at 1.0 (never generous -- NS_FORM_SPEC B.2's uprating
    trap: two-sided scaling made higher sigma EASIER on winter chronics).
  * the process is reproducible: same (seed, episode) gives the same weather,
    so an arm difference can never be a weather difference.
  * winter still clips to 1.0, so the placebo regime survives.

WHAT THIS DOES NOT DO
---------------------
It does not change the reward, the action space, or what any arm observes, and
it is applied below the method in the class hierarchy, so every algorithm sees
identical physics.  It also does not make the disturbance unknowable: the
REALISED rating is published to every arm (see dlr_env), exactly as a real DLR
system publishes ratings to the control room.  What stays unavailable is what
peers are doing about it right now, which is the thing under test.
"""
import numpy as np

from . import dlr

# Day-to-day departure of ambient from its climatological mean.  A few degrees
# is the right order for a mid-latitude summer: Ile-de-France July daily maxima
# scatter roughly +/- 4 degC around the monthly mean.
A_WEATHER = 4.0            # degC, stationary std of the deviation at sigma=1

# AR(1) retention per DLR update (default update = 30 min).  0.93 gives an
# e-folding time of ~7 hours: synoptic weather, not noise.  A process that
# decorrelates every step would be unlearnable AND uncompensable, which is not
# the regime being studied (NS_FORM_SPEC A.4).
AR_RETAIN = 0.93

# Wind.  IEEE 738 forced convection: the conductor's heat-transfer coefficient
# scales roughly with sqrt(perpendicular wind speed), so ampacity carries a
# sqrt(h) factor.  The static rating assumes a light breeze.
V_REF = 0.61               # m/s, the 2 ft/s the standard rating assumes
V_FLOOR = 0.15             # m/s, still-air offset so h stays finite
WIND_LOG_STD = 0.55        # lognormal scatter of wind at sigma=1
WIND_RETAIN = 0.88         # wind decorrelates faster than temperature

MODES = ("clock", "stochastic", "wind")


class WeatherProcess:
    """Per-line ampacity ratios, optionally stochastic and/or spatial.

    Owned by the environment, stepped once per DLR update.  Deterministic given
    (seed, episode_index), so two arms on the same seed see identical weather
    and any difference between them is the method.
    """

    def __init__(self, n_line, line_point, sigma=1.0, mode="clock",
                 geographic=False, n_points=1, seed=0, derate_only=True):
        self.n_line = int(n_line)
        # line -> field point. For the non-geographic case every line maps to
        # point 0 and the field is regional.
        self.line_point = np.asarray(line_point, dtype=int)
        self.sigma = float(sigma)
        self.mode = str(mode)
        if self.mode not in MODES:
            raise ValueError(f"unknown weather mode {mode!r}; expected {MODES}")
        self.geographic = bool(geographic)
        self.n_points = max(1, int(n_points))
        self.seed = int(seed)
        self.derate_only = bool(derate_only)

        self._episode = -1
        self._rng = None
        self._dev = np.zeros(self.n_points)        # degC, temperature departure
        self._logv = np.zeros(self.n_points)       # log wind multiplier
        self.reset(0)

    # -- state ----------------------------------------------------------

    @property
    def stochastic(self):
        return self.sigma > 0.0 and self.mode in ("stochastic", "wind")

    def reset(self, episode_index):
        """New episode, new weather realisation -- reproducibly.

        Seeded from (seed, episode) rather than from a running counter so that
        replaying episode k gives the same weather whatever order episodes ran
        in, which is what lets a severity sweep pin scenarios (NS_FORM_SPEC
        E.2.5) without the weather drifting underneath it.
        """
        self._episode = int(episode_index)
        self._rng = np.random.RandomState(
            (self.seed * 1_000_003 + self._episode) % (2 ** 31 - 1))
        if not self.stochastic:
            self._dev = np.zeros(self.n_points)
            self._logv = np.zeros(self.n_points)
            return
        # Start from the stationary distribution, not from zero: otherwise
        # every episode opens at exactly climatology and the first hours are
        # systematically easier than the rest.
        s = self.sigma * A_WEATHER
        self._dev = self._rng.normal(0.0, s, size=self.n_points)
        if self.mode == "wind":
            self._logv = self._rng.normal(
                0.0, self.sigma * WIND_LOG_STD, size=self.n_points)
        else:
            self._logv = np.zeros(self.n_points)

    def step(self):
        """Advance one DLR update."""
        if not self.stochastic:
            return
        s = self.sigma * A_WEATHER
        a = AR_RETAIN
        self._dev = (a * self._dev
                     + np.sqrt(max(1.0 - a * a, 0.0)) * s
                     * self._rng.normal(size=self.n_points))
        if self.mode == "wind":
            b = WIND_RETAIN
            sw = self.sigma * WIND_LOG_STD
            self._logv = (b * self._logv
                          + np.sqrt(max(1.0 - b * b, 0.0)) * sw
                          * self._rng.normal(size=self.n_points))

    # -- physics --------------------------------------------------------

    def _point_ratio(self, month, hour):
        """Ampacity ratio at each field point."""
        if self.sigma == 0.0:
            # EXACT identity, before any RNG or arithmetic that could round.
            return np.ones(self.n_points)

        t_clim = dlr.ambient_temp(month, hour, sigma=self.sigma)
        t_amb = t_clim + self._dev                      # zeros unless stochastic
        headroom = np.maximum(dlr.T_CONDUCTOR_MAX - t_amb, 1.0)
        r = np.sqrt(headroom / (dlr.T_CONDUCTOR_MAX - dlr.T_RATING_REF))

        if self.mode == "wind":
            # Forced convection: h ~ sqrt(v), ampacity ~ sqrt(h).
            v = V_REF * np.exp(self._logv)
            r = r * ((v + V_FLOOR) / (V_REF + V_FLOOR)) ** 0.25

        return np.minimum(r, 1.0) if self.derate_only else r

    def line_ratios(self, month, hour):
        """Per-line ampacity ratio, length n_line."""
        if self.sigma == 0.0:
            return np.ones(self.n_line)
        pr = self._point_ratio(month, hour)
        return pr[np.clip(self.line_point, 0, self.n_points - 1)]

    def describe(self):
        bits = [f"mode={self.mode}"]
        if self.geographic:
            bits.append(f"geographic field, {self.n_points} points")
        else:
            bits.append("uniform over the region")
        if self.stochastic:
            bits.append(f"AR(1) dev std {self.sigma * A_WEATHER:.1f} degC, "
                        f"retain {AR_RETAIN}")
        if self.mode == "wind":
            bits.append(f"wind lognormal std {self.sigma * WIND_LOG_STD:.2f}")
        return "; ".join(bits)


def geographic_points(n_line, line_or_sub, line_ex_sub, layout=None,
                      n_points=8, seed=0):
    """Assign each line to a weather cell from the grid's own geometry.

    Uses grid2op's `grid_layout` coordinates when available, so cells are real
    regions of the network rather than index buckets.  Falls back to a fixed
    hash of substation id -- still N-independent, which is the property that
    matters, since dlr.zone_phase() makes the weather a function of how finely
    the grid was partitioned.

    A line's cell is taken from its ORIGIN substation; a line spanning two
    cells is rated by one of them, which is the conservative reading (the
    binding constraint is a property of the whole span).
    """
    n_points = max(1, int(n_points))
    if layout:
        pts = []
        for s in np.unique(np.concatenate([line_or_sub, line_ex_sub])):
            pts.append(layout.get(int(s)))
        coords = {int(s): layout.get(int(s)) for s in
                  np.unique(np.concatenate([line_or_sub, line_ex_sub]))}
        good = [c for c in coords.values() if c is not None]
        if len(good) >= n_points:
            xy = np.array([[c[0], c[1]] for c in good], dtype=float)
            # k cells by angle+radius quantiles around the centroid: cheap,
            # deterministic, and it produces contiguous regions.
            ctr = xy.mean(axis=0)
            ang = np.arctan2(xy[:, 1] - ctr[1], xy[:, 0] - ctr[0])
            order = np.argsort(ang)
            cell_of = {}
            keys = [k for k, c in coords.items() if c is not None]
            keys = [keys[i] for i in order]
            for i, k in enumerate(keys):
                cell_of[k] = int(i * n_points // max(len(keys), 1))
            return np.array([cell_of.get(int(s), 0) for s in line_or_sub],
                            dtype=int), n_points
    rng = np.random.RandomState(int(seed))
    n_sub = int(max(line_or_sub.max(), line_ex_sub.max())) + 1
    cell = rng.randint(0, n_points, size=n_sub)
    return cell[np.asarray(line_or_sub, dtype=int)], n_points


# ----------------------------------------------------------------------

def selfcheck():
    """Arithmetic checks; needs numpy only."""
    n, bad = 0, []

    def ck(name, cond):
        nonlocal n
        n += 1
        if not cond:
            bad.append(name)
            print(f"  FAIL  {name}")

    print("weather self-check (no grid2op required)")
    lp = np.arange(20) % 4

    # -- sigma = 0 is EXACT identity, in every mode ----------------------
    for mode in MODES:
        for geo in (False, True):
            w = WeatherProcess(20, lp, sigma=0.0, mode=mode, geographic=geo,
                               n_points=4)
            ok = True
            for ep in range(3):
                w.reset(ep)
                for _ in range(50):
                    w.step()
                    for m, h in ((1, 3), (7, 15), (12, 23), (4, 9)):
                        ok &= bool(np.all(w.line_ratios(m, h) == 1.0))
            ck(f"sigma=0 exact 1.0 [{mode}, geo={geo}]", ok)

    # -- never generous (B.2 uprating trap) ------------------------------
    for mode in MODES:
        w = WeatherProcess(20, lp, sigma=1.5, mode=mode, n_points=4)
        hi = 0.0
        for ep in range(4):
            w.reset(ep)
            for _ in range(200):
                w.step()
                hi = max(hi, float(w.line_ratios(1, 3).max()),
                         float(w.line_ratios(7, 15).max()))
        ck(f"ratio never exceeds 1.0 [{mode}]", hi <= 1.0 + 1e-12)

    # -- reproducibility: same seed+episode => same weather ---------------
    def trace(seed, ep, mode="wind"):
        w = WeatherProcess(20, lp, sigma=1.0, mode=mode, n_points=4, seed=seed)
        w.reset(ep)
        out = []
        for _ in range(40):
            w.step()
            out.append(w.line_ratios(7, 15).copy())
        return np.array(out)

    ck("same (seed, episode) -> identical weather",
       np.array_equal(trace(3, 5), trace(3, 5)))
    ck("different episode -> different weather",
       not np.array_equal(trace(3, 5), trace(3, 6)))
    ck("different seed -> different weather",
       not np.array_equal(trace(3, 5), trace(4, 5)))
    ck("replaying an episode out of order is reproducible",
       np.array_equal(trace(3, 9), trace(3, 9)))

    # -- the whole point: NOT a function of the clock ---------------------
    w = WeatherProcess(20, lp, sigma=1.0, mode="stochastic", n_points=4)
    w.reset(0)
    at_same_clock = []
    for _ in range(300):
        w.step()
        at_same_clock.append(float(w.line_ratios(7, 15).mean()))
    spread = float(np.std(at_same_clock))
    ck(f"stochastic: ratio varies at a FIXED (month,hour); std={spread:.4f}",
       spread > 1e-3)

    w0 = WeatherProcess(20, lp, sigma=1.0, mode="clock", n_points=4)
    w0.reset(0)
    clock_vals = []
    for _ in range(50):
        w0.step()
        clock_vals.append(float(w0.line_ratios(7, 15).mean()))
    ck("clock mode: ratio is CONSTANT at a fixed (month,hour) -- the control",
       float(np.std(clock_vals)) < 1e-12)
    ck("clock mode reproduces dlr.ampacity_ratio exactly",
       abs(clock_vals[0] - dlr.ampacity_ratio(7, 15, 1.0)) < 1e-12)

    # -- geographic field actually differs across cells -------------------
    w = WeatherProcess(20, lp, sigma=1.0, mode="stochastic", geographic=True,
                       n_points=4)
    w.reset(1)
    w.step()
    r = w.line_ratios(7, 15)
    ck("geographic: lines in different cells get different ratios",
       len(np.unique(np.round(r, 9))) > 1)
    w = WeatherProcess(20, lp, sigma=1.0, mode="stochastic", geographic=False,
                       n_points=1)
    w.reset(1)
    w.step()
    ck("uniform: every line gets the same ratio",
       len(np.unique(np.round(w.line_ratios(7, 15), 12))) == 1)

    # -- the winter placebo, per mode -------------------------------------
    # Asserted as the DOCUMENTED contract, not as a hope: clock and stochastic
    # keep the placebo, wind knowingly trades it away. A silent change here is
    # exactly the kind that invalidates the "the rig switches itself off in
    # February" defence without anyone noticing.
    PLACEBO = {"clock": True, "stochastic": True, "wind": False}
    for mode in MODES:
        w = WeatherProcess(20, lp, sigma=1.0, mode=mode, n_points=4)
        worst = 1.0
        for ep in range(8):
            w.reset(ep)
            for _ in range(200):
                w.step()
                worst = min(worst, float(w.line_ratios(1, 4).min()))
        if PLACEBO[mode]:
            ck(f"winter placebo HOLDS [{mode}] (min ratio {worst:.4f})",
               worst > 0.999)
        else:
            ck(f"winter placebo is KNOWN LOST [{mode}] (min ratio {worst:.4f}) "
               f"-- must be stated in every table", worst < 1.0)

    # -- severity is monotone in the mean ---------------------------------
    def mean_ratio(sig, mode):
        w = WeatherProcess(20, lp, sigma=sig, mode=mode, n_points=4, seed=1)
        acc = []
        for ep in range(6):
            w.reset(ep)
            for _ in range(120):
                w.step()
                acc.append(float(w.line_ratios(7, 15).mean()))
        return float(np.mean(acc))

    for mode in MODES:
        ms = [mean_ratio(s, mode) for s in (0.0, 0.5, 1.0, 1.5)]
        ck(f"mean ratio non-increasing in sigma [{mode}] {['%.3f' % m for m in ms]}",
           all(b <= a + 1e-9 for a, b in zip(ms, ms[1:])))

    # -- unknown mode is refused ------------------------------------------
    try:
        WeatherProcess(20, lp, mode="whatever")
        ck("unknown mode refused", False)
    except ValueError:
        ck("unknown mode refused", True)

    print(f"\n  {n - len(bad)}/{n} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(selfcheck())
