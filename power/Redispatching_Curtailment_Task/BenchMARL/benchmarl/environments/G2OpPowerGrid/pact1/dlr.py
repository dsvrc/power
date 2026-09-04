"""Dynamic Line Rating: the severity dial, as removed simplification.

WHAT THIS IS, AND WHY IT IS NOT A GREMLIN
-----------------------------------------
Grid2Op ships every environment with STATIC thermal limits: one ampacity per
line, constant for the whole year.  Real transmission operators do not work that
way.  A conductor's current-carrying capacity depends on how fast it sheds heat,
so ampacity falls in hot still weather and rises in cold windy weather.  Rating
lines dynamically from ambient conditions -- Dynamic Line Rating -- is standard
practice and is the subject of its own IEEE standard (IEEE 738, "Standard for
Calculating the Current-Temperature Relationship of Bare Overhead Conductors").

So this module does not add a disturbance to the task.  It restores physics the
simulator abstracts away.  The severity dial sigma scales how much ambient
temperature is allowed to vary:

    sigma = 0   ambient pinned at the rating reference -> ratio == 1 for every
                line at every step -> the environment is byte-identical to
                stock grid2op.  This is the published task, unchanged.
    sigma = 1   the real annual temperature range for the grid's own region
                (l2rpn_idf_2023 is Ile-de-France), giving roughly -18% ampacity
                at the summer afternoon peak and +15% on a winter night.
                THIS IS REALITY, not a tuned value.
    sigma > 1   deliberately beyond-physical stress testing.  Must be labelled
                as such in any table it appears in.

The point of anchoring sigma=1 to measured climate rather than to a number that
makes a method look good: the headline experiment can be run at sigma=1 and
described as "under realistic dynamic line ratings", with no tuning to defend.

WHY IT IS A GAIN ON THE COUPLING, NOT AN ADDITIVE TERM
------------------------------------------------------
NS guide I.2 requires the driver to MULTIPLY the cross-agent term rather than
add a term of its own -- otherwise a lone agent feels it and the setting is
category B in disguise.  Derating satisfies this structurally.  Loading is

    rho_l = |flow_l| / limit_l(t)

and flow_l is a linear function of every agent's injection through the PTDF.
Shrinking limit_l scales the WHOLE ratio, so peer j's contribution to agent i's
loading is amplified by exactly the same factor as everything else:

    d(rho_l)/d(inj_j)  =  PTDF[l, j] / limit_l(t)

The coupling term is multiplied by 1/limit(t); no term independent of the peers
is introduced.  At N=1 the peer contribution is still identically zero however
hot it gets.

APPLIED TO EVERY ARM
--------------------
This lives in the environment, not in PACT-1, and is configured by a task-level
`severity` field.  MAPPO, MASAC and PACT-1 all see identical physics at a given
sigma.  A dial that only the method's arm experienced would be worthless.
"""
import numpy as np

# IEEE 738 steady-state reference conditions.  ACSR conductor, standard summer
# rating assumptions; these are published defaults, not fitted quantities.
T_CONDUCTOR_MAX = 75.0     # degC, conductor design temperature
T_RATING_REF = 20.0        # degC, ambient the static rating assumes

# Ile-de-France climate: mean daily max ~25degC in July, ~7degC in January;
# diurnal swing ~10degC.  Amplitudes below reproduce roughly 2-38degC over a
# year at sigma=1.
A_SEASONAL = 12.0          # degC, peak-to-mean seasonal amplitude
A_DIURNAL = 6.0            # degC, peak-to-mean daily amplitude
PEAK_MONTH = 7.5           # late July
PEAK_HOUR = 15.0           # mid-afternoon


def ambient_temp(month, hour, sigma=1.0):
    """Ambient temperature from the calendar alone.

    Deterministic in (month, hour): no agent action and no simulator RNG can
    move it, which is what makes it an exogenous driver rather than a feedback
    path.  Both fields are already in the observation.
    """
    seasonal = np.cos(2.0 * np.pi * (np.asarray(month) - PEAK_MONTH) / 12.0)
    diurnal = np.cos(2.0 * np.pi * (np.asarray(hour) - PEAK_HOUR) / 24.0)
    return T_RATING_REF + float(sigma) * (A_SEASONAL * seasonal
                                          + A_DIURNAL * diurnal)


def driver_level(month, hour, sigma=1.0):
    """A(t) in [0, 1]: the normalised driver, 1 at the hottest point.

    Reported so the diagnostics can bin on a MATCHED driver level (I.3's escape
    test) without re-deriving the climate model.
    """
    t = ambient_temp(month, hour, sigma=1.0)      # shape, not amplitude
    lo = T_RATING_REF - (A_SEASONAL + A_DIURNAL)
    hi = T_RATING_REF + (A_SEASONAL + A_DIURNAL)
    return float(np.clip((t - lo) / (hi - lo), 0.0, 1.0))


def ampacity_ratio(month, hour, sigma=1.0, derate_only=True):
    """limit(t) / limit_static, from the IEEE 738 convective-cooling relation.

        I(T_amb) / I(T_ref) = sqrt( (T_max - T_amb) / (T_max - T_ref) )

    At sigma = 0 this is exactly 1.0 -- the identity that makes the stock task
    recoverable byte for byte, and which the self-check asserts.

    DERATE-ONLY, and this is a substantive choice, not a detail.
    -----------------------------------------------------------
    The raw relation both derates in heat and UPRATES in cold, and sigma scales
    both.  That makes sigma a realism dial whose difficulty depends on season,
    not a severity dial: measured on winter chronics, raising sigma to 2.0
    applied x1.269 to every limit and made the task strictly EASIER (reference
    survival 350 -> 901 steps, rho 0.905 -> 0.784).

    Clipping at 1.0 keeps only the derating half:

      * it is the conservative choice -- the grid is never credited with more
        capacity than its own published static rating, so no headroom is
        invented anywhere;
      * it makes sigma monotone in difficulty, which a severity dial must be;
      * it is what a cautious operator does, since static ratings are the
        contractual limit and DLR uprating requires extra instrumentation
        before it may be relied on.

    Pass derate_only=False only to inspect the raw two-sided physics.
    """
    if sigma == 0.0:
        return 1.0
    t_amb = ambient_temp(month, hour, sigma=sigma)
    # Keep the conductor headroom positive even under absurd sigma.
    headroom = max(T_CONDUCTOR_MAX - float(t_amb), 1.0)
    r = float(np.sqrt(headroom / (T_CONDUCTOR_MAX - T_RATING_REF)))
    return min(1.0, r) if derate_only else r


def describe(sigma):
    """One-line banner text: what this sigma means physically."""
    if sigma == 0.0:
        return ("severity 0.0 -> static ratings; environment is byte-identical "
                "to stock grid2op")
    hot = ampacity_ratio(PEAK_MONTH, PEAK_HOUR, sigma)
    cold = ampacity_ratio((PEAK_MONTH + 6) % 12, (PEAK_HOUR + 12) % 24, sigma)
    t_hot = ambient_temp(PEAK_MONTH, PEAK_HOUR, sigma)
    t_cold = ambient_temp((PEAK_MONTH + 6) % 12, (PEAK_HOUR + 12) % 24, sigma)
    tag = "REALISTIC (IEEE 738, Ile-de-France)" if abs(sigma - 1.0) < 1e-9 \
        else "BEYOND-PHYSICAL stress test" if sigma > 1.0 else "sub-realistic"
    return (f"severity {sigma:.2f} -> ambient {t_cold:+.1f}..{t_hot:+.1f} degC, "
            f"ampacity x{hot:.3f} (summer peak) .. x{cold:.3f} (winter, clipped "
            f"at the static rating)   [{tag}]")


# ---------------------------------------------------------------------------
# SPATIAL HETEROGENEITY
# ---------------------------------------------------------------------------
# Ile-de-France spans roughly 100 km.  Ambient temperature, wind and insolation
# are not uniform over that: a summer afternoon typically runs several degrees
# hotter inland than on the western, more ventilated side, and cloud cover moves
# across the region during the day.  Operators using dynamic ratings therefore
# rate lines from LOCAL conditions, not one regional number.
#
# Uniform derating is the simplification; per-zone derating is the physics.
#
# WHY IT MATTERS FOR COORDINATION, and this is the point:
# under uniform derating every agent sees the same 1/g amplification, so an
# agent's own loading tells it everything it needs -- the fix is local and the
# coordination gap stays small (measured 9.5% at sigma=1).  Under heterogeneous
# derating the cheap fix for a hot zone's overload is often generation in a
# COOLER neighbouring zone, where the same MW of curtailment buys more headroom.
# An agent cannot identify that lever from its own loading alone.  The
# coordination-recoverable fraction rises by construction, and it rises for a
# physical reason rather than because a knob was turned until baselines fell.
SPATIAL_AMP = 0.6          # fraction of the seasonal amplitude that varies by zone
SPATIAL_PERIOD_H = 30.0    # hours for the hot cell to traverse the region


def zone_phase(zone_index, n_zones):
    """Position of a zone along the region's thermal gradient, in [0, 1).

    Derived from the zone index so it is deterministic and reproducible; a
    deployment would use each zone's real centroid.
    """
    return (float(zone_index) / max(int(n_zones), 1)) % 1.0


def ambient_temp_zone(month, hour, zone_index, n_zones, sigma=1.0,
                      spatial=True):
    """Ambient at one zone: the regional cycle plus a travelling spatial term.

    The spatial term is a wave crossing the region over SPATIAL_PERIOD_H hours,
    so which zone is hottest CHANGES through the episode.  A fixed gradient
    would let an agent learn a static "zone 7 is always hot" rule from its own
    observations; a moving one cannot be inferred without knowing peers' state,
    which is what makes the residual difficulty coordination rather than
    memorisation.
    """
    base = ambient_temp(month, hour, sigma=sigma)
    if not spatial or sigma == 0.0:
        return base
    ph = zone_phase(zone_index, n_zones)
    wave = np.cos(2.0 * np.pi * (float(hour) / SPATIAL_PERIOD_H - ph))
    return base + float(sigma) * SPATIAL_AMP * A_SEASONAL * wave


def ampacity_ratio_zone(month, hour, zone_index, n_zones, sigma=1.0,
                        derate_only=True, spatial=True):
    """Per-zone ampacity ratio.  sigma = 0 still returns exactly 1.0."""
    if sigma == 0.0:
        return 1.0
    t_amb = ambient_temp_zone(month, hour, zone_index, n_zones, sigma=sigma,
                              spatial=spatial)
    headroom = max(T_CONDUCTOR_MAX - float(t_amb), 1.0)
    r = float(np.sqrt(headroom / (T_CONDUCTOR_MAX - T_RATING_REF)))
    return min(1.0, r) if derate_only else r


def spatial_spread(month, hour, n_zones, sigma=1.0):
    """max/min ampacity ratio across zones at one instant.

    1.0 means uniform derating (no coordination pressure from heterogeneity).
    Report this alongside the coordination gap: it is the knob that connects
    them, and it is a property of the weather model, not of any method.
    """
    rs = [ampacity_ratio_zone(month, hour, i, n_zones, sigma)
          for i in range(n_zones)]
    lo = min(rs)
    return (max(rs) / lo) if lo > 0 else float("inf")


def is_thermally_active(month, sigma=1.0, threshold=0.99):
    """Does derating bite in this month at all?

    Winter months clip to exactly 1.0, so they are a natural PLACEBO condition:
    the dial is configured and running, and provably does nothing.  Reporting a
    winter sweep alongside the summer one is the cheapest available control
    against "your severity knob is doing something you have not described".
    """
    return ampacity_ratio(month, PEAK_HOUR, sigma) < threshold
