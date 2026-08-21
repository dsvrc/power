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


def ampacity_ratio(month, hour, sigma=1.0):
    """limit(t) / limit_static, from the IEEE 738 convective-cooling relation.

        I(T_amb) / I(T_ref) = sqrt( (T_max - T_amb) / (T_max - T_ref) )

    At sigma = 0 this is exactly 1.0 -- the identity that makes the stock task
    recoverable byte for byte, and which the self-check asserts.
    """
    if sigma == 0.0:
        return 1.0
    t_amb = ambient_temp(month, hour, sigma=sigma)
    # Keep the conductor headroom positive even under absurd sigma.
    headroom = max(T_CONDUCTOR_MAX - float(t_amb), 1.0)
    return float(np.sqrt(headroom / (T_CONDUCTOR_MAX - T_RATING_REF)))


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
            f"ampacity x{hot:.3f}..x{cold:.3f}   [{tag}]")
