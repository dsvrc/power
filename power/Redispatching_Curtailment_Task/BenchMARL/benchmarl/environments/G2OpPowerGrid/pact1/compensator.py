"""The certified channel inverse -- pure arithmetic, no grid2op.

Kept free of simulator imports on purpose so the self-check gate (IV.4) can
exercise the real code path rather than a reimplementation of it.

Channel class (I.4): additive in curtailment-limit units, so the inverse is
`u = a - beta*ell` and lives inside the action set.  This is what makes T3
conjugacy available in POWER and unavailable in URB, where a route index has
no inverse -- a difference derived in advance, not discovered.
"""
import numpy as np


class StandingLevel:
    """Removes the standing pedestal from the CONTROL signal only.

    Why this exists.  III.6 requires the regressor to be centred on the basis's
    own geometric zero point (peers acting uniformly at random), never on the
    sample mean -- the sample mean is run data and would turn the declared
    model class into a fit.  That rule governs IDENTIFICATION, and it is kept
    exactly: the RLS regressor is untouched.

    But a trained policy does not act uniformly at random.  Curtailment limits
    sit near 1.0 unless the grid needs relief, while the reference assumes 0.5,
    so psi carries a large constant offset.  Measured on the synthetic grid:
    psi mean [-0.50, -2.01] against temporal std [0.30, 0.17] -- about 85% of
    ell_hat is a pedestal.  Compensating it burns the entire bounded actuator
    budget on a constant and leaves the time-varying part -- the only part the
    policy cannot anticipate, because it depends on unobserved peer actions --
    uncorrected.  That is exactly backwards, and it measured as a 5.6% residual
    reduction where the mechanism should give far more.

    The standing level is the policy's job; the fluctuation is PACT-1's.  So
    the control law cancels ell_hat - ell_bar, with ell_bar a slow EMA.  tau
    must be long compared with the driver period, or this removes the driver
    content it is supposed to correct -- it is a declared constant and belongs
    in the ablation table (II.1), never tuned per result.
    """

    def __init__(self, tau=2000.0):
        self.tau = float(tau)
        self.level = None

    def update(self, ell_hat):
        if not np.isfinite(ell_hat):
            return 0.0
        if self.level is None:
            self.level = float(ell_hat)
            return 0.0                     # nothing to act on until a level exists
        alpha = 1.0 / max(self.tau, 1.0)
        self.level += alpha * (float(ell_hat) - self.level)
        return float(ell_hat) - self.level

    def reset(self):
        self.level = None


def action_sensitivity(own_gain, pmax):
    """d(rho_i)/d(a_i) from the RLS own-gain coefficient.

    The regression learns d(rho)/d(own_col), but own_col is built from
    exertion, and exertion runs BACKWARDS to the action -- curtailing more
    means a LOWER limit ratio.  The chain, written out once so the sign lives
    in exactly one place:

        own_ex  = mean(1 - a) * pmax          =>  d(own_ex)/d(a) = -pmax
        own_col = (own_ex - 0.5*pmax) / s     with s = max(pmax, 1)
                                              =>  d(own_col)/d(a) = -pmax/s
        d(rho)/d(a) = own_gain * (-pmax/s)
    """
    return float(own_gain) * (-float(pmax) / max(float(pmax), 1.0))


def compensation_delta(ell_hat, drho_da, trust, conf,
                       max_delta=0.25, sensitivity_floor=1e-3):
    """Return (delta, applied_gain).

    delta is the additive shift to the curtailment-limit vector that cancels
    the predicted peer-induced loading.

    Parameters
    ----------
    ell_hat : predicted peer-induced change in the agent's own loading, in rho
        units.  Positive means peers have pushed this agent's lines UP.
    drho_da : d(rho_i) / d(a_i) -- the derivative of realized loading with
        respect to the CURTAILMENT-LIMIT ACTION, not with respect to exertion.
        Callers must do the exertion->action conversion themselves (see
        env.PACT1Env._sensitivity); this signature takes one unambiguous
        quantity so the sign convention lives in exactly one place.

    The inverse solves  ell_hat + drho_da * delta = 0, so

        delta = -ell_hat / drho_da

    which is the additive row of I.4's table.  drho_da is LEARNED, not assumed:
    whether curtailing raises or lowers a zone's loading depends on whether the
    zone is a net exporter, which is a property of the grid.

    Floor property (III.4): if no usable inverse exists yet the gain is exactly
    0, and the executed action is exactly the blind action.  A diverging
    estimate can fail to help but cannot go below baseline.
    """
    if not np.isfinite(ell_hat) or not np.isfinite(drho_da):
        return 0.0, 0.0
    if abs(drho_da) < sensitivity_floor:
        return 0.0, 0.0
    g = float(trust) * float(conf)
    delta = -g * float(ell_hat) / float(drho_da)
    if not np.isfinite(delta):
        return 0.0, 0.0
    return float(np.clip(delta, -max_delta, max_delta)), g


def apply_inverse(curtail, delta):
    """Execute the inverse and report rail contact.

    Returns (executed, n_saturated).  Curtailment limits live in [0, 1].
    """
    curtail = np.asarray(curtail, dtype=np.float64)
    out = np.clip(curtail + delta, 0.0, 1.0)
    n_sat = int(np.sum((out <= 0.0) | (out >= 1.0)))
    return out, n_sat
