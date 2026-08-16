"""Recursive least squares with forgetting, and the confidence gate.

Guide refs: III.1 (track beta* online, decentralized), III.5 (the PREDICTION
gate -- the trace gate silently disarms the compensator), III.9 (tracking floor).

RLS is not the contribution; the reduction that makes it applicable is (III.2).
This file is deliberately small and boring.
"""
import numpy as np


class RLSEstimator:
    """One estimator per agent.  Tracks beta* = c(t)*theta(t) from the agent's
    own proprioceptive residual.

    The parameter vector is [intercept, own_gain, beta_1..beta_r]:
      - intercept and own_gain are the NULL MODEL.  They must be carried
        explicitly, because a pooled R^2 against no baseline measures nothing
        (III.11: URB read 0.9998 until the intercept-only model was scored at
        0.656).
      - beta_1..beta_r are the peer channels, and only these are compensated.
    """

    def __init__(self, r, mu=0.9995, p0=100.0, n_null=2, fit_warmup=200,
                 ready_updates=2000):
        self.r = r
        self.n_null = n_null              # intercept + own-gain columns
        self.dim = n_null + r
        # Forgetting factor.  III.9's floor is
        #     E||beta~||^2 ~ (1-mu)*sigma^2*tr(R^-1) + ||beta_dot||^2/(1-mu)^2
        # so mu trades parameter noise against tracking lag, with an interior
        # optimum set by the physics.  Measured on the synthetic grid, sweeping
        # mu against corr(ell_hat, ell_true) and the noise in the divisor:
        #     mu     0.990  0.995  0.997  0.999  0.9995  0.9999
        #     corr   0.556  0.500  0.481  0.525  0.555   0.572
        #     dg_std 0.395  0.317  0.231  0.093  0.055   0.030
        # Aggressive forgetting buys nothing here and injects noise straight
        # into the coefficient the channel inverse divides by.  This default
        # must be re-measured on the real chronics, whose drift is faster.
        self.mu = mu
        self.p0 = p0
        self.ready_updates = int(ready_updates)
        self.beta = np.zeros(self.dim)
        self.P = p0 * np.eye(self.dim)
        self.n_updates = 0
        # Running sums for the fit_gain null-model comparison (III.11).
        # Accumulation starts only after fit_warmup updates: beta starts at 0,
        # so the cold-start transient otherwise dominates both SSEs and the
        # difference of two huge negative R^2 values is noise.
        self.fit_warmup = int(fit_warmup)
        self._sse_full = 0.0
        self._sse_null = 0.0
        self._sy = 0.0
        self._syy = 0.0
        self._n = 0

    # ------------------------------------------------------------------
    def predict(self, psi):
        return float(self.beta @ psi)

    def peer_component(self, psi):
        """The part of the prediction that is peer-induced -- what we cancel.

        Deliberately excludes the intercept and the agent's own-action gain:
        compensating those would be fighting its own control, not the coupling.
        """
        return float(self.beta[self.n_null:] @ psi[self.n_null:])

    def update(self, psi, y):
        """One RLS step.  psi is the full regressor, y the measured loading."""
        psi = np.asarray(psi, dtype=np.float64)

        # Score the PRIOR predictions, before this sample moves beta.  A
        # posterior fit flatters itself; one-step-ahead does not.
        pred_full = float(self.beta @ psi)
        pred_null = float(self.beta[:self.n_null] @ psi[:self.n_null])

        Pp = self.P @ psi
        denom = self.mu + float(psi @ Pp)
        if denom < 1e-12:
            return 0.0
        k = Pp / denom
        err = float(y) - pred_full
        self.beta = self.beta + k * err
        self.P = (self.P - np.outer(k, Pp)) / self.mu
        # Keep P symmetric; asymmetry accumulates and the gate reads P.
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1

        if self.n_updates > self.fit_warmup:
            self._sse_full += (float(y) - pred_full) ** 2
            self._sse_null += (float(y) - pred_null) ** 2
            self._sy += float(y)
            self._syy += float(y) ** 2
            self._n += 1
        return err

    # ------------------------------------------------------------------
    def confidence(self):
        """PREDICTION gate -- III.5.

            conf = 1 / (1 + r * psi^T P psi / (p0 * ||psi||^2))

        NOT the trace gate 1/(1 + tr(P)/p0).  tr(P) is dominated by the least
        excited direction, and RLS with forgetting inflates exactly those by
        1/mu every update without bound.  So once the policy converges and the
        regressor stops varying, the trace gate disarms a compensator whose
        prediction is still perfect.  Measured on URB: applied trust 0.173 ->
        0.003, i.e. the arm silently became plain IPPO while reporting
        fit_r2 = 0.9998.

        The compensator only ever uses beta.psi, so the uncertainty that
        matters is that of the scalar, not of the parameter vector.
        """
        return self._confidence_for(self._last_psi) if hasattr(self, "_last_psi") \
            else 1.0 / (1.0 + self.dim)

    def _confidence_for(self, psi):
        nrm = float(psi @ psi)
        if nrm < 1e-12:
            return 1.0 / (1.0 + self.dim)
        pred_var = float(psi @ (self.P @ psi))
        return 1.0 / (1.0 + self.dim * pred_var / (self.p0 * nrm))

    def confidence_at(self, psi):
        psi = np.asarray(psi, dtype=np.float64)
        self._last_psi = psi
        return self._confidence_for(psi)

    def ready_confidence(self):
        """Ramps 0 -> 1 over the first `ready_updates` samples.

        Not a hand-set warmup schedule in the sense III.5 warns against -- it
        gates on the estimator's own sample count, which is the one thing that
        cannot be faked by a confident-looking covariance.  Measured need: the
        same run scored corr(ell_hat, ell_true) = 0.10 when read from step 1000
        and 0.55 from step 2000, so acting early is acting on noise.
        """
        if self.ready_updates <= 0:
            return 1.0
        return float(min(1.0, self.n_updates / self.ready_updates))

    def divisor_confidence(self, idx=1):
        """Reliability of the coefficient the channel inverse DIVIDES by.

        III.5 gates on the variance of the prediction beta.psi, which is right
        for the numerator: the compensator uses ell_hat and nothing else about
        the peer parameters.  But the additive inverse is

            delta = -ell_hat / (d rho / d a)

        and that denominator is *also* a learned coefficient.  Prediction
        variance can be small while the own-gain coefficient is still badly
        split against a collinear peer channel -- measured here at 6000 steps:
        ell_hat tracked ell_true in sign and rough magnitude while own_gain
        came out at +0.37 against a true -1.13.  Dividing by a wrong-signed
        estimate steers the compensation backwards, and the III.5 gate cannot
        see it, because nothing is wrong with the prediction.

        So gate the divisor on its own relative standard error:

            conf_div = 1 / (1 + (se / |coef|)^2),   se = sqrt(P[idx, idx])

        -> 1 when the coefficient is pinned down, -> 0 while its sign is still
        in doubt.  Multiplies the prediction gate rather than replacing it;
        the two cover different halves of the same expression.
        """
        var = float(self.P[idx, idx])
        coef = float(self.beta[idx])
        if not np.isfinite(var) or var < 0.0:
            return 0.0
        se = np.sqrt(var)
        if abs(coef) < 1e-12:
            return 0.0
        rse = se / abs(coef)
        return float(1.0 / (1.0 + rse * rse))

    def trace_confidence(self):
        """The WRONG gate, retained so the ablation can be run rather than
        described.  Never wire this into the control path."""
        return 1.0 / (1.0 + np.trace(self.P) / self.p0)

    # ------------------------------------------------------------------
    def fit_gain(self):
        """R^2(full) - R^2(intercept+own only).  III.11.

        The honest quantity.  A raw pooled R^2 inherits the per-agent mean and
        reads near 1 whatever the basis does.
        """
        if self._n < 32:
            return np.nan
        var = self._syy - self._sy ** 2 / self._n
        # A target with no variance has nothing to explain, and dividing by it
        # manufactures an arbitrarily large "gain".  NaN averages away; a
        # number does not (II.7's rule, applied to a non-ratio).
        if var <= 1e-9 * max(1.0, abs(self._sy / self._n)):
            return np.nan
        r2_full = 1.0 - self._sse_full / var
        r2_null = 1.0 - self._sse_null / var
        return float(r2_full - r2_null)
