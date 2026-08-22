"""The columns that decide whether a run means anything.

Guide refs: II.7 (NS-liveness on EVERY arm; guard ratios with NaN, not eps;
banner every resolved knob), III.11 (what each column decides; read applied
trust first), I.3 (the escape test at MATCHED driver level).
"""
import os
import threading
import time

import numpy as np

# Written in this order.  applied_trust sits early on purpose: a safety
# property that silently engages is indistinguishable from a bug, and this is
# the column that catches it (III.5).
COLUMNS = [
    "step", "episode",
    "applied_trust", "trust_pol", "conf", "conf_trace",
    # The gate is a product of three terms.  Logging only the product tells you
    # trust is low but not WHICH half of the ratio is unsure, and they have
    # different fixes: conf_pred low = the peer prediction is noisy, conf_div
    # low = the coefficient the inverse divides by is not pinned down,
    # conf_ready low = simply not enough samples yet.
    "conf_pred", "conf_div", "conf_ready", "own_gain", "own_gain_se",
    "fit_gain_now",               # windowed lift; the compensator gates on this
    # Is the compensator moving actions at all, and is it RESPONDING to the
    # estimate or pinned at the rail?  A rail-pinned delta is a constant bias,
    # not a compensation, and it is invisible in every other column.
    "analytic_frac",              # share of steps using the PTDF divisor
    "dlr_ratio", "dlr_skip",      # severity liveness: is the dial reaching physics?
    "delta_abs", "delta_clip_frac", "delta_nonzero_frac",
    "trP", "clamp_frac",          # covariance windup: tr(P) and how often bounded
    "state",                      # INERT / ASLEEP / ALIVE
    "ell_mean", "ell_max",
    "ell_matched",                # E||ell|| at MATCHED driver level -- I.3
    "matched_n",
    "cancel",                     # 1 - felt/blind, NaN-guarded
    "beta_err_proxy", "fit_gain", "cond_psi",
    "sat_frac",                   # compensation hitting the [0,1] rail
    "A_mean", "A_min", "A_max",   # the native driver: grid stress
    "rho_mean", "rho_max",
    # Is there anything to identify at all?  If the exertion functional never
    # moves, phi is constant, the peer columns are collinear with the intercept
    # and fit_gain is 0 BY CONSTRUCTION -- no estimator can help.  Log this
    # before concluding anything about the method from a flat fit_gain.
    "phi_mean", "phi_std", "phi_frac_active",
    "own_col_std",                # own-action excitation, drives the divisor gate
    "rho_own_mean", "rho_own_max",  # the sensor's own-zone aggregate
    "g2op_per_gym",               # grid2op steps consumed per agent decision
    "n_eff_r",
]


class PactLogger:
    """Append-only CSV, one row per env step-batch.  Thread-safe because the
    collector may fork several envs into one file."""

    _lock = threading.Lock()

    def __init__(self, path, extra_note=""):
        self.path = path
        self._note = extra_note
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

        # Appending a run whose column set differs from the existing header
        # silently misaligns every field in the new segment -- it produced a
        # cond_psi of 0.02 (impossible; cond >= 1) and cost a full analysis
        # pass before anyone noticed.  If the header on disk does not match,
        # roll the old file aside rather than corrupting both runs.
        if os.path.exists(path):
            existing = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.startswith("#"):
                            existing = line.strip().split(",")
                            break
            except OSError:
                existing = None
            if existing != list(COLUMNS):
                stamp = time.strftime("%Y%m%d-%H%M%S")
                os.replace(path, f"{path}.{stamp}.bak")
                print(f"[pact1] log schema changed; previous log moved to "
                      f"{os.path.basename(path)}.{stamp}.bak")

        if not os.path.exists(path):
            with self._lock, open(path, "w", encoding="utf-8") as f:
                if extra_note:
                    f.write(f"# {extra_note}\n")
                f.write(",".join(COLUMNS) + "\n")

    def write(self, row):
        vals = []
        for c in COLUMNS:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append("" if v is None else f"{v:.6g}")
            else:
                vals.append(str(v))
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(vals) + "\n")


def classify(ell_max_seen, A_span, ell_max_now):
    """II.7's automatic classification.  A return that looks *good* is the
    failure nobody investigates, so the disturbance must self-report."""
    if ell_max_seen < 1e-8:
        return "INERT"        # env not deployed, phi masked, severity 0, stale .pyc
    if A_span < 0.05:
        return "ASLEEP"       # driver never left its trough in this window
    return "ALIVE"


def safe_ratio(numer, denom, floor=1e-6):
    """II.7: guard every ratio HARD, not with an epsilon.

    At the driver trough the disturbance is genuinely ~0 and the ratio is
    meaningless.  Return NaN, which averaging drops.  A 1e-12 floor once
    produced a -1011 that poisoned a column average until it was caught.
    """
    if not np.isfinite(denom) or abs(denom) < floor:
        return np.nan
    return float(numer / denom)


class MatchedDriverTracker:
    """The escape test -- I.3, 'the single highest-value diagnostic in the
    whole programme'.

    Accumulates E||ell|| only over steps whose driver level falls in a fixed
    band, so the number is comparable across training.  Falls over training =>
    the policy found a way to make the coupling stop happening, and the
    benchmark is measuring nothing.
    """

    def __init__(self, lo=0.69, hi=0.70):
        self.lo, self.hi = lo, hi
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._n = 0

    def observe(self, driver_level, ell_norm):
        if self.lo <= driver_level <= self.hi:
            self._sum += float(ell_norm)
            self._n += 1

    @property
    def value(self):
        return self._sum / self._n if self._n else np.nan

    @property
    def count(self):
        return self._n


def banner(cfg_lines):
    """II.7: print an unmissable banner of every resolved knob from the
    constructor.  Hardcoding a severity as a literal silently disables the env
    var and runs a whole 'sweep' at one setting."""
    width = 72
    out = ["=" * width, "  PACT-1 ACTIVE".ljust(width)]
    out += [f"  {ln}" for ln in cfg_lines]
    out.append("=" * width)
    return "\n".join(out)
