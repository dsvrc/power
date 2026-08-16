"""The columns that decide whether a run means anything.

Guide refs: II.7 (NS-liveness on EVERY arm; guard ratios with NaN, not eps;
banner every resolved knob), III.11 (what each column decides; read applied
trust first), I.3 (the escape test at MATCHED driver level).
"""
import os
import threading

import numpy as np

# Written in this order.  applied_trust sits early on purpose: a safety
# property that silently engages is indistinguishable from a bug, and this is
# the column that catches it (III.5).
COLUMNS = [
    "step", "episode",
    "applied_trust", "trust_pol", "conf", "conf_trace",
    "state",                      # INERT / ASLEEP / ALIVE
    "ell_mean", "ell_max",
    "ell_matched",                # E||ell|| at MATCHED driver level -- I.3
    "matched_n",
    "cancel",                     # 1 - felt/blind, NaN-guarded
    "beta_err_proxy", "fit_gain", "cond_psi",
    "sat_frac",                   # compensation hitting the [0,1] rail
    "A_mean", "A_min", "A_max",   # the native driver: grid stress
    "rho_mean", "rho_max",
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
