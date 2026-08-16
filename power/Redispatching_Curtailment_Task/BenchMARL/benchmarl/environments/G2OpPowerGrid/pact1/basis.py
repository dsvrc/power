"""The declared basis: zone geometry -> r channels, and the reference load.

Guide refs: I.7 (hand over a basis, not an operator), III.6 (centre on the
basis's own zero point; cond=inf is a VALUE), IV.4 (report the effective r,
do not force channels to survive).

Nothing here reads run data.  The channel partition and x_ref are functions of
zones_definitions.json and the action box alone, which is what keeps the model
class *declared* rather than *fitted*.
"""
import json
import os

import numpy as np

ENV_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Channel ids.  CORE is kept in the enumeration even though the l2rpn_idf_2023
# zones never populate it (they partition the lines, so no two zones own the
# same one) -- a different zoning would, and the spread check drops it cleanly.
CORE, HALO, DIST = 0, 1, 2
CHANNEL_NAMES = ("core", "halo", "distant")

# A channel whose relative spread across agents falls below this is collinear
# with the intercept and cannot be identified.  URB's dead arterial channel
# measured 0.00796.
SPREAD_DEAD = 0.05


def _load_zones():
    with open(os.path.join(ENV_PATH, "zones_definitions.json"), "r",
              encoding="utf-8") as f:
        return json.load(f)


class CouplingBasis:
    """Zero-diagonal channel partition over zone agents, plus x_ref.

    Parameters
    ----------
    zone_names : ordered zone names, one per local agent (agent_1..agent_n)
    gen_curtail_inside : {zone_name: array of generator ids it may curtail}
    gen_pmax : full grid gen_pmax vector, used to weight exertion in MW
    """

    def __init__(self, zone_names, gen_curtail_inside, gen_pmax):
        self.zone_names = list(zone_names)
        self.n = len(self.zone_names)
        zones = _load_zones()

        line_in = {z: set(zones[z]["line_in_zone_idx"]) for z in self.zone_names}
        line_nb = {z: set(zones[z]["line_neighboring_idx"]) for z in self.zone_names}

        # --- channel matrix -------------------------------------------------
        chan = np.full((self.n, self.n), DIST, dtype=np.int8)
        for a, i in enumerate(self.zone_names):
            for b, j in enumerate(self.zone_names):
                if a == b:
                    chan[a, b] = -1           # zero diagonal (I.7 rule 1)
                elif line_in[i] & line_in[j]:
                    chan[a, b] = CORE         # j drives a line i owns
                elif line_nb[i] & line_in[j]:
                    chan[a, b] = HALO         # j drives a line i only watches
        self.chan = chan

        # I.7 rule 1 is load-bearing: assert it, never argue it.
        assert (np.diag(chan) == -1).all(), "basis is not zero-diagonal"

        # --- exertion scale, and x_ref (III.6) -----------------------------
        # phi_j = sum_g (1 - a_g) * pmax_g  = MW of renewable withheld by j.
        # A magnitude, so it has no sign-cancellation escape (I.3, the Ant trap),
        # and the team can only drive it down by not curtailing at all, which
        # overloads the grid.
        self.pmax_per_zone = np.array(
            [float(np.sum(gen_pmax[np.asarray(gen_curtail_inside[z], dtype=int)]))
             if len(gen_curtail_inside[z]) else 0.0
             for z in self.zone_names])
        # a_g ~ U[0,1] => E[1 - a_g] = 0.5.  Geometry and action box only.
        phi_ref = 0.5 * self.pmax_per_zone

        x_ref = np.zeros((self.n, 3))
        for a in range(self.n):
            for c in (CORE, HALO, DIST):
                x_ref[a, c] = phi_ref[np.where(chan[a] == c)[0]].sum()
        self.x_ref_full = x_ref

        # --- spread test -> which channels are identifiable at all ----------
        self.spread = np.zeros(3)
        for c in (CORE, HALO, DIST):
            col = x_ref[:, c]
            m = np.abs(col).mean()
            self.spread[c] = (col.std() / m) if m > 1e-9 else np.inf
        # inf means the channel is identically zero -> dead, not "very spread".
        self.keep = [c for c in (CORE, HALO, DIST)
                     if np.isfinite(self.spread[c]) and self.spread[c] >= SPREAD_DEAD]
        if not self.keep:
            raise RuntimeError(
                "every coupling channel is degenerate; the basis cannot be "
                "identified on this zoning")

        self.r = len(self.keep)
        self.kept_names = [CHANNEL_NAMES[c] for c in self.keep]
        self.x_ref = x_ref[:, self.keep]
        # Scale per channel so no column dominates the Gram (III.6).
        sc = self.x_ref.std(axis=0)
        sc[sc < 1e-9] = 1.0
        self.scale = sc

        # Masks used per step: peers of agent i on kept channel c.
        self.masks = np.zeros((self.n, self.r, self.n), dtype=np.float64)
        for a in range(self.n):
            for ci, c in enumerate(self.keep):
                self.masks[a, ci, np.where(chan[a] == c)[0]] = 1.0

    # -----------------------------------------------------------------------
    def exertion(self, curtail_actions):
        """phi_j for every zone agent, in MW withheld.

        curtail_actions : {zone_name: action vector in [0,1]} -- the curtailment
        limit ratios agent j commanded.  Absent agents contribute 0.
        """
        phi = np.zeros(self.n)
        for a, z in enumerate(self.zone_names):
            act = curtail_actions.get(z)
            if act is None or len(act) == 0:
                continue
            # (1 - ratio) * pmax, summed.  pmax_per_zone already holds the sum,
            # so weight by the mean withheld fraction.
            phi[a] = float(np.mean(1.0 - np.clip(act, 0.0, 1.0))) * self.pmax_per_zone[a]
        return phi

    def waveforms(self, phi):
        """x[i, c] = channel-c load agent i sees, then centred on x_ref/scaled.

        Returns the *centred* regressor columns psi (n, r).  Centring on the
        geometric reference rather than the sample mean is what keeps cond
        finite -- III.6, measured 1.3e5 -> 7.1 on this grid.
        """
        x = self.masks @ phi              # (n, r)
        return (x - self.x_ref) / self.scale

    def report(self):
        lines = ["[pact1] declared coupling basis"]
        for c in (CORE, HALO, DIST):
            n_pairs = int((self.chan == c).sum())
            s = self.spread[c]
            state = "kept" if c in self.keep else "DROPPED (degenerate)"
            lines.append(f"    {CHANNEL_NAMES[c]:8s} pairs={n_pairs:4d} "
                         f"spread={s:8.4f}  {state}")
        lines.append(f"    effective r = {self.r}  {self.kept_names}")
        return "\n".join(lines)


def gram_cond(psi_history):
    """cond(E[psi psi^T]).  Non-finite is a value: it means some channel is
    exactly collinear and theta cannot be decomposed (III.6).  Callers must
    test non-finite FIRST -- `isfinite(c) and c > thr` lets the most degenerate
    basis possible pass silently.
    """
    if len(psi_history) < 2:
        return np.inf
    M = np.asarray(psi_history, dtype=np.float64)
    G = M.T @ M / M.shape[0]
    ev = np.linalg.eigvalsh(G)
    if ev.min() <= 1e-12:
        return np.inf
    return float(ev.max() / ev.min())
