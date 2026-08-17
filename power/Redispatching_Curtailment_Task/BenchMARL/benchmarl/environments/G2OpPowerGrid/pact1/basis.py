"""The declared basis: the grid's own load-transfer operator -> r channels.

Guide refs: IV.1 (in a capacitated flow network the cross-agent coupling IS the
network's own load-transfer operator -- PTDF for a power grid), I.7 (hand over
a basis, not a fitted operator), III.6 (centre on the basis's own zero point;
cond=inf is a VALUE), IV.4 (report the effective r, do not force channels).

PTDF is not privileged information: it is the network model, which every system
operator has, and it is a *declared* quantity computed once from the grid --
never fitted from run data.  What stays unknown is theta, how load actually
distributes across the channels, and that is what the RLS tracks.

HISTORY, because it cost a run.  The first version of this file built the
channels from zone GEOMETRY -- "peer owns a line I watch" (halo) vs everything
else (distant) -- giving every pair inside a bucket the same weight.  Measured
on l2rpn_idf_2023 that produced fit_gain = -0.0045, i.e. the peer channels made
the one-step-ahead prediction WORSE than an intercept-only null model, because
the true PTDF weights span orders of magnitude (spread std/mean 1.35 against
0.55 / 0.14 for the two geometric buckets) and are strongly asymmetric, which a
symmetric geometric relation cannot represent at all.
"""
import json
import os

import numpy as np

ENV_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHANNEL_NAMES = ("strong", "weak", "residual")

# A channel whose relative spread across agents falls below this is collinear
# with the intercept and cannot be identified.  URB's dead arterial channel
# measured 0.00796.
SPREAD_DEAD = 0.05
# Below this share of the agent's total coupling, a peer is numerically zero.
W_FLOOR = 1e-9


def _load_zones():
    with open(os.path.join(ENV_PATH, "zones_definitions.json"), "r",
              encoding="utf-8") as f:
        return json.load(f)


def get_ptdf(env):
    """Injection -> flow sensitivities, or None if this backend cannot give them.

    lightsim2grid returns (n_line, n_bus) with n_bus = 2 * n_sub, because
    grid2op doubles every substation into two busbars.  At the reference
    topology everything sits on busbar 1, so substation id indexes the first
    n_sub columns directly.
    """
    try:
        return np.asarray(env.backend._grid.get_ptdf())
    except Exception:                                        # noqa: BLE001
        pass
    for attr in ("get_ptdf", "get_PTDF", "ptdf"):
        try:
            cand = getattr(env.backend, attr)
            return np.asarray(cand() if callable(cand) else cand)
        except Exception:                                    # noqa: BLE001
            continue
    return None


def ptdf_zone_coupling(env, zone_names, ptdf=None):
    """W[i, j] = mean |PTDF| from j's curtailable generators onto i's OWN lines.

    Zero-diagonal by construction (I.7 rule 1): j == i is never accumulated, so
    the single-agent projection of the coupling is exactly 0 whatever the grid
    does.
    """
    if ptdf is None:
        ptdf = get_ptdf(env)
    if ptdf is None:
        return None

    zones = _load_zones()
    n = len(zone_names)
    gen_bus = env.gen_to_subid
    renewable = np.where(env.gen_renewable)[0]
    n_cols = ptdf.shape[1]

    W = np.zeros((n, n))
    for a, zi in enumerate(zone_names):
        lines_i = np.asarray(zones[zi]["line_in_zone_idx"], dtype=int)
        lines_i = lines_i[lines_i < ptdf.shape[0]]
        if not len(lines_i):
            continue
        for b, zj in enumerate(zone_names):
            if a == b:
                continue
            gj = np.intersect1d(
                np.asarray(zones[zj]["gen_inside_idx"], dtype=int), renewable)
            if not len(gj):
                continue
            buses = gen_bus[gj]
            buses = buses[buses < n_cols]
            if not len(buses):
                continue
            W[a, b] = float(np.mean(np.abs(ptdf[np.ix_(lines_i, buses)])))
    return W


class CouplingBasis:
    """PTDF-weighted, zero-diagonal channel decomposition over zone agents.

    Channels are a per-agent split of that agent's peers by coupling strength.
    Splitting rather than using one lumped channel lets theta express how the
    load actually distributes between tightly and loosely coupled peers, which
    is the unknown the guide asks the estimator to track; the PTDF weights
    inside each channel carry the physics the geometric version threw away.
    """

    def __init__(self, zone_names, gen_curtail_inside, gen_pmax,
                 W=None, r_target=2):
        self.zone_names = list(zone_names)
        self.n = len(self.zone_names)
        self.r_target = int(r_target)

        if W is None:
            raise RuntimeError(
                "no PTDF available; refusing to fall back to a geometric basis "
                "-- it measured fit_gain = -0.0045 on this grid")
        W = np.asarray(W, dtype=np.float64).copy()
        np.fill_diagonal(W, 0.0)
        self.W = W
        assert np.allclose(np.diag(self.W), 0.0), "basis is not zero-diagonal"

        # --- per-agent channel assignment by coupling strength --------------
        # chan[i, j] = channel id of peer j for agent i, -1 on the diagonal and
        # for peers this agent is not coupled to at all.
        chan = np.full((self.n, self.n), -1, dtype=np.int8)
        for a in range(self.n):
            row = W[a].copy()
            row[a] = 0.0
            live = np.where(row > W_FLOOR * max(row.max(), 1e-30))[0]
            if not len(live):
                continue                      # e.g. Zone9: no coupling at all
            if len(live) < self.r_target:
                chan[a, live] = 0
                continue
            order = live[np.argsort(-row[live])]
            for c, part in enumerate(np.array_split(order, self.r_target)):
                chan[a, part] = c
        self.chan = chan

        # --- exertion scale and x_ref (III.6) -------------------------------
        # phi_j = sum_g (1 - a_g) * pmax_g = MW of renewable withheld by j.
        # A magnitude, so it has no sign-cancellation escape (I.3).
        self.pmax_per_zone = np.array(
            [float(np.sum(gen_pmax[np.asarray(gen_curtail_inside[z], dtype=int)]))
             if len(gen_curtail_inside[z]) else 0.0
             for z in self.zone_names])
        phi_ref = 0.5 * self.pmax_per_zone     # a_g ~ U[0,1] => E[1-a_g] = 0.5

        n_ch = self.r_target
        masks = np.zeros((self.n, n_ch, self.n))
        for a in range(self.n):
            for c in range(n_ch):
                peers = np.where(chan[a] == c)[0]
                masks[a, c, peers] = W[a, peers]      # PTDF-WEIGHTED, not binary
        x_ref_full = masks @ phi_ref                  # (n, n_ch)

        # --- PER-AGENT, PER-CHANNEL scaling ---------------------------------
        # range[a, c] = sum_{j in c} W[a,j] * pmax_j is the largest x that
        # agent a's channel c can take (every peer curtailing fully), so
        # psi = (x - range/2) / range lands in [-0.5, +0.5] for EVERY agent and
        # channel whatever the PTDF magnitudes are.  Still purely declared:
        # the operator and the action box, no run data.
        #
        # A single scale shared across agents -- x_ref.std(axis=0), which is
        # what this used to do -- is fine only while all agents have comparable
        # coupling magnitude.  Under PTDF weights they do not: measured on
        # l2rpn_idf_2023 the rows span orders of magnitude and the shared scale
        # drove cond(E[psi psi^T]) to 72,148 against 57 for the geometric
        # basis, with fit_gain still negative.  Each agent runs its own
        # estimator, so per-agent scaling costs nothing and leaks nothing.
        rng_full = 2.0 * x_ref_full                   # (n, n_ch)

        # A channel is identifiable for an agent only if it has live peers.
        # Report per-channel coverage; drop a channel only if it is dead for
        # essentially everyone (IV.4: report the effective r, do not force
        # channels to survive).
        self.coverage = (rng_full > 1e-12).mean(axis=0)
        self.spread = np.zeros(n_ch)
        for c in range(n_ch):
            col = rng_full[:, c]
            live = col[col > 1e-12]
            self.spread[c] = (live.std() / live.mean()) if len(live) > 1 else np.inf
        self.keep = [c for c in range(n_ch) if self.coverage[c] >= 2.0 / self.n]
        if not self.keep:
            raise RuntimeError(
                "every coupling channel is degenerate; the basis cannot be "
                "identified on this grid")

        self.r = len(self.keep)
        self.kept_names = [CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"
                           for c in self.keep]
        self.masks = masks[:, self.keep, :]
        self.x_ref = x_ref_full[:, self.keep]
        sc = rng_full[:, self.keep].copy()
        sc[sc < 1e-12] = 1.0                          # dead agent-channel -> psi = 0
        self.scale = sc

        # Agents with no live coupling at all: their peer prediction is
        # identically 0, so the floor property keeps them at plain MAPPO.
        self.dead_agents = [a for a in range(self.n) if (chan[a] < 0).all()]

    # -----------------------------------------------------------------------
    def exertion(self, curtail_actions):
        """phi_j for every zone agent, in MW withheld."""
        phi = np.zeros(self.n)
        for a, z in enumerate(self.zone_names):
            act = curtail_actions.get(z)
            if act is None or len(act) == 0:
                continue
            phi[a] = float(np.mean(1.0 - np.clip(act, 0.0, 1.0))) \
                * self.pmax_per_zone[a]
        return phi

    def waveforms(self, phi):
        """Centred, scaled regressor columns psi (n, r).

        Centring on the geometric/PTDF reference rather than the sample mean is
        what keeps cond finite AND keeps the model class declared rather than
        fitted (III.6).
        """
        x = self.masks @ phi
        return (x - self.x_ref) / self.scale

    def report(self):
        off = self.W[~np.eye(self.n, dtype=bool)]
        nz = off[off > 0]
        lines = ["[pact1] declared coupling basis (PTDF-weighted)"]
        lines.append(f"    W off-diagonal: nonzero={len(nz)}/{off.size}  "
                     f"spread std/mean={nz.std()/nz.mean():.4f}"
                     if len(nz) else "    W off-diagonal: ALL ZERO")
        for c in range(self.r_target):
            n_pairs = int((self.chan == c).sum())
            state = "kept" if c in self.keep else "DROPPED (degenerate)"
            nm = CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"ch{c}"
            lines.append(f"    {nm:9s} pairs={n_pairs:4d} "
                         f"coverage={self.coverage[c]:5.2f} "
                         f"spread={self.spread[c]:8.4f}  {state}")
        lines.append(f"    effective r = {self.r}  {self.kept_names}")
        if self.dead_agents:
            lines.append(f"    agents with NO coupling (stay blind): "
                         f"{[self.zone_names[a] for a in self.dead_agents]}")
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
