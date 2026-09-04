"""Generate a zone partition of arbitrary size from the grid's own adjacency.

    python make_zones.py --n-zones 22                    # write zones_definitions_N22.json
    python make_zones.py --n-zones 11 22 33              # several at once
    python make_zones.py --n-zones 11 --compare-shipped  # audit against the hand-drawn file
    python make_zones.py --dump-topology topo.json       # snapshot the grid, once
    python make_zones.py --n-zones 22 --topology topo.json   # regenerate with NO grid2op
    python make_zones.py --selfcheck                     # arithmetic checks, NO grid2op

WHY THIS EXISTS
---------------
`zones_definitions.json` hardcodes 11 hand-drawn zones and the environment
builds one agent per entry, so N is frozen at 11.  The coordination-recoverable
fraction of the derating damage -- the quantity that bounds every result in this
project -- rises steeply with N, because splitting the grid finer shrinks each
agent's LOCAL authority over its own binding line while leaving the coupling
untouched.  Testing that requires a real partition at N != 11.

`theory_scaling.py` approximated one with contiguous substation-index blocks.
That is fast but it is not electrical: two substations with adjacent ids need
not be connected.  This file partitions the substation graph itself, so the
numbers it feeds the theory are the numbers a real control-area split would
give.

METHOD
------
Recursive spectral bisection of the substation adjacency graph.  At each step
the part is split along its Fiedler vector (the eigenvector of the second
smallest Laplacian eigenvalue), which is the standard continuous relaxation of
minimum edge cut, and the cut point is placed so the two halves get their share
of the remaining zone quota.  That yields balanced parts with few tie-lines,
which is what a system operator drawing control areas is also trying to do.

Deterministic by construction: no k-means, no random restarts, eigenvector
signs canonicalised.  Two runs on the same grid give byte-identical output, so
the partition can be committed BEFORE the gates run and the git history is the
evidence that N was not chosen after seeing a method's score.

Repairs applied after bisection, in this order:
  1. connectivity  -- a zone must be a connected subgraph or it is not a zone;
                      stray fragments are given to the adjacent zone they are
                      most strongly tied to
  2. emptiness     -- a zone with no substations would build an agent with no
                      observation, so the largest zone is split to refill it
  3. balance       -- bounded boundary moves, largest zone -> smallest adjacent
                      zone, only while they strictly improve the spread and
                      never breaking connectivity

SEMANTICS OF THE OUTPUT
-----------------------
The 18 fields are the ones the shipped file carries, and they mean:

  sub_inside_ids        substations assigned to this zone.  A FULL partition:
                        every substation is in exactly one zone.
  sub_border_in_ids     inside substations with at least one line leaving
  sub_border_out_ids    outside substations exactly one line away (the halo)
  line_in_zone_idx      lines with BOTH endpoints inside     <- the zone OWNS these
  line_neighboring_idx  lines with exactly one endpoint inside (tie-lines)
  line_large_idx        line_in_zone  U  line_neighboring    (asserted disjoint)
  gen_inside_idx        generators at inside substations
  gen_border_idx        generators at halo substations
  gen_large_idx         inside U border -- the OBSERVATION halo
  gen_curtail_inside_idx  gen_inside   n  renewable          <- the ACTION head
  gen_curtail_large_idx   gen_large    n  renewable
  gen_redisp_large_idx    gen_large    n  redispatchable
  load_* / storage_*      same inside / border / large pattern

NOTE, and it matters when comparing to the shipped file: the shipped 11 zones
are HAND-DRAWN and do not follow these rules.  They cover only 101 of the 118
substations, leaving 17 on the seams owned by nobody, and their border fields
are mutually inconsistent (Zone0 lists gen_border 7,8,9 while its
sub_border_in_ids is empty, and those same generators are Zone1's gen_inside).
So `--compare-shipped` reports structure, not equality.  The consumers that
actually read these files -- utils.get_obs_act_attr_and_kwargs, pact1/basis.py,
pact1/env.py, pact1/dlr_env.py, theory_ceiling.py -- use only line_in_zone_idx,
line_large_idx, gen_inside_idx, gen_large_idx, load_large_idx,
storage_inside_idx and storage_border_idx, and all seven are well defined here.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not look at any algorithm's return, and it cannot: nothing here loads a
policy, a reward or a training log.  The only inputs are the grid's topology
and its declared generator flags.
"""
import argparse
import json
import os
import sys

import numpy as np

# Fields every zone must carry.  utils.add_missing_keys() is a no-op only when
# every zone has every key, so emitting the full set is not cosmetic -- see
# assert_valid().
ZONE_FIELDS = (
    "sub_inside_ids", "sub_border_in_ids", "sub_border_out_ids",
    "line_in_zone_idx", "line_neighboring_idx", "line_large_idx",
    "gen_inside_idx", "gen_border_idx", "gen_large_idx",
    "gen_curtail_inside_idx", "gen_curtail_large_idx", "gen_redisp_large_idx",
    "load_inside_idx", "load_border_idx", "load_large_idx",
    "storage_inside_idx", "storage_border_idx", "storage_large_idx",
)

DEFAULT_ENV_DIR_NAME = "l2rpn_idf_2023"


# --------------------------------------------------------------------------
# topology: a plain-data description of the grid, so everything below this
# line runs without grid2op
# --------------------------------------------------------------------------

class Topology:
    """Exactly the grid facts the partition needs, as plain arrays.

    Kept separate from grid2op on purpose: the generator is then testable with
    a synthetic grid (--selfcheck), reproducible from a committed snapshot
    (--topology), and cheap to reason about.
    """

    def __init__(self, n_sub, line_or_to_subid, line_ex_to_subid,
                 gen_to_subid, load_to_subid, storage_to_subid,
                 gen_renewable, gen_redispatchable, thermal_limit=None,
                 name="<unnamed>"):
        self.name = str(name)
        self.n_sub = int(n_sub)
        self.line_or = np.asarray(line_or_to_subid, dtype=int)
        self.line_ex = np.asarray(line_ex_to_subid, dtype=int)
        self.gen_sub = np.asarray(gen_to_subid, dtype=int)
        self.load_sub = np.asarray(load_to_subid, dtype=int)
        self.storage_sub = np.asarray(storage_to_subid, dtype=int)
        self.gen_renewable = np.asarray(gen_renewable, dtype=bool)
        self.gen_redispatchable = np.asarray(gen_redispatchable, dtype=bool)
        if thermal_limit is None:
            thermal_limit = np.ones(len(self.line_or), dtype=float)
        self.thermal_limit = np.asarray(thermal_limit, dtype=float)
        self._validate()

    @property
    def n_line(self):
        return len(self.line_or)

    @property
    def n_gen(self):
        return len(self.gen_sub)

    @property
    def n_load(self):
        return len(self.load_sub)

    @property
    def n_storage(self):
        return len(self.storage_sub)

    def _validate(self):
        if len(self.line_ex) != self.n_line:
            raise ValueError("line_or / line_ex length mismatch")
        if len(self.thermal_limit) != self.n_line:
            raise ValueError("thermal_limit length mismatch")
        if len(self.gen_renewable) != self.n_gen or \
                len(self.gen_redispatchable) != self.n_gen:
            raise ValueError("generator flag length mismatch")
        for nm, arr in (("line_or", self.line_or), ("line_ex", self.line_ex),
                        ("gen", self.gen_sub), ("load", self.load_sub),
                        ("storage", self.storage_sub)):
            if len(arr) and (arr.min() < 0 or arr.max() >= self.n_sub):
                raise ValueError(f"{nm}_to_subid out of range [0, {self.n_sub})")

    # -- io -----------------------------------------------------------------

    @classmethod
    def from_env(cls, env, name=None):
        """Read the topology off a live grid2op environment."""
        try:
            limits = np.asarray(env.get_thermal_limit(), dtype=float)
        except Exception:                                     # noqa: BLE001
            # A thermal-limit read is refused on an uninitialised or game-over
            # env.  Unweighted adjacency is a worse proxy but never a crash,
            # and the weighting actually used is recorded in the meta file.
            limits = None
        return cls(n_sub=int(env.n_sub),
                   line_or_to_subid=env.line_or_to_subid,
                   line_ex_to_subid=env.line_ex_to_subid,
                   gen_to_subid=env.gen_to_subid,
                   load_to_subid=env.load_to_subid,
                   storage_to_subid=env.storage_to_subid,
                   gen_renewable=env.gen_renewable,
                   gen_redispatchable=env.gen_redispatchable,
                   thermal_limit=limits,
                   name=name or getattr(env, "env_name", "<env>"))

    def to_dict(self):
        return {
            "name": self.name,
            "n_sub": self.n_sub,
            "line_or_to_subid": self.line_or.tolist(),
            "line_ex_to_subid": self.line_ex.tolist(),
            "gen_to_subid": self.gen_sub.tolist(),
            "load_to_subid": self.load_sub.tolist(),
            "storage_to_subid": self.storage_sub.tolist(),
            "gen_renewable": self.gen_renewable.astype(int).tolist(),
            "gen_redispatchable": self.gen_redispatchable.astype(int).tolist(),
            "thermal_limit": self.thermal_limit.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(n_sub=d["n_sub"],
                   line_or_to_subid=d["line_or_to_subid"],
                   line_ex_to_subid=d["line_ex_to_subid"],
                   gen_to_subid=d["gen_to_subid"],
                   load_to_subid=d["load_to_subid"],
                   storage_to_subid=d["storage_to_subid"],
                   gen_renewable=np.asarray(d["gen_renewable"], dtype=bool),
                   gen_redispatchable=np.asarray(d["gen_redispatchable"],
                                                 dtype=bool),
                   thermal_limit=d.get("thermal_limit"),
                   name=d.get("name", "<snapshot>"))


# --------------------------------------------------------------------------
# adjacency
# --------------------------------------------------------------------------

def adjacency(topo, weights="thermal"):
    """Substation adjacency, symmetric, zero diagonal.

    weights:
      thermal  sum of thermal limits of the circuits joining two substations.
               Corridor strength: two substations joined by a 1000 MW double
               circuit are electrically closer than two joined by one 200 MW
               line, and a partition should be reluctant to cut the former.
      count    number of parallel circuits
      unit     1 if joined at all
    """
    A = np.zeros((topo.n_sub, topo.n_sub), dtype=float)
    if weights == "thermal":
        w = np.asarray(topo.thermal_limit, dtype=float)
        # Scale out the units; only ratios reach the eigenvectors.
        pos = w[w > 0]
        w = w / (float(np.median(pos)) if len(pos) else 1.0)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    elif weights in ("count", "unit"):
        w = np.ones(topo.n_line, dtype=float)
    else:
        raise ValueError(f"unknown weights {weights!r}")

    for l in range(topo.n_line):
        a, b = int(topo.line_or[l]), int(topo.line_ex[l])
        if a == b:
            continue                       # a loop line couples nothing
        A[a, b] += w[l]
        A[b, a] += w[l]
    if weights == "unit":
        A = (A > 0).astype(float)
    return A


def _components(A, nodes):
    """Connected components of the subgraph induced on `nodes`.

    Returned largest-first, ties broken by lowest member id, so every caller
    that says "keep comps[0]" gets a reproducible answer.
    """
    nodes = np.asarray(nodes, dtype=int)
    if not len(nodes):
        return []
    sub = A[np.ix_(nodes, nodes)] > 0
    seen = np.zeros(len(nodes), dtype=bool)
    out = []
    for s in range(len(nodes)):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(sub[u] & ~seen)[0]:
                seen[v] = True
                stack.append(int(v))
        out.append(nodes[np.sort(np.asarray(comp, dtype=int))])
    return sorted(out, key=lambda c: (-len(c), int(c[0])))


def _fiedler(Asub):
    """Fiedler vector of a dense weighted Laplacian, sign-canonicalised.

    The eigenvector of the second-smallest eigenvalue.  An eigenvector is only
    defined up to sign, and an arbitrary sign would make the whole partition
    non-reproducible, so the sign is pinned: the largest-magnitude entry is
    made positive, ties broken by index.  n_sub is ~120 here, so a dense
    symmetric eigendecomposition costs microseconds and there is no reason to
    reach for anything sparse.
    """
    n = Asub.shape[0]
    if n <= 2:
        return np.arange(n, dtype=float)
    L = np.diag(Asub.sum(axis=1)) - Asub
    L = 0.5 * (L + L.T)                      # kill asymmetry from rounding
    _vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1].astype(float)
    k = int(np.lexsort((np.arange(n), -np.abs(f)))[0])
    if f[k] < 0:
        f = -f
    return f


def _split2(A, nodes, order_key, frac, min_left, min_right):
    """Cut `nodes` in two along the Fiedler vector, then repair the halves.

    The Fiedler cut minimises a CONTINUOUS relaxation, so nothing stops it from
    stranding a fragment on the wrong side of the sweep.  Repairing that at the
    very end -- after every level of the recursion has already committed to its
    quota -- is far too late: on a 118-substation grid it moved 65 substations
    and left zones of 3 and 30 where 11 and 11 were asked for.  So both repairs
    happen HERE, while there are still only two parts to reason about:

      1. connectivity  each fragment joins the side its edges pull it towards
      2. balance       boundary nodes move heavy -> light until the split hits
                       `frac`, and only while both sides stay connected
    """
    nodes = np.asarray(nodes, dtype=int)
    f = _fiedler(A[np.ix_(nodes, nodes)])
    # lexsort tie-break on substation id, so equal Fiedler entries -- which a
    # symmetric grid produces plenty of -- do not depend on sort stability
    order = nodes[np.lexsort((nodes, f))]

    wts = order_key[order].astype(float)
    tot = float(wts.sum())
    if tot <= 0:
        cut = int(round(len(order) * frac))
    else:
        cut = int(np.argmin(np.abs(np.cumsum(wts) - tot * frac))) + 1
    cut = int(np.clip(cut, min_left, len(order) - min_right))

    side = np.zeros(len(A), dtype=np.int8) - 1      # -1 = not in this subgraph
    side[order[:cut]] = 0
    side[order[cut:]] = 1

    # -- 1. connectivity ------------------------------------------------
    for _ in range(64):
        changed = False
        for s in (0, 1):
            members = np.where(side == s)[0]
            comps = _components(A, members)
            if len(comps) <= 1:
                continue
            for frag in comps[1:]:               # comps[0] is the largest
                # only move it if the other side would actually hold it
                if not any(side[v] == 1 - s
                           for u in frag for v in np.where(A[u] > 0)[0]):
                    continue
                side[frag] = 1 - s
                changed = True
        if not changed:
            break

    # -- 2. balance, without breaking what step 1 just fixed -------------
    target = tot * frac
    for _ in range(4 * len(nodes) + 8):
        left = np.where(side == 0)[0]
        right = np.where(side == 1)[0]
        if not len(left) or not len(right):
            break
        wl = float(order_key[left].sum())
        hi, lo = (0, 1) if wl > target else (1, 0)
        gap = abs(wl - target)
        if gap <= 1e-9:
            break
        donors = np.where(side == hi)[0]
        if len(donors) <= (min_left if hi == 0 else min_right):
            break
        best = None
        for u in donors:
            nbr = np.where(A[u] > 0)[0]
            if not np.any(side[nbr] == lo):
                continue                          # must border the other side
            k = float(order_key[u])
            # moving u overshoots if it is bigger than twice the gap
            if k > 2.0 * gap:
                continue
            rest = donors[donors != u]
            if not len(rest) or len(_components(A, rest)) != 1:
                continue                          # would disconnect the donor
            score = (2.0 * gap - k, -int(u))      # biggest safe step, then id
            if best is None or score > best[0]:
                best = (score, int(u))
        if best is None:
            break
        side[best[1]] = lo

    return np.where(side == 0)[0], np.where(side == 1)[0]


def _bisect(A, nodes, k, order_key):
    """Split `nodes` into k connected parts by recursive spectral bisection."""
    nodes = np.asarray(nodes, dtype=int)
    if k <= 1 or len(nodes) <= 1:
        return [nodes]
    if k >= len(nodes):
        return [np.array([n], dtype=int) for n in nodes]

    kl = k // 2
    left, right = _split2(A, nodes, order_key, kl / k, kl, k - kl)
    return (_bisect(A, left, kl, order_key) +
            _bisect(A, right, k - kl, order_key))


# --------------------------------------------------------------------------
# repairs
# --------------------------------------------------------------------------

def _repair_connectivity(A, sub_zone, n_zones, max_passes=64):
    """A zone must be a connected subgraph, or it is not a control area.

    Spectral bisection cuts a continuous relaxation, so it can strand a
    fragment on the wrong side.  Each fragment is handed to whichever adjacent
    zone it is most strongly tied to; ties broken by lowest zone id so the
    result stays deterministic.
    """
    moved = 0
    for _ in range(max_passes):
        changed = False
        for z in range(n_zones):
            nodes = np.where(sub_zone == z)[0]
            comps = _components(A, nodes)
            if len(comps) <= 1:
                continue
            for frag in comps[1:]:           # comps[0] is the largest
                tie = np.zeros(n_zones)
                for u in frag:
                    for v in np.where(A[u] > 0)[0]:
                        zv = int(sub_zone[v])
                        if zv != z:
                            tie[zv] += A[u, v]
                if tie.max() <= 0:
                    continue                 # island: nothing to attach it to
                sub_zone[frag] = int(np.argmax(tie))
                moved += len(frag)
                changed = True
        if not changed:
            break
    return moved


def _repair_empty(A, sub_zone, n_zones, order_key):
    """No zone may end up empty: an agent with no substations has no
    observation and no action, and the env would build a degenerate space."""
    refilled = 0
    for _ in range(n_zones):
        counts = np.array([int((sub_zone == z).sum()) for z in range(n_zones)])
        empties = np.where(counts == 0)[0]
        if not len(empties):
            break
        z_empty = int(empties[0])
        z_big = int(np.argmax(counts))
        if counts[z_big] < 2:
            raise RuntimeError(
                f"cannot fill zone {z_empty}: no zone has 2+ substations. "
                f"n_zones={n_zones} exceeds what this grid can carry.")
        halves = _bisect(A, np.where(sub_zone == z_big)[0], 2, order_key)
        # give the empty zone the smaller half, keeping zone ids stable
        halves = sorted(halves, key=lambda h: (len(h), int(h[0])))
        sub_zone[halves[0]] = z_empty
        refilled += 1
        _repair_connectivity(A, sub_zone, n_zones)
    return refilled


def _repair_balance(A, sub_zone, n_zones, order_key, max_passes=400):
    """Boundary moves that flatten the zone weights, over ALL adjacent pairs.

    Objective: sum of squared deviations from the mean weight.  Moving a
    substation of weight k from zone h to zone l changes it by
    `2k(w_l - w_h + k)`, so a move helps exactly when `w_h - w_l > k`, which is
    both the acceptance test and the ranking key.

    Considering only the single (heaviest, lightest) pair -- the obvious first
    implementation -- stalls the moment those two zones do not share a border,
    which on a real grid is most of the time: it made 0 moves and left zones of
    3 and 30 substations side by side.  Scanning every adjacent pair costs
    nothing at this size and actually converges.

    A move is taken only when it also leaves the donor connected, so the
    connectivity guarantee established during bisection survives.
    """
    moved = 0
    w = np.array([order_key[sub_zone == z].sum() for z in range(n_zones)],
                 dtype=float)
    for _ in range(max_passes):
        best, best_gain = None, 1e-12
        for u in range(len(sub_zone)):
            z_hi = int(sub_zone[u])
            if int((sub_zone == z_hi).sum()) <= 1:
                continue                        # never empty a zone
            k = float(order_key[u])
            nbr_zones = {int(sub_zone[v]) for v in np.where(A[u] > 0)[0]}
            nbr_zones.discard(z_hi)
            for z_lo in nbr_zones:
                gain = w[z_hi] - w[z_lo] - k
                if gain <= best_gain:
                    continue
                donors = np.where(sub_zone == z_hi)[0]
                rest = donors[donors != u]
                if not len(rest) or len(_components(A, rest)) != 1:
                    continue                    # would disconnect the donor
                best, best_gain = (int(u), z_hi, z_lo, k), gain
        if best is None:
            break
        u, z_hi, z_lo, k = best
        sub_zone[u] = z_lo
        w[z_hi] -= k
        w[z_lo] += k
        moved += 1
    return moved


# --------------------------------------------------------------------------
# partition -> the 18 fields
# --------------------------------------------------------------------------

def _order_key(topo, criterion):
    """Per-substation weight the partition tries to spread evenly."""
    if criterion == "sub":
        return np.ones(topo.n_sub, dtype=float)
    k = np.zeros(topo.n_sub, dtype=float)
    if criterion == "gen":
        src = topo.gen_sub
    elif criterion == "curtail":
        src = topo.gen_sub[topo.gen_renewable]
    elif criterion == "load":
        src = topo.load_sub
    else:
        raise ValueError(f"unknown balance criterion {criterion!r}")
    for s in src:
        k[int(s)] += 1.0
    # A substation with none of the thing being balanced still has to weigh
    # something, or bisection sweeps them all into one part.
    return k + 0.25


def partition_substations(topo, n_zones, weights="thermal", balance_by="sub",
                          balance=True):
    """Assign every substation to exactly one of n_zones connected zones."""
    if n_zones < 1:
        raise ValueError("n_zones must be >= 1")
    if n_zones > topo.n_sub:
        raise ValueError(f"n_zones={n_zones} exceeds n_sub={topo.n_sub}")

    A = adjacency(topo, weights=weights)
    key = _order_key(topo, balance_by)

    # Disconnected grids: give each component a share of the zones, so no zone
    # straddles two components (it could not be connected if it did).
    comps = _components(A, np.arange(topo.n_sub))
    sub_zone = np.full(topo.n_sub, -1, dtype=int)
    if len(comps) == 1:
        parts = _bisect(A, comps[0], n_zones, key)
    else:
        sizes = np.array([len(c) for c in comps], dtype=float)
        quota = np.maximum(1, np.round(sizes / sizes.sum() * n_zones)).astype(int)
        quota = np.minimum(quota, sizes.astype(int))
        order = np.argsort(-sizes)
        i = 0
        while quota.sum() != n_zones:
            c = int(order[i % len(order)])
            if quota.sum() > n_zones and quota[c] > 1:
                quota[c] -= 1
            elif quota.sum() < n_zones and quota[c] < len(comps[c]):
                quota[c] += 1
            i += 1
            if i > 10000:
                raise RuntimeError("could not distribute zones over components")
        parts = []
        for c, q in zip(comps, quota):
            parts.extend(_bisect(A, c, int(q), key))

    for z, part in enumerate(parts):
        sub_zone[part] = z
    if (sub_zone < 0).any():
        raise RuntimeError("bisection left substations unassigned")

    stats = {"moved_connectivity": _repair_connectivity(A, sub_zone, n_zones)}
    stats["refilled_empty"] = _repair_empty(A, sub_zone, n_zones, key)
    stats["moved_balance"] = (
        _repair_balance(A, sub_zone, n_zones, key) if balance else 0)
    stats["moved_connectivity"] += _repair_connectivity(A, sub_zone, n_zones)

    # A zone emptied by the balance pass would otherwise be a silent failure.
    counts = np.array([int((sub_zone == z).sum()) for z in range(n_zones)])
    if (counts == 0).any():
        raise RuntimeError(
            f"zone(s) {np.where(counts == 0)[0].tolist()} empty after repair")
    return sub_zone, A, stats


def build_zones(topo, sub_zone, n_zones, zone_name=lambda z: f"Zone{z}"):
    """Derive the 18 index lists per zone from a substation assignment."""
    lor, lex = topo.line_or, topo.line_ex

    def _il(a):
        """Sorted, de-duplicated, plain python ints.  json needs the last one:
        np.int64 is not serialisable and the failure is at write time, after
        the expensive part."""
        a = np.asarray(list(a), dtype=int)
        return [int(x) for x in np.unique(a)] if a.size else []

    def _at(sub_ids, elem_sub):
        if not len(elem_sub) or not len(sub_ids):
            return np.array([], dtype=int)
        return np.where(np.isin(elem_sub, sub_ids))[0]

    renew = np.where(topo.gen_renewable)[0]
    redisp = np.where(topo.gen_redispatchable)[0]

    out = {}
    for z in range(n_zones):
        inside = np.where(sub_zone == z)[0]
        in_set = np.zeros(topo.n_sub, dtype=bool)
        in_set[inside] = True

        n_end_in = in_set[lor].astype(int) + in_set[lex].astype(int)
        line_in = np.where(n_end_in == 2)[0]
        line_nb = np.where(n_end_in == 1)[0]

        # halo: substations one line away, on the far side of a tie-line
        halo, border_in = set(), set()
        for l in line_nb:
            a, b = int(lor[l]), int(lex[l])
            if in_set[a]:
                border_in.add(a)
                halo.add(b)
            else:
                border_in.add(b)
                halo.add(a)
        halo_arr = np.asarray(sorted(halo), dtype=int)

        gen_in = _at(inside, topo.gen_sub)
        gen_bd = _at(halo_arr, topo.gen_sub)
        gen_lg = np.union1d(gen_in, gen_bd)
        load_in = _at(inside, topo.load_sub)
        load_bd = _at(halo_arr, topo.load_sub)
        sto_in = _at(inside, topo.storage_sub)
        sto_bd = _at(halo_arr, topo.storage_sub)

        out[zone_name(z)] = {
            "sub_inside_ids": _il(inside),
            "sub_border_in_ids": _il(border_in),
            "sub_border_out_ids": _il(halo_arr),
            "line_in_zone_idx": _il(line_in),
            "line_neighboring_idx": _il(line_nb),
            "line_large_idx": _il(np.union1d(line_in, line_nb)),
            "gen_inside_idx": _il(gen_in),
            "gen_border_idx": _il(gen_bd),
            "gen_large_idx": _il(gen_lg),
            "gen_curtail_inside_idx": _il(np.intersect1d(gen_in, renew)),
            "gen_curtail_large_idx": _il(np.intersect1d(gen_lg, renew)),
            "gen_redisp_large_idx": _il(np.intersect1d(gen_lg, redisp)),
            "load_inside_idx": _il(load_in),
            "load_border_idx": _il(load_bd),
            "load_large_idx": _il(np.union1d(load_in, load_bd)),
            "storage_inside_idx": _il(sto_in),
            "storage_border_idx": _il(sto_bd),
            "storage_large_idx": _il(np.union1d(sto_in, sto_bd)),
        }
    return out


# --------------------------------------------------------------------------
# assertions -- every one of these is something a consumer would otherwise
# discover at training time, or worse, not discover
# --------------------------------------------------------------------------

def assert_valid(zones, topo, n_zones, strict_curtail=False):
    """Raise on any structural violation.  Returns a list of soft warnings."""
    warn = []
    names = list(zones.keys())
    if len(names) != n_zones:
        raise AssertionError(f"{len(names)} zones written, expected {n_zones}")

    # --- schema ---------------------------------------------------------
    # utils.add_missing_keys() only stays a no-op when every zone has every
    # key, so a missing field is not cosmetic: it reaches a code path that
    # indexes a dict with a set.
    for z in names:
        missing = set(ZONE_FIELDS) - set(zones[z].keys())
        extra = set(zones[z].keys()) - set(ZONE_FIELDS)
        if missing:
            raise AssertionError(f"{z} missing fields {sorted(missing)}")
        if extra:
            raise AssertionError(f"{z} has unexpected fields {sorted(extra)}")
        for f in ZONE_FIELDS:
            v = zones[z][f]
            if not isinstance(v, list) or any(not isinstance(x, int) for x in v):
                raise AssertionError(f"{z}.{f} must be a list of python ints")
            if v != sorted(v) or len(set(v)) != len(v):
                raise AssertionError(f"{z}.{f} must be sorted and unique")

    def S(z, f):
        return set(zones[z][f])

    # --- substations partition exactly ----------------------------------
    seen, dupes = set(), set()
    for z in names:
        s = S(z, "sub_inside_ids")
        if not s:
            raise AssertionError(f"{z} has no substations")
        dupes |= (seen & s)
        seen |= s
    if dupes:
        raise AssertionError(f"substations in >1 zone: {sorted(dupes)}")
    if seen != set(range(topo.n_sub)):
        raise AssertionError(
            f"substations not covered: {sorted(set(range(topo.n_sub)) - seen)}")

    zone_of_sub = {}
    for z in names:
        for s in S(z, "sub_inside_ids"):
            zone_of_sub[s] = z

    # --- lines ----------------------------------------------------------
    owned = {}
    for z in names:
        li, nb, lg = (S(z, "line_in_zone_idx"), S(z, "line_neighboring_idx"),
                      S(z, "line_large_idx"))
        if li & nb:
            raise AssertionError(
                f"{z}: line_in and line_neighboring overlap {sorted(li & nb)}")
        if (li | nb) != lg:
            raise AssertionError(f"{z}: line_large != line_in U line_neighboring")
        for l in li:
            if l in owned:
                raise AssertionError(f"line {l} owned by {owned[l]} and {z}")
            owned[l] = z
    # every line is either owned by exactly one zone, or is a tie-line seen as
    # neighbouring by exactly the two zones it joins
    for l in range(topo.n_line):
        za = zone_of_sub[int(topo.line_or[l])]
        zb = zone_of_sub[int(topo.line_ex[l])]
        if za == zb:
            if owned.get(l) != za:
                raise AssertionError(f"line {l} inside {za} but not owned by it")
        else:
            if l in owned:
                raise AssertionError(f"tie-line {l} wrongly owned by {owned[l]}")
            for zz in (za, zb):
                if l not in S(zz, "line_neighboring_idx"):
                    raise AssertionError(
                        f"tie-line {l} missing from {zz}.line_neighboring_idx")

    # --- elements -------------------------------------------------------
    for kind, sub_of, n_elem in (("gen", topo.gen_sub, topo.n_gen),
                                 ("load", topo.load_sub, topo.n_load),
                                 ("storage", topo.storage_sub, topo.n_storage)):
        seen, dupes = set(), set()
        for z in names:
            ins, bd, lg = (S(z, f"{kind}_inside_idx"), S(z, f"{kind}_border_idx"),
                           S(z, f"{kind}_large_idx"))
            if (ins | bd) != lg:
                raise AssertionError(f"{z}: {kind}_large != inside U border")
            if ins & bd:
                raise AssertionError(f"{z}: {kind} inside/border overlap")
            for e in ins:
                if int(sub_of[e]) not in S(z, "sub_inside_ids"):
                    raise AssertionError(
                        f"{z}: {kind} {e} is not at an inside substation")
            for e in bd:
                if int(sub_of[e]) not in S(z, "sub_border_out_ids"):
                    raise AssertionError(
                        f"{z}: {kind} {e} is not at a halo substation")
            dupes |= (seen & ins)
            seen |= ins
        if dupes:
            raise AssertionError(f"{kind} in >1 zone's inside: {sorted(dupes)}")
        if seen != set(range(n_elem)):
            raise AssertionError(
                f"{kind}s not covered: {sorted(set(range(n_elem)) - seen)}")

    # --- derived generator sets ----------------------------------------
    renew = set(np.where(topo.gen_renewable)[0].tolist())
    redisp = set(np.where(topo.gen_redispatchable)[0].tolist())
    for z in names:
        if S(z, "gen_curtail_inside_idx") != S(z, "gen_inside_idx") & renew:
            raise AssertionError(f"{z}: gen_curtail_inside_idx wrong")
        if S(z, "gen_curtail_large_idx") != S(z, "gen_large_idx") & renew:
            raise AssertionError(f"{z}: gen_curtail_large_idx wrong")
        if S(z, "gen_redisp_large_idx") != S(z, "gen_large_idx") & redisp:
            raise AssertionError(f"{z}: gen_redisp_large_idx wrong")

    # --- borders --------------------------------------------------------
    for z in names:
        ins, bin_, bout = (S(z, "sub_inside_ids"), S(z, "sub_border_in_ids"),
                           S(z, "sub_border_out_ids"))
        if not bin_ <= ins:
            raise AssertionError(f"{z}: sub_border_in_ids not inside the zone")
        if bout & ins:
            raise AssertionError(f"{z}: sub_border_out_ids overlaps the zone")

    # --- soft: legal, but decides whether this N is usable ---------------
    for z in names:
        if not S(z, "gen_curtail_inside_idx"):
            warn.append(f"{z} has NO curtailable generator: its action head is "
                        f"shape (0,) and it can never act")
        if not S(z, "line_in_zone_idx"):
            warn.append(f"{z} owns NO line: its own-loading sensor is empty and "
                        f"PACT's self-sensitivity row is all zeros")
    if strict_curtail and any("NO curtailable" in w for w in warn):
        raise AssertionError(
            "--require-curtail: some zone has no curtailable generator")
    return warn


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def summarise(zones, topo):
    names = list(zones.keys())
    owned = set()
    for z in names:
        owned |= set(zones[z]["line_in_zone_idx"])
    tie = topo.n_line - len(owned)
    rows = []
    for z in names:
        v = zones[z]
        rows.append((z, len(v["sub_inside_ids"]), len(v["line_in_zone_idx"]),
                     len(v["line_neighboring_idx"]), len(v["gen_inside_idx"]),
                     len(v["gen_curtail_inside_idx"]),
                     len(v["load_inside_idx"]), len(v["storage_inside_idx"]),
                     len(v["line_large_idx"]), len(v["gen_large_idx"])))
    return rows, tie


def print_report(zones, topo, n_zones, stats, warn):
    rows, tie = summarise(zones, topo)
    n_curt = int(topo.gen_renewable.sum())
    print(f"\n  N={n_zones}  {topo.n_sub} substations, {topo.n_line} lines, "
          f"{topo.n_gen} gens ({n_curt} curtailable), {topo.n_load} loads, "
          f"{topo.n_storage} storage")
    print(f"  repairs: connectivity {stats['moved_connectivity']} subs, "
          f"empty-zone refills {stats['refilled_empty']}, "
          f"balance moves {stats['moved_balance']}")
    print(f"  {'zone':>8} {'subs':>5} {'lines':>6} {'tie':>5} {'gens':>5} "
          f"{'curt':>5} {'loads':>6} {'stor':>5} | {'obs lines':>9} "
          f"{'obs gens':>8}")
    for r in rows:
        print(f"  {r[0]:>8} {r[1]:5d} {r[2]:6d} {r[3]:5d} {r[4]:5d} "
              f"{r[5]:5d} {r[6]:6d} {r[7]:5d} | {r[8]:9d} {r[9]:8d}")
    subs = np.array([r[1] for r in rows])
    curt = np.array([r[5] for r in rows])
    print(f"  substations/zone  min {subs.min()} max {subs.max()} "
          f"mean {subs.mean():.1f}")
    print(f"  curtailable/zone  min {curt.min()} max {curt.max()} "
          f"mean {curt.mean():.2f}   zones with none: {int((curt == 0).sum())}")
    print(f"  TIE-LINES (owned by nobody): {tie}/{topo.n_line} "
          f"= {tie / max(topo.n_line, 1):.1%}")
    print("  ^ the mechanism behind PEER rising with N: an overload on a line")
    print("    nobody owns cannot be fixed by any single agent.")
    for w in warn:
        print(f"  WARNING: {w}")


def compare_shipped(zones, shipped_path):
    """Structural comparison against the hand-drawn 11-zone file.

    Deliberately not an equality test.  The shipped zones cover 101 of 118
    substations, leave 17 on the seams owned by nobody, and their border fields
    follow no single rule -- so an exact diff would be noise.  What is worth
    checking is that the generated file is not structurally WORSE: comparable
    tie-line fraction, comparable per-zone sizes, no zone starved of lines.
    """
    with open(shipped_path, "r", encoding="utf-8") as f:
        ship = json.load(f)
    s_sub, s_owned = set(), set()
    for v in ship.values():
        s_sub |= set(v["sub_inside_ids"])
        s_owned |= set(v["line_in_zone_idx"])
    g_sub, g_owned = set(), set()
    for v in zones.values():
        g_sub |= set(v["sub_inside_ids"])
        g_owned |= set(v["line_in_zone_idx"])
    print("\n  --- against the shipped hand-drawn partition ---")
    print(f"  zones            shipped {len(ship):3d}   generated {len(zones):3d}")
    print(f"  substations held shipped {len(s_sub):3d}   generated {len(g_sub):3d}"
          f"   (shipped leaves {len(g_sub - s_sub)} on the seams)")
    print(f"  lines owned      shipped {len(s_owned):3d}   "
          f"generated {len(g_owned):3d}")
    ss = np.array([len(v['sub_inside_ids']) for v in ship.values()])
    gs = np.array([len(v['sub_inside_ids']) for v in zones.values()])
    print(f"  subs/zone spread shipped {ss.min()}-{ss.max()} (sd {ss.std():.1f})"
          f"   generated {gs.min()}-{gs.max()} (sd {gs.std():.1f})")
    sc = np.array([len(v['gen_curtail_inside_idx']) for v in ship.values()])
    gc = np.array([len(v['gen_curtail_inside_idx']) for v in zones.values()])
    print(f"  curtailable/zone shipped {sc.min()}-{sc.max()} "
          f"({int((sc == 0).sum())} zones with none)   "
          f"generated {gc.min()}-{gc.max()} "
          f"({int((gc == 0).sum())} zones with none)")
    print("  NOTE: not an equality test -- the shipped zones are hand-drawn and")
    print("  follow no single documented rule.  See this file's docstring.")


# --------------------------------------------------------------------------
# environment loading (the only part that needs grid2op)
# --------------------------------------------------------------------------

def load_topology(args):
    if args.topology:
        with open(args.topology, "r", encoding="utf-8") as f:
            return Topology.from_dict(json.load(f))
    import grid2op                                            # noqa: PLC0415
    from grid2op.Action import PlayableAction                 # noqa: PLC0415
    try:
        from lightsim2grid import LightSimBackend             # noqa: PLC0415
        backend = LightSimBackend()
    except ImportError:
        from grid2op.Backend import PandaPowerBackend         # noqa: PLC0415
        backend = PandaPowerBackend()
    path = args.env
    if path is None:
        try:
            from utils import G2OP_ENV_DIR                    # noqa: PLC0415
            path = os.path.join(G2OP_ENV_DIR, DEFAULT_ENV_DIR_NAME)
        except Exception:                                     # noqa: BLE001
            path = DEFAULT_ENV_DIR_NAME
    env = grid2op.make(path, action_class=PlayableAction, backend=backend)
    return Topology.from_env(env, name=str(path))


def partition_path(n_zones, env_dir=None):
    """Where the partition file for N zones lives. N=11 is the shipped file."""
    if env_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        env_dir = os.path.join(os.path.dirname(here), "BenchMARL", "benchmarl",
                               "environments", "G2OpPowerGrid")
    name = ("zones_definitions.json" if int(n_zones) == 11
            else f"zones_definitions_N{int(n_zones)}.json")
    return os.path.abspath(os.path.join(env_dir, name))


def select_partition(n_zones, env_dir=None):
    """Point this process (and any child it spawns) at the N-zone partition.

    Shared by gate_severity.py, theory_ceiling.py and theory_scaling.py so the
    gates, the ceiling decomposition and the scaling curve cannot silently run
    on different partitions -- which would make the numbers they print
    incomparable while every one of them looked fine on its own.

    Returns (path, zone_names) with zone_names in the SAME lexicographic order
    the environment assigns agents in.
    """
    path = partition_path(n_zones, env_dir)
    if not os.path.exists(path):
        raise SystemExit(
            f"--n-zones {n_zones}: {path} does not exist.\n"
            f"Generate it first:  python make_zones.py --n-zones {n_zones}")
    with open(path, "r", encoding="utf-8") as f:
        zones = json.load(f)
    if len(zones) != int(n_zones):
        raise SystemExit(
            f"{path} defines {len(zones)} zones, not {n_zones}")
    os.environ["G2OP_ZONES_FILE"] = path
    return path, sorted(zones.keys())


def _git_rev():
    try:
        import subprocess                                     # noqa: PLC0415
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                         # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------
# self-check: the arithmetic, on a synthetic grid, with NO grid2op
# --------------------------------------------------------------------------

def _toy_topology(rows=4, cols=5, seed=0):
    """A rows x cols lattice of substations with a line on every lattice edge.

    Chosen over a random graph because its structure is known: a lattice has an
    obvious balanced partition, so a bisection that produces something wild is
    visibly wrong rather than merely unfamiliar.
    """
    rng = np.random.RandomState(seed)
    n_sub = rows * cols

    def idx(r, c):
        return r * cols + c

    lor, lex, lim = [], [], []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                lor.append(idx(r, c))
                lex.append(idx(r, c + 1))
                lim.append(300.0)
            if r + 1 < rows:
                lor.append(idx(r, c))
                lex.append(idx(r + 1, c))
                lim.append(200.0)
    n_gen = 12
    gen_sub = rng.choice(n_sub, size=n_gen, replace=True)
    gen_sub[:6] = np.arange(6)                 # make coverage predictable
    renew = np.zeros(n_gen, dtype=bool)
    renew[::2] = True
    redisp = np.zeros(n_gen, dtype=bool)
    redisp[1::2] = True
    return Topology(n_sub=n_sub, line_or_to_subid=lor, line_ex_to_subid=lex,
                    gen_to_subid=gen_sub,
                    load_to_subid=rng.choice(n_sub, size=15, replace=True),
                    storage_to_subid=rng.choice(n_sub, size=3, replace=True),
                    gen_renewable=renew, gen_redispatchable=redisp,
                    thermal_limit=lim, name=f"toy{rows}x{cols}")


def _raises(zones, topo, n):
    try:
        assert_valid(zones, topo, n)
        return False
    except AssertionError:
        return True


def selfcheck():
    """Arithmetic checks that run without grid2op, torch or a dataset.

    Same purpose as pact1/selfcheck.py: this cannot prove the partition is a
    good one, it proves the arithmetic is not the reason if it is not.
    """
    import copy as _copy
    checks, failed = 0, []

    def ck(name, cond):
        nonlocal checks
        checks += 1
        if not cond:
            failed.append(name)
            print(f"  FAIL  {name}")

    print("make_zones self-check (no grid2op required)")

    topo = _toy_topology()
    print(f"  toy grid: {topo.n_sub} subs, {topo.n_line} lines, "
          f"{topo.n_gen} gens, {topo.n_load} loads, {topo.n_storage} storage")

    # -- adjacency -------------------------------------------------------
    for w in ("thermal", "count", "unit"):
        A = adjacency(topo, weights=w)
        ck(f"adjacency[{w}] symmetric", np.allclose(A, A.T))
        ck(f"adjacency[{w}] zero diagonal", np.allclose(np.diag(A), 0.0))
        ck(f"adjacency[{w}] finite", bool(np.isfinite(A).all()))
        ck(f"adjacency[{w}] non-negative", bool((A >= 0).all()))
    Au = adjacency(topo, weights="unit")
    ck("adjacency[unit] is 0/1", set(np.unique(Au).tolist()) <= {0.0, 1.0})
    ck("lattice degree <= 4", bool((Au.sum(1) <= 4).all()))

    # -- fiedler determinism ---------------------------------------------
    At = adjacency(topo, "thermal")
    f1, f2 = _fiedler(At), _fiedler(At)
    ck("fiedler deterministic", np.array_equal(f1, f2))
    ck("fiedler sign canonical", f1[int(np.argmax(np.abs(f1)))] > 0)

    # -- partition over a range of N -------------------------------------
    for n in (1, 2, 3, 4, 5, 7, 11, 19, topo.n_sub):
        sub_zone, A_, _st = partition_substations(topo, n)
        ck(f"N={n}: every substation assigned", bool((sub_zone >= 0).all()))
        ck(f"N={n}: exactly {n} non-empty zones", len(np.unique(sub_zone)) == n)
        ck(f"N={n}: every zone connected",
           all(len(_components(A_, np.where(sub_zone == z)[0])) == 1
               for z in range(n)))
        zones = build_zones(topo, sub_zone, n)
        try:
            assert_valid(zones, topo, n)
            ck(f"N={n}: all structural invariants", True)
        except AssertionError as exc:
            ck(f"N={n}: all structural invariants -- {exc}", False)
        ck(f"N={n}: all 18 fields on every zone",
           all(set(v.keys()) == set(ZONE_FIELDS) for v in zones.values()))
        ck(f"N={n}: json round-trips", json.loads(json.dumps(zones)) == zones)

    # -- N=1 and N=n_sub edges -------------------------------------------
    z1 = build_zones(topo, partition_substations(topo, 1)[0], 1)
    ck("N=1: one zone holds every substation",
       len(z1["Zone0"]["sub_inside_ids"]) == topo.n_sub)
    ck("N=1: no tie-lines", z1["Zone0"]["line_neighboring_idx"] == [])
    ck("N=1: owns every line",
       len(z1["Zone0"]["line_in_zone_idx"]) == topo.n_line)
    ck("N=1: halo empty", z1["Zone0"]["sub_border_out_ids"] == [])
    zn = build_zones(topo, partition_substations(topo, topo.n_sub)[0], topo.n_sub)
    ck("N=n_sub: every zone is one substation",
       all(len(v["sub_inside_ids"]) == 1 for v in zn.values()))
    ck("N=n_sub: nobody owns any line",
       all(v["line_in_zone_idx"] == [] for v in zn.values()))

    # -- monotonicity: a finer partition cannot own more lines ------------
    ties = []
    for n in (2, 4, 8, 16):
        zz = build_zones(topo, partition_substations(topo, n)[0], n)
        owned = set()
        for v in zz.values():
            owned |= set(v["line_in_zone_idx"])
        ties.append(topo.n_line - len(owned))
    ck(f"tie-lines non-decreasing in N {ties}",
       all(b >= a for a, b in zip(ties, ties[1:])))
    ck("tie-lines strictly rise from N=2 to N=16", ties[-1] > ties[0])

    # -- determinism across calls ----------------------------------------
    a1 = build_zones(topo, partition_substations(topo, 6)[0], 6)
    a2 = build_zones(topo, partition_substations(topo, 6)[0], 6)
    ck("generation is deterministic", json.dumps(a1) == json.dumps(a2))

    # -- every balance criterion still produces a valid partition ---------
    for crit in ("sub", "gen", "curtail", "load"):
        sz, Ab, _ = partition_substations(topo, 6, balance_by=crit)
        zb = build_zones(topo, sz, 6)
        try:
            assert_valid(zb, topo, 6)
            ok = all(len(_components(Ab, np.where(sz == z)[0])) == 1
                     for z in range(6))
        except AssertionError:
            ok = False
        ck(f"balance-by {crit} gives a valid connected partition", ok)

    # -- assert_valid actually catches corruption -------------------------
    good = build_zones(topo, partition_substations(topo, 4)[0], 4)
    ck("assert_valid passes the good one", not _raises(good, topo, 4))

    bad = _copy.deepcopy(good)
    bad["Zone0"].pop("storage_large_idx")
    ck("catches a missing field", _raises(bad, topo, 4))

    bad = _copy.deepcopy(good)
    bad["Zone1"]["sub_inside_ids"] = sorted(
        set(bad["Zone1"]["sub_inside_ids"]) | {bad["Zone0"]["sub_inside_ids"][0]})
    ck("catches a substation in two zones", _raises(bad, topo, 4))

    bad = _copy.deepcopy(good)
    bad["Zone0"]["line_large_idx"] = [
        x for x in bad["Zone0"]["line_large_idx"]
        if x != bad["Zone0"]["line_in_zone_idx"][0]]
    ck("catches line_large != line_in U line_neighboring", _raises(bad, topo, 4))

    bad = _copy.deepcopy(good)
    bad["Zone0"]["line_neighboring_idx"] = sorted(
        set(bad["Zone0"]["line_neighboring_idx"])
        | {bad["Zone0"]["line_in_zone_idx"][0]})
    ck("catches line_in / line_neighboring overlap", _raises(bad, topo, 4))

    bad = _copy.deepcopy(good)
    bad["Zone0"]["gen_curtail_inside_idx"] = []
    ck("catches a wrong curtailable set",
       _raises(bad, topo, 4) or not good["Zone0"]["gen_curtail_inside_idx"])

    bad = _copy.deepcopy(good)
    bad["Zone0"]["sub_inside_ids"] = list(
        reversed(bad["Zone0"]["sub_inside_ids"]))
    ck("catches an unsorted field",
       _raises(bad, topo, 4) or len(good["Zone0"]["sub_inside_ids"]) < 2)

    bad = _copy.deepcopy(good)
    bad["Zone0"]["sub_inside_ids"] = [float(x)
                                      for x in bad["Zone0"]["sub_inside_ids"]]
    ck("catches non-int indices (json would accept them, grid2op would not)",
       _raises(bad, topo, 4))

    # -- disconnected grid ------------------------------------------------
    iso = Topology(n_sub=8, line_or_to_subid=[0, 1, 2, 4, 5, 6],
                   line_ex_to_subid=[1, 2, 3, 5, 6, 7],
                   gen_to_subid=[0, 4], load_to_subid=[3, 7],
                   storage_to_subid=[], gen_renewable=[True, False],
                   gen_redispatchable=[False, True], name="two-islands")
    sz, Ai, _ = partition_substations(iso, 4)
    ck("disconnected grid: 4 zones", len(np.unique(sz)) == 4)
    ck("disconnected grid: zones stay connected",
       all(len(_components(Ai, np.where(sz == z)[0])) == 1 for z in range(4)))
    try:
        assert_valid(build_zones(iso, sz, 4), iso, 4)
        ck("disconnected grid: invariants hold", True)
    except AssertionError as exc:
        ck(f"disconnected grid: invariants hold -- {exc}", False)

    # -- a zone with no curtailable generator warns, it does not raise ----
    nocurt = Topology(n_sub=6, line_or_to_subid=[0, 1, 2, 3, 4],
                      line_ex_to_subid=[1, 2, 3, 4, 5],
                      gen_to_subid=[0], load_to_subid=[5], storage_to_subid=[],
                      gen_renewable=[True], gen_redispatchable=[False],
                      name="one-gen")
    zc = build_zones(nocurt, partition_substations(nocurt, 3)[0], 3)
    ck("zero-curtail zone warns rather than raises",
       any("NO curtailable" in x for x in assert_valid(zc, nocurt, 3)))
    try:
        assert_valid(zc, nocurt, 3, strict_curtail=True)
        ck("--require-curtail turns that warning into an error", False)
    except AssertionError:
        ck("--require-curtail turns that warning into an error", True)

    # -- topology snapshot round-trip -------------------------------------
    rt = Topology.from_dict(json.loads(json.dumps(topo.to_dict())))
    ck("topology snapshot round-trips",
       json.dumps(build_zones(rt, partition_substations(rt, 5)[0], 5))
       == json.dumps(build_zones(topo, partition_substations(topo, 5)[0], 5)))

    # -- n_zones out of range is refused ----------------------------------
    for bad_n in (0, -1, topo.n_sub + 1):
        try:
            partition_substations(topo, bad_n)
            ck(f"n_zones={bad_n} refused", False)
        except ValueError:
            ck(f"n_zones={bad_n} refused", True)

    print(f"\n  {checks - len(failed)}/{checks} checks passed")
    if failed:
        print("  FAILED: " + "; ".join(failed))
        return 1
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate zones_definitions_N<N>.json from grid adjacency.")
    ap.add_argument("--n-zones", type=int, nargs="+", default=None,
                    help="zone counts to generate, e.g. --n-zones 11 22 33")
    ap.add_argument("--env", default=None,
                    help="grid2op environment path "
                         "(default: G2OP_ENV_DIR/l2rpn_idf_2023)")
    ap.add_argument("--topology", default=None,
                    help="read the grid from a snapshot instead of grid2op")
    ap.add_argument("--dump-topology", default=None,
                    help="write a snapshot of the grid and exit")
    ap.add_argument("--out-dir", default=None,
                    help="where the json files go (default: the G2OpPowerGrid "
                         "env directory, which is where utils.py looks)")
    ap.add_argument("--weights", choices=["thermal", "count", "unit"],
                    default="thermal",
                    help="edge weight for adjacency (default: thermal, i.e. "
                         "corridor MW)")
    ap.add_argument("--balance-by", choices=["sub", "gen", "curtail", "load"],
                    default="sub",
                    help="quantity the zones are balanced on. 'sub' (default) "
                         "is the method-blind choice; 'curtail' spreads "
                         "controllable generation, which is what a real "
                         "operator does but is also the choice a reviewer will "
                         "ask about, so it is not the default")
    ap.add_argument("--no-balance", action="store_true",
                    help="skip the boundary-move balance pass")
    ap.add_argument("--require-curtail", action="store_true",
                    help="fail if any zone has no curtailable generator "
                         "(default: warn -- the shipped Zone6 has none and the "
                         "env tolerates a shape (0,) action head)")
    ap.add_argument("--compare-shipped", action="store_true",
                    help="also print a structural comparison against "
                         "zones_definitions.json")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the arithmetic checks and exit; needs no grid2op")
    ap.add_argument("--dry-run", action="store_true",
                    help="report but write nothing")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    if not args.n_zones and not args.dump_topology:
        ap.error("nothing to do: pass --n-zones, --dump-topology or --selfcheck")

    here = os.path.dirname(os.path.abspath(__file__))
    env_dir = args.out_dir or os.path.join(
        os.path.dirname(here), "BenchMARL", "benchmarl", "environments",
        "G2OpPowerGrid")

    topo = load_topology(args)
    print(f"grid: {topo.name}")

    if args.dump_topology:
        with open(args.dump_topology, "w", encoding="utf-8") as f:
            json.dump(topo.to_dict(), f)
        print(f"topology snapshot -> {args.dump_topology}")
        if not args.n_zones:
            return 0

    for n in args.n_zones:
        sub_zone, _, stats = partition_substations(
            topo, n, weights=args.weights, balance_by=args.balance_by,
            balance=not args.no_balance)
        zones = build_zones(topo, sub_zone, n)
        warn = assert_valid(zones, topo, n, strict_curtail=args.require_curtail)
        print_report(zones, topo, n, stats, warn)

        if args.compare_shipped:
            shipped = os.path.join(env_dir, "zones_definitions.json")
            if os.path.exists(shipped):
                compare_shipped(zones, shipped)
            else:
                print(f"  (no shipped file at {shipped})")

        if args.dry_run:
            continue
        out = os.path.join(env_dir, f"zones_definitions_N{n}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(zones, f, indent=4)
        meta = {
            "n_zones": n,
            "zone_names": list(zones.keys()),
            "grid": topo.name,
            "n_sub": topo.n_sub, "n_line": topo.n_line, "n_gen": topo.n_gen,
            "method": "recursive spectral bisection "
                      "+ connectivity/empty/balance repair",
            "weights": args.weights,
            "balance_by": args.balance_by,
            "balance": not args.no_balance,
            "repairs": stats,
            "warnings": warn,
            "generator": os.path.basename(__file__),
            "git_rev": _git_rev(),
        }
        with open(os.path.join(env_dir, f"zones_definitions_N{n}.meta.json"),
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
