"""The recoverable-fraction decomposition: how much of the derating loss can
ANY method get back, and how much of that requires coordination.

    python theory_ceiling.py --sigmas 0.5 1.0 1.5 --episodes 8

THEORY
------
Line flow is linear in injections through the PTDF:  f_l = sum_b H[l,b] inj_b.
Derating scales the limit, so loading rho_l = f_l / (L_l r(t)) and the excess
over the sigma=0 counterfactual is

    Delta_l(t) = rho0_l(t) * (1/r(t) - 1)

Every MW of that excess is attributable to a specific injection, and the
injections partition into three classes by WHO CAN MOVE THEM:

    Delta_fixed   loads and non-curtailable generation   -> nobody
    Delta_own     the agent's own-zone renewables        -> any decentralized
                                                            policy, unilaterally
    Delta_peer    other zones' renewables                -> only by coordinating

Hence three ceilings, all computable from the network model and the observed
dispatch, with no training and no method:

    irreducible              = Delta_fixed / Delta_total
    decentralized ceiling    = 1 - Delta_fixed / Delta_total
    non-coordinating ceiling = 1 - (Delta_fixed + Delta_peer) / Delta_total
    COORDINATION GAP         = Delta_peer / Delta_total

The last line is the quantity a coupling-aware method claims.  A blind
decentralized learner gets Delta_own for free -- it only has to notice its own
loading.  It cannot get Delta_peer without knowing what peers are doing, which
is precisely the category-C structure.

ATTRIBUTION, and why it is not arbitrary
----------------------------------------
Excess is attributed to an injection in proportion to its SIGNED contribution
to the flow on the binding line, and only where curtailment could actually
help: a generator whose PTDF pushes flow the other way, or which is already at
zero output, contributes no recoverable headroom.  Concretely, for binding line
l the recoverable MW from a set S of generators is

    R_S(l) = sum_{g in S}  max(0, H[l, bus(g)] * sign(f_l)) * p_g(t)

bounded by what is actually being generated.  This is a DC (first-order)
approximation: it ignores losses and the AC reactive coupling, which is the
standard assumption behind PTDF and is stated as a limitation rather than
hidden.

INFORMATION
-----------
The PTDF is the network model.  It is not privileged: every system operator
has it, and IV.1 names it as the domain's own load-transfer operator.  What
makes the comparison fair is not withholding it but MATCHING it -- the
information-matched baseline gets the same PTDF-derived features, so any gap
is attributable to the mechanism rather than to the data.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                            # noqa: E402

import grid2op                                                # noqa: E402
from grid2op.Action import PlayableAction                     # noqa: E402
from utils import G2OP_ENV_DIR                                # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "BenchMARL", "benchmarl", "environments", "G2OpPowerGrid"))
from pact1 import dlr                                          # noqa: E402
from pact1.basis import _load_zones, get_ptdf                  # noqa: E402

ZONES = [f"Zone{i}" for i in range(11)]
PRESETS = {"summer": r".*-07-.*$", "winter": r".*-02-.*$"}


def safe_set_limit(env, limits):
    """set_thermal_limit, guarded.

    grid2op refuses the call whenever the environment is not initialised --
    freshly built, or sitting on a game over -- and raises.  This has now bitten
    three separate scripts, so every call in this file goes through here rather
    than relying on the caller to remember which states are legal.
    """
    try:
        env.set_thermal_limit(limits)
        return True
    except Exception:                                         # noqa: BLE001
        return False


def make_env(regex):
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    except ImportError:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
    import re
    from grid2op.Chronics import MultifolderWithCache
    env = grid2op.make(os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023"),
                       action_class=PlayableAction, backend=backend,
                       chronics_class=MultifolderWithCache)
    pat = re.compile(regex)
    env.chronics_handler.real_data.set_filter(lambda n: bool(pat.match(n)))
    env.chronics_handler.reset()
    return env


def line_ratios(obs, line_zone, n_zones, sigma, spatial):
    """Per-line ampacity ratio vector (or a scalar broadcast when uniform)."""
    if not spatial or line_zone is None:
        return None, dlr.ampacity_ratio(obs.month, obs.hour_of_day, sigma)
    pz = np.array([dlr.ampacity_ratio_zone(obs.month, obs.hour_of_day, z,
                                           n_zones, sigma)
                   for z in range(n_zones)], dtype=np.float64)
    reg = float(pz.mean())
    return np.where(line_zone >= 0, pz[np.clip(line_zone, 0, n_zones - 1)],
                    reg), reg


def decompose(obs, ptdf, zone_of_gen, curtail_ids, sigma,
              line_zone=None, n_zones=11, spatial=False):
    """Split the derating-induced excess on the binding line into the three
    classes.  Returns (delta_total, frac_fixed, frac_own, frac_peer) or None
    when there is no excess to attribute at this step.
    """
    rvec, rreg = line_ratios(obs, line_zone, n_zones, sigma, spatial)
    l = int(np.argmax(obs.rho))
    # The ratio that matters is the one on the BINDING line, which under
    # spatial heterogeneity is its own zone's, not the regional average.
    ratio = float(rvec[l]) if rvec is not None else float(rreg)
    if ratio >= 1.0 - 1e-12:
        return None                        # no derating on the binding line
    rho0 = float(obs.rho[l]) * ratio       # loading the static rating would show
    excess = rho0 * (1.0 / ratio - 1.0)
    if excess <= 1e-9:
        return None

    f_l = float(obs.p_or[l])
    if abs(f_l) < 1e-6:
        return None
    sgn = np.sign(f_l)

    # Recoverable MW by class: only generators whose PTDF pushes flow DOWN on
    # this line, and only up to what they are actually producing.
    rec = {"own": 0.0, "peer": 0.0}
    # "own" is evaluated for the zone that owns the binding line; every other
    # zone's contribution is peer.  Which zone owns line l is fixed geometry.
    owner = None
    Z = _load_zones()
    for z in ZONES:
        if l in Z[z]["line_in_zone_idx"]:
            owner = z
            break

    for g in curtail_ids:
        b = int(min(zone_of_gen["bus"][g], ptdf.shape[1] - 1))
        h = float(ptdf[l, b]) * sgn
        if h <= 1e-9:
            continue                       # curtailing this gen does not help
        avail = float(obs.gen_p[g])
        if avail <= 1e-3:
            continue                       # nothing left to curtail
        key = "own" if zone_of_gen["zone"][g] == owner else "peer"
        rec[key] += h * avail

    # Convert MW of flow reduction into rho units on this line.
    limit_mw = abs(f_l) / max(float(obs.rho[l]), 1e-6)
    need_mw = excess * limit_mw * ratio

    r_own = min(rec["own"], need_mw)
    r_peer = min(rec["peer"], max(need_mw - r_own, 0.0))
    r_fixed = max(need_mw - r_own - r_peer, 0.0)
    tot = max(need_mw, 1e-9)
    return excess, r_fixed / tot, r_own / tot, r_peer / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--chronics", default="summer")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--spatial", action="store_true",
                    help="per-zone ampacity (heterogeneous weather). Compare "
                         "against the uniform run: the PEER column is the "
                         "quantity that should rise.")
    args = ap.parse_args()

    env = make_env(PRESETS.get(args.chronics, args.chronics))
    ptdf = get_ptdf(env)
    if ptdf is None:
        print("no PTDF available; cannot compute the ceiling")
        return 1

    Z = _load_zones()
    renew = np.where(env.gen_renewable)[0]
    curtail_ids = np.unique(np.concatenate(
        [np.intersect1d(np.asarray(Z[z]["gen_inside_idx"], dtype=int), renew)
         for z in ZONES]))
    zone_of_gen = {"bus": env.gen_to_subid, "zone": {}}
    for z in ZONES:
        for g in np.intersect1d(np.asarray(Z[z]["gen_inside_idx"], dtype=int),
                                renew):
            zone_of_gen["zone"][int(g)] = z

    # line -> owning zone, for per-zone ratings
    n_zones = len(ZONES)
    line_zone = np.full(env.n_line, -1, dtype=int)
    for zi, z in enumerate(ZONES):
        for l in Z[z]["line_in_zone_idx"]:
            if 0 <= int(l) < env.n_line:
                line_zone[int(l)] = zi

    print(f"grid: {env.n_line} lines, {len(curtail_ids)} curtailable renewables")
    print(f"ampacity: {'PER-ZONE (spatial)' if args.spatial else 'uniform'}")
    print(f"chronics: {args.chronics}   {args.episodes} episodes x "
          f"{args.max_steps} steps\n")
    print(f"{'sigma':>6} {'n_steps':>8} {'excess':>8} | {'irreducible':>12} "
          f"{'own (free)':>11} {'PEER (coord)':>13} | {'decentr ceil':>13}")

    for s in args.sigmas:
        fixed, own, peer, exc, n = [], [], [], [], 0
        base = np.array(env.get_thermal_limit(), dtype=np.float64, copy=True)
        for ep in range(args.episodes):
            env.set_id(ep)
            obs = env.reset()
            rv, rg = line_ratios(obs, line_zone, n_zones, s, args.spatial)
            safe_set_limit(env, base * (rv if rv is not None else rg))
            done, steps = False, 0
            while not done and steps < args.max_steps:
                d = decompose(obs, ptdf, zone_of_gen, curtail_ids, s,
                              line_zone, n_zones, args.spatial)
                if d is not None:
                    exc.append(d[0])
                    fixed.append(d[1])
                    own.append(d[2])
                    peer.append(d[3])
                    n += 1
                obs, _, done, _ = env.step(env.action_space({}))
                steps += 1
                if not done:
                    rv, rg = line_ratios(obs, line_zone, n_zones, s,
                                         args.spatial)
                    safe_set_limit(env, base * (rv if rv is not None else rg))
            # Restoring here is best-effort: after a game over grid2op refuses,
            # and it does not matter because the next episode resets first and
            # re-applies the ratio from a clean state.
            safe_set_limit(env, base)
        if not n:
            print(f"{s:6.2f}  no derating-induced excess observed")
            continue
        fx, ow, pe = np.mean(fixed), np.mean(own), np.mean(peer)
        print(f"{s:6.2f} {n:8d} {np.mean(exc):8.4f} | {fx:11.1%} "
              f"{ow:10.1%} {pe:12.1%} | {1-fx:12.1%}")

    print("\nREADING THIS TABLE")
    print("  irreducible   loads and non-curtailable generation: no method")
    print("                recovers this. It is the lossy part of the dial.")
    print("  own (free)    a blind decentralized policy gets this by watching")
    print("                its OWN loading. MAPPO should reach it.")
    print("  PEER (coord)  requires knowing what other zones are doing. This")
    print("                is the coordination gap, and it is what a")
    print("                coupling-aware method claims. Report PACT-1's")
    print("                recovery AGAINST THIS NUMBER, not against B0.")
    print("\n  Computed from the network model and observed dispatch only:")
    print("  no training, no method, no reward. It can be pre-registered.")
    print("  DC approximation (losses and reactive coupling ignored) -- state")
    print("  that as a limitation rather than hiding it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
