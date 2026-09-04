"""Does the coordination gap grow with the number of agents?

    python theory_scaling.py --n-zones 6 11 22 33 --episodes 6

METHOD_design.md section 8 predicts it must: a blind learner's cooperative
gradient component drowns as N grows, while a coupling-aware method's is
N-independent.  Here that prediction is testable WITHOUT TRAINING ANYTHING,
because the coordination-recoverable fraction is computable from the network
operator and the observed dispatch alone.

Why it should grow, mechanically: the excess on a binding line is recoverable
"for free" by whichever agent owns enough generation with favourable PTDF onto
that line.  With few, large zones an agent usually owns enough and the fix is
local.  Split the grid finer and each agent's local authority shrinks while the
coupling does not, so a larger share of every fix must come from peers.

Measured on the shipped 11-zone partition: PEER = 8.3% (uniform ratings),
11.0% (spatial).  If PEER rises materially with N, the finer partition is the
regime where a coordination method has room -- and the choice was made by
physics, before any method ran, which is what makes it defensible.

The partition here is built from electrical adjacency, not hand-drawn: a
line belongs to a zone when BOTH endpoints are inside it, a generator when its
substation is.  Real grids are operated with many more control areas than 11,
so finer partitions are the realistic direction, not an artificial one.
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
from pact1.basis import get_ptdf                               # noqa: E402

PRESETS = {"summer": r".*-07-.*$", "winter": r".*-02-.*$"}


def make_env(regex):
    import re
    from grid2op.Chronics import MultifolderWithCache
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    except ImportError:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
    env = grid2op.make(os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023"),
                       action_class=PlayableAction, backend=backend,
                       chronics_class=MultifolderWithCache)
    pat = re.compile(regex)
    env.chronics_handler.real_data.set_filter(lambda n: bool(pat.match(n)))
    env.chronics_handler.reset()
    return env


def build_partition(env, n_zones):
    """Contiguous substation blocks -> line and generator ownership.

    Substations are numbered along the grid, so contiguous blocks approximate
    electrical regions.  A line is owned only when BOTH endpoints are inside a
    block; lines spanning blocks are TIE-LINES and belong to nobody, which is
    exactly the situation where no single agent can fix an overload alone.
    """
    n_sub = env.n_sub
    edges = np.linspace(0, n_sub, n_zones + 1).astype(int)
    sub_zone = np.full(n_sub, -1, dtype=int)
    for z in range(n_zones):
        sub_zone[edges[z]:edges[z + 1]] = z

    lor, lex = env.line_or_to_subid, env.line_ex_to_subid
    line_zone = np.where(sub_zone[lor] == sub_zone[lex], sub_zone[lor], -1)
    gen_zone = sub_zone[env.gen_to_subid]
    return sub_zone, line_zone, gen_zone


def decompose(obs, ptdf, line_zone, gen_zone, curtail_ids, sigma, n_zones,
              spatial):
    l = int(np.argmax(obs.rho))
    if spatial:
        zi = int(line_zone[l]) if line_zone[l] >= 0 else 0
        ratio = dlr.ampacity_ratio_zone(obs.month, obs.hour_of_day, zi,
                                        n_zones, sigma)
    else:
        ratio = dlr.ampacity_ratio(obs.month, obs.hour_of_day, sigma)
    if ratio >= 1.0 - 1e-12:
        return None
    rho0 = float(obs.rho[l]) * ratio
    excess = rho0 * (1.0 / ratio - 1.0)
    if excess <= 1e-9:
        return None
    f_l = float(obs.p_or[l])
    if abs(f_l) < 1e-6:
        return None
    sgn = np.sign(f_l)
    owner = int(line_zone[l])          # -1 for a tie-line: nobody owns it

    rec = {"own": 0.0, "peer": 0.0}
    for g in curtail_ids:
        b = int(min(env_gen_bus[g], ptdf.shape[1] - 1))
        h = float(ptdf[l, b]) * sgn
        if h <= 1e-9:
            continue
        avail = float(obs.gen_p[g])
        if avail <= 1e-3:
            continue
        key = "own" if (owner >= 0 and int(gen_zone[g]) == owner) else "peer"
        rec[key] += h * avail

    limit_mw = abs(f_l) / max(float(obs.rho[l]), 1e-6)
    need = excess * limit_mw * ratio
    r_own = min(rec["own"], need)
    r_peer = min(rec["peer"], max(need - r_own, 0.0))
    r_fixed = max(need - r_own - r_peer, 0.0)
    tot = max(need, 1e-9)
    return excess, r_fixed / tot, r_own / tot, r_peer / tot, owner < 0


def main():
    global env_gen_bus
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-zones", type=int, nargs="+", default=[6, 11, 22, 33])
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--chronics", default="summer")
    ap.add_argument("--spatial", action="store_true")
    args = ap.parse_args()

    env = make_env(PRESETS.get(args.chronics, args.chronics))
    env_gen_bus = env.gen_to_subid
    ptdf = get_ptdf(env)
    if ptdf is None:
        print("no PTDF; cannot compute the ceiling")
        return 1
    curtail_ids = np.where(env.gen_renewable)[0]
    base = np.array(env.get_thermal_limit(), dtype=np.float64, copy=True)

    print(f"grid: {env.n_sub} substations, {env.n_line} lines, "
          f"{len(curtail_ids)} curtailable renewables")
    print(f"sigma={args.sigma}  ratings="
          f"{'per-zone' if args.spatial else 'uniform'}  "
          f"chronics={args.chronics}\n")
    print(f"{'N zones':>8} {'gens/zone':>10} {'tie-line%':>10} {'n_steps':>8} | "
          f"{'irreduc':>8} {'own':>8} {'PEER':>8}")

    for nz in args.n_zones:
        sub_zone, line_zone, gen_zone = build_partition(env, nz)
        fixed, own, peer, tie, n = [], [], [], [], 0
        for ep in range(args.episodes):
            env.set_id(ep)
            obs = env.reset()
            done, steps = False, 0
            while not done and steps < args.max_steps:
                d = decompose(obs, ptdf, line_zone, gen_zone, curtail_ids,
                              args.sigma, nz, args.spatial)
                if d is not None:
                    fixed.append(d[1]); own.append(d[2]); peer.append(d[3])
                    tie.append(float(d[4])); n += 1
                obs, _, done, _ = env.step(env.action_space({}))
                steps += 1
            try:
                env.set_thermal_limit(base)
            except Exception:                                # noqa: BLE001
                pass
        if not n:
            print(f"{nz:8d}  no derating-induced excess observed")
            continue
        gpz = len(curtail_ids) / nz
        print(f"{nz:8d} {gpz:10.1f} {np.mean(tie):9.1%} {n:8d} | "
              f"{np.mean(fixed):7.1%} {np.mean(own):7.1%} {np.mean(peer):7.1%}")

    print("\nPEER is the coordination gap. If it rises with N, that confirms")
    print("METHOD_design.md section 8's scaling prediction -- and no competing")
    print("credit-assignment method predicts it, which makes the curve itself")
    print("a result rather than just a configuration choice.")
    print("\ntie-line%% is the share of binding constraints owned by NOBODY.")
    print("Those are unfixable without coordination by construction, so they")
    print("are the mechanism behind any rise in PEER.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
