"""Choose sigma from method-independent gates, BEFORE any method runs.

    python gate_severity.py --sigmas 0 0.5 1.0 1.5 2.0 --episodes 6

Runs no learning and never imports PACT-1's compensator.  Every quantity here is
a property of the environment that could have been measured before the method
existed -- which is the whole point.  Commit its output BEFORE the training runs
so the choice of sigma is auditable (NS guide I.10, pitfall 8: "retuning the NS
after seeing a method fail means you have planted the problem").

Gates, per NS guide section 6:

  G1  N=1 identity      the cross-agent term must be exactly 0 with one agent at
                        every sigma.  Structural here (zero-diagonal W).
  G2  sigma=0 identity  the environment must be byte-identical to stock.
  G3  it must hurt      the reference controller's survival must fall with
                        sigma, or the dial does nothing (I.2 constraint 4).
  G4  not throughput-limited  (BLOCKING)
                        a privileged full-information controller must still
                        survive well.  If it cannot, the task is
                        throughput-limited at that sigma and NO method can
                        recover it (3.1).
  G5  coupling strength the grid must actually run closer to its limits, since
                        the peer term scales as 1/limit(t).

Recommended sigma = the SMALLEST passing G3+G4+G5.  Smallest, not
best-performing: nothing here can see any method's score.
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
from pact1.basis import (_load_zones, get_ptdf,                # noqa: E402
                         ptdf_zone_coupling)

ENV = os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023")
ZONES = [f"Zone{i}" for i in range(11)]


def make_env():
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    except ImportError:
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
    return grid2op.make(ENV, action_class=PlayableAction, backend=backend)


def reconnect_actions(env, obs):
    """What the task's own heuristic does: put lines back when allowed.

    The reference controller has to include this or it is not the environment's
    baseline behaviour, it is a strawman -- and a strawman makes every gate
    meaningless.
    """
    to_reco = (obs.time_before_cooldown_line == 0) & (~obs.line_status)
    if not np.any(to_reco):
        return None
    line_id = int(np.where(to_reco)[0][0])
    return env.action_space({"set_line_status": [(line_id, +1)]})


def privileged_action(env, obs, ptdf, curtail_ids, gen_bus, gen_pmax,
                      target_rho=0.95):
    """Full-information controller: PTDF-targeted curtailment.

    Uses its privilege properly.  Finds the single most loaded line, works out
    how much flow must come off it, and curtails only the renewables whose PTDF
    onto THAT line actually helps -- in the right direction, largest lever
    first.  A blunt "curtail everything by 20% whenever anything is loaded"
    controller destabilises the grid and scores below do-nothing, which makes
    G4 impossible to pass for reasons that have nothing to do with sigma.
    """
    rho_max = float(obs.rho.max())
    if rho_max <= target_rho or ptdf is None or not len(curtail_ids):
        return None
    l = int(np.argmax(obs.rho))
    flow = float(obs.p_or[l])
    if abs(flow) < 1e-6:
        return None
    limit_mw = abs(flow) / max(rho_max, 1e-6)
    need = (rho_max - target_rho) * limit_mw          # MW to remove from line l

    cols = np.clip(gen_bus[curtail_ids], 0, ptdf.shape[1] - 1)
    sens = ptdf[l, cols] * np.sign(flow)              # >0 => cutting helps
    order = np.argsort(-sens)
    setpoints, removed = [], 0.0
    for k in order:
        if removed >= need or sens[k] <= 1e-6:
            break
        g = int(curtail_ids[k])
        avail = float(obs.gen_p[g])
        if avail <= 1e-3:
            continue
        # MW of generation to cut at this bus for the remaining need
        cut = min(avail, (need - removed) / max(sens[k], 1e-6))
        ratio = float(np.clip((avail - cut) / max(gen_pmax[g], 1e-6), 0.0, 1.0))
        setpoints.append((g, ratio))
        removed += cut * sens[k]
    if not setpoints:
        return None
    act = env.action_space({})
    act.curtail = setpoints
    return act


def roll(env, base_limits, sigma, n_episodes, privileged, ptdf, curtail_ids,
         gen_bus, gen_pmax, seed=0, max_steps=2016):
    env.seed(seed)
    lens, rhos, near, overl = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        if sigma > 0:
            env.set_thermal_limit(base_limits * dlr.ampacity_ratio(
                obs.month, obs.hour_of_day, sigma))
            obs = env.reset()          # limits take effect from a clean state
        done, steps = False, 0
        while not done and steps < max_steps:
            act = None
            if privileged:
                act = privileged_action(env, obs, ptdf, curtail_ids, gen_bus,
                                        gen_pmax)
            if act is None:
                act = reconnect_actions(env, obs) or env.action_space({})
            act.limit_curtail_storage(obs, margin=30)
            obs, _, done, _ = env.step(act)
            steps += 1
            rhos.append(float(obs.rho.max()))
            near.append(float((obs.rho > 0.95).mean()))
            overl.append(float((obs.rho > 1.0).sum()))
            # Only touch limits on a live environment: after a game over
            # grid2op refuses, and that is what crashed the first version.
            if sigma > 0 and not done:
                env.set_thermal_limit(base_limits * dlr.ampacity_ratio(
                    obs.month, obs.hour_of_day, sigma))
        lens.append(steps)
    return (float(np.mean(lens)), float(np.mean(rhos)),
            float(np.mean(near)), float(np.mean(overl)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=2016)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env()
    base_limits = np.array(env.get_thermal_limit(), dtype=np.float64, copy=True)
    print(f"grid: n_line={env.n_line} n_gen={env.n_gen}  "
          f"static limits {base_limits.min():.0f}..{base_limits.max():.0f}")

    ptdf = get_ptdf(env)
    W = ptdf_zone_coupling(env, ZONES, ptdf=ptdf)
    Z = _load_zones()
    renew = np.where(env.gen_renewable)[0]
    curtail_ids = np.unique(np.concatenate(
        [np.intersect1d(np.asarray(Z[z]["gen_inside_idx"], dtype=int), renew)
         for z in ZONES]))
    print(f"ptdf: {'yes' if ptdf is not None else 'NO'}   "
          f"curtailable renewables: {len(curtail_ids)}")

    print("\n[G1] N=1 identity")
    ok_g1 = np.allclose(np.diag(W), 0.0)
    print(f"  coupling matrix zero-diagonal: {ok_g1}"
          f"  -> peer term exactly 0 at N=1 for every sigma")

    print("\n[G2] sigma = 0 identity")
    ok_g2 = (dlr.ampacity_ratio(7, 15, 0.0) == 1.0)
    print(f"  ampacity ratio at sigma=0 is exactly 1.0: {ok_g2}"
          f"  -> stock task byte-identical")

    print(f"\nrolling {args.episodes} episodes per sigma, "
          f"max {args.max_steps} steps ...")
    rows = []
    for s in args.sigmas:
        print(f"\n  {dlr.describe(s)}")
        ln_b, rho_b, nr_b, ov_b = roll(
            env, base_limits, s, args.episodes, False, ptdf, curtail_ids,
            env.gen_to_subid, env.gen_pmax, args.seed, args.max_steps)
        ln_p, rho_p, nr_p, ov_p = roll(
            env, base_limits, s, args.episodes, True, ptdf, curtail_ids,
            env.gen_to_subid, env.gen_pmax, args.seed, args.max_steps)
        rows.append(dict(sigma=s, len_b=ln_b, rho_b=rho_b, near_b=nr_b,
                         len_p=ln_p, rho_p=rho_p))
        print(f"    reference : steps={ln_b:7.1f} max_rho={rho_b:.3f} "
              f"near_limit={nr_b:.4f} overloaded/step={ov_b:.3f}")
        print(f"    privileged: steps={ln_p:7.1f} max_rho={rho_p:.3f} "
              f"near_limit={nr_p:.4f} overloaded/step={ov_p:.3f}")

    base = rows[0]
    if base["len_p"] < base["len_b"]:
        print("\n  !! WARNING: at sigma=0 the privileged controller is WORSE")
        print("     than the reference. G4 is then meaningless -- fix the")
        print("     controller before reading any gate below.")

    print("\n" + "=" * 74)
    print(f"{'sigma':>6} {'ref steps':>10} {'vs s=0':>8} {'priv steps':>11} "
          f"{'priv/ref':>9} {'max_rho':>8} {'near_lim':>9}  gates")
    recommended = None
    for r in rows:
        hurt = r["len_b"] < 0.90 * base["len_b"]
        priv_ok = r["len_p"] >= 0.95 * base["len_p"]
        coupling_up = r["near_b"] > base["near_b"] * 1.10
        tags = ["G3+" if hurt else "G3-",
                "G4+" if priv_ok else "G4-BLOCK",
                "G5+" if coupling_up else "G5-"]
        if r["sigma"] > 0 and hurt and priv_ok and coupling_up \
                and recommended is None:
            recommended = r["sigma"]
        print(f"{r['sigma']:6.2f} {r['len_b']:10.1f} "
              f"{100*(r['len_b']/max(base['len_b'],1e-9)-1):+7.1f}% "
              f"{r['len_p']:11.1f} {r['len_p']/max(r['len_b'],1e-9):9.2f} "
              f"{r['rho_b']:8.3f} {r['near_b']:9.4f}  {' '.join(tags)}")

    print("\nG3 the dial must hurt the reference controller (I.2 constraint 4)")
    print("G4 BLOCKING: privileged controller must still survive; failing it")
    print("   means throughput-limited and no method can recover (3.1)")
    print("G5 the grid must run nearer its limits: the peer term scales as")
    print("   1/limit(t), so this is where dormant coupling wakes up")
    print("\n" + "=" * 74)
    if recommended is None:
        print("  NO sigma PASSES ALL GATES at this episode budget.")
        print("  Do not proceed to method runs; widen --sigmas or --episodes.")
    else:
        tag = "REALISTIC" if abs(recommended - 1.0) < 1e-9 else "see label above"
        print(f"  RECOMMENDED sigma = {recommended:.2f}   ({tag})")
        print("  Smallest sigma passing G3+G4+G5, chosen without reference to")
        print("  ANY method's score -- nothing here can see one.")
    print("  Commit this output before running MAPPO or PACT-1.")
    print("=" * 74)


if __name__ == "__main__":
    main()
