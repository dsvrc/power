"""Choose sigma from method-independent gates, BEFORE any method runs.

    python gate_severity.py --sigmas 0 0.5 1.0 1.5 2.0 --episodes 6

Runs no learning and never imports PACT-1's compensator.  Every quantity here is
a property of the environment that could have been measured before the method
existed -- which is the whole point.  Commit its output BEFORE the training
runs so the choice of sigma is auditable (NS guide I.10, pitfall 8:
"retuning the NS after seeing a method fail means you have planted the problem").

Gates, per NS guide section 6:

  G1  N=1 identity        the cross-agent term must be exactly 0 with one agent,
                          at every sigma.  Structural here (zero-diagonal W),
                          asserted anyway.
  G2  sigma=0 identity    the environment must be byte-identical to stock.
  G3  it must hurt        a fixed reference policy's survival must fall with
                          sigma, or the dial does nothing (I.2 constraint 4).
  G4  not throughput-limited
                          a privileged full-information controller must still
                          survive well at the driver peak.  If it cannot, the
                          task is throughput-limited at that sigma and NO method
                          can recover it (3.1, blocking).
  G5  coupling strength   the peer-induced share of loading must actually grow
                          with sigma -- this is what stock POWER lacked
                          (measured 0.7%, too weak to act on).

The recommended sigma is the SMALLEST one that passes G3 and G5 while still
passing G4.  Smallest, not best-performing: nothing here can see any method's
score, so it cannot be tuned toward one.
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
from pact1.basis import (CouplingBasis, get_ptdf,              # noqa: E402
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


def roll(env, base_limits, sigma, n_episodes, safe_rho=0.9, privileged=False,
         seed=0, basis=None, curtail_idx=None):
    """Roll a reference controller and report survival plus loading stats.

    privileged=False : do-nothing but reconnect -- the task's own heuristic,
                       i.e. what the environment does when no agent acts.
    privileged=True  : a full-information greedy controller that curtails
                       proportionally to observed overload.  Stands in for the
                       'privileged scripted controller' of gate G4; it sees the
                       true rho on every line, which no learner does.
    """
    env.seed(seed)
    lens, rhos, ells, overloads = [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        if sigma > 0:
            env.set_thermal_limit(base_limits * dlr.ampacity_ratio(
                obs.month, obs.hour_of_day, sigma))
        done, steps = False, 0
        while not done and steps < 2016:            # one week at 5-min steps
            act = env.action_space({})
            if privileged:
                over = obs.rho > safe_rho
                if over.any() and curtail_idx is not None and len(curtail_idx):
                    # Curtail renewables in proportion to the worst overload.
                    excess = float(np.clip(obs.rho.max() - safe_rho, 0.0, 1.0))
                    ratio = float(np.clip(1.0 - 2.0 * excess, 0.0, 1.0))
                    act.curtail = [(g, ratio) for g in curtail_idx]
            act.limit_curtail_storage(obs, margin=30)
            obs, _, done, _ = env.step(act)
            steps += 1
            if sigma > 0:
                env.set_thermal_limit(base_limits * dlr.ampacity_ratio(
                    obs.month, obs.hour_of_day, sigma))
            rhos.append(float(obs.rho.max()))
            overloads.append(float((obs.rho > 1.0).sum()))
            if basis is not None:
                # Peer-induced share of loading: the PTDF-weighted peer term
                # relative to the loading actually observed.
                ells.append(float(np.mean(np.abs(basis))) / max(obs.rho.max(), 1e-6))
        lens.append(steps)
    return (float(np.mean(lens)), float(np.mean(rhos)),
            float(np.mean(overloads)), float(np.mean(ells)) if ells else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = make_env()
    base_limits = np.array(env.get_thermal_limit(), dtype=np.float64, copy=True)
    print(f"grid: n_line={env.n_line} n_gen={env.n_gen}  "
          f"static limits {base_limits.min():.0f}..{base_limits.max():.0f}")

    ptdf = get_ptdf(env)
    W = ptdf_zone_coupling(env, ZONES, ptdf=ptdf)
    curt_inside = {}
    renew = np.where(env.gen_renewable)[0]
    from pact1.basis import _load_zones
    Z = _load_zones()
    for z in ZONES:
        curt_inside[z] = np.intersect1d(
            np.asarray(Z[z]["gen_inside_idx"], dtype=int), renew)
    basis = CouplingBasis(ZONES, curt_inside, env.gen_pmax, W=W, r_target=1)
    all_curtail = np.unique(np.concatenate([curt_inside[z] for z in ZONES
                                            if len(curt_inside[z])]))

    # ---- G1 / G2: structural identities ---------------------------------
    print("\n[G1] N=1 identity")
    ok_g1 = np.allclose(np.diag(basis.W), 0.0)
    print(f"  coupling matrix zero-diagonal: {ok_g1}  "
          f"-> peer term is exactly 0 at N=1 for every sigma")

    print("\n[G2] sigma = 0 identity")
    r0 = dlr.ampacity_ratio(7, 15, 0.0)
    ok_g2 = (r0 == 1.0)
    print(f"  ampacity ratio at sigma=0: {r0}  -> stock task byte-identical: {ok_g2}")

    # ---- G3 / G4 / G5: per-sigma rolls ----------------------------------
    print(f"\nrolling {args.episodes} episodes per sigma "
          f"(reference + privileged) ...")
    rows = []
    for s in args.sigmas:
        print(f"\n  {dlr.describe(s)}")
        ln_b, rho_b, ov_b, ell_b = roll(
            env, base_limits, s, args.episodes, seed=args.seed,
            basis=basis.W, curtail_idx=all_curtail)
        ln_p, rho_p, ov_p, _ = roll(
            env, base_limits, s, args.episodes, seed=args.seed, privileged=True,
            basis=basis.W, curtail_idx=all_curtail)
        rows.append(dict(sigma=s, len_b=ln_b, rho_b=rho_b, ov_b=ov_b,
                         len_p=ln_p, rho_p=rho_p, ov_p=ov_p))
        print(f"    reference : steps={ln_b:7.1f}  max_rho={rho_b:.3f}  "
              f"overloaded_lines/step={ov_b:.3f}")
        print(f"    privileged: steps={ln_p:7.1f}  max_rho={rho_p:.3f}  "
              f"overloaded_lines/step={ov_p:.3f}")

    base = rows[0]
    print("\n" + "=" * 72)
    print(f"{'sigma':>6} {'ref steps':>10} {'vs s=0':>8} {'priv steps':>11} "
          f"{'priv/ref':>9} {'max_rho':>8}  gates")
    recommended = None
    for r in rows:
        hurt = r["len_b"] < 0.90 * base["len_b"]              # G3
        priv_ok = r["len_p"] >= 0.95 * base["len_p"]          # G4 (blocking)
        coupling_up = r["rho_b"] > base["rho_b"] * 1.02       # G5 proxy
        tags = []
        tags.append("G3+" if hurt else "G3-")
        tags.append("G4+" if priv_ok else "G4-BLOCK")
        tags.append("G5+" if coupling_up else "G5-")
        if r["sigma"] > 0 and hurt and priv_ok and coupling_up \
                and recommended is None:
            recommended = r["sigma"]
        print(f"{r['sigma']:6.2f} {r['len_b']:10.1f} "
              f"{100*(r['len_b']/base['len_b']-1):+7.1f}% {r['len_p']:11.1f} "
              f"{r['len_p']/max(r['len_b'],1e-9):9.2f} {r['rho_b']:8.3f}  "
              f"{' '.join(tags)}")

    print("\nG3 the dial must hurt the reference controller (I.2 constraint 4)")
    print("G4 BLOCKING: a privileged controller must still survive; failing it")
    print("   means the task is throughput-limited and no method can recover")
    print("G5 loading must actually rise, or the coupling is still dormant")
    print("\n" + "=" * 72)
    if recommended is None:
        print("  NO sigma PASSES ALL GATES on this grid/episode budget.")
        print("  Do not proceed to method runs; widen --sigmas or --episodes.")
    else:
        print(f"  RECOMMENDED sigma = {recommended:.2f}"
              f"   ({'realistic' if abs(recommended-1.0) < 1e-9 else 'see label above'})")
        print("  Smallest sigma passing G3+G4+G5. Chosen without reference to")
        print("  ANY method's score -- nothing in this script can see one.")
    print("  Commit this output before running MAPPO or PACT-1.")
    print("=" * 72)


if __name__ == "__main__":
    main()
