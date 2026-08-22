"""Build the training env in-process and exercise it, with no multiprocessing.

    python debug_env.py --severity 1.0 --chronics summer

The collector hides worker exceptions behind `EOFError` from a closed pipe, so
the actual failure is invisible.  This constructs the SAME env the collector
builds, in this process, and lets the real traceback surface.  It also reports
how many chronics the filter selected and how much memory the cache costs,
because MultifolderWithCache loads every matching chronic into RAM and the
collector pays that once PER WORKER.
"""
import argparse
import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                            # noqa: E402

from utils import G2OP_ENV_DIR                                # noqa: E402

CHRONICS_PRESETS = {
    "summer": r".*-0[678]-.*$",
    "winter": r".*-(12|01|02)-.*$",
    "feb": r".*-02-.*$",
    "all": None,
}


def rss_mb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1024.0            # Linux reports KB
    except Exception:                                         # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--chronics", default="summer")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--pact1", action="store_true")
    args = ap.parse_args()

    regex = CHRONICS_PRESETS.get(args.chronics, args.chronics)
    print(f"severity={args.severity}  chronics={args.chronics} -> {regex!r}")
    print(f"RSS before build: {rss_mb():.0f} MB")

    cfg = dict(
        env_name=os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023"),
        zone_names=[f"Zone{j}" for j in range(11)],
        use_global_obs=False,
        use_redispatching_agent=True,
        env_g2op_config={},
        local_rewards=None,
        shuffle_chronics=True,
        regex_filter_chronics=regex,
        safe_max_rho=0.9,
        curtail_margin=30,
    )

    try:
        if args.pact1:
            from benchmarl.environments.G2OpPowerGrid.pact1.env import PACT1Env
            env = PACT1Env(pact1_enabled=True, severity=args.severity, **cfg)
        else:
            from benchmarl.environments.G2OpPowerGrid.pact1.dlr_env import (
                PZMAEnvDLR)
            env = PZMAEnvDLR(severity=args.severity, **cfg)
    except Exception:                                         # noqa: BLE001
        print("\n!! CONSTRUCTION FAILED -- this is the hidden worker error:\n")
        traceback.print_exc()
        return 1

    print(f"RSS after build : {rss_mb():.0f} MB")
    n_chron = len(env.env_g2op.chronics_handler.real_data.subpaths)
    print(f"chronics selected by filter: {n_chron}")
    print(f"chronics_class: {type(env.env_g2op.chronics_handler.real_data).__name__}")
    if n_chron > 100:
        print(f"  NOTE: MultifolderWithCache holds all {n_chron} in RAM, and the")
        print(f"        collector pays that PER WORKER. With 12 workers that is")
        print(f"        12x this process's cache footprint.")

    try:
        obs, _ = env.reset()
        print(f"\nreset OK. RSS after reset: {rss_mb():.0f} MB")
        print(f"agents: {len(env.agents)}")
    except Exception:                                         # noqa: BLE001
        print("\n!! RESET FAILED -- this is the hidden worker error:\n")
        traceback.print_exc()
        return 1

    try:
        for i in range(args.steps):
            act = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, done, trunc, info = env.step(act)
            if any(done.values()):
                obs, _ = env.reset()
        print(f"{args.steps} steps OK. RSS: {rss_mb():.0f} MB")
    except Exception:                                         # noqa: BLE001
        print("\n!! STEP FAILED -- this is the hidden worker error:\n")
        traceback.print_exc()
        return 1

    if args.severity > 0:
        print(f"\nDLR liveness: ratio={getattr(env, '_last_ratio', None)}  "
              f"applied={getattr(env, '_dlr_applied', 0)}  "
              f"skipped={getattr(env, '_dlr_skipped', 0)}")
        lim = np.asarray(env.env_g2op.get_thermal_limit(), dtype=float)
        base = np.asarray(env._base_limits, dtype=float)
        print(f"limits vs static: mean ratio {float((lim/base).mean()):.4f}")

    print("\nAll good in-process. If the collector still dies, the cause is")
    print("multiprocessing-specific: most likely memory (see chronics count")
    print("above) rather than logic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
