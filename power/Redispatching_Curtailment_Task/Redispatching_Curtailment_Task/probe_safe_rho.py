"""Pick safe_max_rho from how often the AGENTS act, not from any return.

    python probe_safe_rho.py --severity 0.0 --chronics summer

The heuristic in PZMAEnvRecoDNLimit plays do-nothing while max rho stays below
safe_max_rho, stepping grid2op without consulting the policy.  Measured on July
chronics at the shipped 0.9: 7.52 grid2op steps per agent decision, i.e. the
learned policy is asked for an action on 13% of steps and the other 87% is the
built-in heuristic.  That is bad twice over -- the benchmark mostly measures the
heuristic, and it costs ~3x the wall clock per training frame to do so
(collection is 95% of iteration time).

This script reports, per candidate threshold:

    steps/decision   grid2op steps burned per agent action  (lower = faster)
    agent_share      fraction of simulated time the policy actually drives
    est_speedup      wall-clock gain relative to the shipped 0.9

No policy is trained and no return is computed, so the choice cannot be tuned
toward any method.  Pick a threshold, freeze it, use it for EVERY arm.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                            # noqa: E402

from utils import G2OP_ENV_DIR                                # noqa: E402

CHRONICS_PRESETS = {
    "summer": r".*-07-.*$",
    "winter": r".*-02-.*$",
    "jja": r".*-0[678]-.*$",
    "all": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--chronics", default="summer")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.9, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    regex = CHRONICS_PRESETS.get(args.chronics, args.chronics)
    print(f"severity={args.severity}  chronics={args.chronics} -> {regex!r}")
    print(f"{args.steps} agent decisions per threshold\n")

    from benchmarl.environments.G2OpPowerGrid.pact1.dlr_env import PZMAEnvDLR

    base = None
    print(f"{'safe_max_rho':>13} {'steps/decision':>15} {'agent_share':>12} "
          f"{'sec/decision':>13} {'est_speedup':>12}")
    for thr in args.thresholds:
        cfg = dict(
            env_name=os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023"),
            zone_names=[f"Zone{j}" for j in range(11)],
            use_global_obs=False, use_redispatching_agent=True,
            env_g2op_config={}, local_rewards=None, shuffle_chronics=True,
            regex_filter_chronics=regex, safe_max_rho=thr, curtail_margin=30,
        )
        env = PZMAEnvDLR(severity=args.severity, **cfg)
        env.reset()
        n_g2op, t0 = 0, time.time()
        for _ in range(args.steps):
            before = getattr(env.env_g2op, "nb_time_step", 0)
            act = {a: env.action_space(a).sample() for a in env.agents}
            _, _, done, _, _ = env.step(act)
            after = getattr(env.env_g2op, "nb_time_step", 0)
            n_g2op += max(after - before, 1)
            if any(done.values()):
                env.reset()
        dt = time.time() - t0
        spd = n_g2op / args.steps
        share = 1.0 / spd
        sec = dt / args.steps
        if base is None:
            base = sec
        print(f"{thr:13.2f} {spd:15.2f} {share:11.1%} {sec:13.3f} "
              f"{base/sec:11.2f}x")
        del env

    print("\nPick the LARGEST threshold at which the agents drive a clear")
    print("majority of steps -- largest, because lowering it further changes")
    print("the task more than necessary. Then freeze it and pass the same")
    print("--safe_max_rho to every arm, at every severity.")
    print("\nThis is a measurement of the ENVIRONMENT, made without training")
    print("anything, so it could have been run before any method existed.")


if __name__ == "__main__":
    main()
