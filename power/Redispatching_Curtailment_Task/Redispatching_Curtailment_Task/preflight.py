"""Configure a run from the environment, before the run, without touching return.

    python preflight.py --n_zones 22 --severity 1.0 --chronics summer \
        --safe_max_rho 0.7 --dlr_spatial false
    python preflight.py ... --calibrate --calib-seeds 100 101 --eval-seeds 0 1 2 3 4
    python preflight.py --selfcheck            # logic checks, no grid2op

Writes a MANIFEST that `main.py --preflight <manifest>` consumes, so the
campaign runs at settings chosen here rather than at whatever was last typed on
a command line.

THE LINE THIS FILE DRAWS
------------------------
There are two kinds of "hyperparameter tuning" and they are not the same thing.

  BLIND (phase A).  Determined by the environment and the declared operator,
  with no policy, no reward and no return anywhere in the computation.  The
  effective number of channels `r`, which agents have no live coupling, whether
  the exertion functional actually varies, whether the dial reaches the
  physics, how often agents are consulted.  These are measurements OF THE TASK.
  Automating them is free: run it, commit the manifest, and the choice is
  auditable and reproducible.

  CALIBRATED (phase B).  Determined by watching return.  `max_trust` is the
  only one here, and PACT_PIPELINE_SPEC 8.4 is explicit about it: it is a
  Phase-1 calibration parameter, swept, reported WITH the sweep, "calibrated on
  one seed and validated on held-out seeds. Calibrating and reporting on the
  same seed is fitting to the test set."

So phase B refuses to run on any seed listed as an evaluation seed, records
which seeds it used, and `main.py --preflight` refuses to evaluate on a seed
the manifest was calibrated on.  That constraint is enforced in code rather
than in a README, because a README does not stop anyone.

TWO THINGS THIS FILE DELIBERATELY DOES NOT DO
---------------------------------------------
1. It does not calibrate the SEVERITY, the partition, the chronics, or
   safe_max_rho.  Those are task physics, fixed by gate_severity.py and
   theory_scaling.py before any method runs, and re-picking them here after
   seeing a method's score is the exact move NS_FORM_SPEC E.2.10 forbids.
2. It does not give PACT a calibration budget the baseline does not get.  Pass
   --calibrate-baseline to sweep the host's own hyperparameters over the same
   grid size on the same held-out seeds; if you calibrate one arm and not the
   other, the gap you measure is tuning budget, not mechanism.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(os.path.dirname(HERE), "BenchMARL", "benchmarl",
                       "environments", "G2OpPowerGrid")

MANIFEST_VERSION = 1

# Task keys that must match between a manifest and the run consuming it.  A
# manifest calibrated at severity 1.0 says nothing about severity 1.5.
TASK_KEYS = ("n_zones", "severity", "chronics", "safe_max_rho", "dlr_spatial")


# ---------------------------------------------------------------------------
# pure logic -- testable without grid2op
# ---------------------------------------------------------------------------

def choose_r(cond_by_r, max_cond=1e4):
    """Pick the channel count from CONDITIONING, never from return.

    cond_by_r maps r -> a summary condition number of [1, psi_peer] measured
    over live agents under the basis's own reference distribution.

    Rule: smallest cond wins; ties and non-finite both lose.  A non-finite cond
    means a channel is exactly collinear and theta cannot be decomposed at all
    (PACT_PIPELINE_SPEC 2.4), so it is tested FIRST -- `isfinite(c) and c > thr`
    is the comparison that lets the most degenerate basis possible pass
    silently, and it has already cost this project one full analysis pass.
    """
    ok = {r: c for r, c in cond_by_r.items()
          if c is not None and np.isfinite(c) and c <= max_cond}
    if not ok:
        return None, "no candidate r gave a finite, well-conditioned basis"
    best = min(ok, key=lambda r: (ok[r], r))
    return best, f"cond={ok[best]:.4g} over {sorted(ok)}"


def check_seed_disjoint(calib_seeds, eval_seeds):
    """Calibration and evaluation seeds must not overlap.  Spec 8.4."""
    overlap = sorted(set(calib_seeds) & set(eval_seeds))
    if overlap:
        raise SystemExit(
            f"calibration seeds {sorted(calib_seeds)} overlap evaluation seeds "
            f"{sorted(eval_seeds)} at {overlap}.\n"
            "Calibrating and reporting on the same seed is fitting to the test "
            "set (PACT_PIPELINE_SPEC 8.4). Pick disjoint sets.")
    return True


def task_block(args):
    return {
        "n_zones": int(args.n_zones),
        "severity": float(args.severity),
        "chronics": str(args.chronics),
        "safe_max_rho": float(args.safe_max_rho),
        "dlr_spatial": (args.dlr_spatial == "true"),
    }


def manifest_matches(manifest, task, strict=True):
    """Return a list of mismatched keys between a manifest and a run's task."""
    mine = manifest.get("task", {})
    bad = []
    for k in TASK_KEYS:
        a, b = mine.get(k), task.get(k)
        if isinstance(a, float) or isinstance(b, float):
            same = (a is not None and b is not None
                    and abs(float(a) - float(b)) < 1e-9)
        else:
            same = (a == b)
        if not same:
            bad.append(f"{k}: manifest={a!r} run={b!r}")
    if bad and not strict:
        return []
    return bad


def pick_calibrated(table, metric="return"):
    """argmax over the sweep, with the whole table kept.

    The sweep IS the T4 evidence (spec 10: "max_trust sweep | the T4 inverted-U
    -- evidence, not tuning"), so the table is the deliverable and the argmax is
    a by-product.  Reporting only the winner throws away the result.
    """
    rows = [r for r in table if r.get(metric) is not None
            and np.isfinite(r[metric])]
    if not rows:
        return None, "no sweep point produced a usable return"
    best = max(rows, key=lambda r: r[metric])
    shape = ", ".join(f"{r['max_trust']:.3g}->{r[metric]:.1f}" for r in rows)
    return best, shape


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                         # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# phase A -- blind measurements of the task
# ---------------------------------------------------------------------------

def _make_env(args, zones_file, zone_names, g2op_env_dir):
    """Build the same env class the training run builds.

    `g2op_env_dir` is passed in rather than imported here on purpose: phase_a
    puts the G2OpPowerGrid directory on sys.path so `pact1.*` resolves, and that
    directory contains its OWN utils.py (the zone helpers). A later
    `from utils import ...` then binds to the wrong module. gate_severity.py and
    theory_*.py dodge this by importing the task-level utils BEFORE the insert;
    phase_a now does the same and hands the value down.
    """
    CHRONICS = {
        "summer": r".*-07-.*$", "winter": r".*-02-.*$",
        "jja": r".*-0[678]-.*$", "djf": r".*-(12|01|02)-.*$",
        "feb": r".*-02-.*$", "all": None,
    }
    cfg = dict(
        env_name=os.path.join(g2op_env_dir, "l2rpn_idf_2023"),
        zone_names=zone_names, zones_file=zones_file,
        use_global_obs=False, use_redispatching_agent=True,
        env_g2op_config={}, local_rewards=None, shuffle_chronics=True,
        regex_filter_chronics=CHRONICS.get(args.chronics, args.chronics),
        safe_max_rho=float(args.safe_max_rho), curtail_margin=30,
    )
    from pact1.dlr_env import PZMAEnvDLR                      # noqa: PLC0415
    return PZMAEnvDLR(severity=float(args.severity),
                      dlr_spatial=(args.dlr_spatial == "true"), **cfg)


def phase_a(args):
    """Everything determinable without a policy, a reward or a return."""
    # ORDER MATTERS. The task directory and ENV_DIR both contain a utils.py,
    # and they are different modules. Bind the task-level one FIRST, so it is
    # in sys.modules before ENV_DIR joins sys.path; everything after this line
    # that says `utils` then still means the right file.
    from utils import G2OP_ENV_DIR                            # noqa: PLC0415
    sys.path.insert(0, ENV_DIR)
    from make_zones import select_partition                   # noqa: PLC0415
    from pact1.basis import (CouplingBasis, get_ptdf,         # noqa: PLC0415
                             gram_cond, ptdf_zone_coupling)
    from pact1 import dlr                                     # noqa: PLC0415

    out = {}
    zones_file, zone_names = select_partition(args.n_zones)
    with open(zones_file, "r", encoding="utf-8") as f:
        zones = json.load(f)
    out["zones_file"] = os.path.basename(zones_file)

    dead_curtail = [z for z in zone_names
                    if not zones[z]["gen_curtail_inside_idx"]]
    dead_lines = [z for z in zone_names if not zones[z]["line_in_zone_idx"]]
    out["zones_without_lever"] = dead_curtail
    out["zones_without_own_line"] = dead_lines
    print(f"\n[A1] partition: {len(zone_names)} zones from {out['zones_file']}")
    print(f"     no curtailable generator : {len(dead_curtail)} "
          f"{dead_curtail if dead_curtail else ''}")
    print(f"     owns no line             : {len(dead_lines)} "
          f"{dead_lines if dead_lines else ''}")
    if dead_curtail:
        print("     ^ these agents can never act. They still sit in every "
              "peer's channel\n       contributing a CONSTANT zero, which is "
              "intercept mass, not signal.")

    print("\n[A2] building env (this is the slow part) ...")
    env_pz = _make_env(args, zones_file, zone_names, G2OP_ENV_DIR)
    env = env_pz.env_g2op
    ptdf = get_ptdf(env)
    out["ptdf"] = ptdf is not None
    if ptdf is None:
        raise SystemExit("no PTDF from this backend; PACT cannot run. "
                         "See probe_ptdf.py.")
    W = ptdf_zone_coupling(env, zone_names, ptdf=ptdf)
    out["W_zero_diagonal"] = bool(np.allclose(np.diag(W), 0.0))
    print(f"     PTDF ok; W zero-diagonal: {out['W_zero_diagonal']}")

    renew = np.where(env.gen_renewable)[0]
    gen_curtail_inside = {
        z: np.intersect1d(np.asarray(zones[z]["gen_inside_idx"], dtype=int),
                          renew) for z in zone_names}
    live = [i for i, z in enumerate(zone_names) if len(gen_curtail_inside[z])]

    # ---- A3: choose r by CONDITIONING, under the basis's own reference -----
    # Section 2.3 defines x_ref as "peers acting uniformly at random", so that
    # is the distribution the basis was designed against and the honest one to
    # measure conditioning under.  No policy is involved, so no return can leak
    # into the choice of r.
    print(f"\n[A3] conditioning sweep over r, under uniform-random exertion")
    rng = np.random.RandomState(0)
    n_samples = int(args.cond_samples)
    cond_by_r, detail = {}, {}
    for r_try in args.r_grid:
        basis = CouplingBasis(zone_names, gen_curtail_inside, env.gen_pmax,
                              W=W, r_target=int(r_try))
        rows = {a: [] for a in live}
        for _ in range(n_samples):
            curt = {z: rng.uniform(0.0, 1.0, size=len(gen_curtail_inside[z]))
                    for z in zone_names}
            phi = basis.exertion(curt)
            psi = np.asarray(basis.waveforms(phi))
            for a in live:
                rows[a].append(np.concatenate([[1.0], np.atleast_1d(psi[a])]))
        per_agent = {}
        for a in live:
            per_agent[zone_names[a]] = float(gram_cond(rows[a]))
        finite = [c for c in per_agent.values() if np.isfinite(c)]
        summary = float(np.median(finite)) if finite else np.inf
        cond_by_r[int(basis.r)] = summary
        detail[int(basis.r)] = {
            "requested_r": int(r_try), "effective_r": int(basis.r),
            "kept_channels": list(basis.kept_names),
            "median_cond_live_agents": summary,
            "worst_cond_live_agents":
                (float(max(finite)) if finite else None),
            "n_live_agents_nonfinite": int(len(per_agent) - len(finite)),
            "dead_agents": [zone_names[a] for a in basis.dead_agents],
        }
        print(f"     r_target={r_try} -> effective r={basis.r} "
              f"{basis.kept_names}  median cond={summary:.5g}  "
              f"non-finite on {len(per_agent) - len(finite)}/{len(per_agent)} "
              f"live agents")
    r_choice, why = choose_r(cond_by_r, max_cond=float(args.max_cond))
    out["cond_by_r"] = detail
    out["r"] = r_choice
    out["r_reason"] = why
    print(f"     -> r = {r_choice}   ({why})")
    if r_choice is None:
        print("     !! no usable basis. Do NOT run PACT at this N until this "
              "is resolved.")

    # The agent whose conditioning is worth logging: one that can actually act.
    # env.py logs cond_psi for agent index 0 only, and at N=22 index 0 is a
    # zone with zero curtailable generators, whose own_col is a constant and
    # whose Gram is therefore singular BY CONSTRUCTION -- inf on every row,
    # telling you nothing about the agents that can act.
    out["cond_agent"] = zone_names[live[0]] if live else None
    out["cond_agent_index"] = int(live[0]) if live else None
    print(f"     representative agent for cond_psi: {out['cond_agent']} "
          f"(index {out['cond_agent_index']})")

    # ---- A4: does the exertion functional vary? (NS_FORM_SPEC A.5) --------
    basis = CouplingBasis(zone_names, gen_curtail_inside, env.gen_pmax, W=W,
                          r_target=r_choice or 1)
    phis = []
    for _ in range(n_samples):
        curt = {z: rng.uniform(0.0, 1.0, size=len(gen_curtail_inside[z]))
                for z in zone_names}
        phis.append(basis.exertion(curt))
    phis = np.asarray(phis)
    cv = float(np.std(phis) / max(abs(np.mean(phis)), 1e-12))
    out["phi_cv"] = cv
    out["phi_cv_ok"] = bool(cv > 0.05)
    print(f"\n[A4] exertion variability std/mean = {cv:.4f} "
          f"({'OK' if cv > 0.05 else 'TOO FLAT -- unidentifiable'}; "
          f"need > 0.05, POWER measured 0.28)")

    # ---- A5: is the dial live? -------------------------------------------
    ratio = float(dlr.ampacity_ratio(dlr.PEAK_MONTH, dlr.PEAK_HOUR,
                                     float(args.severity)))
    out["dial_ratio_at_peak"] = ratio
    out["dial_ok"] = bool(float(args.severity) == 0.0 or ratio < 1.0)
    print(f"[A5] ampacity at the summer peak: x{ratio:.4f} "
          f"({'derating' if ratio < 1 else 'INERT'})")

    # ---- A6: are agents consulted on a majority of steps? (G7 / D.3) -----
    print(f"[A6] steps per agent decision at safe_max_rho="
          f"{args.safe_max_rho} ...")
    env_pz.reset()
    n_g2op = 0
    for _ in range(int(args.probe_steps)):
        before = getattr(env_pz.env_g2op, "nb_time_step", 0)
        act = {a: env_pz.action_space(a).sample() for a in env_pz.agents}
        _, _, done, _, _ = env_pz.step(act)
        n_g2op += max(getattr(env_pz.env_g2op, "nb_time_step", 0) - before, 1)
        if any(done.values()):
            env_pz.reset()
    spd = n_g2op / float(args.probe_steps)
    out["steps_per_decision"] = float(spd)
    out["agents_act_majority"] = bool(spd < 2.0)
    print(f"     {spd:.2f} steps/decision -> agents drive "
          f"{100.0 / spd:.0f}% of steps "
          f"({'majority OK' if spd < 2.0 else 'MINORITY -- lower safe_max_rho'})")
    return out


# ---------------------------------------------------------------------------
# phase B -- calibration on held-out seeds
# ---------------------------------------------------------------------------

def _run_one(args, seed, extra, tag):
    """Invoke main.py exactly as the campaign would, and return its run dir."""
    cmd = [sys.executable, os.path.join(HERE, "main.py"),
           "--n_frames", str(int(args.calib_frames)),
           "--lr", str(args.lr), "--MAPPO_n_episode", str(args.mappo_n_episode),
           "--seeds", str(seed), "--chronics", args.chronics,
           "--safe_max_rho", str(args.safe_max_rho),
           "--n_zones", str(args.n_zones),
           "--dlr_spatial", args.dlr_spatial,
           "--severity", str(args.severity)] + extra
    print(f"     [{tag}] " + " ".join(cmd[1:]))
    before = _run_dirs()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE)
    if proc.returncode != 0:
        print(f"     [{tag}] FAILED rc={proc.returncode}")
        return None
    new = sorted(set(_run_dirs()) - set(before))
    print(f"     [{tag}] done in {time.time() - t0:.0f}s -> "
          f"{os.path.basename(new[-1]) if new else '??'}")
    return new[-1] if new else None


def _run_dirs():
    import glob                                               # noqa: PLC0415
    return glob.glob(os.path.join(HERE, "saved_models", "*"))


def _return_of(run_dir):
    from read_results import (find_scalars_dir, load_metric,  # noqa: PLC0415
                              pick_metric, tail_mean)
    try:
        d = find_scalars_dir(run_dir)
        m = pick_metric(d)
        got = load_metric(d, m) if m else None
        return float(tail_mean(got[1])) if got is not None else None
    except SystemExit:
        return None
    except Exception:                                         # noqa: BLE001
        return None


def phase_b(args, blind):
    """Sweep max_trust on HELD-OUT seeds.  The table is the deliverable."""
    check_seed_disjoint(args.calib_seeds, args.eval_seeds)
    print(f"\n[B] calibrating max_trust on seeds {args.calib_seeds} "
          f"({args.calib_frames:,} frames each)")
    print(f"    evaluation seeds {args.eval_seeds} are EXCLUDED and will be "
          f"refused by main.py --preflight")

    table = []
    for mt in args.trust_grid:
        rets = []
        for s in args.calib_seeds:
            extra = ["--alg", "PACT1", "--pact1_gate", args.gate,
                     "--pact1_max_trust", str(mt),
                     "--pact1_ff_gain", str(args.ff_gain),
                     "--pact1_log", f"calib_mt{mt}_s{s}.csv"]
            d = _run_one(args, s, extra, f"max_trust={mt} seed={s}")
            r = _return_of(d) if d else None
            if r is not None:
                rets.append(r)
        row = {"max_trust": float(mt),
               "return": float(np.mean(rets)) if rets else None,
               "per_seed": rets, "n_seeds": len(rets)}
        table.append(row)
        shown = "n/a" if row["return"] is None else f"{row['return']:.1f}"
        print(f"    max_trust={mt:<5g} -> return {shown}  "
              f"({row['n_seeds']}/{len(args.calib_seeds)} seeds)")

    best, shape = pick_calibrated(table)
    print(f"\n    sweep: {shape}")
    if best is None:
        print("    !! no usable point; leaving max_trust uncalibrated")
        return None
    print(f"    -> max_trust = {best['max_trust']}  (return {best['return']:.1f})")
    print("    Report the WHOLE table: the inverted-U is the T4 evidence, and")
    print("    quoting only the winner throws the result away.")
    return {
        "max_trust": best["max_trust"],
        "table": table,
        "calib_seeds": list(args.calib_seeds),
        "eval_seeds": list(args.eval_seeds),
        "calib_frames": int(args.calib_frames),
        "gate": args.gate,
        "ff_gain": float(args.ff_gain),
        "metric": "tail_mean(episode_reward_mean, last 20%)",
    }


# ---------------------------------------------------------------------------

def emit_commands(path, manifest):
    """Write the campaign as a script, so the chosen settings are what runs."""
    t = manifest["task"]
    cal = manifest.get("calibrated") or {}
    mt = cal.get("max_trust")
    seeds = cal.get("eval_seeds") or [0, 1, 2, 3, 4]
    mf = os.path.basename(path)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by preflight.py -- do not hand-edit; regenerate instead.",
        f"# manifest: {mf}   git {manifest.get('git_rev')}",
        "set -euo pipefail",
        "",
        f"SEEDS=\"{' '.join(str(s) for s in seeds)}\"",
        f"MF={mf}",
        "",
        "for s in $SEEDS; do",
        "  # 1. blind host -- the matched baseline",
        f"  python main.py --preflight $MF --seeds $s --alg MAPPO",
        "",
        "  # 2. information-matched baseline: host + the SAME analytic",
        "  #    feedforward, no coordination. max_trust=0 makes the gate",
        "  #    return g=0 every step, so the peer term is exactly zero and",
        "  #    what is left is the local term any operator could compute.",
        "  #    PACT_PIPELINE_SPEC 12.3 makes this arm mandatory.",
        f"  python main.py --preflight $MF --seeds $s --alg PACT1 \\",
        "      --pact1_max_trust 0.0 --pact1_ff_gain 1.0 \\",
        "      --pact1_log ff_only_s${s}.csv",
        "",
        "  # 3. peer-only -- the coordination term alone. Run this EARLY:",
        "  #    it decides whether the coordination claim survives.",
        f"  python main.py --preflight $MF --seeds $s --alg PACT1 \\",
        "      --pact1_ff_gain 0.0 \\",
        "      --pact1_log peer_only_s${s}.csv",
        "",
        "  # 4. PACT full",
        f"  python main.py --preflight $MF --seeds $s --alg PACT1 \\",
        "      --pact1_log pact_s${s}.csv",
        "done",
        "",
    ]
    if mt is None:
        lines.insert(3, "# WARNING: manifest carries no calibrated max_trust; "
                        "arms 3 and 4 use the default.")
    sh = os.path.splitext(path)[0] + "_campaign.sh"
    with open(sh, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"campaign script -> {sh}")
    return sh


def run_campaign(args, manifest, manifest_path):
    """Phase A -> calibration -> every arm, in one invocation.

    This is the "just do it in one command" path.  What it does NOT do is pick
    hyperparameters per evaluation seed: the calibration ran on
    --calib-seeds, the evaluation runs on --eval-seeds, and the two are
    disjoint by construction.  Tuning inside the evaluation run would mean
    each seed's PACT was fitted to the seed it is reported on, while the
    baseline was not -- which measures tuning budget, not mechanism.
    """
    cal = manifest.get("calibrated") or {}
    mt = cal.get("max_trust")
    arms = [
        ("mappo", ["--alg", "MAPPO"], "blind baseline"),
        ("ff_only", ["--alg", "PACT1", "--pact1_max_trust", "0.0",
                     "--pact1_ff_gain", "1.0"],
         "information-matched: host + the same analytic feedforward (spec 12.3)"),
        ("peer_only", ["--alg", "PACT1", "--pact1_ff_gain", "0.0"],
         "coordination term alone -- this one decides the framing"),
        ("pact", ["--alg", "PACT1"], "PACT full"),
    ]
    print(f"\n[C] campaign: {len(arms)} arms x {len(args.eval_seeds)} seeds "
          f"x {args.frames:,} frames")
    if mt is None:
        print("    WARNING: no calibrated max_trust in the manifest; the PACT "
              "arms will use\n             the command-line default. Run with "
              "--calibrate to fix that.")
    results = {}
    for tag, extra, why in arms:
        print(f"\n    -- {tag}: {why}")
        for s in args.eval_seeds:
            cmd = list(extra) + ["--preflight", os.path.abspath(manifest_path)]
            if tag != "mappo":
                cmd += ["--pact1_log", f"{tag}_s{s}.csv"]
            saved = args.calib_frames
            args.calib_frames = args.frames        # _run_one reads this
            d = _run_one(args, s, cmd, f"{tag} seed={s}")
            args.calib_frames = saved
            results.setdefault(tag, {})[s] = _return_of(d) if d else None

    print("\n" + "=" * 62)
    print(f"  {'seed':>6}" + "".join(f"{t:>13s}" for t, _, _ in arms))
    for s in args.eval_seeds:
        row = "".join(
            f"{results.get(t, {}).get(s):13.1f}"
            if isinstance(results.get(t, {}).get(s), float) else f"{'n/a':>13s}"
            for t, _, _ in arms)
        print(f"  {s:>6}" + row)
    base = [results.get("mappo", {}).get(s) for s in args.eval_seeds]
    for t, _, _ in arms[1:]:
        d = [(results.get(t, {}).get(s), b) for s, b in
             zip(args.eval_seeds, base)]
        d = [a - b for a, b in d if isinstance(a, float) and isinstance(b, float)]
        if d:
            print(f"  paired mean {t} - mappo: {np.mean(d):+.1f} "
                  f"over {len(d)} seeds")
    print("=" * 62)
    print("  Paired differences over a handful of seeds are not a result on")
    print("  their own -- bootstrap the mean of d, and compare at a COMMON")
    print("  horizon: runs stop at different iteration counts and the longer")
    print("  one flatters itself.")
    return results


def selfcheck():
    """Logic checks that need no grid2op, no dataset and no training."""
    n, bad = 0, []

    def ck(name, cond):
        nonlocal n
        n += 1
        if not cond:
            bad.append(name)
            print(f"  FAIL  {name}")

    print("preflight self-check (no grid2op required)")

    # choose_r: non-finite must lose, and must be tested before the threshold
    r, _ = choose_r({1: np.inf, 2: 810.0, 3: 3006.0})
    ck("choose_r picks the best-conditioned r", r == 2)
    r, _ = choose_r({1: np.inf, 2: np.nan})
    ck("choose_r returns None when nothing is finite", r is None)
    r, _ = choose_r({1: 1e9, 2: 500.0}, max_cond=1e4)
    ck("choose_r rejects a basis above max_cond", r == 2)
    r, _ = choose_r({1: np.inf, 2: 1e9}, max_cond=1e4)
    ck("choose_r refuses rather than accept a degenerate basis", r is None)
    r, _ = choose_r({2: 100.0, 1: 100.0})
    ck("choose_r breaks ties toward the smaller r", r == 1)
    r, _ = choose_r({})
    ck("choose_r handles an empty sweep", r is None)

    # seed disjointness is the whole point of phase B
    ck("disjoint seeds accepted", check_seed_disjoint([100, 101], [0, 1, 2]))
    try:
        check_seed_disjoint([0, 100], [0, 1, 2])
        ck("overlapping calib/eval seeds refused", False)
    except SystemExit:
        ck("overlapping calib/eval seeds refused", True)
    try:
        check_seed_disjoint([1], [1])
        ck("single overlapping seed refused", False)
    except SystemExit:
        ck("single overlapping seed refused", True)

    # manifest/task matching
    m = {"task": {"n_zones": 22, "severity": 1.0, "chronics": "summer",
                  "safe_max_rho": 0.7, "dlr_spatial": False}}
    ck("identical task matches", manifest_matches(m, dict(m["task"])) == [])
    t = dict(m["task"], severity=1.5)
    ck("different severity is rejected", manifest_matches(m, t) != [])
    t = dict(m["task"], n_zones=33)
    ck("different n_zones is rejected", manifest_matches(m, t) != [])
    t = dict(m["task"], dlr_spatial=True)
    ck("different dlr_spatial is rejected", manifest_matches(m, t) != [])
    t = dict(m["task"], safe_max_rho=0.7 + 1e-12)
    ck("float noise does not trigger a mismatch", manifest_matches(m, t) == [])

    # sweep selection keeps the table
    tbl = [{"max_trust": 0.1, "return": 300.0},
           {"max_trust": 0.2, "return": 700.0},
           {"max_trust": 0.5, "return": 250.0}]
    best, shape = pick_calibrated(tbl)
    ck("pick_calibrated finds the interior optimum", best["max_trust"] == 0.2)
    ck("pick_calibrated reports the whole shape", shape.count("->") == 3)
    best, _ = pick_calibrated([{"max_trust": 0.1, "return": None}])
    ck("pick_calibrated survives an all-failed sweep", best is None)
    best, _ = pick_calibrated([{"max_trust": 0.1, "return": float("nan")},
                               {"max_trust": 0.2, "return": 5.0}])
    ck("pick_calibrated ignores NaN points", best["max_trust"] == 0.2)

    print(f"\n  {n - len(bad)}/{n} checks passed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        description="Measure the task, then calibrate on held-out seeds.")
    ap.add_argument("--n_zones", type=int, default=22)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--chronics", default="summer")
    ap.add_argument("--safe_max_rho", type=float, default=0.7)
    ap.add_argument("--dlr_spatial", choices=["true", "false"], default="false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--r-grid", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-cond", type=float, default=1e4,
                    help="reject any basis worse conditioned than this")
    ap.add_argument("--cond-samples", type=int, default=2000)
    ap.add_argument("--probe-steps", type=int, default=120)
    ap.add_argument("--calibrate", action="store_true",
                    help="also sweep max_trust on held-out seeds (slow)")
    ap.add_argument("--calib-seeds", type=int, nargs="+", default=[100, 101])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--calib-frames", type=int, default=400_000)
    ap.add_argument("--trust-grid", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--gate", default="binary")
    ap.add_argument("--ff-gain", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--mappo-n-episode", type=int, default=30)
    ap.add_argument("--emit-commands", action="store_true")
    ap.add_argument("--run-campaign", action="store_true",
                    help="after writing the manifest, run every arm on "
                         "--eval-seeds. One command for the whole thing; the "
                         "calibration/evaluation seed split is still enforced.")
    ap.add_argument("--frames", type=int, default=2_000_000,
                    help="frames per EVALUATION run in --run-campaign "
                         "(default 2M; PACT was still climbing at 2M on N=11, "
                         "and a short run measures the warm-up, not the method)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    task = task_block(args)
    print("=" * 74)
    print("  PREFLIGHT -- phase A measures the task, phase B calibrates on")
    print("  held-out seeds. Nothing here re-picks severity, the partition,")
    print("  the chronics or safe_max_rho: those are fixed by the gates.")
    print("=" * 74)
    print(f"  task: {task}")

    blind = phase_a(args)
    manifest = {
        "version": MANIFEST_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_rev": _git_rev(),
        "task": task,
        "blind": blind,
        "calibrated": None,
    }
    if args.calibrate:
        manifest["calibrated"] = phase_b(args, blind)

    out = args.out or os.path.join(
        HERE, f"preflight_N{args.n_zones}_sev{args.severity:g}"
              f"_{args.chronics}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {out}")
    if args.emit_commands:
        emit_commands(out, manifest)
    print("Commit this manifest BEFORE the evaluation runs: it is the record "
          "that\nthe settings were chosen without seeing the evaluation seeds.")
    if args.run_campaign:
        check_seed_disjoint(args.calib_seeds, args.eval_seeds)
        run_campaign(args, manifest, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
