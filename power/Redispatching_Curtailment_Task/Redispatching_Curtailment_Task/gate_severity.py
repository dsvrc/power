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
                      sigma=0.0, target_rho=0.85, act_above=0.88,
                      lookahead_steps=12):
    """Full-information ANTICIPATORY controller: PTDF-targeted curtailment.

    Privileged in strategy as well as in information, which the first version
    was not.  Three things it does that a reactive controller cannot:

    1. Keeps MARGIN.  Acts from rho > 0.88 down to 0.85 rather than
       firefighting at 0.95.  Overload timers and cascading trips make late
       intervention much more expensive than early intervention.
    2. ANTICIPATES the derating.  It knows the ambient model, so it can see the
       afternoon ampacity trough coming an hour out and pre-curtail into it.
       This is the whole point of 3.1's slack condition: derating is
       time-varying (x0.82 at 3pm in July, x1.00 at night and all winter), so
       the work is shiftable -- but only a controller that looks ahead can
       shift it.  A purely reactive controller makes a schedulable problem look
       throughput-limited, which is what blocked G4 everywhere.
    3. Targets by PTDF onto the specific binding line, largest lever first.
    """
    if ptdf is None or not len(curtail_ids):
        return None
    rho_max = float(obs.rho.max())

    # Anticipation: if ampacity is about to fall, tighten the target now.
    if sigma > 0:
        now = dlr.ampacity_ratio(obs.month, obs.hour_of_day, sigma)
        soon = dlr.ampacity_ratio(
            obs.month, (obs.hour_of_day + lookahead_steps / 12.0) % 24, sigma)
        if soon < now - 1e-9:
            shrink = soon / max(now, 1e-9)
            target_rho *= shrink
            act_above *= shrink

    if rho_max <= act_above:
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
         gen_bus, gen_pmax, seed=0, max_steps=2016, chronic_ids=None,
         window=200):
    """Roll a fixed set of chronics at one severity.

    Two things here are load-bearing and were wrong in the first version:

    1. The thermal limit is set AFTER reset and never followed by another
       reset.  reset() restores the configured limits, so setting-then-
       resetting silently discarded the entire dial.
    2. The chronics are pinned with set_id, so every sigma sees THE SAME weeks.
       Previously the extra reset also advanced the scenario, so each sigma was
       measured on a different sample of the year -- which is what produced
       max_rho FALLING as limits shrank, a physical impossibility.
    """
    env.seed(seed)
    lens, rhos, near, overl, rho_win = [], [], [], [], []
    for ep in range(n_episodes):
        if chronic_ids is not None:
            try:
                env.set_id(int(chronic_ids[ep % len(chronic_ids)]))
            except Exception:                                # noqa: BLE001
                pass
        obs = env.reset()
        if sigma > 0:
            env.set_thermal_limit(base_limits * dlr.ampacity_ratio(
                obs.month, obs.hour_of_day, sigma))
        done, steps = False, 0
        while not done and steps < max_steps:
            act = None
            if privileged:
                act = privileged_action(env, obs, ptdf, curtail_ids, gen_bus,
                                        gen_pmax, sigma=sigma)
            if act is None:
                act = reconnect_actions(env, obs) or env.action_space({})
            act.limit_curtail_storage(obs, margin=30)
            obs, _, done, _ = env.step(act)
            steps += 1
            rhos.append(float(obs.rho.max()))
            near.append(float((obs.rho > 0.95).mean()))
            overl.append(float((obs.rho > 1.0).sum()))
            # Loading over a FIXED early window, so a run that dies early is
            # not credited with a low mean rho just because it never saw the
            # hard part.  Survivorship bias otherwise makes higher sigma look
            # calmer, which is the opposite of the truth.
            if steps <= window:
                rho_win.append(float(obs.rho.max()))
            # Only touch limits on a live environment: after a game over
            # grid2op refuses, and that is what crashed the first version.
            if sigma > 0 and not done:
                env.set_thermal_limit(base_limits * dlr.ampacity_ratio(
                    obs.month, obs.hour_of_day, sigma))
        lens.append(steps)
    # Per-episode lengths are returned, not just their mean: the gap statistic
    # is a ratio of two noisy means, and with 8 episodes a reading of "1.30
    # against a 1.30 threshold" is a coin flip.  Anything that goes into the
    # commit history needs an interval attached.
    return (float(np.mean(lens)), float(np.mean(rhos)),
            float(np.mean(near)), float(np.mean(overl)),
            float(np.mean(rho_win)) if rho_win else float("nan"),
            np.asarray(lens, dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=2016)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--season", choices=["summer", "winter", "any"],
                    default="summer",
                    help="summer = where derating bites; winter = placebo arm "
                         "where the dial provably does nothing")
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

    # ---- G0: does the dial reach the physics AT ALL? --------------------
    # Cheap, decisive, and it is the check whose absence let a silently
    # discarded dial produce five sigma rows of pure scenario noise.
    print("\n[G0] dial actually changes rho (same chronic, limits only)")
    env.set_id(0)
    obs = env.reset()
    for _ in range(20):
        obs, _, done, _ = env.step(env.action_space({}))
        if done:
            obs = env.reset()
    rho_static = float(obs.rho.max())
    # Evaluate at the SUMMER PEAK, not at whatever timestamp the rollout
    # happened to reach.  The first version read a January observation, got an
    # uprate of x1.269, and reported the dial as working while it was making
    # the grid easier.
    ratio = dlr.ampacity_ratio(dlr.PEAK_MONTH, dlr.PEAK_HOUR, 2.0)
    env.set_thermal_limit(base_limits * ratio)
    obs2, _, _, _ = env.step(env.action_space({}))
    applied = np.asarray(env.get_thermal_limit(), dtype=np.float64)
    took = np.allclose(applied, base_limits * ratio, rtol=1e-6)
    print(f"  summer-peak ampacity ratio: x{ratio:.3f} (must be < 1)   "
          f"limits took effect: {took}")
    print(f"  max_rho {rho_static:.4f} -> {float(obs2.rho.max()):.4f}  "
          f"(expect roughly x{1/ratio:.2f} if the dial is live)")
    if not took or ratio >= 1.0:
        print("  !! dial not derating; every gate below is noise. STOP.")
    env.set_thermal_limit(base_limits)

    # ---- pick chronics from the season where derating exists -------------
    # Thermal stress is a summer phenomenon.  Winter chronics clip to a ratio
    # of exactly 1.0, so a sweep run on them measures nothing -- which is what
    # happened when ids 0-5 (January) were used, and is also why the shipped
    # training filter ".*-02-.*$" (February) would have been blind to the dial.
    # Winter is therefore kept as an explicit PLACEBO arm rather than discarded.
    names = list(env.chronics_handler.real_data.subpaths)
    def month_of(path):
        base = os.path.basename(str(path))
        for part in base.replace("_", "-").split("-"):
            if len(part) == 2 and part.isdigit() and 1 <= int(part) <= 12:
                return int(part)
        return None

    months = [month_of(p) for p in names]
    summer = [i for i, m in enumerate(months) if m in (6, 7, 8)]
    winter = [i for i, m in enumerate(months) if m in (12, 1, 2)]
    def spread(pool, k):
        """Even coverage of the pool, not its first k entries.

        Taking summer[:8] gave eight consecutive June weeks; peak heat is late
        July (PEAK_MONTH = 7.5), so that samples the mild end of the season and
        understates the derating the dial is meant to apply.
        """
        if not pool or k <= 0:
            return []
        step = max(1, len(pool) // k)
        return [pool[min(i * step, len(pool) - 1)] for i in range(k)]

    if args.season == "summer" and summer:
        chronic_ids = spread(summer, args.episodes)
    elif args.season == "winter" and winter:
        chronic_ids = spread(winter, args.episodes)
    else:
        chronic_ids = spread(list(range(len(names))), args.episodes)
    used_months = sorted({months[i] for i in chronic_ids if months[i]})
    print(f"\nchronics: {len(names)} available, "
          f"{len(summer)} summer / {len(winter)} winter")
    print(f"season={args.season}  ids={chronic_ids}  months={used_months}")
    if args.season == "winter":
        print("  NOTE: winter is the PLACEBO arm -- derating clips to 1.0, so")
        print("        every sigma should look identical here.  That is the point.")
    print(f"\nrolling {args.episodes} episodes per sigma, "
          f"max {args.max_steps} steps ...")
    rows = []
    for s in args.sigmas:
        print(f"\n  {dlr.describe(s)}")
        ln_b, rho_b, nr_b, ov_b, rw_b, eps_b = roll(
            env, base_limits, s, args.episodes, False, ptdf, curtail_ids,
            env.gen_to_subid, env.gen_pmax, args.seed, args.max_steps,
            chronic_ids)
        ln_p, rho_p, nr_p, ov_p, _, eps_p = roll(
            env, base_limits, s, args.episodes, True, ptdf, curtail_ids,
            env.gen_to_subid, env.gen_pmax, args.seed, args.max_steps,
            chronic_ids)
        rows.append(dict(sigma=s, len_b=ln_b, rho_b=rho_b, near_b=nr_b,
                         len_p=ln_p, rho_p=rho_p, rw_b=rw_b,
                         eps_b=eps_b, eps_p=eps_p))
        print(f"    reference : steps={ln_b:7.1f} max_rho={rho_b:.3f} "
              f"rho(first {200})={rw_b:.3f} near_limit={nr_b:.4f} "
              f"overloaded/step={ov_b:.3f}")
        print(f"    privileged: steps={ln_p:7.1f} max_rho={rho_p:.3f} "
              f"near_limit={nr_p:.4f} overloaded/step={ov_p:.3f}")

    base = rows[0]
    if base["len_p"] < base["len_b"]:
        print("\n  !! WARNING: at sigma=0 the privileged controller is WORSE")
        print("     than the reference. G4 is then meaningless -- fix the")
        print("     controller before reading any gate below.")

    # Sanity: rho over the fixed early window MUST rise with sigma. If it does
    # not, the dial is not reaching the physics and no gate below means
    # anything -- this is exactly the failure the first version hid.
    rw = [r["rw_b"] for r in rows]
    monotone = all(b >= a - 1e-9 for a, b in zip(rw, rw[1:]))
    print(f"\n[G0b] rho over first 200 steps vs sigma: "
          f"{' -> '.join(f'{v:.3f}' for v in rw)}")
    print(f"      rises with sigma: {monotone}"
          f"{'' if monotone else '   !! DIAL NOT REACHING PHYSICS -- STOP'}")

    # Bootstrap the coordination gap: it is a ratio of two noisy means, and a
    # point estimate landing on the threshold decides nothing.
    rng = np.random.default_rng(0)
    for r in rows:
        b, p = r["eps_b"], r["eps_p"]
        boot = []
        for _ in range(5000):
            bb = rng.choice(b, len(b), replace=True).mean()
            pp = rng.choice(p, len(p), replace=True).mean()
            boot.append(pp / max(bb, 1e-9))
        r["gap_lo"], r["gap_hi"] = np.percentile(boot, [2.5, 97.5])

    print("\n" + "=" * 86)
    print(f"{'sigma':>6} {'ref':>8} {'vs s=0':>8} {'priv':>8} {'G4a':>6} "
          f"{'G4b gap':>8} {'gap 95% CI':>18} {'rho200':>7}  gates")
    rec_strict, rec_relaxed = None, None
    for r in rows:
        hurt = r["len_b"] < 0.90 * base["len_b"]                    # G3
        g4a = r["len_p"] / max(base["len_p"], 1e-9)                 # capacity
        gap = r["len_p"] / max(r["len_b"], 1e-9)                    # coordination
        g4a_ok = g4a >= 0.95
        # Require the CI's lower bound to clear the bar, not the point estimate.
        g4b_ok = r["gap_lo"] >= 1.30
        g5 = r["rw_b"] > base["rw_b"] * 1.02
        tags = ["G3+" if hurt else "G3-",
                "G4a+" if g4a_ok else "G4a-",
                "G4b+" if g4b_ok else "G4b-",
                "G5+" if g5 else "G5-"]
        if r["sigma"] > 0 and hurt and g5:
            if g4a_ok and rec_strict is None:
                rec_strict = r["sigma"]
            if g4b_ok and rec_relaxed is None:
                rec_relaxed = r["sigma"]
        print(f"{r['sigma']:6.2f} {r['len_b']:8.1f} "
              f"{100*(r['len_b']/max(base['len_b'],1e-9)-1):+7.1f}% "
              f"{r['len_p']:8.1f} {g4a:6.2f} {gap:8.2f} "
              f"[{r['gap_lo']:6.2f},{r['gap_hi']:6.2f}] {r['rw_b']:7.3f}  "
              f"{' '.join(tags)}")

    print("\nG4 has two readings and they answer different questions:")
    print("  G4a  priv(sigma) / priv(sigma=0) >= 0.95   -- is TOTAL CAPACITY")
    print("       preserved?  This is 3.1 read literally.  A derating dial")
    print("       removes capacity, so this is expected to fail past some")
    print("       sigma however good the controller is.")
    print("  G4b  priv(sigma) / ref(sigma) >= 1.30      -- is there a")
    print("       COORDINATION GAP for a method to close at this sigma?")
    print("       This is what METHOD_design.md section 8 actually normalises")
    print("       against: the centralized ceiling AT that severity, not at 0.")

    print("\nG3 the dial must hurt the reference controller (I.2 constraint 4)")
    print("G4 BLOCKING: privileged controller must still survive; failing it")
    print("   means throughput-limited and no method can recover (3.1)")
    print("G5 the grid must run nearer its limits: the peer term scales as")
    print("   1/limit(t), so this is where dormant coupling wakes up")
    print("\n" + "=" * 78)
    if rec_strict is not None:
        tag = "REALISTIC" if abs(rec_strict - 1.0) < 1e-9 else "see label above"
        print(f"  RECOMMENDED sigma = {rec_strict:.2f}   ({tag})   [strict: G3+G4a+G5]")
        print("  Capacity is preserved AND a coordination gap exists: the")
        print("  cleanest case, and 3.1 holds in its literal form.")
    elif rec_relaxed is not None:
        tag = "REALISTIC" if abs(rec_relaxed - 1.0) < 1e-9 else "see label above"
        print(f"  RECOMMENDED sigma = {rec_relaxed:.2f}   ({tag})   "
              f"[relaxed: G3+G4b+G5]")
        print("  NO sigma preserves total capacity (G4a), which is expected of")
        print("  a derating dial.  This sigma still leaves a large coordination")
        print("  gap, so a method has room -- but you MUST then normalise")
        print("  against the centralized ceiling measured AT this sigma, per")
        print("  METHOD_design.md section 8, and say so explicitly.  Reporting")
        print("  'percent of B0 at sigma=0' would be dishonest here.")
    else:
        print("  NO sigma PASSES G3+G5 plus either reading of G4.")
        print("  Do not proceed to method runs; widen --sigmas or --episodes.")
    print("\n  Commit this output before running MAPPO or PACT-1.")
    print("=" * 78)


if __name__ == "__main__":
    main()
