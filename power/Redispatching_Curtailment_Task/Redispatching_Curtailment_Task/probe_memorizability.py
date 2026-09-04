"""Can a policy just MEMORISE the disturbance? Answered in minutes, no training.

    python probe_memorizability.py --modes clock stochastic wind
    python probe_memorizability.py --modes stochastic --geographic
    python probe_memorizability.py --selfcheck          # no grid2op

WHY THIS EXISTS
---------------
"Is MAPPO going to catch up?" was being answered by 24-hour training runs. It
does not need to be. MAPPO catches up when the disturbance is a learnable
function of what it observes, and that is a SUPERVISED question:

    fit  ampacity_ratio(t)  from the clock fields the agent already sees
         (month, hour_of_day, minute_of_hour, day_of_week)
    report R^2

R^2 near 1 means a lookup table exists, a blind learner will find it with
enough frames, and no amount of coordination pressure elsewhere will stop it.
R^2 near 0 means the disturbance cannot be anticipated from the clock and has
to be handled online -- which is the regime a compensator is for.

On the shipped dial this should return R^2 = 1.000, because dlr.ambient_temp
is documented as "deterministic in (month, hour)" and both fields are in the
observation. That number is the diagnosis, and it costs two minutes.

WHAT IT DOES NOT TELL YOU
-------------------------
A low clock-R^2 says a blind learner cannot ANTICIPATE the disturbance. It does
not say a compensator can fix it, and it does not say the residual needs
coordination rather than local feedback. Those are theory_ceiling.py's PEER
column and the --pact1_ff_gain 0 ablation respectively. This screens obstacles;
it does not certify them.

The regression is deliberately generous to the baseline: a rich Fourier basis
on the clock, fit and scored on the SAME data. That is an upper bound on what a
policy could memorise, so a low R^2 here is a strong statement and a high R^2
is not surprising.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# the regression -- pure numpy, testable without grid2op
# ---------------------------------------------------------------------------

def clock_features(month, hour, minute, dow, n_harm=6):
    """Fourier basis on the calendar: what a network can represent easily.

    Harmonics rather than raw values because a policy net sees normalised
    scalars and can compose periodic responses; giving the regression only
    (hour, month) would understate what a policy can learn and would flatter
    the obstacle.
    """
    month = np.asarray(month, float)
    hour = np.asarray(hour, float)
    minute = np.asarray(minute, float)
    dow = np.asarray(dow, float)
    # Harmonics on hour_of_day AND on the continuous hour+minute/60. The two
    # are different signals and the agent observes both fields separately:
    # dlr's ratio is a function of hour_of_day alone, while anything driven by
    # the chronic's 5-minute cadence lives in the finer one. Spanning only the
    # continuous version cost 0.006 of R^2 on a target that is exactly a
    # function of hour_of_day -- small, but it understates the baseline, which
    # is the wrong direction for a screen whose job is to be generous to it.
    frac = hour + minute / 60.0
    cols = [np.ones_like(hour)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * hour / 24.0),
                 np.sin(2 * np.pi * k * hour / 24.0),
                 np.cos(2 * np.pi * k * frac / 24.0),
                 np.sin(2 * np.pi * k * frac / 24.0)]
    for k in range(1, 3):
        cols += [np.cos(2 * np.pi * k * month / 12.0),
                 np.sin(2 * np.pi * k * month / 12.0),
                 np.cos(2 * np.pi * k * dow / 7.0),
                 np.sin(2 * np.pi * k * dow / 7.0)]
    return np.column_stack(cols)


def r2(X, y, ridge=1e-8):
    """In-sample R^2 of a ridge fit. Upper bound on what a policy could learn."""
    y = np.asarray(y, float)
    if len(y) < X.shape[1] + 2:
        return float("nan")
    var = float(np.var(y))
    if var <= 1e-18:
        # A constant target is perfectly predictable and perfectly useless;
        # calling that R^2=1 would be true but misleading, so name it.
        return 1.0
    A = X.T @ X + ridge * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X.T @ y)
    resid = y - X @ beta
    return float(1.0 - np.var(resid) / var)


def verdict(clock_r2):
    if not np.isfinite(clock_r2):
        return "??  not enough samples"
    if clock_r2 > 0.98:
        return "MEMORISABLE -- a blind learner will match any compensator"
    if clock_r2 > 0.80:
        return "mostly memorisable -- expect the baseline to catch up"
    if clock_r2 > 0.40:
        return "partly memorisable -- some room, not much"
    return "NOT memorisable from the clock -- has to be handled online"


# ---------------------------------------------------------------------------

def collect(args, mode, geographic):
    """Roll do-nothing episodes and record (clock, realised ratio)."""
    from utils import G2OP_ENV_DIR                            # noqa: PLC0415
    import grid2op                                            # noqa: PLC0415
    from grid2op.Action import PlayableAction                 # noqa: PLC0415
    from benchmarl.environments.G2OpPowerGrid.pact1 import (   # noqa: PLC0415
        weather as wx)
    try:
        from lightsim2grid import LightSimBackend             # noqa: PLC0415
        backend = LightSimBackend()
    except ImportError:
        from grid2op.Backend import PandaPowerBackend         # noqa: PLC0415
        backend = PandaPowerBackend()

    import re                                                 # noqa: PLC0415
    from grid2op.Chronics import MultifolderWithCache         # noqa: PLC0415
    env = grid2op.make(os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023"),
                       action_class=PlayableAction, backend=backend,
                       chronics_class=MultifolderWithCache)
    pat = re.compile({"summer": r".*-07-.*$",
                      "winter": r".*-02-.*$"}.get(args.chronics, args.chronics))
    env.chronics_handler.real_data.set_filter(lambda n: bool(pat.match(n)))
    env.chronics_handler.reset()

    layout = None
    try:
        gl = env.grid_layout
        names = list(env.name_sub)
        layout = {i: gl[n] for i, n in enumerate(names) if n in gl}
    except Exception:                                         # noqa: BLE001
        layout = None

    if geographic:
        line_point, n_points = wx.geographic_points(
            env.n_line, env.line_or_to_subid, env.line_ex_to_subid,
            layout=layout, n_points=args.n_points)
    else:
        line_point, n_points = np.zeros(env.n_line, dtype=int), 1

    proc = wx.WeatherProcess(env.n_line, line_point, sigma=args.severity,
                             mode=mode, geographic=geographic,
                             n_points=n_points, seed=args.seed)

    rows, per_line = [], []
    for ep in range(args.episodes):
        env.set_id(ep)
        obs = env.reset()
        proc.reset(ep)
        done, steps = False, 0
        while not done and steps < args.max_steps:
            if steps % args.update_every == 0:
                proc.step()
            r = proc.line_ratios(float(obs.month), float(obs.hour_of_day))
            rows.append((float(obs.month), float(obs.hour_of_day),
                         float(obs.minute_of_hour), float(obs.day_of_week),
                         float(np.mean(r))))
            per_line.append(r.copy())
            obs, _, done, _ = env.step(env.action_space({}))
            steps += 1
    return np.asarray(rows), np.asarray(per_line), n_points


def run(args):
    out = {}
    print(f"{'mode':>22s} {'clock R^2':>10s} {'ratio mean':>11s} "
          f"{'ratio std':>10s} {'spread':>8s}  verdict")
    for mode in args.modes:
        for geo in ([False, True] if args.geographic else [False]):
            rows, per_line, npts = collect(args, mode, geo)
            X = clock_features(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3],
                               n_harm=args.harmonics)
            y = rows[:, 4]
            score = r2(X, y)
            spread = float(np.mean(per_line.max(axis=1)
                                   / np.maximum(per_line.min(axis=1), 1e-9)))
            tag = f"{mode}{'+geo' if geo else ''}"
            print(f"{tag:>22s} {score:10.4f} {y.mean():11.4f} {y.std():10.4f} "
                  f"{spread:8.4f}  {verdict(score)}")
            out[tag] = {"clock_r2": score, "ratio_mean": float(y.mean()),
                        "ratio_std": float(y.std()), "spatial_spread": spread,
                        "n_samples": int(len(y)), "n_points": int(npts)}
    return out


# ---------------------------------------------------------------------------

def selfcheck():
    n, bad = 0, []

    def ck(name, cond):
        nonlocal n
        n += 1
        if not cond:
            bad.append(name)
            print(f"  FAIL  {name}")

    print("probe_memorizability self-check (no grid2op required)")
    rng = np.random.RandomState(0)
    T = 4000
    hour = rng.uniform(0, 24, T)
    month = np.full(T, 7.0)
    minute = rng.uniform(0, 60, T)
    dow = rng.randint(0, 7, T).astype(float)
    X = clock_features(month, hour, minute, dow)

    # a pure function of the clock must score ~1
    y_clock = 0.82 + 0.06 * np.cos(2 * np.pi * (hour - 15) / 24.0)
    ck(f"deterministic clock signal -> R^2 ~ 1 ({r2(X, y_clock):.4f})",
       r2(X, y_clock) > 0.999)
    ck("verdict names it MEMORISABLE", "MEMORISABLE" in verdict(r2(X, y_clock)))

    # pure noise must score ~0
    y_noise = rng.normal(0, 1, T)
    ck(f"pure noise -> R^2 ~ 0 ({r2(X, y_noise):.4f})", abs(r2(X, y_noise)) < 0.05)
    ck("verdict names it NOT memorisable",
       "NOT memorisable" in verdict(r2(X, y_noise)))

    # half and half lands in between and is ordered correctly
    scores = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = (1 - w) * y_clock + w * 0.06 * rng.normal(0, 1, T)
        scores.append(r2(X, y))
    ck(f"R^2 decreases as noise replaces signal "
       f"{['%.3f' % s for s in scores]}",
       all(b <= a + 1e-9 for a, b in zip(scores, scores[1:])))

    # a constant target is degenerate, not informative
    ck("constant target reported as 1.0", r2(X, np.full(T, 0.9)) == 1.0)

    # an AR(1) process -- what the stochastic obstacle actually produces
    dev = np.zeros(T)
    for i in range(1, T):
        dev[i] = 0.93 * dev[i - 1] + np.sqrt(1 - 0.93 ** 2) * rng.normal()
    y_ar = y_clock + 0.02 * dev
    ck(f"AR(1) weather on top of climatology drops R^2 "
       f"({r2(X, y_ar):.4f} < 0.98)", r2(X, y_ar) < 0.98)

    ck("too few samples -> NaN rather than a wrong number",
       not np.isfinite(r2(X[:3], y_clock[:3])))
    print(f"\n  {n - len(bad)}/{n} checks passed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        description="Is the disturbance memorisable from the clock?")
    ap.add_argument("--modes", nargs="+", default=["clock", "stochastic", "wind"])
    ap.add_argument("--geographic", action="store_true",
                    help="also probe the spatial-field variant of each mode")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--chronics", default="summer")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--update-every", type=int, default=6,
                    help="steps per DLR update; match dlr_update_every")
    ap.add_argument("--n-points", type=int, default=8,
                    help="weather cells for --geographic")
    ap.add_argument("--harmonics", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    print(f"severity={args.severity}  chronics={args.chronics}  "
          f"{args.episodes} episodes x {args.max_steps} steps\n")
    print("R^2 of predicting the realised ampacity ratio from the clock fields")
    print("the agent already observes (month, hour, minute, day_of_week).")
    print("High = a lookup table exists = the baseline will find it.\n")
    out = run(args)
    print("\nclock is the CONTROL, not a failure: it is the condition under")
    print("which a blind learner should match a compensator. Report it.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"severity": args.severity, "chronics": args.chronics,
                       "results": out}, f, indent=2)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
