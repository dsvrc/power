"""Read and compare BenchMARL csv-logger runs.

    python read_results.py <run_dir> [<run_dir> ...]

<run_dir> is any path at or above the 'scalars' folder -- the script finds it.

    # list what was logged
    python read_results.py saved_models/mappo_..._21_01_25

    # compare two arms on the headline metric
    python read_results.py saved_models/mappo_... saved_models/pact1_...

    # pick a metric explicitly
    python read_results.py --metric train/reward/episode_reward_mean run_a run_b

Reports the mean over the last 20% of training rather than the final point:
a single last value on a noisy curve is not a result.
"""
import argparse
import glob
import os

import numpy as np

# Ordered preference for "the headline metric". Evaluation is off in the
# shipped config, so a collection/train reward key is normally what exists.
PREFERRED = (
    "eval/reward/episode_reward_mean",
    "collection/reward/episode_reward_mean",
    "train/reward/episode_reward_mean",
    "collection/reward/reward_mean",
    "train/reward/reward_mean",
)


def find_scalars_dir(path):
    if os.path.basename(path.rstrip(os.sep)) == "scalars":
        return path
    hits = glob.glob(os.path.join(path, "**", "scalars"), recursive=True)
    if not hits:
        raise SystemExit(f"no 'scalars' directory found under {path}")
    return sorted(hits)[0]


def load_metric(scalars_dir, name):
    """torchrl's CSVLogger writes one file per metric, '/' flattened to '_'."""
    for cand in (name, name.replace("/", "_")):
        p = os.path.join(scalars_dir, cand + ".csv")
        if os.path.exists(p):
            return read_csv(p)
    return None


def read_csv(path):
    steps, vals = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                s, v = float(parts[0]), float(parts[-1])
            except (ValueError, IndexError):
                continue          # header or malformed row
            steps.append(s)
            vals.append(v)
    return np.asarray(steps), np.asarray(vals)


def available(scalars_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(scalars_dir, "*.csv"))):
        s, v = read_csv(p)
        if len(v):
            out.append((os.path.basename(p)[:-4], len(v), v[-1]))
    return out


def pick_metric(scalars_dir):
    names = {n for n, _, _ in available(scalars_dir)}
    for want in PREFERRED:
        for cand in (want, want.replace("/", "_")):
            if cand in names:
                return want
    # fall back to anything that looks like an episode return
    for n in sorted(names):
        if "reward" in n.lower():
            return n
    return None


def tail_mean(vals, frac=0.2):
    if not len(vals):
        return np.nan
    k = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-k:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--list", action="store_true",
                    help="list every logged metric and exit")
    args = ap.parse_args()

    dirs = [find_scalars_dir(r) for r in args.runs]

    if args.list:
        for r, d in zip(args.runs, dirs):
            print(f"\n=== {r}\n    {d}")
            for n, k, last in available(d):
                print(f"    {n:55s} n={k:5d}  last={last:.4g}")
        return

    metric = args.metric or pick_metric(dirs[0])
    if metric is None:
        raise SystemExit("could not identify a reward metric; rerun with --list")
    print(f"metric: {metric}\n")

    rows = []
    for r, d in zip(args.runs, dirs):
        got = load_metric(d, metric)
        if got is None:
            print(f"  {os.path.basename(r.rstrip(os.sep)):50s}  MISSING")
            continue
        s, v = got
        rows.append((r, s, v))
        print(f"  {os.path.basename(r.rstrip(os.sep))}")
        print(f"      points   {len(v)}   frames {s[-1]:.0f}" if len(s) else "")
        print(f"      final    {v[-1]:.4f}")
        print(f"      last 20% {tail_mean(v):.4f}   <- quote this, not the final point")
        print(f"      best     {v.max():.4f}")

    if len(rows) == 2:
        (ra, _, va), (rb, _, vb) = rows
        a, b = tail_mean(va), tail_mean(vb)
        print(f"\n  {os.path.basename(rb.rstrip(os.sep))} vs "
              f"{os.path.basename(ra.rstrip(os.sep))}: "
              f"{b - a:+.4f}  ({100 * (b - a) / abs(a):+.1f}%)")
        print("\n  One seed each is not evidence. Repeat over >=5 seeds and")
        print("  report bootstrap CIs before treating this gap as real.")


if __name__ == "__main__":
    main()
