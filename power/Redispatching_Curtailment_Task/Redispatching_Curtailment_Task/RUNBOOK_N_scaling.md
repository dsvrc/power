# Runbook — choosing N and running the paired head-to-head

Run these on the machine that has grid2op and `l2rpn_idf_2023` (the Linux box;
`main.py` falls back to serial collection anywhere else). `cd` into this
directory first — every path below is relative to it.

Steps 1–4 are **method-blind**: nothing in them can see an algorithm's return.
Commit their output *before* step 5, because the git history is the evidence
that N was not chosen after seeing a score (HANDOFF §5.3).

---

## 0. Preflight — 30 seconds, no grid2op needed

```bash
python make_zones.py --selfcheck
```

Expect `100/100 checks passed`. It exercises the partition arithmetic on a
synthetic lattice: invariants, determinism, the N=1 and N=n_sub edges, the
disconnected-grid case, and that the validator actually rejects corrupted
input. It cannot prove the partition is a *good* one; it proves the arithmetic
is not the reason if it is not.

```bash
python -m pact1.selfcheck
```

From `../BenchMARL/benchmarl/environments/G2OpPowerGrid`. Expect **53/53**.
Confirms `pact1/` did not regress — the only file touched there was
`basis.py::_load_zones`, and only its path resolution.

---

## 1. Generate the partitions

```bash
python make_zones.py --n-zones 22 33 --compare-shipped
```

Writes `zones_definitions_N22.json`, `zones_definitions_N33.json` and a
`.meta.json` beside each (method, weights, repair counts, warnings, git rev)
into `../BenchMARL/benchmarl/environments/G2OpPowerGrid/`.

**N=11 is deliberately absent**: `--n_zones 11` keeps the shipped hand-drawn
`zones_definitions.json` byte for byte, so every existing 11-zone result
(MAPPO 303 / ~707 / ~773, PACT 705) stays comparable.

Also snapshot the topology, so the partitions can be regenerated and audited
anywhere without grid2op:

```bash
python make_zones.py --dump-topology topology_l2rpn_idf_2023.json
```

**Read in the report, in this order:**

| line | what it decides |
|---|---|
| `repairs: connectivity N subs` | large N means spectral bisection fought the graph; the zones are still valid and connected, but check the size spread below |
| `substations/zone min/max` | a partition where one agent owns half the grid is not the regime the N-scaling argument is about |
| `curtailable/zone ... zones with none` | **the G4 number.** A zone with none has a shape `(0,)` action head and can never act. The shipped Zone6 already has none, so it is legal — but at N=33 this is what makes agents lever-less |
| `TIE-LINES ... = X%` | the mechanism. Must rise with N, or the whole premise is wrong |

If N=33 shows many zero-curtail zones, that is not a bug to route around — it
is the measurement telling you the task is fragmenting, and it is what G4 will
confirm or deny in step 3.

**Do not** reach for `--balance-by curtail` to make that number look better.
It is a legitimate, physically motivated criterion (real operators do draw
control areas around controllable generation) and it is implemented — but it
is also a knob that improves the method's prospects, so switching to it after
seeing an unfavourable count is exactly the move HANDOFF §5 and NS_FORM_SPEC
§E.2.10 forbid. If you want it, choose it now, before step 3, and say so.

---

## 2. Confirm the environment actually assembles at each N

```bash
python debug_env.py --n_zones 22 --severity 1.0 --chronics summer --steps 25
python debug_env.py --n_zones 33 --severity 1.0 --chronics summer --steps 25
python debug_env.py --n_zones 22 --severity 1.0 --pact1 --steps 25
```

Cheapest way to find out whether the observation and action spaces build. A
failure here surfaces as a real traceback; the same failure inside a collector
worker surfaces only as `EOFError`.

Then re-measure how often agents are actually consulted (gate G7 / NS_FORM_SPEC
D.3) — it is a property of the environment, so it must be re-measured at the
chosen N and then frozen identically for every arm:

```bash
python probe_safe_rho.py --n_zones 22 --severity 1.0 --chronics summer
```

`safe_max_rho 0.7` gave 2.2 steps/decision at N=11. Agents must drive a clear
majority of steps.

---

## 3. The gates — this is what chooses N

```bash
for N in 11 22 33; do
  python gate_severity.py --n-zones $N --sigmas 0 0.5 1.0 1.5 \
      --episodes 24 --season summer | tee gates_N${N}_summer.txt
  python gate_severity.py --n-zones $N --sigmas 0 0.5 1.0 1.5 \
      --episodes 24 --season winter | tee gates_N${N}_winter.txt
done
```

**G4 is blocking.** At N=33 a privileged full-information controller must still
survive. If it does not, the task is throughput-limited by fragmentation, no
method can recover it, and N=33 is out — regardless of how attractive its PEER
fraction looks. N=22 is the safe candidate; N=33 has to earn its place.

**Winter is the placebo.** Every σ row must be byte-identical there (N=11 gave
reference 305.8 / privileged 441.5 at all five severities). If winter rows
differ, something other than the dial is moving and every summer number is
suspect.

Then the theory, on the real partitions:

```bash
python theory_scaling.py --n-zones 11 22 33 --sigma 1.0 --episodes 8 \
    --chronics summer | tee theory_scaling_generated.txt
```

`--partition generated` is the default and is the number to trust. The table in
HANDOFF Part 1 (PEER 22.6% → 28.0% → 73.1%) came from contiguous
substation-index *blocks*, which are not electrical — two substations with
adjacent ids need not be connected. Reproduce it if you want the comparison
with `--partition blocks`, but **quote the generated numbers**.

Per-N ceiling decomposition, for the same reason:

```bash
for N in 11 22 33; do
  python theory_ceiling.py --n-zones $N --sigmas 0.5 1.0 1.5 --episodes 8 \
      | tee theory_ceiling_N${N}.txt
done
```

---

## 4. Freeze

Pick N from steps 1–3 only. Then:

```bash
git add -A
git commit -m "Zone partition N=<chosen>; gates re-certified"
git tag frozen-N<chosen>
```

Freeze `pact1/` here and do not touch it until the campaign ends — five method
revisions have already invalidated earlier runs.

---

## 5. The paired head-to-head

Per seed in `{0,1,2,3,4}`, two runs differing **only** in `--alg`:

```bash
N=22
for s in 0 1 2 3 4; do
  python main.py --n_frames 2_000_000 --lr 3e-5 --MAPPO_n_episode 30 \
    --seeds $s --chronics summer --safe_max_rho 0.7 --n_zones $N \
    --dlr_spatial false --severity 1.0 \
    --pact1_gate binary --pact1_max_trust 0.5 --alg MAPPO

  python main.py --n_frames 2_000_000 --lr 3e-5 --MAPPO_n_episode 30 \
    --seeds $s --chronics summer --safe_max_rho 0.7 --n_zones $N \
    --dlr_spatial false --severity 1.0 \
    --pact1_gate binary --pact1_max_trust 0.5 --alg PACT1 \
    --pact1_log pact_N${N}_s${s}.csv
done
```

### `--dlr_spatial false` is not optional for an N sweep

`dlr.zone_phase()` takes `n_zones`, so with per-zone ampacity — **the env class
default** — changing N changes *which region the heat wave hits when*. Measured
at σ=1: the mean derating is N-invariant (0.82279 at N=11, 22 and 33 alike), so
this is a pattern change rather than a severity change, but it still rides
along with the independent variable and a reviewer will call it exactly that.
Uniform ratings are also what `theory_scaling.py` and `theory_ceiling.py`
default to, so this keeps the theory and the training arms on the same physics.
`main.py` prints a warning if you pass `--n_zones` at σ>0 without it.

### Reading the result

Statistic: `d_i = PACT_i - MAPPO_i` per seed, bootstrap the mean of `d`.
Pairing removes the shared seed effect, which is what makes MAPPO's spread
(240 / 713 / 828 — std 312, ~99% of the mean) tractable.

**Always compare at a common horizon.** Runs stop at different iteration
counts, and comparing a 334-iteration run to a 241-iteration one flatters the
longer one.

Before anything else, in every `pact_N*_s*.csv`: read `applied_trust` and
`delta_nonzero_frac`. A silently disabled compensator produces a healthy
learning curve that beats baselines while the method is off.

---

## 6. Ablations

| arm | flag | isolates |
|---|---|---|
| **peer-only** | `--pact1_ff_gain 0` | **run this first.** The feedforward is 79% of the correction and is a *local* term. If peer-only ties MAPPO, the coordination claim on this environment is thin and the paper should lead with whichever of URB/Ant/SMAC/VMAS has the largest PEER fraction. It decides the framing |
| information-matched baseline | MAPPO + the same analytic feedforward | **mandatory.** Without it the gap is information, not mechanism |
| T4 inverted-U | `--pact1_max_trust` sweep | the sweep *is* the evidence, not tuning. Calibrate on one seed, validate on held-out seeds |
| scaling law | `--n_zones` sweep at fixed σ | the prediction |
| trivial ablation | recurrent host + raw peer actions in obs | if it matches PACT, the estimator is decoration |

---

## What changed to make any of this possible

| file | change |
|---|---|
| `make_zones.py` | **new.** Partition generator, validator, offline self-check |
| `main.py` | `--n_zones`, `--dlr_spatial`; both applied after the yaml merge |
| `G2OpPowerGrid/utils.py` | swappable `ZONES_DICT` via `G2OP_ZONES_FILE`; fixed a latent `add_missing_keys` crash |
| `G2OpPowerGrid/PZMultiAgentEnv.py` | `zones_file` kwarg |
| `G2OpPowerGrid/common.py` | routes `dlr_spatial` only to the classes that accept it |
| `G2OpPowerGrid/my_power_grid.py` + yaml | `zones_file`, `dlr_spatial` config fields |
| `pact1/basis.py` | `_load_zones()` honours the override; cached on (path, mtime) |
| `gate_severity.py`, `theory_ceiling.py`, `theory_scaling.py` | `--n-zones` |
| `debug_env.py`, `probe_safe_rho.py` | `--n_zones` |

Defaults are unchanged everywhere: with no new flag passed, every one of these
files behaves exactly as before.

### The partition rule, and where the shipped file departs from it

Generated zones are a **full partition** — every substation in exactly one
zone, a line owned when both endpoints are inside, tie-lines owned by nobody.

The shipped 11 are hand-drawn and follow no single rule: they cover 101 of 118
substations (17 sit on the seams, owned by nobody) and their border fields
contradict each other — Zone0 lists `gen_border_idx` 7, 8, 9 while its
`sub_border_in_ids` is empty, and those same generators are Zone1's
`gen_inside_idx`. `make_zones.py --compare-shipped` reports structure, not
equality, for that reason.

This does not affect correctness: the only fields any consumer reads are
`line_in_zone_idx`, `line_large_idx`, `gen_inside_idx`, `gen_large_idx`,
`load_large_idx`, `storage_inside_idx` and `storage_border_idx`, and all seven
are well defined in both. It does mean the N=11 row of the scaling curve is
drawn under a slightly different rule than N=22 and N=33 — worth one sentence
in the paper. If a reviewer presses, regenerate N=11 with
`make_zones.py --n-zones 11 --out-dir <somewhere-else>` and re-run step 3
against it; the numbers will differ and the *shape* should not.
