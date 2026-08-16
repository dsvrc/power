# PACT-1 on the Grid2Op redispatching / curtailment task

Implements `NS_and_PACT1_complete_guide.md` Part III on the MARL2Grid
continuous-action task. Section refs below are to that guide.

## Run

```bash
python -m pact1.selfcheck
```

The arithmetic gate (IV.4). Needs numpy and `zones_definitions.json` — no
grid2op, no torch, no dataset. **Run it before the simulator, every time.**

```bash
python main.py --alg PACT1 --seeds 0 --n_envs 1
```

`--alg MAPPO` is the exactly-matched blind arm: PACT-1 is an env wrapper and
never touches the host (III.4), so both arms share host hyperparameters and an
arm difference cannot be an algorithm difference.

## Why this task and not Topology

| requirement | redispatch/curtail | topology |
|---|---|---|
| invertible harm channel (I.4) | ✅ additive in curtailment limits | ❌ discrete switching has no inverse |
| partial trust buys partial recovery (III.5) | ✅ continuous | ❌ permutation: β=0.5 measured *worse* than β=0 |
| excitation survives (IV.4) | ✅ continuous setpoints keep moving | ❌ discrete fleets converge and the regressor freezes |
| codebase supports it | ✅ | ❌ `core.py` raises on continuous actions |

## The declared basis — measured, not assumed

Zone geometry only; no run data, no PTDF call. Three candidate channels:

- **core** — peer drives a line this agent owns
- **halo** — peer drives a line this agent only watches
- **distant** — everything else

Measured on `l2rpn_idf_2023`, 11 zones:

```
core     pairs=  0   spread=inf      DROPPED (zones partition the lines)
halo     pairs= 33   spread=0.5211   kept
distant  pairs= 77   spread=0.1376   kept
effective r = 2
cond(Gram)  raw 3.9e4  ->  centred+scaled 7.15
```

`core` is empty because no two zones own the same line. Reporting **effective
r = 2** rather than forcing three channels is IV.4's rule; the centring result
reproduces III.6's URB fix (1.3e5 → 24 there, 3.9e4 → 7.15 here).

## Three deviations from the guide, each forced by a measurement

**1. The gate is a product of three terms, not III.5's one.**
The inverse is a ratio and III.5 only gates the numerator. Measured: `ell_hat`
tracked `ell_true` in sign and magnitude while the own-gain coefficient read
**+0.37 against a true −1.13**, and the prediction gate reported `conf=0.90`
throughout — nothing was wrong with the prediction. Dividing by a wrong-signed
estimate steers compensation backwards. Added `divisor_confidence` (relative
standard error of the divisor) and `ready_confidence` (sample count).

**2. Compensation cancels the fluctuation, not the pedestal.**
III.6's centring on the geometric reference is kept exactly for identification.
But a trained policy does not act uniformly at random, so `psi` carries a large
constant offset — measured mean `[−0.50, −2.01]` against temporal std
`[0.30, 0.17]`, i.e. ~85% pedestal. Cancelling it burns the whole bounded
actuator budget on a constant and leaves the time-varying part uncorrected.
`StandingLevel` removes the standing term from the **control path only**.
`tau` is a declared constant and belongs in the ablation table (II.1).

**3. Forgetting is much slower than the guide's default.**
III.9's floor has an interior optimum. Swept:

```
mu      0.990  0.995  0.997  0.999  0.9995  0.9999
corr    0.556  0.500  0.481  0.525  0.555   0.572
dg_std  0.395  0.317  0.231  0.093  0.055   0.030
```

Aggressive forgetting buys nothing here and injects noise straight into the
divisor. Default `mu=0.9995`. **Re-measure on the real chronics** — their drift
is faster than the synthetic driver.

## What the self-check establishes

32/32, including end-to-end on a synthetic linear grid:

- **No loop coupling**: std 0.2734 → 0.2470, **9.6% of the fluctuation removed**,
  against a ceiling of 9.2% implied by `corr(ell_hat, ell_true) = 0.42`. The
  method extracts essentially everything its estimate permits; the binding
  constraint is identifiability, not the control law.
- **Loop coupled (T4)**: interior optimum at `g=0.15` beating blind
  (0.2618 vs 0.2734), degrading to 0.4150 at `g=0.90`. III.7's β-sweep shape
  reproduced, including over-compensation ending up worse than doing nothing.
  `g* < 1` as III.9 predicts.

Also caught, and worth keeping: a **single-tone** policy makes `own_col` and
every `psi` column sinusoids at one frequency, so the regressor spans a 2-D
subspace and nothing is identifiable (smallest singular value ratio 0.053 vs
0.259 broadband). A converged narrowband policy would do this in the real env.
That is what `cond_psi` is logged for.

**A green self-check does not show the method works on the grid.** It shows the
arithmetic is not the reason if it does not.

## Read `applied_trust` before any other column

III.5's URB case: the arm ran to completion, produced a healthy learning curve,
beat the baselines and reported `fit_r2 = 0.9998` — while the compensator had
been switched off the whole time. A safety property that silently engages is
indistinguishable from a bug.

`pact_debug.csv` columns are documented in `diagnostics.py`. Gate on `fit_gain`
(lift over the intercept-only null model), never raw R². Treat non-finite
`cond_psi` as a **value**, and test it first — `isfinite(c) and c > thr` lets
the most degenerate basis possible pass silently.

## Honest limits for POWER specifically

- **N=1 is not byte-identical to a stationary task.** Weather still stresses a
  lone operator's grid. What vanishes at N=1 is the *coupling component*, which
  is what the category-C claim covers. This is a natural environment, so IV.5's
  weaker headline applies — do not claim an injected category-C NS here.
- **The driver does not algebraically multiply the coupling.** The chronics
  multiply it only through convexity of harm in loading: near a thermal limit
  the same peer action costs more. Physically real, but an approximation to
  I.2's exact form, and a reviewer will ask.
- **`r=2`, and the distant channel is weak** (spread 0.1376). θ may be
  predictable without being decomposable (III.14 limit 3). Report `cond_psi`
  before claiming to identify θ itself.
- **The redispatching agent is not modelled as a peer.** It acts on every
  generator at once, so its effect is common across zones and lands in the
  intercept.
