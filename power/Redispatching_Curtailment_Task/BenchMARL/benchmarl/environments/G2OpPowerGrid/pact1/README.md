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

## The declared basis — PTDF, and why the first version failed

**Current:** `W[i,j] = mean |PTDF|` from peer *j*'s curtailable generators onto
agent *i*'s own lines, computed once from the grid model via
`lightsim2grid gridmodel.get_ptdf()`. Peers are then split per agent into
`r=2` channels by coupling strength, with the PTDF magnitudes carried as
weights inside each channel. This is IV.1's prescription — the coupling
operator *is* the network's own load-transfer operator, a declared quantity
every operator has, never fitted from run data. θ, how load distributes across
the channels, stays unknown and is what the RLS tracks.

**The first version used zone geometry instead and it did not work.** Channels
were "peer owns a line I watch" (halo) vs everything else (distant), every pair
inside a bucket weighted equally. Measured over 19k steps on `l2rpn_idf_2023`:

```
fit_gain  mean = -0.0045, only 9.7% of rows > 0
```

Negative fit_gain means the peer columns made the one-step-ahead prediction
*worse* than an intercept-only null model. Diagnosis, from `probe_ptdf.py`:

```
PTDF W spread std/mean = 1.3499
geometric halo spread  = 0.5463
geometric distant      = 0.1447
W is strongly ASYMMETRIC:  W[1,0]=0.075 vs W[0,1]=0.031
                           W[6,10]=0.171 vs W[10,6]=0
```

True coupling spans orders of magnitude and is asymmetric; a symmetric
two-bucket geometric relation cannot represent it, and flattening it to two
weights destroyed the signal. `basis.py` now refuses to fall back to the
geometric version rather than silently degrading to the one that fails.

Two structural facts the real grid forces, both reproduced in the self-check:
**Zone6 exerts nothing** (no curtailable generators) and **Zone9 receives
nothing** (its own lines are insensitive to every peer). Zone9 is reported in
`dead_agents` and stays at `g=0` by the floor property rather than being
silently zero-filled.

### Scaling: per agent, per channel — and why

The first PTDF version kept the old normalisation (`x_ref.std(axis=0)`: one
scale per channel, shared across agents) and measured **`cond_psi` = 72,148**
against 57 for the geometric basis, with `fit_gain` still negative. Cause:
Zone0's PTDF row runs 0.0306 down to 0.0001, so under a shared scale the weak
channel becomes a near-zero column and the Gram goes singular.

Each agent-channel is now scaled by its own declared range,
`sum_j W[i,j] * pmax_j`, putting `psi` in [-0.5, +0.5] for every agent whatever
the PTDF magnitude. Still declared — operator plus action box, no run data —
and each agent runs its own estimator, so per-agent scaling leaks nothing.

**`r = 1` is the default.** Splitting peers into strong/weak channels measures
*worse* conditioned than a single PTDF-weighted channel (3006 vs 810 on the
calibrated fixture), because after each channel is normalised by its own range
both columns collapse to a weighted mean of peer curtailment fractions and go
near-collinear. IV.4: report the effective r, do not force channels to survive.

**The self-check cannot calibrate an absolute `cond`** — synthetic ~8e2 where
the real run measured 7.2e4. It tests non-singularity and the absence of zero
columns; `fit_gain` on the real environment is the decisive measurement.

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

- **No loop coupling**: std 0.1336 → 0.1174, **12.1% of the fluctuation
  removed**, against a ceiling of 14.0% implied by
  `corr(ell_hat, ell_true) = 0.510` — i.e. 87% of what the estimate permits.
  The binding constraint is identifiability, not the control law.
  (With the old geometric-shaped weights: corr 0.42, 9.6% removed.)
- **Loop coupled (T4)**: interior optimum at `g=0.30` beating blind
  (0.1314 vs 0.1336), degrading to 0.1437 at `g=0.90`. III.7's β-sweep shape
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
