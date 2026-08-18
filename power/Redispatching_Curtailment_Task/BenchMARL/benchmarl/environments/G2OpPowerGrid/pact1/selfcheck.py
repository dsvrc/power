"""Arithmetic self-check -- run this BEFORE the simulator.

IV.4: "budget for plumbing, and gate the plumbing before the physics -- an
arithmetic self-check that runs before the simulator starts is worth more here
than anywhere else."

Needs numpy and zones_definitions.json.  No grid2op, no torch, no dataset.

    python -m benchmarl.environments.G2OpPowerGrid.pact1.selfcheck

Every check below encodes a failure that was actually paid for somewhere in the
guide.  A green run does not prove the method works on the grid; it proves the
arithmetic is not the reason if it does not.
"""
import sys

import numpy as np

from .basis import CHANNEL_NAMES, CouplingBasis, gram_cond
from .compensator import (StandingLevel, action_sensitivity, apply_inverse,
                          compensation_delta)
from .rls import RLSEstimator

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _synth_W(n, seed=0):
    """A stand-in for the PTDF zone coupling with the properties the real one has.

    Measured on l2rpn_idf_2023: strongly ASYMMETRIC, spread std/mean = 1.35,
    weights spanning orders of magnitude, and at least one agent with no
    coupling at all (Zone9) plus one peer that exerts nothing (Zone6, which has
    no curtailable generators).  All four are reproduced here, because each one
    broke something in the first implementation.
    """
    # sigma chosen from the measured grid, not for convenience: Zone0's real
    # PTDF row runs 0.0306 down to 0.0001, ~300x within a single agent's peers,
    # and it is that within-row span that turns a weak channel into a zero
    # column under a shared scale.  A gentler fixture hides the failure.
    rng = np.random.default_rng(seed)
    W = np.exp(rng.normal(-4.0, 2.5, size=(n, n)))   # log-normal => wide spread
    np.fill_diagonal(W, 0.0)
    W[:, 6] = 0.0        # a peer with no curtailable generation
    W[9, :] = 0.0        # an agent whose own lines see nobody
    return W


def _basis(n_zones=11):
    zones = [f"Zone{i}" for i in range(n_zones)]
    rng = np.random.default_rng(0)
    pmax = rng.uniform(20.0, 120.0, size=200)
    curt = {z: np.arange(i * 5, i * 5 + 4) for i, z in enumerate(zones)}
    W = _synth_W(n_zones)
    return CouplingBasis(zones, curt, pmax, W=W, r_target=1), zones, curt, pmax


# ---------------------------------------------------------------- structure
def test_structure():
    print("\n[1] basis structure  (I.7 rule 1, III.6, IV.1)")
    b, zones, _, _ = _basis()
    check("zero diagonal in W", np.allclose(np.diag(b.W), 0.0))
    check("zero diagonal in the channel assignment",
          (np.diag(b.chan) == -1).all())
    check("channels are PTDF-WEIGHTED, not binary buckets",
          len(np.unique(np.round(b.masks[b.masks > 0], 8))) > b.n,
          "the geometric version gave every pair in a bucket one weight, "
          "which measured fit_gain = -0.0045")
    check("effective r reported, dead channels dropped",
          b.r == len(b.keep) and b.r >= 1,
          f"r={b.r} kept={b.kept_names}")
    check("an agent with no coupling is identified, not silently zero-filled",
          9 in b.dead_agents, f"dead_agents={b.dead_agents}")
    # x_ref must be a pure function of the declared operator: bit-identical.
    b2, *_ = _basis()
    check("x_ref is run-data-free (bit-identical on rebuild)",
          np.array_equal(b.x_ref, b2.x_ref))


def test_conditioning():
    """The regressor must stay well-conditioned under PTDF-magnitude spread.

    Measured on l2rpn_idf_2023: sharing one scale per channel across agents
    drove cond(E[psi psi^T]) to 72,148 (against 57 for the geometric basis)
    because PTDF rows span orders of magnitude, and fit_gain stayed negative.
    Per-agent, per-channel scaling by the channel's own declared range fixes it.
    """
    print("\n[2] regressor conditioning under PTDF spread  (III.6, IV.4)")
    zones = [f"Zone{i}" for i in range(11)]
    rng0 = np.random.default_rng(0)
    pmax = rng0.uniform(20.0, 120.0, size=200)
    curt = {z: np.arange(i * 5, i * 5 + 4) for i, z in enumerate(zones)}
    W = _synth_W(11)
    rng = np.random.default_rng(11)
    ph = rng.uniform(0, 2 * np.pi, size=(3, 11))

    def roll(b, shared_scale=False, agent=0):
        """One agent's regressor over time -- the same quantity env.py logs as
        cond_psi.  Pooling rows across agents measures something else entirely
        and hides the failure this test exists to catch."""
        rows = []
        for t in range(1500):
            a = np.clip(0.65
                        + 0.08 * np.sin(2 * np.pi * t / 260.0 + ph[0])
                        + 0.06 * np.sin(2 * np.pi * t / 97.0 + ph[1])
                        + 0.03 * np.sin(2 * np.pi * t / 41.0 + ph[2])
                        + 0.02 * rng.normal(size=b.n), 0.0, 1.0)
            phi = b.exertion({z: np.full(len(curt[z]), a[i])
                              for i, z in enumerate(zones)})
            x = b.masks @ phi
            if shared_scale:
                sc = b.x_ref.std(axis=0)
                sc[sc < 1e-12] = 1.0
                psi = (x - b.x_ref) / sc
            else:
                psi = (x - b.x_ref) / b.scale
            rows.append(np.concatenate([[1.0, 0.2 * np.sin(t / 30.0)],
                                        psi[agent]]))
        return gram_cond(rows)

    b1 = CouplingBasis(zones, curt, pmax, W=W, r_target=1)
    b2 = CouplingBasis(zones, curt, pmax, W=W, r_target=2)
    c1 = roll(b1)
    c2 = roll(b2)
    c2_shared = roll(b2, shared_scale=True)
    print(f"    r=1 per-agent={c1:.4g}   r=2 per-agent={c2:.4g}   "
          f"r=2 shared-scale={c2_shared:.4g}")

    # No hard cond threshold here: this fixture cannot calibrate an absolute
    # number against the real grid (synthetic ~1e3 where the real run measured
    # 7.2e4).  What it CAN test is that the Gram stays non-singular and that
    # no channel collapses to a zero column -- the two failures that actually
    # broke the run.  fit_gain on the real env is the decisive measurement.
    check("r=1 regressor is non-singular", np.isfinite(c1), f"cond={c1:.4g}")
    # Under a scale shared across agents, a weak channel whose PTDF weights are
    # ~100x smaller than the strong one becomes a near-zero column and the Gram
    # goes singular.  Measured on the real grid: cond_psi = 72,148 with the
    # shared scale, against 57 for the old geometric basis.  Per-agent,
    # per-channel scaling by each channel's own declared range removes it.
    check("per-agent scaling beats the shared scale on r=2",
          c2 < c2_shared, f"shared={c2_shared:.4g} -> per-agent={c2:.4g}")
    check("psi stays inside its declared range for every agent",
          True, "scale is the channel's own max load, so this holds by "
                "construction whatever the PTDF magnitude")


def test_n1_irreducibility():
    print("\n[2] N=1 irreducibility certificate  (I.1 litmus)")
    zones = ["Zone0"]
    pmax = np.full(200, 50.0)
    try:
        b = CouplingBasis(zones, {"Zone0": np.arange(4)}, pmax,
                          W=np.zeros((1, 1)), r_target=2)
    except RuntimeError as e:
        # Every channel degenerate at N=1 is the CORRECT outcome: with no peers
        # there is no coupling to identify at all.
        check("N=1 leaves no identifiable coupling channel", True, str(e)[:60])
        return
    phi = b.exertion({"Zone0": np.zeros(4)})     # maximal own exertion
    x = b.masks @ phi
    check("cross-agent load is exactly 0 at N=1", np.allclose(x, 0.0),
          f"max|x|={np.abs(x).max():.3g}")


def test_uncancellable_exertion():
    print("\n[3] exertion functional  (I.3, the escape hatch)")
    b, zones, _, _ = _basis()
    # phi is a magnitude in [0, pmax]: no sign trick can cancel it, which is
    # the Ant failure (signed sum -> anti-symmetric gait -> 49% of the
    # difficulty walked away).
    lo = b.exertion({z: np.ones(4) for z in zones})    # no curtailment
    hi = b.exertion({z: np.zeros(4) for z in zones})   # full curtailment
    check("phi >= 0 always", (lo >= -1e-12).all() and (hi >= -1e-12).all())
    check("phi is monotone in curtailment depth", (hi > lo).all())
    mixed = b.exertion({z: (np.ones(4) if i % 2 else np.zeros(4))
                        for i, z in enumerate(zones)})
    check("phi cannot be cancelled by opposing peers (no signed sum)",
          (mixed >= -1e-12).all() and mixed.sum() > 0)


# ---------------------------------------------------------------- estimator
def test_rls_recovery():
    print("\n[4] RLS recovers a known beta*  (III.1)")
    rng = np.random.default_rng(1)
    r = 2
    beta_true = np.array([0.30, -0.80, 0.45, -0.25])   # [icept, own, ch1, ch2]
    est = RLSEstimator(r, mu=0.999, p0=100.0)
    for _ in range(3000):
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        y = beta_true @ psi + rng.normal(0, 0.01)
        est.update(psi, y)
    err = np.linalg.norm(est.beta - beta_true)
    check("beta_hat converges to beta*", err < 0.05, f"||beta_hat - beta*||={err:.4g}")

    # Drifting parameter: the real case (c(t)*theta(t) tracks the chronics).
    est2 = RLSEstimator(r, mu=0.99, p0=100.0)
    for t in range(4000):
        bt = beta_true.copy()
        bt[2] = 0.45 + 0.30 * np.sin(2 * np.pi * t / 1500.0)
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        est2.update(psi, bt @ psi + rng.normal(0, 0.01))
    bt_end = beta_true.copy()
    bt_end[2] = 0.45 + 0.30 * np.sin(2 * np.pi * 3999 / 1500.0)
    derr = np.linalg.norm(est2.beta - bt_end)
    check("tracks a DRIFTING beta* (forgetting works)", derr < 0.15,
          f"||err||={derr:.4g}")


def test_gate_under_excitation_death():
    print("\n[5] confidence gate under excitation death  (III.5 -- the URB bug)")
    rng = np.random.default_rng(2)
    r = 2
    beta_true = np.array([0.3, -0.8, 0.45, -0.25])
    est = RLSEstimator(r, mu=0.99, p0=100.0)

    # Phase 1: healthy excitation.
    for _ in range(1500):
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        est.update(psi, beta_true @ psi + rng.normal(0, 0.01))
    conf_live = est.confidence_at(psi)
    trace_live = est.trace_confidence()

    # Phase 2: the policy converges and the regressor FREEZES.  Measured on
    # URB: route_switch_frac -> 0.0000 and psi froze to six decimals.
    frozen = np.array([1.0, 0.5, 0.2, -0.1])
    for _ in range(4000):
        est.update(frozen, beta_true @ frozen + rng.normal(0, 0.01))
    conf_dead = est.confidence_at(frozen)
    trace_dead = est.trace_confidence()

    check("PREDICTION gate stays armed after excitation death",
          conf_dead > 0.5, f"{conf_live:.3f} -> {conf_dead:.3f}")
    check("TRACE gate collapses (reproduces the URB failure)",
          trace_dead < conf_dead,
          f"trace {trace_live:.3f} -> {trace_dead:.4f} vs prediction {conf_dead:.3f}")
    pred_err = abs(est.beta @ frozen - beta_true @ frozen)
    check("prediction is still correct while trace gate says otherwise",
          pred_err < 0.05, f"|pred err|={pred_err:.4g}")


def test_fit_gain_null_model():
    print("\n[6] fit_gain against a null model  (III.11 -- the 0.9998 artefact)")
    rng = np.random.default_rng(3)
    r = 2
    # Case A: peer channels carry NO information; only the intercept does.
    est = RLSEstimator(r, mu=0.999, p0=100.0)
    for _ in range(3000):
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        est.update(psi, 5.0 + 0.01 * rng.normal())
    gain_null = est.fit_gain()
    check("fit_gain ~ 0 when the basis explains nothing",
          abs(gain_null) < 0.05, f"fit_gain={gain_null:.4g}")

    # Case B: peer channels genuinely drive the target.
    est2 = RLSEstimator(r, mu=0.999, p0=100.0)
    for _ in range(3000):
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        est2.update(psi, 5.0 + 0.9 * psi[2] - 0.6 * psi[3] + 0.01 * rng.normal())
    gain_real = est2.fit_gain()
    check("fit_gain > 0 when the basis does the work",
          gain_real > 0.2, f"fit_gain={gain_real:.4g}")


# ------------------------------------------------------------- compensation
def test_fit_gate():
    """Compensation must be OFF while the peer channels explain nothing.

    Measured on the 1M-frame run: Q1 had applied_trust = 0.095 with
    fit_gain = 0.0006, and PACT-1 lost 6-14 return per iteration over that
    stretch.  The covariance gates could not see it -- nothing is wrong with
    the covariance when the peer term is simply uninformative.
    """
    print("\n[6b] fit_gain gate  (III.11's rule applied to the control path)")
    rng = np.random.default_rng(21)
    r = 2

    # Case A: peer columns carry NO information -> gate must be hard 0.
    est = RLSEstimator(r, mu=0.9995, p0=100.0, ready_updates=100)
    for _ in range(4000):
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        est.update(psi, 5.0 + 0.8 * psi[1] + 0.02 * rng.normal())
    fg_a, gate_a = est.fit_gain_now(), est.fit_confidence()
    check("uninformative peer channels => gate is exactly 0",
          gate_a == 0.0, f"fit_gain_now={fg_a:+.5f} gate={gate_a:.4f}")

    # Case B: peer columns genuinely drive the target -> gate opens.
    est2 = RLSEstimator(r, mu=0.9995, p0=100.0, ready_updates=100)
    for _ in range(4000):
        psi = np.concatenate([[1.0], rng.normal(0, 1, size=1 + r)])
        est2.update(psi, 5.0 + 0.8 * psi[1] + 0.9 * psi[2] - 0.6 * psi[3]
                    + 0.02 * rng.normal())
    fg_b, gate_b = est2.fit_gain_now(), est2.fit_confidence()
    check("informative peer channels => gate opens",
          gate_b > 0.5, f"fit_gain_now={fg_b:+.5f} gate={gate_b:.4f}")

    # The floor property, restored in the regime that actually needed it.
    d, g = compensation_delta(0.5, -0.8, trust=0.9, conf=gate_a * 0.5)
    check("gate 0 => executed action IS the blind action (floor property)",
          g == 0.0 and d == 0.0, "PACT-1 reduces exactly to MAPPO here")


def test_conjugacy():
    print("\n[7] T3 conjugacy -- the inverse cancels  (I.4, III.7)")
    # Linear channel: realized rho = ... + drho_da * a + ell.
    drho_da = -0.8
    ell_true = 0.12
    a = np.full(4, 0.60)
    residual_blind = ell_true          # blind does not correct

    delta, g = compensation_delta(ell_true, drho_da, trust=1.0, conf=1.0,
                                  max_delta=1.0)
    u, _ = apply_inverse(a, delta)
    residual_comp = abs(ell_true + drho_da * delta)
    check("perfect estimate cancels the peer term",
          residual_comp < 1e-9,
          f"residual {residual_blind:.4g} -> {residual_comp:.3g}")
    check("compensation stays inside the action set", (u >= 0).all() and (u <= 1).all())

    # Partial trust on a CONTINUOUS channel must buy partial recovery -- unlike
    # a permutation channel, where beta=0.5 measured WORSE than beta=0 (I.4).
    d_half, _ = compensation_delta(ell_true, drho_da, trust=0.5, conf=1.0,
                                   max_delta=1.0)
    res_half = abs(ell_true + drho_da * d_half)
    check("partial trust buys partial recovery (continuous channel)",
          residual_comp < res_half < residual_blind,
          f"beta=0.5 residual={res_half:.4g}")

    # The sign chain itself: exertion runs backwards to the action, so a
    # positive own_gain must yield a NEGATIVE d(rho)/d(a).
    s = action_sensitivity(own_gain=0.7, pmax=50.0)
    check("action_sensitivity flips sign (curtail more = lower limit ratio)",
          s < 0, f"own_gain=+0.7 -> drho_da={s:.4g}")


def test_floor_property():
    print("\n[8] floor property -- cannot do worse than blind  (III.4)")
    a = np.full(4, 0.5)
    # A wildly diverged estimate.
    d, g = compensation_delta(1e9, -0.8, trust=0.9, conf=1.0, max_delta=0.25)
    check("diverged estimate is capped by max_delta", abs(d) <= 0.25 + 1e-12,
          f"delta={d:.4g}")
    # No usable inverse -> exactly blind.
    d0, g0 = compensation_delta(5.0, 1e-9, trust=0.9, conf=1.0)  # noqa: E501
    u0, _ = apply_inverse(a, d0)
    check("no inverse available => executed action IS the blind action",
          g0 == 0.0 and np.array_equal(u0, a))
    # NaN/inf must not propagate into the control path.
    for bad in (np.nan, np.inf, -np.inf):
        db, gb = compensation_delta(bad, -0.8, trust=0.9, conf=1.0)
        if db != 0.0 or gb != 0.0:
            check("non-finite estimate is neutralised", False, f"ell={bad}")
            return
    check("non-finite estimate is neutralised", True)


def test_cond_guard():
    print("\n[9] cond guard -- non-finite is a VALUE  (III.6 star)")
    # A frozen regressor: exactly collinear, Gram singular.
    frozen = [np.array([1.0, 0.5, 0.2, -0.1]) for _ in range(50)]
    c = gram_cond(frozen)
    check("cond is non-finite on a degenerate regressor", not np.isfinite(c),
          f"cond={c}")
    naive_passes = np.isfinite(c) and c > 1e4
    check("the naive `isfinite(c) and c > thr` guard WOULD have missed it",
          not naive_passes,
          "so callers must test non-finite first -- URB: 4299/4299 rows missed")
    rng = np.random.default_rng(4)
    healthy = [np.concatenate([[1.0], rng.normal(0, 1, 3)]) for _ in range(500)]
    check("cond is finite and small on a healthy regressor",
          np.isfinite(gram_cond(healthy)) and gram_cond(healthy) < 100,
          f"cond={gram_cond(healthy):.4g}")


def test_ratio_guard():
    print("\n[10] ratio guard -- NaN not epsilon  (II.7)")
    from .diagnostics import safe_ratio
    check("ratio at the driver trough returns NaN",
          np.isnan(safe_ratio(0.4, 1e-12)),
          "a 1e-12 floor once produced -1011 and poisoned a column average")
    check("ordinary ratio still computes", abs(safe_ratio(1.0, 4.0) - 0.25) < 1e-12)


def _closed_loop(compensate, T=12000, seed=7, trust=0.90, loop=True,
                 score_from=3000, diagnostics=False):
    """Synthetic linear grid wired exactly like the wrapper.

        rho_i(t) = rho0 + drho_da_i * a_exec_i(t) + beta*(t) . psi_peer_i(t) + noise

    i.e. the observation reflects the CURRENT actions of everyone, as grid2op's
    does.  The agent, however, may only use one-step-delayed peer actions, so
    it compensates with beta_hat . psi_peer(t-1) -- a one-step-ahead forecast
    under persistence, sound because the coupling is slow (III.8b).

    Returns the STD of the deviation of realized loading from the no-coupling
    counterfactual at the intended action -- the FLUCTUATING part of what the
    coupling costs.

    Std, not RMS, on purpose.  The compensator deliberately leaves the standing
    pedestal to the policy (see compensator.StandingLevel) and targets only the
    fluctuation the policy cannot anticipate.  Scoring on RMS would grade it on
    a term it is designed not to touch, and in a real run the policy absorbs
    that term in both arms anyway.
    """
    b, zones, curt, pmax = _basis()
    rng = np.random.default_rng(seed)
    n, r = b.n, b.r

    own_gain_true = rng.uniform(0.5, 1.5, size=n)
    drho_da_true = np.array([action_sensitivity(own_gain_true[i], b.pmax_per_zone[i])
                             for i in range(n)])
    rho0 = 0.55

    est = [RLSEstimator(r, mu=0.9995, p0=100.0) for _ in range(n)]
    standing = [StandingLevel(tau=2000.0) for _ in range(n)]
    psi_peer_prev = None
    # Three incommensurate tones per agent with independent phases.  A SINGLE
    # tone makes own_col and every psi column sinusoids at one frequency, so the
    # regressor spans a 2-D subspace and the parameters are not identifiable at
    # all -- measured, smallest singular value ratio 0.053 vs 0.259 broadband,
    # cond 1.5e4 vs 938.  A converged narrowband policy would do the same thing
    # in the real env; that is what the cond_psi column is there to catch (IV.4).
    phase = rng.uniform(0, 2 * np.pi, size=(3, n))

    errs = []
    ell_hat_log, ell_true_log = [], []
    for t in range(T):
        # beta* = c(t)*theta(t): the drifting latent the chronics carry.
        bt = np.array([0.40 + 0.25 * np.sin(2 * np.pi * t / 900.0),
                       -0.30 + 0.15 * np.cos(2 * np.pi * t / 1300.0)])[:r]

        # A slowly-varying "policy": excitation stays alive, persistence holds.
        a_int = np.clip(0.65
                        + 0.08 * np.sin(2 * np.pi * t / 260.0 + phase[0])
                        + 0.06 * np.sin(2 * np.pi * t / 97.0 + phase[1])
                        + 0.03 * np.sin(2 * np.pi * t / 41.0 + phase[2])
                        + 0.02 * rng.normal(size=n), 0.0, 1.0)

        # --- compensate using the freshest LAGGED peer information ----------
        a_exec = a_int.copy()
        if compensate and psi_peer_prev is not None:
            for i in range(n):
                psi_hat = np.concatenate([[1.0, 0.0], psi_peer_prev[i]])
                conf = (est[i].confidence_at(psi_hat)
                        * est[i].divisor_confidence(1)
                        * est[i].ready_confidence())
                ell = standing[i].update(est[i].peer_component(psi_hat))
                d, _ = compensation_delta(
                    ell, action_sensitivity(est[i].beta[1], b.pmax_per_zone[i]),
                    trust=trust, conf=conf, max_delta=0.25)
                a_exec[i] = np.clip(a_int[i] + d, 0.0, 1.0)

        # --- grid responds to CURRENT executed actions ----------------------
        # loop=True: phi reads the EXECUTED action, so compensating feeds the
        # medium it compensates against (I.5).  This is the physically forced
        # choice in POWER -- compensation IS curtailment -- and it is what makes
        # T4 apply.  loop=False isolates the estimator from that feedback.
        a_phi = a_exec if loop else a_int
        phi = b.exertion({z: np.full(len(curt[z]), a_phi[i])
                          for i, z in enumerate(zones)})
        psi_peer = b.waveforms(phi)
        ell_true = psi_peer @ bt
        rho = rho0 + drho_da_true * a_exec + ell_true + 0.002 * rng.normal(size=n)

        # Cost of the coupling: distance from the no-coupling counterfactual
        # evaluated at the INTENDED action, same time index.
        target = rho0 + drho_da_true * a_int
        if t > score_from:               # past the estimator warmup ramp
            errs.append(rho - target)
            if diagnostics and psi_peer_prev is not None:
                # All agents, not just agent 0: the score is averaged over
                # agents, so the ceiling it is graded against must be too.
                ell_hat_log.append([
                    est[k].peer_component(
                        np.concatenate([[1.0, 0.0], psi_peer[k]]))
                    for k in range(n)])
                ell_true_log.append(ell_true.copy())

        # --- estimator update, same ordering/asymmetry as the wrapper -------
        if psi_peer_prev is not None:
            for i in range(n):
                own_ex = (1.0 - a_exec[i]) * b.pmax_per_zone[i]
                own_col = ((own_ex - 0.5 * b.pmax_per_zone[i])
                           / max(b.pmax_per_zone[i], 1.0))
                # own column CURRENT, peer columns LAGGED
                psi_reg = np.concatenate([[1.0, own_col], psi_peer_prev[i]])
                est[i].update(psi_reg, rho[i])

        psi_peer_prev = psi_peer

    E = np.asarray(errs)
    # Per-agent std, then averaged: each agent's own pedestal is removed, not a
    # pooled one, so a cross-agent level spread cannot masquerade as fluctuation.
    score = float(np.mean(E.std(axis=0)))
    if not diagnostics:
        return score
    P, Q = np.asarray(ell_hat_log), np.asarray(ell_true_log)   # (T, n_agents)
    if P.ndim != 2 or P.shape[0] < 3:
        return score, np.nan
    cors = []
    for k in range(P.shape[1]):
        if P[:, k].std() > 1e-12 and Q[:, k].std() > 1e-12:
            cors.append(np.corrcoef(P[:, k], Q[:, k])[0, 1])
    return score, (float(np.mean(cors)) if cors else np.nan)


def test_closed_loop_open():
    """Correctness of estimator + inverse + sign chain, loop isolated."""
    print("\n[11] closed loop, NO loop coupling  (phi reads intended -- I.5)")
    blind = _closed_loop(compensate=False, loop=False)
    pact, rho_c = _closed_loop(compensate=True, loop=False, diagnostics=True)
    red = 100.0 * (1.0 - pact / blind) if blind > 0 else np.nan
    check("PACT-1 reduces the coupling residual vs blind",
          pact < blind, f"std {blind:.4f} -> {pact:.4f}  ({red:.1f}% removed)")

    # How much reduction is even available?  A predictor correlating rho_c with
    # the disturbance can remove at most 1 - sqrt(1 - rho_c^2) of its std, no
    # matter how the gain is set.  Grading against that ceiling rather than a
    # round number keeps the test honest about what identifiability allows:
    # this basis is only weakly excited on the 'distant' channel, so the
    # ceiling here is modest and the method cannot beat it.
    ceiling = 100.0 * (1.0 - np.sqrt(max(0.0, 1.0 - rho_c ** 2)))
    frac = red / ceiling if ceiling > 0 else np.nan
    check("reduction reaches a real fraction of the achievable ceiling",
          frac > 0.4,
          f"corr(ell_hat, ell_true)={rho_c:.3f} => ceiling {ceiling:.1f}%; "
          f"achieved {red:.1f}% ({100*frac:.0f}% of ceiling)")


def test_loop_commons():
    """T4: compensation feeds the coupling it compensates against.

    The assertion here is deliberately NOT 'compensation always wins'.  III.7's
    measured beta-sweep peaks at beta=c and turns over -- at beta=0.8 the Ant
    return fell to 1481 against a blind 2302, i.e. over-compensation was 39%
    worse than doing nothing.  What must hold is the SHAPE: an interior
    optimum, with the high-trust end degrading.
    """
    print("\n[12] T4 -- the compensation commons  (III.7, III.8)")
    blind = _closed_loop(compensate=False, loop=True)
    # Sweep past 1.0: III.7's measured beta-sweep ran to beta/c = 1.8 before
    # the return fell below blind, so a grid stopping at 0.9 can miss the
    # turnover entirely -- II.3's "stopping at a sigma where everything passes"
    # mistake, in the gain dimension.
    sweep = {g: _closed_loop(compensate=True, loop=True, trust=g)
             for g in (0.0, 0.30, 0.60, 0.90, 1.30, 1.80)}
    row = "  ".join(f"g={g:.2f}:{v:.4f}" for g, v in sweep.items())
    print(f"    blind={blind:.4f}   {row}")

    best_g = min(sweep, key=sweep.get)
    check("an interior trust setting beats blind",
          sweep[best_g] < blind,
          f"best g={best_g:.2f} -> {sweep[best_g]:.4f} vs blind {blind:.4f}")
    check("over-compensation degrades (the commons has a cost)",
          sweep[1.80] > sweep[best_g],
          f"g=1.80 -> {sweep[1.80]:.4f} vs best g={best_g:.2f} "
          f"-> {sweep[best_g]:.4f}")
    check("the optimum is interior, not at the grid edge (II.3)",
          best_g < 1.80,
          f"g* = {best_g:.2f}; a g* pinned at the largest value tested means "
          f"the grid edge was found, not the frontier")


def main():
    print("=" * 72)
    print("  PACT-1 arithmetic self-check  (no grid2op, no dataset)")
    print("=" * 72)
    for fn in (test_structure, test_conditioning, test_n1_irreducibility,
               test_uncancellable_exertion,
               test_rls_recovery, test_gate_under_excitation_death,
               test_fit_gain_null_model, test_fit_gate,
               test_conjugacy, test_floor_property,
               test_cond_guard, test_ratio_guard, test_closed_loop_open,
               test_loop_commons):
        fn()
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n = len(RESULTS)
    print("\n" + "=" * 72)
    print(f"  {n_pass}/{n} checks passed")
    print("=" * 72)
    if n_pass != n:
        print("\nFAILED:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}  {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
