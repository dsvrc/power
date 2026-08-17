"""PACT-1 on the l2rpn_idf_2023 redispatching / curtailment task.

The host RL algorithm is untouched (III.4).  Everything lives here, in an
environment wrapper, so every arm shares host hyperparameters and an arm
difference cannot be an algorithm difference.

Per-step mechanism, one agent at a time:

    peers' EXECUTED curtailment (t-1) --> phi --> psi_1..psi_r   exact arithmetic
    own rho (t)                        --> y                      the sensor
    RLS([1, own(t), psi_peer(t-1)], y) --> beta_hat               tracks c(t)*theta(t)
    ell_hat  = beta_hat[peer part] . psi_peer,  minus its standing level
    g        = trust * conf_pred * conf_divisor * conf_ready
    u        = clip(a - g * ell_hat / (d rho / d a))              the channel inverse

The gate is a product of three, not III.5's single prediction term.  The inverse
is a RATIO: conf_pred covers the numerator, conf_divisor covers the learned
denominator (a prediction can be fine while the own-gain coefficient is still
wrong-signed against a collinear peer channel), and conf_ready covers sample
count.  Measured justification is in rls.py.

Loop coupling (I.5).  Compensation here is itself curtailment -- the same
actuator that generates the exertion -- so phi reads the EXECUTED action, not
the intended one.  That is not a modelling choice we are free to make:
"pushing back is pushing".  POWER is therefore loop-coupled like Ant and unlike
SMAC, which is what makes T4 (the compensation commons) and the Pigouvian CTDE
prediction of III.10 testable in this environment.

What is NOT claimed here: this is a natural environment, so the driver is the
chronics rather than an injected A(t)*sigma, and the N=1 projection is not
byte-identical to a stationary task (weather still stresses a lone operator's
grid).  What IS category-C is the coupling component -- peer injections
reaching agent i only through the shared network -- and that does vanish at
N=1.  See IV.5 for the honest headline this supports.
"""
import os

import numpy as np

from ..PZMAEnvWithHeuristics import PZMAEnvRecoDNLimit
from .basis import CouplingBasis, gram_cond, ptdf_zone_coupling
from .compensator import (StandingLevel, action_sensitivity, apply_inverse,
                          compensation_delta)
from .rls import RLSEstimator
from . import diagnostics as diag


class PACT1Env(PZMAEnvRecoDNLimit):
    """PZMAEnvRecoDNLimit + the PACT-1 estimator/compensator loop."""

    def __init__(self,
                 pact1_enabled=True,
                 pact1_trust=0.90,          # III.5: INVERT the prior. 0.5 cost ~1800 return.
                 pact1_mu=0.9995,           # RLS forgetting; see rls.py for the sweep
                 pact1_p0=100.0,
                 pact1_max_delta=0.25,      # cap on |compensation| in action units
                 pact1_own_gain_floor=1e-3,
                 pact1_gate="prediction",   # "prediction" | "trace" (ablation) | "none"
                 pact1_hp_tau=2000.0,       # standing-level EMA, in env steps
                 pact1_sensor="max",        # "mean" | "max" over the zone's own lines
                 pact1_r=2,                 # peer channels, split by PTDF strength
                 pact1_log=None,
                 pact1_log_every=200,
                 pact1_matched_band=(0.69, 0.70),
                 **kwargs):
        super().__init__(**kwargs)

        self.pact1_enabled = bool(pact1_enabled)
        self.trust_init = float(pact1_trust)
        self.max_delta = float(pact1_max_delta)
        self.own_gain_floor = float(pact1_own_gain_floor)
        self.gate_kind = str(pact1_gate)
        self.sensor_kind = str(pact1_sensor)
        self.log_every = int(pact1_log_every)

        # Zone agents only.  The global redispatching_agent acts on every gen
        # at once, so its effect is common to all zones and lands in the
        # intercept rather than in a peer channel.
        self._zone_agents = [a for a in self.possible_agents
                             if a != "redispatching_agent"]
        self._zone_of = {a: self._map_agent_id_to_zone(a) for a in self._zone_agents}
        zone_names = [self._zone_of[a] for a in self._zone_agents]

        env = self.env_g2op
        # Recomputed exactly as utils.get_obs_act_attr_and_kwargs does, so the
        # action layout here cannot drift from the real action space.
        from ..utils import ZONES_DICT
        gen_curtail_inside, line_in_zone = {}, {}
        for z in zone_names:
            gi = np.asarray(ZONES_DICT[z]["gen_inside_idx"], dtype=int)
            gen_curtail_inside[z] = np.intersect1d(gi, np.where(env.gen_renewable)[0])
            line_in_zone[z] = np.asarray(ZONES_DICT[z]["line_in_zone_idx"], dtype=int)
        self._n_curtail = {z: len(gen_curtail_inside[z]) for z in zone_names}
        self._line_in_zone = line_in_zone

        # The coupling operator comes from the grid model, computed once, never
        # fitted from run data (IV.1).  A geometric stand-in measured
        # fit_gain = -0.0045 here, so failing to get a PTDF is fatal rather
        # than silently degrading to the version that does not work.
        W = ptdf_zone_coupling(env, zone_names)
        if W is None:
            raise RuntimeError(
                "PACT-1 needs injection->flow sensitivities and this backend "
                "exposes none. Run probe_ptdf.py to see which routes exist.")
        self.basis = CouplingBasis(zone_names, gen_curtail_inside, env.gen_pmax,
                                   W=W, r_target=pact1_r)
        self.n_zones = self.basis.n

        self.est = {a: RLSEstimator(self.basis.r, mu=pact1_mu, p0=pact1_p0)
                    for a in self._zone_agents}
        # The policy owns the standing level; PACT-1 owns the fluctuation.
        self.standing = {a: StandingLevel(tau=pact1_hp_tau)
                         for a in self._zone_agents}

        # Driver: total load from the chronics.  Purely exogenous -- no agent
        # action changes it -- unlike mean rho, which the agents themselves move.
        self._load_lo, self._load_hi = np.inf, -np.inf

        self._prev_psi = {a: None for a in self._zone_agents}
        self._psi_peer_prev = None
        self._exec_curtail = {z: None for z in zone_names}
        self._ell_hat = {a: 0.0 for a in self._zone_agents}
        self._applied_trust = {a: 0.0 for a in self._zone_agents}
        self._last_conf = {a: 0.0 for a in self._zone_agents}
        self._sat_hits = 0
        self._sat_total = 0
        self._ell_max_seen = 0.0
        self._A_window = []
        self._psi_hist = []
        self._phi_hist = []
        self._own_col_hist = []
        self._g2op_steps = []
        self._step = 0
        self._episode = 0
        self._matched = diag.MatchedDriverTracker(*pact1_matched_band)

        self.logger = None
        if pact1_log and self.pact1_enabled:
            self.logger = diag.PactLogger(
                pact1_log,
                extra_note=(f"basis r={self.basis.r} {self.basis.kept_names}; "
                            f"trust_init={self.trust_init}; gate={self.gate_kind}; "
                            f"mu={pact1_mu}; loop_coupled=True (phi reads executed)"))

        print(diag.banner([
            f"enabled          : {self.pact1_enabled}",
            f"zone agents      : {len(self._zone_agents)}",
            f"trust init       : {self.trust_init}   (III.5 inverted prior)",
            f"confidence gate  : {self.gate_kind}",
            f"own-harm sensor  : {self.sensor_kind} rho over the zone's own lines",
            f"RLS mu / p0      : {pact1_mu} / {pact1_p0}",
            f"max |delta|      : {self.max_delta} action units",
            f"log              : {pact1_log or '(off)'}",
        ]) + "\n" + self.basis.report())

    # ------------------------------------------------------------------
    # driver
    def _driver_level(self, g2op_obs):
        """A(t) in [0,1] from total load.  Running min/max, so the first
        episode is warmup and the matched band is only meaningful after it."""
        tot = float(np.sum(g2op_obs.load_p))
        self._load_lo = min(self._load_lo, tot)
        self._load_hi = max(self._load_hi, tot)
        span = self._load_hi - self._load_lo
        if span < 1e-6:
            return 0.0
        return float(np.clip((tot - self._load_lo) / span, 0.0, 1.0))

    def _sensor(self, g2op_obs, zone):
        """Agent's own realized loading -- the proprioceptive measurement.

        rho is already in the observation (IV.1), so this is not privileged
        information: every operator measures the loading of its own lines.
        It reports t, while compensating t+1 needs the next value.  That gap is
        exactly where the unknown lives and it is preserved (I.8).

        Congestion is a property of the BINDING line, so "max" is the
        physically meaningful aggregate; "mean" over 23-35 lines dilutes any
        peer effect toward zero, which is a candidate explanation for a flat
        fit_gain.  Both are logged every row (rho_own_mean / rho_own_max) so
        the choice can be made on measured variance rather than argument.
        """
        idx = self._line_in_zone[zone]
        if len(idx) == 0:
            return 0.0
        vals = g2op_obs.rho[idx]
        return float(np.max(vals) if self.sensor_kind == "max" else np.mean(vals))

    # ------------------------------------------------------------------
    def _split_action(self, agent, act):
        """Curtailment head of the agent's flat action vector."""
        z = self._zone_of[agent]
        n = self._n_curtail[z]
        act = np.asarray(act, dtype=np.float64).ravel()
        return act[:n], act[n:]

    def _sensitivity(self, agent_idx, own_gain):
        """d(rho_i)/d(a_i).  Derivation lives in compensator.action_sensitivity
        so the wrapper and the self-check cannot drift apart on the sign."""
        return action_sensitivity(own_gain, self.basis.pmax_per_zone[agent_idx])

    def _compensate(self, action_dict):
        """Apply the certified channel inverse.  Returns the executed dict.

        Floor property (III.4): with g = 0 the executed action is exactly the
        blind action, whatever beta_hat says.  The estimator sits outside the
        worst-case control path, so a diverging estimate can fail to help but
        cannot do worse than the baseline.
        """
        if not self.pact1_enabled:
            return action_dict, {}

        out = dict(action_dict)
        info = {}
        for a_i, agent in enumerate(self._zone_agents):
            if agent not in action_dict:
                continue
            z = self._zone_of[agent]
            curt, rest = self._split_action(agent, action_dict[agent])
            if len(curt) == 0:
                continue

            psi = self._prev_psi[agent]
            g = 0.0
            delta = 0.0
            if psi is not None:
                est = self.est[agent]
                if self.gate_kind == "prediction":
                    # Numerator gate (III.5) x divisor gate: the inverse is a
                    # ratio, and each half needs its own confidence.
                    conf = (est.confidence_at(psi)
                            * est.divisor_confidence(1)
                            * est.ready_confidence())
                elif self.gate_kind == "prediction_only":
                    conf = est.confidence_at(psi)      # ablation: no divisor gate
                elif self.gate_kind == "trace":
                    conf = est.trace_confidence()      # ablation, the URB bug
                else:
                    conf = 1.0
                self._last_conf[agent] = conf

                # Cancel the FLUCTUATION, not the pedestal: the standing offset
                # between the geometric reference and the policy's operating
                # point is the policy's to absorb, and chasing it saturates the
                # bounded actuator on a constant.  See compensator.StandingLevel.
                ell = self.standing[agent].update(est.peer_component(psi))
                self._ell_hat[agent] = ell

                delta, g = compensation_delta(
                    ell, self._sensitivity(a_i, est.beta[1]),
                    self.trust_init, conf,
                    max_delta=self.max_delta,
                    sensitivity_floor=self.own_gain_floor)
            self._applied_trust[agent] = g

            new_curt, n_sat = apply_inverse(curt, delta)
            self._sat_hits += n_sat
            self._sat_total += int(new_curt.size)
            out[agent] = np.concatenate([new_curt, rest]).astype(
                np.asarray(action_dict[agent]).dtype, copy=False)
            info[agent] = (delta, g)
        return out, info

    # ------------------------------------------------------------------
    def _update_estimators(self, g2op_obs, exec_curtail):
        """RLS on the residual, then stage psi(t) for the next step's inverse.

        The regressor is deliberately ASYMMETRIC in time, and getting this
        wrong makes compensation actively harmful:

          own column  -- CURRENT executed action.  grid2op returns the
              observation *after* applying the action, so the measured rho
              already reflects it, and the agent knows its own action exactly.
          peer columns -- PREVIOUS step.  An agent cannot observe its peers'
              current actions; one-step-delayed peer actions are all III.1
              allows, and using them for the *current* rho is a persistence
              assumption, sound because the coupling is slow (III.8b).

        Regressing rho(t) on the whole of psi(t-1) instead lags the own-gain
        column by one step, which corrupts the very coefficient the channel
        inverse divides by.
        """
        phi = self.basis.exertion(exec_curtail)
        psi_peer_now = self.basis.waveforms(phi)      # (n_zones, r), centred

        A = self._driver_level(g2op_obs)
        self._A_window.append(A)

        ell_norms = []
        for a_i, agent in enumerate(self._zone_agents):
            z = self._zone_of[agent]
            y = self._sensor(g2op_obs, z)

            cur = exec_curtail.get(z)
            own_ex = 0.0
            if cur is not None and len(cur):
                own_ex = float(np.mean(1.0 - np.clip(cur, 0.0, 1.0))) \
                    * self.basis.pmax_per_zone[a_i]
            own_col = (own_ex - 0.5 * self.basis.pmax_per_zone[a_i]) / \
                      max(self.basis.pmax_per_zone[a_i], 1.0)

            if self._psi_peer_prev is not None:
                psi_reg = np.concatenate(
                    [[1.0, own_col], self._psi_peer_prev[a_i]])
                self.est[agent].update(psi_reg, y)

            # Staged for the next step's inverse: the freshest peer information
            # that will be available when the next action is chosen.
            psi_next = np.concatenate([[1.0, own_col], psi_peer_now[a_i]])
            self._prev_psi[agent] = psi_next
            ell_norms.append(abs(self.est[agent].peer_component(psi_next)))

            if a_i == 0:
                self._psi_hist.append(psi_next.copy())
                if len(self._psi_hist) > 4000:
                    self._psi_hist.pop(0)
            self._own_col_hist.append(own_col)

        self._psi_peer_prev = psi_peer_now
        self._phi_hist.append(phi.copy())
        if len(self._phi_hist) > 2000:
            self._phi_hist.pop(0)
        if len(self._own_col_hist) > 4000:
            del self._own_col_hist[:-4000]

        ell_arr = np.asarray(ell_norms)
        self._ell_max_seen = max(self._ell_max_seen, float(ell_arr.max(initial=0.0)))
        self._matched.observe(A, float(np.linalg.norm(ell_arr)))
        return ell_arr, A

    def _log(self, ell_arr, A, g2op_obs):
        if self.logger is None or self._step % self.log_every:
            return
        Aw = np.asarray(self._A_window) if self._A_window else np.array([0.0])
        P = np.asarray(self._phi_hist) if self._phi_hist else np.zeros((0, 0))
        own_rhos = [self._sensor(g2op_obs, self._zone_of[a])
                    for a in self._zone_agents]
        trusts = np.asarray([self._applied_trust[a] for a in self._zone_agents])
        confs = np.asarray([self._last_conf[a] for a in self._zone_agents])
        gains = [self.est[a].fit_gain() for a in self._zone_agents]
        self.logger.write({
            "step": self._step,
            "episode": self._episode,
            "applied_trust": float(trusts.mean()),
            "trust_pol": self.trust_init,
            "conf": float(confs.mean()),
            "conf_trace": float(np.mean([self.est[a].trace_confidence()
                                         for a in self._zone_agents])),
            "state": diag.classify(self._ell_max_seen,
                                   float(Aw.max() - Aw.min()),
                                   float(ell_arr.max(initial=0.0))),
            "ell_mean": float(ell_arr.mean()),
            "ell_max": float(ell_arr.max(initial=0.0)),
            "ell_matched": self._matched.value,
            "matched_n": self._matched.count,
            "cancel": diag.safe_ratio(float(ell_arr.mean()), self._ell_max_seen),
            "beta_err_proxy": float(np.mean(
                [np.linalg.norm(self.est[a].beta[2:]) for a in self._zone_agents])),
            "fit_gain": float(np.nanmean(gains)) if len(gains) else np.nan,
            "cond_psi": gram_cond(self._psi_hist),
            "sat_frac": diag.safe_ratio(self._sat_hits, self._sat_total),
            "A_mean": float(Aw.mean()), "A_min": float(Aw.min()), "A_max": float(Aw.max()),
            "rho_mean": float(np.mean(g2op_obs.rho)),
            "rho_max": float(np.max(g2op_obs.rho)),
            "phi_mean": float(np.mean(P)) if P.size else np.nan,
            "phi_std": float(np.mean(P.std(axis=0))) if P.shape[0] > 1 else np.nan,
            "phi_frac_active": float(np.mean(P > 1e-6)) if P.size else np.nan,
            "own_col_std": (float(np.std(self._own_col_hist[-2000:]))
                            if len(self._own_col_hist) > 1 else np.nan),
            "rho_own_mean": float(np.mean(own_rhos)) if own_rhos else np.nan,
            "rho_own_max": float(np.max(own_rhos)) if own_rhos else np.nan,
            "g2op_per_gym": (float(np.mean(self._g2op_steps))
                             if self._g2op_steps else np.nan),
            "n_eff_r": self.basis.r,
        })
        self._A_window = []

    # ------------------------------------------------------------------
    def step(self, gym_action):
        # The heuristic base class plays do-nothing while max rho <= safe_max_rho
        # and only surfaces to the agent when the grid is stressed, so ONE gym
        # step can span many grid2op steps.  That matters: PACT-1 compensates
        # with one-gym-step-delayed peer actions under a persistence assumption
        # (III.8b), and persistence over 30 simulated minutes is a very
        # different claim from persistence over 5.  If this ratio is large and
        # variable, the lag -- not the basis -- is what is killing fit_gain.
        t0 = getattr(self.env_g2op, "nb_time_step", 0)
        exec_action, _ = self._compensate(gym_action)

        for agent in self._zone_agents:
            if agent in exec_action:
                curt, _ = self._split_action(agent, exec_action[agent])
                # phi reads the EXECUTED action -- the loop, I.5.
                self._exec_curtail[self._zone_of[agent]] = curt

        obs, rew, done, trunc, info = super().step(exec_action)

        self._step += 1
        t1 = getattr(self.env_g2op, "nb_time_step", 0)
        if t1 >= t0:
            self._g2op_steps.append(int(t1 - t0))
            if len(self._g2op_steps) > 2000:
                self._g2op_steps.pop(0)
        if self.pact1_enabled:
            g2op_obs = self._previous_act        # the post-step grid2op obs
            ell_arr, A = self._update_estimators(
                g2op_obs, {z: v for z, v in self._exec_curtail.items()})
            self._log(ell_arr, A, g2op_obs)
        return obs, rew, done, trunc, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._episode += 1
        # Estimator state persists across episodes on purpose: beta* = c*theta
        # drifts with the chronics, and the point is to TRACK it, not to relearn
        # it from cold every episode.  Only the per-episode action memory resets.
        self._prev_psi = {a: None for a in self._zone_agents}
        self._psi_peer_prev = None
        self._exec_curtail = {z: None for z in self._exec_curtail}
        return obs, info
