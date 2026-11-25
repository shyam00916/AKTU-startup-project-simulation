"""Contextual bandit simulation for collection action optimization.

IMPORTANT EPISTEMIC BOUNDARY:
All outcomes, propensities, and policy evaluations in this module come from a
synthetic collection simulator and off-policy estimators (IPS, SNIPS, Doubly Robust).
Simulated policy lift MUST NOT be interpreted as real-world causal effectiveness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from receivables_ml.config import BanditConfig

ACTIONS: tuple[str, ...] = (
    "whatsapp_reminder",
    "email_reminder",
    "phone_call",
    "escalation",
)

ACTION_COSTS: dict[str, float] = {
    "whatsapp_reminder": 0.40,
    "email_reminder": 0.15,
    "phone_call": 6.50,
    "escalation": 28.00,
}

EPISTEMIC_DISCLAIMER: str = (
    "NON-CAUSAL SIMULATION NOTICE: Payment probabilities and expected collection rewards "
    "are estimated on synthetic historical episodes using off-policy evaluation (IPS, SNIPS, "
    "Doubly Robust) and a synthetic response model. They demonstrate algorithmic policy "
    "optimization mechanics and DO NOT prove real-world causal effectiveness. Randomized "
    "field experiments (A/B testing) are required prior to autonomous production rollout."
)


@dataclass
class BanditEvaluationReport:
    """Complete outputs from contextual bandit simulation and off-policy evaluation."""

    disclaimer: str
    n_train_episodes: int
    n_eval_episodes: int
    action_costs: dict[str, float]
    policy_comparison: list[dict[str, Any]]
    action_probability_by_segment: list[dict[str, Any]]
    recommended_action_distribution: dict[str, dict[str, float]]
    linucb_learning_curve: list[dict[str, float]]


class SyntheticCollectionEnvironment:
    """Synthetic response simulator for B2B collection actions."""

    def __init__(self, goodwill_penalty_weight: float = 1.0, seed: int = 42) -> None:
        self.goodwill_penalty_weight = goodwill_penalty_weight
        self.rng = np.random.default_rng(seed)

    def build_context_matrix(self, customer_meta: pd.DataFrame) -> np.ndarray:
        """Build standardized raw context matrix for bandit policies."""
        amt = customer_meta["invoice_amount"].to_numpy(dtype=np.float64)
        util = customer_meta["balance_utilization"].to_numpy(dtype=np.float64)
        delay_mean = customer_meta["hist_delay_mean"].to_numpy(dtype=np.float64)
        delay_slope = customer_meta["hist_delay_slope"].to_numpy(dtype=np.float64)
        partial_rate = customer_meta["hist_partial_rate"].to_numpy(dtype=np.float64)
        tx_int = customer_meta["hist_tx_interval_mean"].to_numpy(dtype=np.float64)

        return np.column_stack(
            [
                np.log1p(amt),
                util,
                delay_mean / 15.0,
                delay_slope / 3.0,
                partial_rate,
                tx_int / 30.0,
                np.minimum(amt / 50000.0, 2.0),
                (delay_mean > 7.0).astype(np.float64),
                (delay_slope > 0.8).astype(np.float64),
            ]
        )

    def true_payment_probabilities(self, customer_meta: pd.DataFrame) -> np.ndarray:
        """Compute synthetic ground-truth P(pay_within_window | x, a) for all 4 actions."""
        n = len(customer_meta)
        probs = np.zeros((n, len(ACTIONS)), dtype=np.float64)

        log_amt = np.log1p(customer_meta["invoice_amount"].to_numpy(dtype=np.float64))
        util = customer_meta["balance_utilization"].to_numpy(dtype=np.float64)
        delay_mean = customer_meta["hist_delay_mean"].to_numpy(dtype=np.float64)
        delay_slope = customer_meta["hist_delay_slope"].to_numpy(dtype=np.float64)
        partial_rate = customer_meta["hist_partial_rate"].to_numpy(dtype=np.float64)
        arch = customer_meta["true_archetype"].to_numpy()

        # Baseline propensity to resolve promptly without aggressive friction
        base_logit = (
            1.15
            - 0.055 * np.maximum(delay_mean, -5.0)
            - 0.55 * util
            - 0.35 * np.maximum(delay_slope, 0.0)
            - 0.45 * partial_rate
        )

        is_irregular = (arch == "irregular_payer").astype(np.float64)
        is_high_val = (arch == "high_value_delayed_payer").astype(np.float64)
        is_deteriorating = (arch == "deteriorating_customer").astype(np.float64)
        is_early = (arch == "consistent_early_payer").astype(np.float64)

        # Action 0: whatsapp_reminder (effective on irregular & moderate-ticket digital accounts)
        logit_wa = (
            base_logit
            + 0.48
            + 0.85 * is_irregular
            + 0.25 * (log_amt < 9.6).astype(np.float64)
            - 0.20 * (delay_mean > 22.0).astype(np.float64)
        )
        # Action 1: email_reminder (baseline low-cost nudge, works well for early/mild delays)
        logit_em = (
            base_logit
            + 0.28
            + 0.40 * is_early
            - 0.25 * is_deteriorating
            - 0.22 * (delay_mean > 14.0).astype(np.float64)
        )
        # Action 2: phone_call (high efficacy for high-value delayed & deteriorating accounts)
        logit_ph = (
            base_logit
            + 0.65
            + 0.90 * is_high_val
            + 0.80 * is_deteriorating
            + 0.45 * (log_amt >= 9.6).astype(np.float64)
            + 0.35 * (partial_rate > 0.25).astype(np.float64)
        )
        # Action 3: escalation (highest recovery pressure on severe chronic delinquency, backfires slightly on mild accounts)
        logit_es = (
            base_logit
            + 0.55
            + 0.95 * (delay_mean > 16.0).astype(np.float64)
            + 0.55 * is_deteriorating
            - 0.45 * is_early
        )

        logits = np.column_stack([logit_wa, logit_em, logit_ph, logit_es])
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -8.0, 8.0)))
        return probs

    def expected_rewards(self, customer_meta: pd.DataFrame, true_probs: np.ndarray) -> np.ndarray:
        """Compute synthetic expected net reward E[R | x, a] in normalized value units ($ scaled)."""
        amt = customer_meta["invoice_amount"].to_numpy(dtype=np.float64)
        delay_mean = customer_meta["hist_delay_mean"].to_numpy(dtype=np.float64)
        arch = customer_meta["true_archetype"].to_numpy()
        is_early = (arch == "consistent_early_payer").astype(np.float64)

        # Value of accelerating collection: liquidity carrying value (~3.2% of invoice, capped)
        recovery_value = np.clip(0.032 * amt, 40.0, 1400.0)
        rewards = np.zeros_like(true_probs)

        for a_idx, action in enumerate(ACTIONS):
            cost = ACTION_COSTS[action]
            goodwill_pen = np.zeros(len(customer_meta), dtype=np.float64)
            if action == "escalation":
                # Escalating early/low-delay customers damages retention & goodwill
                goodwill_pen = self.goodwill_penalty_weight * (
                    45.0 * is_early + 24.0 * (delay_mean <= 6.0).astype(np.float64) + 0.008 * amt * (delay_mean <= 8.0)
                )
            elif action == "phone_call":
                goodwill_pen = self.goodwill_penalty_weight * (6.0 * is_early)

            rewards[:, a_idx] = true_probs[:, a_idx] * recovery_value - cost - goodwill_pen

        return rewards

    def logging_policy_probs(self, customer_meta: pd.DataFrame, epsilon: float = 0.15) -> np.ndarray:
        """Historical heuristic logging policy with guaranteed positivity for OPE."""
        amt = customer_meta["invoice_amount"].to_numpy(dtype=np.float64)
        delay_mean = customer_meta["hist_delay_mean"].to_numpy(dtype=np.float64)
        n = len(customer_meta)
        n_actions = len(ACTIONS)

        logits = np.zeros((n, n_actions), dtype=np.float64)
        # Heuristic scores biased toward email on low delay, phone on large invoices, escalation on high delay
        logits[:, 0] = 0.6 + 0.5 * ((delay_mean > 4) & (delay_mean <= 14))
        logits[:, 1] = 1.1 + 0.8 * (delay_mean <= 6)
        logits[:, 2] = 0.3 + 1.0 * (amt > 18000)
        logits[:, 3] = -0.3 + 1.3 * (delay_mean > 16)

        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_p = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        # Epsilon-mix ensures strict positivity (p >= epsilon / n_actions)
        return (1.0 - epsilon) * softmax_p + (epsilon / n_actions)

    def sample_logged_episodes(
        self, pool_meta: pd.DataFrame, n_episodes: int, epsilon: float = 0.15
    ) -> dict[str, Any]:
        """Sample logged historical episodes under the biased logging policy."""
        idx = self.rng.integers(0, len(pool_meta), size=n_episodes)
        ep_meta = pool_meta.iloc[idx].reset_index(drop=True)
        X = self.build_context_matrix(ep_meta)

        true_probs = self.true_payment_probabilities(ep_meta)
        true_exp_rewards = self.expected_rewards(ep_meta, true_probs)
        log_pi = self.logging_policy_probs(ep_meta, epsilon=epsilon)

        n_actions = len(ACTIONS)
        chosen_actions = np.array(
            [self.rng.choice(n_actions, p=log_pi[i]) for i in range(n_episodes)],
            dtype=np.int64,
        )
        propensities = log_pi[np.arange(n_episodes), chosen_actions]

        p_chosen = true_probs[np.arange(n_episodes), chosen_actions]
        paid = (self.rng.random(n_episodes) < p_chosen).astype(np.int64)

        amt = ep_meta["invoice_amount"].to_numpy(dtype=np.float64)
        recovery_value = np.clip(0.032 * amt, 40.0, 1400.0)
        costs = np.array([ACTION_COSTS[ACTIONS[a]] for a in chosen_actions], dtype=np.float64)
        # Realized reward matches true expected reward expectation plus outcome noise
        pen_term = true_probs[np.arange(n_episodes), chosen_actions] * recovery_value - costs - true_exp_rewards[
            np.arange(n_episodes), chosen_actions
        ]
        realized_rewards = paid * recovery_value - costs - pen_term + self.rng.normal(0.0, 2.5, size=n_episodes)

        return {
            "meta": ep_meta,
            "X": X,
            "actions": chosen_actions,
            "propensities": propensities,
            "paid": paid,
            "rewards": realized_rewards,
            "true_probs": true_probs,
            "true_exp_rewards": true_exp_rewards,
            "logging_probs": log_pi,
        }


def _evaluate_policy_ope(
    target_actions: np.ndarray,
    logged_actions: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    mu_hat: np.ndarray,
    true_probs: np.ndarray,
    true_exp_rewards: np.ndarray,
    is_logging_policy: bool = False,
) -> dict[str, float]:
    """Compute IPS, SNIPS, Doubly Robust OPE, and synthetic simulator oracle metrics."""
    n = len(logged_actions)
    if is_logging_policy:
        # Target policy == logging policy => importance weight pi_0(a|x) / pi_0(a|x) = 1.0
        w = np.ones(n, dtype=np.float64)
    else:
        match = (target_actions == logged_actions).astype(np.float64)
        w = match / np.clip(propensities, 1e-4, 1.0)

    ips = float(np.mean(w * rewards))
    snips = float(np.sum(w * rewards) / max(np.sum(w), 1e-8))

    mu_chosen = mu_hat[np.arange(n), logged_actions]
    mu_target = mu_hat[np.arange(n), target_actions]
    dr_scores = mu_target + w * (rewards - mu_chosen)
    dr_est = float(np.mean(dr_scores))
    dr_se = float(np.std(dr_scores, ddof=1) / np.sqrt(max(n, 1)))

    # Synthetic oracle metrics (strictly labeled as simulator-only)
    sim_oracle_reward = float(np.mean(true_exp_rewards[np.arange(n), target_actions]))
    sim_oracle_pay_prob = float(np.mean(true_probs[np.arange(n), target_actions]))

    return {
        "ope_ips_reward": round(ips, 3),
        "ope_snips_reward": round(snips, 3),
        "ope_doubly_robust_reward": round(dr_est, 3),
        "ope_dr_std_err": round(dr_se, 3),
        "sim_oracle_expected_reward": round(sim_oracle_reward, 3),
        "sim_oracle_payment_prob": round(sim_oracle_pay_prob, 4),
    }


def run_collection_bandit_simulation(
    customer_pool_meta: pd.DataFrame,
    config: BanditConfig,
    seed: int = 42,
) -> BanditEvaluationReport:
    """Run logged contextual bandit simulation, policy learning, and off-policy evaluation."""
    env = SyntheticCollectionEnvironment(
        goodwill_penalty_weight=config.goodwill_penalty_weight,
        seed=seed,
    )
    logged = env.sample_logged_episodes(
        pool_meta=customer_pool_meta,
        n_episodes=config.n_episodes,
        epsilon=config.exploration_epsilon,
    )

    n_total = len(logged["actions"])
    split_idx = int(0.65 * n_total)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(logged["X"][:split_idx])
    X_ev = scaler.transform(logged["X"][split_idx:])
    a_tr, a_ev = logged["actions"][:split_idx], logged["actions"][split_idx:]
    p_tr, p_ev = logged["propensities"][:split_idx], logged["propensities"][split_idx:]
    y_tr = logged["paid"][:split_idx]
    r_tr, r_ev = logged["rewards"][:split_idx], logged["rewards"][split_idx:]

    ev_meta = logged["meta"].iloc[split_idx:].reset_index(drop=True)
    true_probs_ev = logged["true_probs"][split_idx:]
    true_rew_ev = logged["true_exp_rewards"][split_idx:]
    log_pi_ev = logged["logging_probs"][split_idx:]

    n_actions = len(ACTIONS)

    # 1. Direct Method Outcome Models (Payment probability & Reward per action)
    prob_models: list[LogisticRegression] = []
    rew_models: list[Ridge] = []
    mu_hat_tr = np.zeros((split_idx, n_actions), dtype=np.float64)
    mu_hat_ev = np.zeros((n_total - split_idx, n_actions), dtype=np.float64)
    p_hat_ev = np.zeros((n_total - split_idx, n_actions), dtype=np.float64)

    for a_idx in range(n_actions):
        mask_a = a_tr == a_idx
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X_tr[mask_a], y_tr[mask_a])
        prob_models.append(clf)

        reg = Ridge(alpha=5.0, random_state=seed)
        reg.fit(X_tr[mask_a], r_tr[mask_a])
        rew_models.append(reg)

        mu_hat_tr[:, a_idx] = reg.predict(X_tr)
        mu_hat_ev[:, a_idx] = reg.predict(X_ev)
        p_hat_ev[:, a_idx] = clf.predict_proba(X_ev)[:, 1]

    dm_actions_ev = np.argmax(mu_hat_ev, axis=1)

    # 2. Disjoint LinUCB Contextual Bandit (sequential online updates on training episodes)
    d_ctx = X_tr.shape[1] + 1
    X_tr_b = np.hstack([X_tr, np.ones((split_idx, 1))])
    X_ev_b = np.hstack([X_ev, np.ones((len(X_ev), 1))])
    A_mats = [np.eye(d_ctx) * 2.0 for _ in range(n_actions)]
    b_vecs = [np.zeros(d_ctx) for _ in range(n_actions)]

    # Normalize rewards for stable LinUCB alpha scaling, then update sequentially on logged stream
    r_scale = max(float(np.std(r_tr)), 1.0)
    linucb_curve: list[dict[str, float]] = []
    cum_bandit_rew = 0.0
    cum_log_rew = 0.0
    matched_count = 0

    for i in range(split_idx):
        xi = X_tr_b[i]
        ucb_scores = np.zeros(n_actions)
        for a_idx in range(n_actions):
            A_inv = np.linalg.inv(A_mats[a_idx])
            theta = A_inv @ b_vecs[a_idx]
            ucb_scores[a_idx] = float(theta @ xi + config.linucb_alpha * np.sqrt(xi @ A_inv @ xi))

        pred_a = int(np.argmax(ucb_scores))
        obs_a = int(a_tr[i])
        obs_r = float(r_tr[i])

        # Replay / online logged update when bandit matches logged action
        if pred_a == obs_a:
            matched_count += 1
            cum_bandit_rew += obs_r
            A_mats[obs_a] += np.outer(xi, xi)
            b_vecs[obs_a] += (obs_r / r_scale) * xi
        else:
            # Also update arm with inverse-propensity damped observation so all arms learn reliably
            w_damp = min(1.0 / max(p_tr[i], 0.08), 4.0) * 0.25
            A_mats[obs_a] += w_damp * np.outer(xi, xi)
            b_vecs[obs_a] += w_damp * (obs_r / r_scale) * xi
        cum_log_rew += obs_r

        if (i + 1) % max(split_idx // 10, 1) == 0:
            linucb_curve.append(
                {
                    "episode": float(i + 1),
                    "matched_episodes": float(matched_count),
                    "mean_matched_reward": round(cum_bandit_rew / max(matched_count, 1), 2),
                    "mean_logged_reward": round(cum_log_rew / (i + 1), 2),
                }
            )

    linucb_scores_ev = np.zeros((len(X_ev_b), n_actions))
    for a_idx in range(n_actions):
        A_inv = np.linalg.inv(A_mats[a_idx])
        theta = A_inv @ b_vecs[a_idx]
        ucb_bonus = np.sqrt(np.sum((X_ev_b @ A_inv) * X_ev_b, axis=1))
        linucb_scores_ev[:, a_idx] = (X_ev_b @ theta) * r_scale + config.linucb_alpha * ucb_bonus * 10.0
    linucb_actions_ev = np.argmax(linucb_scores_ev, axis=1)

    # 3. Doubly Robust Policy Learner (fits per-action DR pseudo-reward models)
    dr_models: list[Ridge] = []
    dr_scores_ev = np.zeros((len(X_ev), n_actions), dtype=np.float64)
    for a_idx in range(n_actions):
        ind_a = (a_tr == a_idx).astype(np.float64)
        pseudo_r = mu_hat_tr[:, a_idx] + (ind_a / np.clip(p_tr, 0.04, 1.0)) * (r_tr - mu_hat_tr[:, a_idx])
        dr_reg = Ridge(alpha=10.0, random_state=seed)
        dr_reg.fit(X_tr, pseudo_r)
        dr_models.append(dr_reg)
        dr_scores_ev[:, a_idx] = dr_reg.predict(X_ev)
    dr_actions_ev = np.argmax(dr_scores_ev, axis=1)

    # Logging policy deterministic-sample baseline (argmax or sampled from log_pi_ev)
    log_actions_ev = a_ev

    policies: dict[str, np.ndarray] = {
        "Logging Policy (Historical Heuristic)": log_actions_ev,
        "Always Email": np.full(len(X_ev), ACTIONS.index("email_reminder"), dtype=np.int64),
        "Always WhatsApp": np.full(len(X_ev), ACTIONS.index("whatsapp_reminder"), dtype=np.int64),
        "Always Phone Call": np.full(len(X_ev), ACTIONS.index("phone_call"), dtype=np.int64),
        "Always Escalation": np.full(len(X_ev), ACTIONS.index("escalation"), dtype=np.int64),
        "Direct-Method Outcome Policy": dm_actions_ev,
        "LinUCB Contextual Bandit": linucb_actions_ev,
        "Doubly-Robust Policy Optimizer": dr_actions_ev,
    }

    policy_rows: list[dict[str, Any]] = []
    action_dist: dict[str, dict[str, float]] = {}

    for pol_name, act_arr in policies.items():
        is_log = pol_name == "Logging Policy (Historical Heuristic)"
        metrics = _evaluate_policy_ope(
            target_actions=act_arr,
            logged_actions=a_ev,
            propensities=p_ev,
            rewards=r_ev,
            mu_hat=mu_hat_ev,
            true_probs=true_probs_ev,
            true_exp_rewards=true_rew_ev,
            is_logging_policy=is_log,
        )
        mean_cost = float(np.mean([ACTION_COSTS[ACTIONS[a]] for a in act_arr]))
        est_pay_prob = float(np.mean(p_hat_ev[np.arange(len(act_arr)), act_arr]))
        row = {
            "policy": pol_name,
            "mean_action_cost": round(mean_cost, 2),
            "estimated_payment_prob_model": round(est_pay_prob, 4),
            **metrics,
        }
        policy_rows.append(row)
        action_dist[pol_name] = {
            act: round(100.0 * float(np.mean(act_arr == idx)), 2) for idx, act in enumerate(ACTIONS)
        }

    # Segment-level estimated payment probability & expected reward under each action
    seg_rows: list[dict[str, Any]] = []
    ev_meta["est_best_dr_action"] = [ACTIONS[a] for a in dr_actions_ev]
    for arch, grp_idx in ev_meta.groupby("true_archetype").groups.items():
        idx_arr = np.asarray(list(grp_idx), dtype=int)
        for a_idx, act in enumerate(ACTIONS):
            seg_rows.append(
                {
                    "customer_archetype": str(arch),
                    "action": act,
                    "action_cost": ACTION_COSTS[act],
                    "estimated_payment_prob": round(float(np.mean(p_hat_ev[idx_arr, a_idx])), 4),
                    "estimated_expected_reward": round(float(np.mean(dr_scores_ev[idx_arr, a_idx])), 2),
                    "sim_oracle_payment_prob": round(float(np.mean(true_probs_ev[idx_arr, a_idx])), 4),
                }
            )

    return BanditEvaluationReport(
        disclaimer=EPISTEMIC_DISCLAIMER,
        n_train_episodes=split_idx,
        n_eval_episodes=n_total - split_idx,
        action_costs=dict(ACTION_COSTS),
        policy_comparison=policy_rows,
        action_probability_by_segment=seg_rows,
        recommended_action_distribution=action_dist,
        linucb_learning_curve=linucb_curve,
    )
