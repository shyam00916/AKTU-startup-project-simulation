"""Longitudinal B2B receivables transaction simulator with behavioral archetypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from receivables_ml.config import DataSimulationConfig

INDUSTRIES: tuple[str, ...] = (
    "SaaS",
    "Healthcare",
    "Manufacturing",
    "Retail",
    "Logistics",
    "Construction",
)

REGIONS: tuple[str, ...] = ("North", "West", "South", "East")

ARCHETYPES: tuple[str, ...] = (
    "consistent_early_payer",
    "consistent_late_payer",
    "irregular_payer",
    "high_value_delayed_payer",
    "deteriorating_customer",
)

ARCHETYPE_PROBS: dict[str, float] = {
    "consistent_early_payer": 0.30,
    "consistent_late_payer": 0.20,
    "irregular_payer": 0.22,
    "high_value_delayed_payer": 0.14,
    "deteriorating_customer": 0.14,
}

INDUSTRY_DELAY_EFFECT: dict[str, float] = {
    "SaaS": -3.0,
    "Healthcare": -0.5,
    "Manufacturing": 1.5,
    "Retail": 2.5,
    "Logistics": 4.0,
    "Construction": 6.0,
}


@dataclass(frozen=True)
class ArchetypeProfile:
    """Generative parameters for a latent customer payment archetype."""

    base_delay_mean: float
    delay_noise_std: float
    ar_coef: float
    mean_cadence_days: float
    cadence_cv: float
    log_amount_mean: float
    log_amount_std: float
    partial_prob_base: float
    credit_limit_multiplier: float
    terms_choices: tuple[int, ...]
    deterioration_slope: float = 0.0


PROFILES: dict[str, ArchetypeProfile] = {
    "consistent_early_payer": ArchetypeProfile(
        base_delay_mean=-5.5,
        delay_noise_std=2.3,
        ar_coef=0.15,
        mean_cadence_days=24.0,
        cadence_cv=0.22,
        log_amount_mean=8.85,
        log_amount_std=0.35,
        partial_prob_base=0.03,
        credit_limit_multiplier=6.0,
        terms_choices=(15, 30),
    ),
    "consistent_late_payer": ArchetypeProfile(
        base_delay_mean=10.5,
        delay_noise_std=3.6,
        ar_coef=0.35,
        mean_cadence_days=26.0,
        cadence_cv=0.28,
        log_amount_mean=9.05,
        log_amount_std=0.40,
        partial_prob_base=0.22,
        credit_limit_multiplier=4.2,
        terms_choices=(30, 45),
    ),
    "irregular_payer": ArchetypeProfile(
        base_delay_mean=1.5,
        delay_noise_std=9.5,
        ar_coef=0.20,
        mean_cadence_days=30.0,
        cadence_cv=0.65,
        log_amount_mean=8.95,
        log_amount_std=0.65,
        partial_prob_base=0.28,
        credit_limit_multiplier=4.0,
        terms_choices=(30, 45),
    ),
    "high_value_delayed_payer": ArchetypeProfile(
        base_delay_mean=11.5,
        delay_noise_std=5.2,
        ar_coef=0.40,
        mean_cadence_days=32.0,
        cadence_cv=0.32,
        log_amount_mean=10.65,
        log_amount_std=0.45,
        partial_prob_base=0.46,
        credit_limit_multiplier=3.8,
        terms_choices=(45, 60),
    ),
    "deteriorating_customer": ArchetypeProfile(
        base_delay_mean=-2.5,
        delay_noise_std=3.6,
        ar_coef=0.48,
        mean_cadence_days=25.0,
        cadence_cv=0.34,
        log_amount_mean=9.15,
        log_amount_std=0.42,
        partial_prob_base=0.12,
        credit_limit_multiplier=4.0,
        terms_choices=(30, 45),
        deterioration_slope=1.35,
    ),
}


class ReceivablesSimulator:
    """Generates longitudinal customer-invoice-payment ledgers for ReceivablesOS."""

    def __init__(self, config: DataSimulationConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate customer metadata and chronological invoice/payment transactions."""
        customers = self._sample_customers()
        transactions = self._simulate_transactions(customers)
        return customers, transactions

    def _sample_customers(self) -> pd.DataFrame:
        n = self.config.n_customers
        archetype_names = list(ARCHETYPE_PROBS.keys())
        archetype_weights = [ARCHETYPE_PROBS[k] for k in archetype_names]
        sampled_archetypes = self.rng.choice(archetype_names, size=n, p=archetype_weights)

        rows: list[dict[str, Any]] = []
        for idx in range(n):
            arch = str(sampled_archetypes[idx])
            prof = PROFILES[arch]
            # Industry distribution correlated lightly with archetype for realism
            if arch == "high_value_delayed_payer":
                ind_probs = [0.10, 0.12, 0.28, 0.10, 0.18, 0.22]
            elif arch == "consistent_early_payer":
                ind_probs = [0.30, 0.24, 0.16, 0.14, 0.10, 0.06]
            elif arch == "deteriorating_customer":
                ind_probs = [0.12, 0.10, 0.20, 0.22, 0.18, 0.18]
            else:
                ind_probs = [0.18, 0.16, 0.18, 0.18, 0.16, 0.14]

            industry = str(self.rng.choice(INDUSTRIES, p=ind_probs))
            region = str(self.rng.choice(REGIONS))
            typical_amt = float(np.exp(prof.log_amount_mean))
            credit_limit = float(
                np.round(
                    typical_amt * prof.credit_limit_multiplier * self.rng.uniform(0.8, 1.35),
                    -2,
                )
            )
            latent_risk_offset = float(self.rng.normal(0.0, 2.0))
            deterioration_onset = (
                float(self.rng.uniform(0.28, 0.55)) if arch == "deteriorating_customer" else 1.0
            )
            n_invoices = int(
                self.rng.integers(
                    self.config.min_invoices_per_customer,
                    self.config.max_invoices_per_customer + 1,
                )
            )
            rows.append(
                {
                    "customer_id": f"CUST_{idx:04d}",
                    "true_archetype": arch,
                    "industry": industry,
                    "region": region,
                    "credit_limit": max(credit_limit, 5000.0),
                    "latent_risk_offset": latent_risk_offset,
                    "deterioration_onset": deterioration_onset,
                    "planned_n_invoices": n_invoices,
                }
            )
        return pd.DataFrame(rows)

    def _simulate_transactions(self, customers: pd.DataFrame) -> pd.DataFrame:
        start_ts = pd.Timestamp(self.config.start_date)
        records: list[dict[str, Any]] = []

        for cust in customers.itertuples(index=False):
            arch = cust.true_archetype
            prof = PROFILES[arch]
            horizon_days = int(self.config.horizon_days)
            ind_eff = INDUSTRY_DELAY_EFFECT[cust.industry]

            # Stagger onboarding across first 45 days so every cohort stays active across the full horizon
            current_day = int(self.rng.integers(0, 45))
            prev_delay = float(prof.base_delay_mean + ind_eff * 0.4)

            # Open ledger items: list of dicts tracking unsettled portions
            open_ledger: list[dict[str, float]] = []

            seq_idx = 0
            while current_day <= horizon_days and seq_idx < self.config.max_invoices_per_customer:
                progress = float(np.clip(current_day / max(horizon_days, 1), 0.0, 1.0))
                if seq_idx > 0:
                    cadence = prof.mean_cadence_days
                    if arch == "deteriorating_customer" and progress > cust.deterioration_onset:
                        # Deteriorating customers order more erratically as cash tightens
                        cadence *= 1.0 + 0.40 * (progress - cust.deterioration_onset)
                    gamma_shape = 1.0 / (prof.cadence_cv**2)
                    gamma_scale = cadence / gamma_shape
                    delta_days = int(np.clip(np.round(self.rng.gamma(gamma_shape, gamma_scale)), 4, 100))
                    current_day += delta_days
                    if current_day > horizon_days and seq_idx >= self.config.min_invoices_per_customer:
                        break
                else:
                    delta_days = int(round(prof.mean_cadence_days))

                terms = int(self.rng.choice(prof.terms_choices))
                due_day = current_day + terms

                # Compute exact ledger state at current_day before issuing new invoice
                outstanding_before = 0.0
                open_count = 0
                overdue_open_balance = 0.0
                for item in open_ledger:
                    if item["final_paid_day"] > current_day:
                        # Remaining unpaid amount at current_day
                        unpaid = item["amount"]
                        if item["first_partial_day"] <= current_day:
                            unpaid -= item["first_partial_amt"]
                        if unpaid > 1e-6:
                            outstanding_before += unpaid
                            open_count += 1
                            if current_day > item["due_day"]:
                                overdue_open_balance += unpaid

                utilization = outstanding_before / cust.credit_limit

                # Invoice amount with optional spike
                amt_log = self.rng.normal(prof.log_amount_mean, prof.log_amount_std)
                if arch == "irregular_payer" and self.rng.random() < 0.16:
                    amt_log += self.rng.uniform(0.45, 0.95)
                invoice_amount = float(np.round(np.exp(amt_log), 2))
                amount_to_limit = invoice_amount / cust.credit_limit

                # Seasonal / macro month effect (quarter-end crunch in Mar, Jun, Sep, Dec)
                issue_ts = start_ts + pd.Timedelta(days=current_day)
                is_quarter_end = 1.0 if issue_ts.month in (3, 6, 9, 12) else 0.0
                macro_effect = 2.0 * is_quarter_end + 1.0 * np.sin(2 * np.pi * issue_ts.day_of_year / 365.0)

                # Deterioration progression (non-linear acceleration)
                det_effect = 0.0
                if arch == "deteriorating_customer" and progress > cust.deterioration_onset:
                    elapsed_steps = (progress - cust.deterioration_onset) * 16.0
                    det_effect = prof.deterioration_slope * elapsed_steps + 0.08 * (elapsed_steps**2)

                # Irregular regime switching
                regime_jump = 0.0
                if arch == "irregular_payer" and self.rng.random() < 0.25:
                    regime_jump = float(self.rng.uniform(10.0, 24.0))

                # Utilization and size non-linear liquidity pressure
                util_pressure = 6.5 * max(0.0, utilization - 0.38) + 10.0 * max(0.0, utilization - 0.78) ** 1.4
                size_pressure = 3.5 * max(0.0, amount_to_limit - 0.25)

                expected_delay = (
                    prof.base_delay_mean
                    + cust.latent_risk_offset
                    + 0.45 * ind_eff
                    + prof.ar_coef * (prev_delay - prof.base_delay_mean)
                    + util_pressure
                    + size_pressure
                    + det_effect
                    + regime_jump
                    + macro_effect
                )
                raw_delay = expected_delay + float(self.rng.normal(0.0, prof.delay_noise_std))
                payment_delay = float(np.clip(np.round(raw_delay, 1), -15.0, 90.0))

                # Partial payment probability depends on archetype, utilization, amount, delay
                p_partial_logit = (
                    np.log(prof.partial_prob_base / max(1.0 - prof.partial_prob_base, 1e-4))
                    + 1.3 * max(0.0, utilization - 0.4)
                    + 0.8 * max(0.0, amount_to_limit - 0.3)
                    + 0.055 * max(0.0, payment_delay)
                    + (0.9 if det_effect > 4.0 else 0.0)
                )
                p_partial = float(1.0 / (1.0 + np.exp(-p_partial_logit)))
                is_partial = bool(self.rng.random() < p_partial)

                if is_partial:
                    n_installments = int(self.rng.choice([2, 3, 4], p=[0.60, 0.28, 0.12]))
                    first_paid_ratio = float(np.round(self.rng.uniform(0.28, 0.72), 3))
                    # First installment happens earlier than final settlement
                    first_delay = float(
                        np.clip(
                            np.round(payment_delay - self.rng.uniform(4.0, 14.0), 1),
                            -10.0,
                            payment_delay,
                        )
                    )
                    first_paid_day = float(max(current_day + 1, due_day + first_delay))
                else:
                    n_installments = 1
                    first_paid_ratio = 1.0
                    first_paid_day = float(max(current_day + 1, due_day + payment_delay))

                final_paid_day = float(max(current_day + 1, due_day + payment_delay))
                actual_delay_from_terms = float(final_paid_day - due_day)
                is_late = int(actual_delay_from_terms > self.config.late_threshold_days)

                open_ledger.append(
                    {
                        "amount": invoice_amount,
                        "due_day": float(due_day),
                        "first_partial_day": first_paid_day,
                        "first_partial_amt": invoice_amount * first_paid_ratio,
                        "final_paid_day": final_paid_day,
                    }
                )
                # Prune settled ledger items for speed
                open_ledger = [x for x in open_ledger if x["final_paid_day"] > current_day - 5]

                prev_delay = actual_delay_from_terms

                records.append(
                    {
                        "invoice_id": f"{cust.customer_id}_INV_{seq_idx:03d}",
                        "customer_id": cust.customer_id,
                        "seq_idx": seq_idx,
                        "true_archetype": arch,
                        "industry": cust.industry,
                        "region": cust.region,
                        "credit_limit": cust.credit_limit,
                        "issue_day": current_day,
                        "issue_date": (start_ts + pd.Timedelta(days=current_day)).strftime("%Y-%m-%d"),
                        "delta_days": delta_days,
                        "payment_terms_days": terms,
                        "due_day": due_day,
                        "invoice_amount": invoice_amount,
                        "outstanding_balance_before": float(np.round(outstanding_before, 2)),
                        "open_invoices_before": open_count,
                        "overdue_open_balance_before": float(np.round(overdue_open_balance, 2)),
                        "balance_utilization": float(utilization),
                        "is_partial_payment": int(is_partial),
                        "n_installments": n_installments,
                        "first_paid_ratio": first_paid_ratio,
                        "first_paid_day": first_paid_day,
                        "final_paid_day": final_paid_day,
                        "payment_delay_days": actual_delay_from_terms,
                        "is_late": is_late,
                    }
                )
                seq_idx += 1

        df = pd.DataFrame(records).sort_values(["issue_day", "customer_id", "seq_idx"]).reset_index(drop=True)
        return df


def generate_receivables_dataset(
    config: DataSimulationConfig, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience entry point to generate customer and transaction tables."""
    sim = ReceivablesSimulator(config=config, seed=seed)
    return sim.simulate()
