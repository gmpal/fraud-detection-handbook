"""
Causal fraud scenarios with explicit SCM definitions

Each scenario implements a specific fraud mechanism as a Structural Causal Model.
These SCMs can be used to generate synthetic fraud data with known causal structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

try:
    from .scm import StructuralCausalModel
except ImportError:
    from scm import StructuralCausalModel


def create_stolen_credentials_scm(
    customer_mean_amount: float = 100.0,
    customer_std_amount: float = 50.0,
    customer_mean_freq: float = 2.0,
    compromise_rate: float = 0.01,
    testing_multiplier: float = 0.5,
    exploitation_multiplier: float = 5.0,
    testing_duration: int = 3,
    exploitation_duration: int = 10
) -> StructuralCausalModel:
    """
    Create SCM for Stolen Credentials fraud scenario.

    Causal mechanism:
        COMPROMISED* → DAYS_SINCE_COMPROMISE* → FRAUD_PHASE*
                ↓                                      ↓
        FRAUDSTER_INTENT* ←────────────────────────────┘
                ↓              ↓            ↓
        TX_FRAUD    TX_AMOUNT    GEOGRAPHIC_DISTANCE

    Phases:
        - Testing (days 0-2): Small transactions (0.5x)
        - Exploitation (days 3-10): Large transactions (5x)
        - Abandonment (days 11+): Stop

    Args:
        customer_mean_amount: Customer's normal transaction amount
        customer_std_amount: Standard deviation of amount
        customer_mean_freq: Customer's normal transaction frequency
        compromise_rate: Probability of being compromised
        testing_multiplier: Amount multiplier during testing phase
        exploitation_multiplier: Amount multiplier during exploitation
        testing_duration: Days in testing phase
        exploitation_duration: Days in exploitation phase (total)

    Returns:
        StructuralCausalModel for stolen credentials scenario
    """
    scm = StructuralCausalModel(name="stolen_credentials")

    # Latent: Is customer compromised?
    scm.add_variable(
        "COMPROMISED",
        is_latent=True,
        mechanism=lambda n: np.random.binomial(1, compromise_rate, n)
    )

    # Latent: Days since compromise (for compromised customers)
    # For non-compromised, set to -1 (indicator of not compromised)
    def days_since_mechanism(COMPROMISED):
        n = len(COMPROMISED)
        days = np.full(n, -1, dtype=int)  # -1 for not compromised
        compromised_idx = np.where(COMPROMISED == 1)[0]
        if len(compromised_idx) > 0:
            # Random days since compromise (0 to 14)
            days[compromised_idx] = np.random.randint(0, 15, len(compromised_idx))
        return days

    scm.add_variable(
        "DAYS_SINCE_COMPROMISE",
        is_latent=True,
        parents=["COMPROMISED"],
        mechanism=days_since_mechanism
    )

    # Latent: Fraud phase (testing/exploitation/abandonment)
    def fraud_phase_mechanism(DAYS_SINCE_COMPROMISE):
        n = len(DAYS_SINCE_COMPROMISE)
        phase = np.full(n, -1, dtype=int)  # -1: not compromised, 0: testing, 1: exploitation, 2: abandonment

        # Testing phase
        testing_mask = (DAYS_SINCE_COMPROMISE >= 0) & (DAYS_SINCE_COMPROMISE < testing_duration)
        phase[testing_mask] = 0

        # Exploitation phase
        exploitation_mask = (DAYS_SINCE_COMPROMISE >= testing_duration) & (DAYS_SINCE_COMPROMISE < exploitation_duration)
        phase[exploitation_mask] = 1

        # Abandonment phase
        abandonment_mask = DAYS_SINCE_COMPROMISE >= exploitation_duration
        phase[abandonment_mask] = 2

        return phase

    scm.add_variable(
        "FRAUD_PHASE",
        is_latent=True,
        parents=["DAYS_SINCE_COMPROMISE"],
        mechanism=fraud_phase_mechanism
    )

    # Latent: Fraudster intent (active during testing and exploitation, not abandonment)
    def fraudster_intent_mechanism(FRAUD_PHASE):
        # Intent is 1 during testing (0) and exploitation (1), 0 during abandonment (2) or not compromised (-1)
        return np.where((FRAUD_PHASE == 0) | (FRAUD_PHASE == 1), 1, 0)

    scm.add_variable(
        "FRAUDSTER_INTENT",
        is_latent=True,
        parents=["FRAUD_PHASE"],
        mechanism=fraudster_intent_mechanism
    )

    # Observable: Transaction amount
    def tx_amount_mechanism(FRAUD_PHASE, noise):
        n = len(FRAUD_PHASE)
        amount = np.full(n, customer_mean_amount, dtype=float)

        # Testing phase: smaller amounts
        testing_mask = FRAUD_PHASE == 0
        amount[testing_mask] = customer_mean_amount * testing_multiplier

        # Exploitation phase: large amounts
        exploitation_mask = FRAUD_PHASE == 1
        amount[exploitation_mask] = customer_mean_amount * exploitation_multiplier

        # Add noise
        amount = amount + noise

        # Ensure positive
        amount = np.maximum(amount, 1.0)

        return amount

    scm.add_variable(
        "TX_AMOUNT",
        parents=["FRAUD_PHASE"],
        mechanism=tx_amount_mechanism,
        noise_dist=lambda n: np.random.normal(0, customer_std_amount, n)
    )

    # Observable: Geographic distance from customer's usual location
    def geo_distance_mechanism(FRAUDSTER_INTENT, noise):
        # Legitimate: Near home (mean 0 km, std 5 km)
        # Testing/Exploitation: Far from home (mean 50 km, std 20 km)
        distance = np.where(
            FRAUDSTER_INTENT == 1,
            50 + noise * 4,  # Fraud: far
            0 + noise         # Legitimate: near
        )
        return np.maximum(distance, 0)  # Ensure non-negative

    scm.add_variable(
        "GEOGRAPHIC_DISTANCE",
        parents=["FRAUDSTER_INTENT"],
        mechanism=geo_distance_mechanism,
        noise_dist=lambda n: np.random.normal(0, 5, n)
    )

    # Observable: Transaction frequency multiplier
    def tx_freq_mechanism(FRAUD_PHASE):
        n = len(FRAUD_PHASE)
        freq_multiplier = np.ones(n, dtype=float)

        # Testing: slightly more frequent (1.5x)
        testing_mask = FRAUD_PHASE == 0
        freq_multiplier[testing_mask] = 1.5

        # Exploitation: much more frequent (4x)
        exploitation_mask = FRAUD_PHASE == 1
        freq_multiplier[exploitation_mask] = 4.0

        return freq_multiplier

    scm.add_variable(
        "TX_FREQUENCY_MULTIPLIER",
        parents=["FRAUD_PHASE"],
        mechanism=tx_freq_mechanism
    )

    # Observable: Fraud label
    scm.add_variable(
        "TX_FRAUD",
        parents=["FRAUDSTER_INTENT"],
        mechanism=lambda FRAUDSTER_INTENT: FRAUDSTER_INTENT
    )

    return scm


def create_high_frequency_attack_scm(
    customer_mean_amount: float = 100.0,
    customer_std_amount: float = 50.0,
    customer_mean_freq: float = 2.0,
    compromise_rate: float = 0.01,
    frequency_multiplier: float = 4.0,
    amount_multiplier: float = 1.2
) -> StructuralCausalModel:
    """
    Create SCM for High-Frequency Attack fraud scenario (for concept drift).

    This is an alternative fraud mechanism where fraudsters make many
    small transactions instead of few large ones.

    Causal mechanism:
        COMPROMISED* → FRAUDSTER_ACTIVE*
                ↓              ↓
        TX_FREQUENCY    TX_AMOUNT (slightly elevated)
                ↓              ↓
                TX_FRAUD ←─────┘

    Args:
        customer_mean_amount: Customer's normal amount
        customer_std_amount: Standard deviation
        customer_mean_freq: Customer's normal frequency
        compromise_rate: Probability of being compromised
        frequency_multiplier: How much more frequent (e.g., 4x)
        amount_multiplier: How much higher amount (e.g., 1.2x)

    Returns:
        StructuralCausalModel for high-frequency attack
    """
    scm = StructuralCausalModel(name="high_frequency_attack")

    # Latent: Is customer compromised?
    scm.add_variable(
        "COMPROMISED",
        is_latent=True,
        mechanism=lambda n: np.random.binomial(1, compromise_rate, n)
    )

    # Latent: Is fraudster actively exploiting? (Same as COMPROMISED for this simple scenario)
    scm.add_variable(
        "FRAUDSTER_ACTIVE",
        is_latent=True,
        parents=["COMPROMISED"],
        mechanism=lambda COMPROMISED: COMPROMISED
    )

    # Observable: Transaction frequency multiplier
    def freq_mechanism(FRAUDSTER_ACTIVE):
        return np.where(FRAUDSTER_ACTIVE == 1, frequency_multiplier, 1.0)

    scm.add_variable(
        "TX_FREQUENCY_MULTIPLIER",
        parents=["FRAUDSTER_ACTIVE"],
        mechanism=freq_mechanism
    )

    # Observable: Transaction amount (only slightly elevated)
    def amount_mechanism(FRAUDSTER_ACTIVE, noise):
        base = np.where(
            FRAUDSTER_ACTIVE == 1,
            customer_mean_amount * amount_multiplier,
            customer_mean_amount
        )
        return np.maximum(base + noise, 1.0)

    scm.add_variable(
        "TX_AMOUNT",
        parents=["FRAUDSTER_ACTIVE"],
        mechanism=amount_mechanism,
        noise_dist=lambda n: np.random.normal(0, customer_std_amount, n)
    )

    # Observable: Fraud label
    scm.add_variable(
        "TX_FRAUD",
        parents=["FRAUDSTER_ACTIVE"],
        mechanism=lambda FRAUDSTER_ACTIVE: FRAUDSTER_ACTIVE
    )

    return scm


# Example usage
if __name__ == "__main__":
    print("Testing Stolen Credentials SCM...")
    scm = create_stolen_credentials_scm(
        customer_mean_amount=100,
        customer_std_amount=30,
        compromise_rate=0.05  # Higher for testing
    )

    samples = scm.sample(n_samples=10000, random_state=42)

    print(f"Total samples: {len(samples)}")
    print(f"Fraud rate: {samples.TX_FRAUD.mean():.3f}")
    print(f"\nAmount statistics:")
    print(samples.groupby('TX_FRAUD')['TX_AMOUNT'].agg(['mean', 'std', 'min', 'max']))
    print(f"\nGeographic distance statistics:")
    print(samples.groupby('TX_FRAUD')['GEOGRAPHIC_DISTANCE'].agg(['mean', 'std']))

    print("\n" + "="*50)
    print("Testing High-Frequency Attack SCM...")
    scm2 = create_high_frequency_attack_scm(
        customer_mean_amount=100,
        customer_std_amount=30,
        compromise_rate=0.05
    )

    samples2 = scm2.sample(n_samples=10000, random_state=42)

    print(f"Total samples: {len(samples2)}")
    print(f"Fraud rate: {samples2.TX_FRAUD.mean():.3f}")
    print(f"\nAmount statistics:")
    print(samples2.groupby('TX_FRAUD')['TX_AMOUNT'].agg(['mean', 'std']))
    print(f"\nFrequency multiplier statistics:")
    print(samples2.groupby('TX_FRAUD')['TX_FREQUENCY_MULTIPLIER'].agg(['mean', 'std']))

    print("\n[SUCCESS] Both scenarios working!")
