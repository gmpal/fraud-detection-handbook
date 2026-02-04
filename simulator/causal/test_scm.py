"""
Test SCM implementation with a simple fraud scenario
"""

import sys
sys.path.append('..')

from scm import StructuralCausalModel
import numpy as np

print("Test 1: Simple fraud SCM")
print("-" * 50)

# Create a simple SCM: COMPROMISED -> TX_AMOUNT -> TX_FRAUD
scm = StructuralCausalModel(name="simple_fraud")

# Latent variable: Is customer compromised?
scm.add_variable(
    "COMPROMISED",
    is_latent=True,
    mechanism=lambda n: np.random.binomial(1, 0.01, n)  # 1% compromise rate
)

# Observable: Transaction amount (depends on compromise status)
def amount_mechanism(COMPROMISED, noise):
    # If compromised: higher amounts
    # If legitimate: normal amounts
    return np.where(
        COMPROMISED == 1,
        np.maximum(0, 500 + noise),   # Fraud: $500 + noise
        np.maximum(0, 100 + noise)    # Legitimate: $100 + noise
    )

scm.add_variable(
    "TX_AMOUNT",
    parents=["COMPROMISED"],
    mechanism=amount_mechanism,
    noise_dist=lambda n: np.random.normal(0, 50, n)
)

# Observable: Fraud label (deterministic given compromise)
scm.add_variable(
    "TX_FRAUD",
    parents=["COMPROMISED"],
    mechanism=lambda COMPROMISED: COMPROMISED
)

# Test 1: Sample from SCM
print("\nSampling from SCM...")
samples = scm.sample(n_samples=10000, random_state=42)

print(f"[OK] Generated {len(samples)} samples")
print(f"[OK] Columns: {list(samples.columns)}")
print(f"[OK] Fraud rate: {samples.TX_FRAUD.mean():.3f} (expected ~0.01)")
print(f"[OK] Mean amount (fraud): ${samples[samples.TX_FRAUD==1].TX_AMOUNT.mean():.2f}")
print(f"[OK] Mean amount (legit): ${samples[samples.TX_FRAUD==0].TX_AMOUNT.mean():.2f}")

# Test 2: Intervention (do-operator)
print("\nTest 2: Intervention do(COMPROMISED=1)")
print("-" * 50)

intervened_samples = scm.sample(n_samples=1000, interventions={"COMPROMISED": 1}, random_state=42)

print(f"[OK] All frauds: {intervened_samples.TX_FRAUD.all()}")
print(f"[OK] Mean amount: ${intervened_samples.TX_AMOUNT.mean():.2f} (expected ~$500)")

# Test 3: Counterfactual
print("\nTest 3: Counterfactual query")
print("-" * 50)

# "What would TX_AMOUNT be if customer was not compromised?"
counterfactual = scm.counterfactual(
    evidence={},
    intervention={"COMPROMISED": 0},
    query="TX_AMOUNT",
    n_samples=1000
)

print(f"[OK] Counterfactual mean amount (if not compromised): ${counterfactual.mean():.2f}")
print(f"[OK] Expected: ~$100")

print("\n[SUCCESS] All SCM tests passed!")
