"""
Generate toy dataset for causal continual learning experiment

This script generates a 60-day dataset with:
- Period 1 (days 0-29): Stolen Credentials fraud (high amount)
- Period 2 (days 30-59): High-Frequency Attack fraud (high frequency)

The concept drift at day 30 tests whether causal models can adapt
better than correlational models.
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

from simulator.core import (
    generate_customer_profiles_table,
    generate_terminal_profiles_table,
    get_list_terminals_within_radius,
    generate_transactions_table
)
from simulator.causal.scenarios import (
    create_stolen_credentials_scm,
    create_high_frequency_attack_scm
)

# Configuration
N_CUSTOMERS = 1000
N_TERMINALS = 100
N_DAYS = 60
START_DATE = "2024-01-01"
RADIUS = 50
DRIFT_DAY = 30
RANDOM_SEED = 42

print("=" * 70)
print("GENERATING TOY DATASET FOR CAUSAL CONTINUAL LEARNING")
print("=" * 70)

np.random.seed(RANDOM_SEED)

# Step 1: Generate customer and terminal profiles
print(f"\n[1/6] Generating {N_CUSTOMERS} customer profiles...")
customer_profiles = generate_customer_profiles_table(N_CUSTOMERS, random_state=RANDOM_SEED)

print(f"[2/6] Generating {N_TERMINALS} terminal profiles...")
terminal_profiles = generate_terminal_profiles_table(N_TERMINALS, random_state=RANDOM_SEED + 1)

# Associate terminals to customers
print(f"[3/6] Associating terminals (radius={RADIUS})...")
x_y_terminals = terminal_profiles[['x_terminal_id', 'y_terminal_id']].values.astype(float)
customer_profiles['available_terminals'] = customer_profiles.apply(
    lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=RADIUS),
    axis=1
)
customer_profiles['nb_terminals'] = customer_profiles.available_terminals.apply(len)

print(f"      Average terminals per customer: {customer_profiles.nb_terminals.mean():.1f}")

# Step 2: Generate legitimate transactions
print(f"\n[4/6] Generating legitimate transactions for {N_DAYS} days...")
transactions_list = []

for day in range(N_DAYS):
    # Generate transactions for this day
    daily_transactions = customer_profiles.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(
            x.iloc[0],
            start_date=START_DATE,
            nb_days=1
        )
    ).reset_index(drop=True)

    if len(daily_transactions) > 0:
        # Adjust day
        daily_transactions['TX_TIME_DAYS'] = day
        daily_transactions['TX_TIME_SECONDS'] = daily_transactions['TX_TIME_SECONDS'] - (day * 86400) + (day * 86400)

        # Fix datetime
        start = datetime.strptime(START_DATE, "%Y-%m-%d")
        daily_transactions['TX_DATETIME'] = daily_transactions.apply(
            lambda row: start + timedelta(days=day, seconds=row['TX_TIME_SECONDS'] % 86400),
            axis=1
        )

        transactions_list.append(daily_transactions)

transactions_df = pd.concat(transactions_list, ignore_index=True)

print(f"      Generated {len(transactions_df)} legitimate transactions")
print(f"      Transactions per day: {len(transactions_df) / N_DAYS:.1f}")

# Step 3: Add fraudulent transactions using causal SCMs
print(f"\n[5/6] Adding fraudulent transactions with causal mechanisms...")

# Initialize fraud labels
transactions_df['TX_FRAUD'] = 0
transactions_df['TX_FRAUD_SCENARIO'] = 0
transactions_df['CAUSAL_GRAPH_ACTIVE'] = 'none'

# Period 1 (days 0-29): Stolen Credentials
print(f"      Period 1 (days 0-{DRIFT_DAY-1}): Stolen Credentials (high amount)")

scm_period1 = create_stolen_credentials_scm(
    customer_mean_amount=100,
    customer_std_amount=50,
    compromise_rate=0.01,  # 1% of customers
    testing_multiplier=0.5,
    exploitation_multiplier=5.0
)

# Select which customers are compromised in Period 1
n_compromised_p1 = int(N_CUSTOMERS * 0.01)  # 1% = 10 customers
np.random.seed(RANDOM_SEED)
compromised_customers_p1 = np.random.choice(customer_profiles.CUSTOMER_ID.values, n_compromised_p1, replace=False)

# Mark their transactions as fraudulent during Period 1
period1_mask = transactions_df.TX_TIME_DAYS < DRIFT_DAY
period1_compromised_mask = period1_mask & transactions_df.CUSTOMER_ID.isin(compromised_customers_p1)

# Sample from SCM to get fraud characteristics
fraud_samples_p1 = scm_period1.sample(n_samples=period1_compromised_mask.sum(), random_state=RANDOM_SEED)
fraud_samples_p1 = fraud_samples_p1[fraud_samples_p1.TX_FRAUD == 1]  # Only actual frauds

# Apply to subset of compromised transactions (since not all phases are fraud)
n_fraud_p1 = len(fraud_samples_p1)
if n_fraud_p1 > 0:
    fraud_indices_p1 = transactions_df[period1_compromised_mask].sample(n=min(n_fraud_p1, period1_compromised_mask.sum()), random_state=RANDOM_SEED).index
    transactions_df.loc[fraud_indices_p1, 'TX_FRAUD'] = 1
    transactions_df.loc[fraud_indices_p1, 'TX_FRAUD_SCENARIO'] = 1
    transactions_df.loc[fraud_indices_p1, 'CAUSAL_GRAPH_ACTIVE'] = 'stolen_credentials'

    # Modify amounts based on SCM
    transactions_df.loc[fraud_indices_p1, 'TX_AMOUNT'] = fraud_samples_p1['TX_AMOUNT'].values[:len(fraud_indices_p1)]

print(f"      -> {transactions_df[period1_mask].TX_FRAUD.sum()} frauds in Period 1")

# Period 2 (days 30-59): High-Frequency Attack
print(f"      Period 2 (days {DRIFT_DAY}-{N_DAYS-1}): High-Frequency Attack (high frequency)")

scm_period2 = create_high_frequency_attack_scm(
    customer_mean_amount=100,
    customer_std_amount=50,
    compromise_rate=0.01,
    frequency_multiplier=4.0,
    amount_multiplier=1.2
)

# Select DIFFERENT customers compromised in Period 2 (concept drift)
np.random.seed(RANDOM_SEED + 1)
available_customers = [c for c in customer_profiles.CUSTOMER_ID.values if c not in compromised_customers_p1]
compromised_customers_p2 = np.random.choice(available_customers, n_compromised_p1, replace=False)

# Mark their transactions as fraudulent during Period 2
period2_mask = transactions_df.TX_TIME_DAYS >= DRIFT_DAY
period2_compromised_mask = period2_mask & transactions_df.CUSTOMER_ID.isin(compromised_customers_p2)

# Sample from SCM
fraud_samples_p2 = scm_period2.sample(n_samples=period2_compromised_mask.sum(), random_state=RANDOM_SEED + 1)
fraud_samples_p2 = fraud_samples_p2[fraud_samples_p2.TX_FRAUD == 1]

n_fraud_p2 = len(fraud_samples_p2)
if n_fraud_p2 > 0:
    fraud_indices_p2 = transactions_df[period2_compromised_mask].sample(n=min(n_fraud_p2, period2_compromised_mask.sum()), random_state=RANDOM_SEED + 1).index
    transactions_df.loc[fraud_indices_p2, 'TX_FRAUD'] = 1
    transactions_df.loc[fraud_indices_p2, 'TX_FRAUD_SCENARIO'] = 2
    transactions_df.loc[fraud_indices_p2, 'CAUSAL_GRAPH_ACTIVE'] = 'high_frequency_attack'

    # Modify amounts (slightly elevated, not obvious)
    transactions_df.loc[fraud_indices_p2, 'TX_AMOUNT'] = fraud_samples_p2['TX_AMOUNT'].values[:len(fraud_indices_p2)]

print(f"      -> {transactions_df[period2_mask].TX_FRAUD.sum()} frauds in Period 2")

# Step 4: Final processing
print(f"\n[6/6] Finalizing dataset...")

# Sort by datetime
transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
transactions_df['TRANSACTION_ID'] = transactions_df.index

# Reorder columns
column_order = [
    'TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID',
    'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS',
    'TX_FRAUD', 'TX_FRAUD_SCENARIO', 'CAUSAL_GRAPH_ACTIVE'
]
transactions_df = transactions_df[column_order]

# Save dataset
output_dir = "../results/data/"
import os
os.makedirs(output_dir, exist_ok=True)

transactions_df.to_pickle(output_dir + "toy_transactions.pkl")
customer_profiles.to_pickle(output_dir + "toy_customers.pkl")
terminal_profiles.to_pickle(output_dir + "toy_terminals.pkl")

# Save causal log (ground truth)
causal_log = {
    'drift_day': DRIFT_DAY,
    'period_1': {
        'days': list(range(0, DRIFT_DAY)),
        'scenario': 'stolen_credentials',
        'scm_name': scm_period1.name,
        'compromised_customers': list(compromised_customers_p1),
        'fraud_count': transactions_df[period1_mask].TX_FRAUD.sum()
    },
    'period_2': {
        'days': list(range(DRIFT_DAY, N_DAYS)),
        'scenario': 'high_frequency_attack',
        'scm_name': scm_period2.name,
        'compromised_customers': list(compromised_customers_p2),
        'fraud_count': transactions_df[period2_mask].TX_FRAUD.sum()
    },
    'config': {
        'n_customers': N_CUSTOMERS,
        'n_terminals': N_TERMINALS,
        'n_days': N_DAYS,
        'random_seed': RANDOM_SEED
    }
}

with open(output_dir + "toy_causal_log.pkl", "wb") as f:
    pickle.dump(causal_log, f)

# Print summary statistics
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"\nTotal transactions: {len(transactions_df)}")
print(f"Total frauds: {transactions_df.TX_FRAUD.sum()}")
print(f"Overall fraud rate: {transactions_df.TX_FRAUD.mean():.3f}")

print(f"\nPeriod 1 (days 0-{DRIFT_DAY-1}, Stolen Credentials):")
period1_stats = transactions_df[period1_mask]
print(f"  Transactions: {len(period1_stats)}")
print(f"  Frauds: {period1_stats.TX_FRAUD.sum()}")
print(f"  Fraud rate: {period1_stats.TX_FRAUD.mean():.3f}")
print(f"  Fraud amount (mean): ${period1_stats[period1_stats.TX_FRAUD==1].TX_AMOUNT.mean():.2f}")
print(f"  Legit amount (mean): ${period1_stats[period1_stats.TX_FRAUD==0].TX_AMOUNT.mean():.2f}")

print(f"\nPeriod 2 (days {DRIFT_DAY}-{N_DAYS-1}, High-Frequency Attack):")
period2_stats = transactions_df[period2_mask]
print(f"  Transactions: {len(period2_stats)}")
print(f"  Frauds: {period2_stats.TX_FRAUD.sum()}")
print(f"  Fraud rate: {period2_stats.TX_FRAUD.mean():.3f}")
print(f"  Fraud amount (mean): ${period2_stats[period2_stats.TX_FRAUD==1].TX_AMOUNT.mean():.2f}")
print(f"  Legit amount (mean): ${period2_stats[period2_stats.TX_FRAUD==0].TX_AMOUNT.mean():.2f}")

print(f"\n[SUCCESS] Dataset saved to {output_dir}")
print("\nFiles created:")
print(f"  - toy_transactions.pkl ({len(transactions_df)} transactions)")
print(f"  - toy_customers.pkl ({len(customer_profiles)} customers)")
print(f"  - toy_terminals.pkl ({len(terminal_profiles)} terminals)")
print(f"  - toy_causal_log.pkl (ground truth causal graphs)")

print("\n" + "=" * 70)
print("READY FOR EXPERIMENT!")
print("=" * 70)
