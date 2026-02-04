"""
Full Continual Learning Experiment: Correlational vs. Causal

This script runs the complete experiment comparing:
1. Correlational Baseline (Random Forest, naive fine-tuning)
2. Causal Baseline (Oracle with causal features, ensemble)

Metrics:
- Forward Transfer (FT): Performance on new task (Period 2)
- Backward Transfer (BT): Forgetting on old task (Period 1)

Expected result: Causal model has BT ~0 (no forgetting),
                 Correlational model has BT < -0.20 (catastrophic forgetting)
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Import models
from models.correlational_baseline import CorrelationalBaseline
from models.causal_baseline import CausalBaseline
from models.utils import extract_features, evaluate_model, print_metrics

print("=" * 80)
print("CONTINUAL LEARNING EXPERIMENT: CAUSAL VS. CORRELATIONAL")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load dataset
print("\n[1/8] Loading toy dataset...")
transactions = pd.read_pickle("../results/data/toy_transactions.pkl")
customers = pd.read_pickle("../results/data/toy_customers.pkl")
terminals = pd.read_pickle("../results/data/toy_terminals.pkl")

with open("../results/data/toy_causal_log.pkl", "rb") as f:
    causal_log = pickle.load(f)

print(f"  Total transactions: {len(transactions)}")
print(f"  Customers: {len(customers)}")
print(f"  Terminals: {len(terminals)}")
print(f"  Drift day: {causal_log['drift_day']}")
print(f"  Period 1 scenario: {causal_log['period_1']['scenario']}")
print(f"  Period 2 scenario: {causal_log['period_2']['scenario']}")

# Split data
print("\n[2/8] Splitting into periods...")
drift_day = causal_log['drift_day']

period1_data = transactions[transactions['TX_TIME_DAYS'] < drift_day].copy()
period2_data = transactions[transactions['TX_TIME_DAYS'] >= drift_day].copy()

print(f"  Period 1 (days 0-{drift_day-1}): {len(period1_data)} transactions, {period1_data.TX_FRAUD.sum()} frauds")
print(f"  Period 2 (days {drift_day}-59): {len(period2_data)} transactions, {period2_data.TX_FRAUD.sum()} frauds")

# Initialize results storage
results = {
    'correlational': {},
    'causal': {},
    'config': {
        'drift_day': drift_day,
        'n_period1': len(period1_data),
        'n_period2': len(period2_data),
        'frauds_period1': int(period1_data.TX_FRAUD.sum()),
        'frauds_period2': int(period2_data.TX_FRAUD.sum())
    }
}

# ============================================================================
# EXPERIMENT 1: CORRELATIONAL BASELINE
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: CORRELATIONAL BASELINE (Random Forest)")
print("=" * 80)

print("\n[3/8] Training correlational model on Period 1...")
corr_model = CorrelationalBaseline(n_estimators=100, max_depth=10, random_state=42)

X_period1_corr = extract_features(period1_data, customers)
y_period1 = period1_data['TX_FRAUD'].values

X_period2_corr = extract_features(period2_data, customers)
y_period2 = period2_data['TX_FRAUD'].values

corr_model.fit(X_period1_corr, y_period1)

# Evaluate on Period 1 (before CL)
print("  Evaluating on Period 1 (before CL)...")
y_pred_p1_before = corr_model.predict_proba(X_period1_corr)
metrics_corr_p1_before = evaluate_model(
    y_period1, y_pred_p1_before,
    customer_ids=period1_data['CUSTOMER_ID'].values,
    day_col=period1_data['TX_TIME_DAYS'].values
)
results['correlational']['period1_before_cl'] = metrics_corr_p1_before
print_metrics(metrics_corr_p1_before, "  Period 1 (before CL)")

# Continual learning on Period 2
print("\n  Continual learning: Naive fine-tuning on Period 2...")
corr_model.continual_update(X_period2_corr, y_period2)

# Evaluate on Period 2 (Forward Transfer)
print("  Evaluating on Period 2 (Forward Transfer)...")
y_pred_p2_corr = corr_model.predict_proba(X_period2_corr)
metrics_corr_p2 = evaluate_model(
    y_period2, y_pred_p2_corr,
    customer_ids=period2_data['CUSTOMER_ID'].values,
    day_col=period2_data['TX_TIME_DAYS'].values
)
results['correlational']['period2_after_cl'] = metrics_corr_p2
print_metrics(metrics_corr_p2, "  Period 2 (after CL) - Forward Transfer")

# Evaluate on Period 1 again (Backward Transfer)
print("\n  Evaluating on Period 1 again (Backward Transfer - FORGETTING)...")
y_pred_p1_after_corr = corr_model.predict_proba(X_period1_corr)
metrics_corr_p1_after = evaluate_model(
    y_period1, y_pred_p1_after_corr,
    customer_ids=period1_data['CUSTOMER_ID'].values,
    day_col=period1_data['TX_TIME_DAYS'].values
)
results['correlational']['period1_after_cl'] = metrics_corr_p1_after
print_metrics(metrics_corr_p1_after, "  Period 1 (after CL) - Backward Transfer")

# ============================================================================
# EXPERIMENT 2: CAUSAL BASELINE
# ============================================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: CAUSAL BASELINE (Oracle with Causal Features)")
print("=" * 80)

print("\n[4/8] Training causal model on Period 1...")
causal_model = CausalBaseline(n_estimators=100, max_depth=10, random_state=42)
causal_model.fit(period1_data, customers)

# Evaluate on Period 1 (before CL)
print("  Evaluating on Period 1 (before CL)...")
y_pred_p1_before_causal = causal_model.predict_proba(period1_data, customers)
metrics_causal_p1_before = evaluate_model(
    y_period1, y_pred_p1_before_causal,
    customer_ids=period1_data['CUSTOMER_ID'].values,
    day_col=period1_data['TX_TIME_DAYS'].values
)
results['causal']['period1_before_cl'] = metrics_causal_p1_before
print_metrics(metrics_causal_p1_before, "  Period 1 (before CL)")

# Continual learning on Period 2
print("\n  Continual learning: Preserving old model (ensemble)...")
causal_model.continual_update(period2_data, customers, preserve_old=True)

# Evaluate on Period 2 (Forward Transfer)
print("  Evaluating on Period 2 (Forward Transfer)...")
y_pred_p2_causal = causal_model.predict_proba(period2_data, customers, use_ensemble=True)
metrics_causal_p2 = evaluate_model(
    y_period2, y_pred_p2_causal,
    customer_ids=period2_data['CUSTOMER_ID'].values,
    day_col=period2_data['TX_TIME_DAYS'].values
)
results['causal']['period2_after_cl'] = metrics_causal_p2
print_metrics(metrics_causal_p2, "  Period 2 (after CL) - Forward Transfer")

# Evaluate on Period 1 again (Backward Transfer)
print("\n  Evaluating on Period 1 again (Backward Transfer - Should be MINIMAL)...")
y_pred_p1_after_causal = causal_model.predict_proba(period1_data, customers, use_ensemble=True)
metrics_causal_p1_after = evaluate_model(
    y_period1, y_pred_p1_after_causal,
    customer_ids=period1_data['CUSTOMER_ID'].values,
    day_col=period1_data['TX_TIME_DAYS'].values
)
results['causal']['period1_after_cl'] = metrics_causal_p1_after
print_metrics(metrics_causal_p1_after, "  Period 1 (after CL) - Backward Transfer")

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY: BACKWARD TRANSFER COMPARISON")
print("=" * 80)

print("\n{:<25} {:<15} {:<15} {:<20}".format("Metric", "Correlational", "Causal", "Difference"))
print("-" * 80)

for metric in ['AUC_ROC', 'Average_Precision', 'Card_Precision@100']:
    # Correlational BT
    before_corr = results['correlational']['period1_before_cl'].get(metric, np.nan)
    after_corr = results['correlational']['period1_after_cl'].get(metric, np.nan)
    bt_corr = after_corr - before_corr

    # Causal BT
    before_causal = results['causal']['period1_before_cl'].get(metric, np.nan)
    after_causal = results['causal']['period1_after_cl'].get(metric, np.nan)
    bt_causal = after_causal - before_causal

    # Difference
    bt_diff = bt_causal - bt_corr

    print(f"{metric:<25} {bt_corr:+.4f}         {bt_causal:+.4f}         {bt_diff:+.4f} {'(Causal better!)' if bt_diff > 0.05 else ''}")

    # Store in results
    results['backward_transfer'] = results.get('backward_transfer', {})
    results['backward_transfer'][metric] = {
        'correlational': float(bt_corr),
        'causal': float(bt_causal),
        'difference': float(bt_diff)
    }

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[5/8] Saving results...")
results_file = "../results/data/continual_learning_results.pkl"
with open(results_file, "wb") as f:
    pickle.dump(results, f)

print(f"  Results saved to: {results_file}")

# ============================================================================
# CONCLUSION
# ============================================================================

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Get key metrics
bt_corr_auc = results['backward_transfer']['AUC_ROC']['correlational']
bt_causal_auc = results['backward_transfer']['AUC_ROC']['causal']

bt_corr_ap = results['backward_transfer']['Average_Precision']['correlational']
bt_causal_ap = results['backward_transfer']['Average_Precision']['causal']

print(f"\nBackward Transfer (Forgetting):")
print(f"  Correlational Model: BT(AUC ROC) = {bt_corr_auc:+.4f}")
print(f"  Causal Model:        BT(AUC ROC) = {bt_causal_auc:+.4f}")
print(f"  Improvement:         {bt_causal_auc - bt_corr_auc:+.4f}")

print(f"\n  Correlational Model: BT(Avg Precision) = {bt_corr_ap:+.4f}")
print(f"  Causal Model:        BT(Avg Precision) = {bt_causal_ap:+.4f}")
print(f"  Improvement:         {bt_causal_ap - bt_corr_ap:+.4f}")

# Determine success
if bt_causal_auc > bt_corr_auc + 0.05:
    print("\n[SUCCESS] Causal model demonstrates significantly less forgetting!")
    print("Hypothesis CONFIRMED: Causal models are more robust to concept drift.")
else:
    print("\n[NOTE] Results inconclusive. May need more data or stronger drift.")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
