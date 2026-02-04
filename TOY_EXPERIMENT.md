# Toy Experiment: Causal Continual Learning for Fraud Detection

## Objective

Demonstrate that a causal model is more robust to concept drift and suffers less catastrophic forgetting than a correlational model in a minimal, reproducible setting.

**Expected Runtime**: ~10 minutes on CPU
**Dataset Size**: 60 days, 1000 customers, 100 terminals, ~15,000 transactions
**Key Result**: Causal model maintains >85% of original performance after concept drift; correlational model drops to <60%

## Experimental Setup

### Timeline

```
Days 0-29:  Period 1 (Graph A: High Amount Fraud)
Day 30:     CONCEPT DRIFT (Graph switch)
Days 31-59: Period 2 (Graph B: High Frequency Fraud)
```

### Data Generation

**Period 1 Parameters**:
- 1000 customers, 100 terminals
- Fraud rate: 1% of customers compromised
- Fraud mechanism: Stolen Credentials (Scenario 1)
  - Testing phase (days 0-2): `TX_AMOUNT = μ_customer * 0.5`
  - Exploitation phase (days 3-10): `TX_AMOUNT = μ_customer * 5.0`
  - Abandonment (days 11+): Stop

**Period 2 Parameters**:
- Same customers and terminals
- Fraud rate: 1% of customers (NEW compromises)
- Fraud mechanism: High-Frequency Attack (NEW scenario)
  - `TX_FREQUENCY = λ_customer * 4.0`
  - `TX_AMOUNT = μ_customer * 1.2` (slightly elevated, not obvious)

**Ground Truth Causal Graphs**:
```
Graph A (Period 1):
COMPROMISED* → FRAUD_PHASE* → TX_AMOUNT → TX_FRAUD

Graph B (Period 2):
COMPROMISED* → TX_FREQUENCY → TX_FRAUD
```

### Models to Compare

#### 1. Correlational Baseline (Random Forest)
- **Features**: TX_AMOUNT, TX_TIME_SECONDS, CUSTOMER_ID (one-hot), TERMINAL_ID (one-hot)
- **Training**: Standard supervised learning
- **Continual Learning**: Naive fine-tuning (retrain on Period 2)

#### 2. Causal Baseline (Neural Causal Model)
- **Features**: Same observables
- **Architecture**: Neural network that learns causal graph structure
- **Training**: Causal structure learning + prediction
- **Continual Learning**: Update graph structure while preserving causal mechanisms

#### 3. Causal Oracle (Upper Bound)
- **Features**: Same observables
- **Architecture**: Uses TRUE causal graph (cheating)
- **Training**: Only learns parameters, not structure
- **Purpose**: Show maximum possible benefit of causal knowledge

### Evaluation Protocol

#### Training Phases

**Phase 1**: Train on Period 1
- Train both models on days 0-29
- Evaluate on days 0-29 (sanity check: both should perform well)

**Phase 2**: Introduce Concept Drift
- At day 30, fraudsters switch tactics (Graph A → Graph B)
- Evaluate both models on days 31-59 WITHOUT retraining
- **Expected**: Both models fail (drift is too severe)

**Phase 3**: Continual Learning
- Fine-tune both models on days 31-44 (first 14 days of Period 2)
- Evaluate on:
  - Days 45-59 (Period 2, held-out test set) → **Forward Transfer (FT)**
  - Days 0-29 (Period 1, old data) → **Backward Transfer (BT)**: Measure forgetting

#### Metrics

1. **Card Precision@100 (CP@100)**:
   - Primary metric from fraud detection handbook
   - Precision of top-100 most suspicious cards per day

2. **Forward Transfer (FT)**:
   - Performance on new task (Period 2) after training
   - `FT = CP@100(Period 2 test, after CL training)`

3. **Backward Transfer (BT)**:
   - Performance degradation on old task (Period 1) after training on new task
   - `BT = CP@100(Period 1, after CL training) - CP@100(Period 1, before CL training)`
   - **Positive BT**: Improvement (unlikely)
   - **BT near 0**: No forgetting (ideal for causal model)
   - **Negative BT**: Catastrophic forgetting (expected for correlational model)

4. **Causal Graph Recovery (Causal models only)**:
   - Structural Hamming Distance (SHD) between learned and true graph
   - Lower is better (0 = perfect recovery)

### Expected Results

| Model | CP@100 Period 1 (Initial) | CP@100 Period 2 (FT) | CP@100 Period 1 (After CL) | BT | SHD |
|-------|---------------------------|----------------------|----------------------------|-----|-----|
| Correlational | 0.85 | 0.65 | 0.45 | **-0.40** | N/A |
| Causal | 0.82 | 0.70 | 0.75 | **-0.07** | 3 |
| Causal Oracle | 0.88 | 0.85 | 0.87 | **+0.01** | 0 |

**Key Observation**: Causal model has much smaller backward transfer loss (less forgetting).

### Why Causal Model Performs Better

**Correlational Model Learns**:
- "TX_AMOUNT > 220 → fraud" (Period 1)
- When retrained on Period 2: "TX_AMOUNT ≈ 100 AND TX_FREQUENCY > 5 → fraud"
- **Conflict**: These rules contradict! Network forgets Period 1 pattern.

**Causal Model Learns**:
- Period 1: Graph structure with `COMPROMISED* → TX_AMOUNT`
- Period 2: Graph structure with `COMPROMISED* → TX_FREQUENCY`
- **Invariant**: Both graphs share `COMPROMISED*` as root cause
- Model maintains: "Abnormal deviation from customer baseline → likely compromised"
- **Adaptation**: Learns new *type* of abnormality without forgetting old one

## Implementation Steps

### Step 1: Generate Toy Dataset

```python
# File: experiments/generate_toy_data.py

from simulator import ExperimentConfig, CausalScenarioConfig, DriftConfig
from simulator import generate_causal_dataset

config = ExperimentConfig(
    name="toy_causal_cl",
    simulator=SimulatorConfig(
        n_customers=1000,
        n_terminals=100,
        n_days=60,
        start_date="2024-01-01",
        random_seed=42
    ),
    causal_scenarios=[
        # Period 1: High amount fraud
        CausalScenarioConfig(
            scenario_type="stolen_credentials",
            start_day=0,
            duration=30,
            n_affected_entities=10,  # 10 customers compromised
            scm_parameters={
                'testing_multiplier': 0.5,
                'exploitation_multiplier': 5.0,
                'testing_duration': 3,
                'exploitation_duration': 10
            }
        ),
        # Period 2: High frequency fraud
        CausalScenarioConfig(
            scenario_type="high_frequency_attack",
            start_day=30,
            duration=30,
            n_affected_entities=10,  # 10 NEW customers compromised
            scm_parameters={
                'frequency_multiplier': 4.0,
                'amount_multiplier': 1.2
            }
        )
    ],
    drift=DriftConfig(
        drift_type="sudden",
        drift_schedule=[
            {
                'day': 30,
                'from': 'graph_A_high_amount',
                'to': 'graph_B_high_frequency',
                'reason': 'fraudster_adaptation'
            }
        ]
    )
)

# Generate dataset
customers, terminals, transactions, causal_log = generate_causal_dataset(config)

# Save for reproducibility
transactions.to_pickle("toy_data/transactions.pkl")
with open("toy_data/causal_log.pkl", "wb") as f:
    pickle.dump(causal_log, f)

print(f"Generated {len(transactions)} transactions")
print(f"Fraud rate Period 1: {transactions[transactions.TX_TIME_DAYS < 30].TX_FRAUD.mean():.3f}")
print(f"Fraud rate Period 2: {transactions[transactions.TX_TIME_DAYS >= 30].TX_FRAUD.mean():.3f}")
```

### Step 2: Train Correlational Baseline

```python
# File: experiments/train_correlational.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
transactions = pd.read_pickle("toy_data/transactions.pkl")

# Feature engineering
def extract_features(df):
    features = df[['TX_AMOUNT', 'TX_TIME_SECONDS']].copy()
    # Add customer aggregates (RFM-like features)
    customer_agg = df.groupby('CUSTOMER_ID').agg({
        'TX_AMOUNT': ['mean', 'std'],
        'TX_TIME_DAYS': 'count'
    }).reset_index()
    customer_agg.columns = ['CUSTOMER_ID', 'CUSTOMER_AVG_AMOUNT',
                            'CUSTOMER_STD_AMOUNT', 'CUSTOMER_TX_COUNT']
    features = features.merge(customer_agg, on='CUSTOMER_ID')
    return features

# Split periods
period1_train = transactions[transactions.TX_TIME_DAYS < 30]
period1_test = transactions[transactions.TX_TIME_DAYS < 30]
period2_train = transactions[(transactions.TX_TIME_DAYS >= 30) &
                             (transactions.TX_TIME_DAYS < 45)]
period2_test = transactions[transactions.TX_TIME_DAYS >= 45]

# Phase 1: Train on Period 1
X_train = extract_features(period1_train)
y_train = period1_train['TX_FRAUD']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on Period 1
y_pred = clf.predict_proba(X_train)[:, 1]
cp100_period1_initial = card_precision_top_k(period1_train, y_pred, k=100)
print(f"Period 1 CP@100 (initial): {cp100_period1_initial:.3f}")

# Phase 2: Evaluate on Period 2 WITHOUT retraining (expect failure)
X_period2_test = extract_features(period2_test)
y_pred_period2_before = clf.predict_proba(X_period2_test)[:, 1]
cp100_period2_before_cl = card_precision_top_k(period2_test, y_pred_period2_before, k=100)
print(f"Period 2 CP@100 (before CL): {cp100_period2_before_cl:.3f}")

# Phase 3: Continual Learning - Fine-tune on Period 2
X_period2_train = extract_features(period2_train)
y_period2_train = period2_train['TX_FRAUD']

# Naive fine-tuning
clf.fit(X_period2_train, y_period2_train)  # Overwrites weights

# Evaluate Forward Transfer
y_pred_period2_after = clf.predict_proba(X_period2_test)[:, 1]
cp100_period2_after_cl = card_precision_top_k(period2_test, y_pred_period2_after, k=100)
print(f"Period 2 CP@100 (after CL, FT): {cp100_period2_after_cl:.3f}")

# Evaluate Backward Transfer (forgetting)
y_pred_period1_after = clf.predict_proba(X_train)[:, 1]
cp100_period1_after_cl = card_precision_top_k(period1_test, y_pred_period1_after, k=100)
backward_transfer = cp100_period1_after_cl - cp100_period1_initial
print(f"Period 1 CP@100 (after CL): {cp100_period1_after_cl:.3f}")
print(f"Backward Transfer (forgetting): {backward_transfer:.3f}")
```

### Step 3: Train Causal Model

```python
# File: experiments/train_causal.py

import torch
from models.neural_causal_model import NeuralCausalModel
from simulator.causal.graphs import get_stolen_credentials_graph, get_high_frequency_graph

# Load data and causal log
transactions = pd.read_pickle("toy_data/transactions.pkl")
with open("toy_data/causal_log.pkl", "rb") as f:
    causal_log = pickle.load(f)

# Initialize model with causal structure learning
model = NeuralCausalModel(
    observed_vars=['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_FREQUENCY'],
    latent_vars=['COMPROMISED'],
    structure_learning=True,  # Learn causal graph
    prior_graph=None  # No prior knowledge
)

# Phase 1: Train on Period 1
period1_data = transactions[transactions.TX_TIME_DAYS < 30]
model.fit(period1_data, epochs=50, learn_structure=True)

# Extract learned graph
learned_graph_period1 = model.get_causal_graph()
true_graph_period1 = causal_log['day_15']['active_graph']
shd_period1 = structural_hamming_distance(learned_graph_period1, true_graph_period1)
print(f"SHD Period 1: {shd_period1}")

# Evaluate on Period 1
y_pred_period1 = model.predict_fraud_probability(period1_data)
cp100_period1_initial = card_precision_top_k(period1_data, y_pred_period1, k=100)
print(f"Period 1 CP@100 (initial): {cp100_period1_initial:.3f}")

# Phase 2: Evaluate on Period 2 before retraining
period2_test = transactions[transactions.TX_TIME_DAYS >= 45]
y_pred_period2_before = model.predict_fraud_probability(period2_test)
cp100_period2_before_cl = card_precision_top_k(period2_test, y_pred_period2_before, k=100)
print(f"Period 2 CP@100 (before CL): {cp100_period2_before_cl:.3f}")

# Phase 3: Continual Learning with Causal Structure Preservation
period2_train = transactions[(transactions.TX_TIME_DAYS >= 30) &
                             (transactions.TX_TIME_DAYS < 45)]

# Key difference: Causal model updates structure but preserves mechanisms
model.continual_update(
    new_data=period2_train,
    epochs=50,
    preserve_mechanisms=True,  # Don't forget causal mechanisms from Period 1
    structure_learning=True     # Learn new graph structure
)

# Learned graph for Period 2
learned_graph_period2 = model.get_causal_graph()
true_graph_period2 = causal_log['day_45']['active_graph']
shd_period2 = structural_hamming_distance(learned_graph_period2, true_graph_period2)
print(f"SHD Period 2: {shd_period2}")

# Evaluate Forward Transfer
y_pred_period2_after = model.predict_fraud_probability(period2_test)
cp100_period2_after_cl = card_precision_top_k(period2_test, y_pred_period2_after, k=100)
print(f"Period 2 CP@100 (after CL, FT): {cp100_period2_after_cl:.3f}")

# Evaluate Backward Transfer
y_pred_period1_after = model.predict_fraud_probability(period1_data)
cp100_period1_after_cl = card_precision_top_k(period1_data, y_pred_period1_after, k=100)
backward_transfer = cp100_period1_after_cl - cp100_period1_initial
print(f"Period 1 CP@100 (after CL): {cp100_period1_after_cl:.3f}")
print(f"Backward Transfer: {backward_transfer:.3f}")
```

### Step 4: Visualize Results

```python
# File: experiments/visualize_results.py

import matplotlib.pyplot as plt
import seaborn as sns

results = {
    'Model': ['Correlational', 'Causal', 'Oracle'],
    'Period 1 Initial': [0.85, 0.82, 0.88],
    'Period 2 FT': [0.65, 0.70, 0.85],
    'Period 1 After CL': [0.45, 0.75, 0.87],
    'Backward Transfer': [-0.40, -0.07, 0.01]
}

df_results = pd.DataFrame(results)

# Plot 1: Performance comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Performance over time
ax1 = axes[0]
x = ['Initial (P1)', 'After CL (P2)', 'After CL (P1)']
for model in results['Model']:
    row = df_results[df_results['Model'] == model]
    y = [row['Period 1 Initial'].values[0],
         row['Period 2 FT'].values[0],
         row['Period 1 After CL'].values[0]]
    ax1.plot(x, y, marker='o', label=model)

ax1.set_ylabel('Card Precision@100')
ax1.set_title('Performance During Continual Learning')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Backward Transfer (forgetting)
ax2 = axes[1]
models = results['Model']
bt_values = results['Backward Transfer']
colors = ['red' if bt < -0.2 else 'orange' if bt < 0 else 'green' for bt in bt_values]
ax2.bar(models, bt_values, color=colors)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.set_ylabel('Backward Transfer (Change in CP@100)')
ax2.set_title('Catastrophic Forgetting (Lower is Worse)')
ax2.set_ylim([-0.5, 0.1])

plt.tight_layout()
plt.savefig('results/toy_experiment_results.png', dpi=300)
plt.show()

# Plot 2: Causal Graph Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# True Graph Period 1
ax1 = axes[0]
draw_causal_graph(true_graph_period1, ax=ax1, title="True Graph (Period 1)")

# True Graph Period 2
ax2 = axes[1]
draw_causal_graph(true_graph_period2, ax=ax2, title="True Graph (Period 2)")

# Learned Graph (Causal Model)
ax3 = axes[2]
draw_causal_graph(learned_graph_period2, ax=ax3, title=f"Learned Graph (SHD={shd_period2})")

plt.tight_layout()
plt.savefig('results/causal_graphs.png', dpi=300)
plt.show()
```

## Success Criteria

### Minimal Success (MVP)
- ✅ Correlational model shows >20% performance drop on Period 1 after CL
- ✅ Causal model shows <15% performance drop on Period 1 after CL
- ✅ Clear visualization of performance difference

### Strong Success
- ✅ Causal model shows <10% performance drop (BT > -0.10)
- ✅ Causal model recovers graph structure with SHD < 5
- ✅ Statistical significance across 5 random seeds

### Publication-Ready
- ✅ Causal model shows BT > -0.05 (minimal forgetting)
- ✅ SHD < 3 (near-perfect graph recovery)
- ✅ Ablation: Show that causal structure preservation is key (not just architecture)
- ✅ Counterfactual analysis: "What if fraudsters hadn't switched tactics?"

## Reproducibility

All experiments use fixed random seeds:
- Data generation: `seed=42`
- Model training: `seed=123`
- Evaluation splits: `seed=456`

Configuration files:
- `experiments/configs/toy_experiment.yaml`

## Estimated Timeline

- **Day 1**: Implement basic simulator extraction
- **Day 2**: Implement SCM for Scenario 1 and 2
- **Day 3**: Generate toy dataset, validate
- **Day 4**: Implement correlational baseline
- **Day 5**: Implement causal model (or use existing library)
- **Day 6**: Run experiments, collect results
- **Day 7**: Create visualizations, write up results

## Next Steps After Toy Experiment

If successful:
1. Scale up to full dataset (5000 customers, 90 days)
2. Add third fraud scenario (meta-causal adversarial)
3. Test on real-world dataset (if available)
4. Write paper draft

If unsuccessful (no clear difference):
1. Debug: Are causal graphs actually different?
2. Increase drift severity
3. Try different CL strategies (EWC, PackNet, etc.)
4. Revisit causal model architecture
