# Causal Structures for Fraud Detection

## Overview

This document defines the explicit causal mechanisms for fraud scenarios in the simulator. Each scenario is specified as a Structural Causal Model (SCM) with:
- **Variables**: Observed and latent
- **Causal Graph**: Directed Acyclic Graph (DAG) of causal relationships
- **Functional Equations**: How causes produce effects
- **Noise Distributions**: Exogenous randomness
- **Interventions**: What can be manipulated (do-operator)
- **Counterfactuals**: "What if" queries we can answer

## Notation

- **Observable variables**: UPPERCASE (e.g., TRANSACTION_AMOUNT)
- **Latent variables**: Uppercase with asterisk (e.g., FRAUDSTER_INTENT*)
- **Causal edge**: X → Y means "X causes Y"
- **Do-operator**: do(X=x) means "set X to x by intervention"
- **Counterfactual**: Y_x(u) means "value of Y if we set X=x, given exogenous U=u"

## Causal Scenario 1: Stolen Credentials (Account Takeover)

### Narrative
A fraudster obtains customer credentials (e.g., phishing, data breach). They follow a pattern:
1. **Testing Phase** (days 0-2): Small transactions to verify credentials work
2. **Exploitation Phase** (days 3-10): Large transactions to maximize gain
3. **Abandonment** (days 11-14): Stop to avoid detection

### Variables

**Observed**:
- `CUSTOMER_ID`: Customer identifier
- `TX_AMOUNT`: Transaction amount
- `TX_FREQUENCY`: Transactions per day
- `TX_TIME_SECONDS`: Time of transaction
- `GEOGRAPHIC_DISTANCE`: Distance from customer's usual location (km)
- `TX_FRAUD`: Fraud label (target)

**Latent**:
- `COMPROMISED*`: Binary, are credentials stolen? (unobserved in practice)
- `DAYS_SINCE_COMPROMISE*`: Days since credentials stolen
- `FRAUDSTER_INTENT*`: Intent of actor (0=legitimate user, 1=fraudster)
- `FRAUD_PHASE*`: Current phase (testing/exploitation/abandonment)

### Causal Graph

```
COMPROMISED* ──────────┐
     │                 │
     ↓                 ↓
DAYS_SINCE_COMPROMISE* → FRAUD_PHASE*
     │                 │         │
     │                 ↓         ↓
     │          FRAUDSTER_INTENT* → TX_FRAUD
     │                 │         │
     │                 ↓         ↓
     └──────────→ TX_AMOUNT ←────┤
                       │         │
                       ↓         ↓
              GEOGRAPHIC_DISTANCE │
                       │         │
                       ↓         ↓
                  TX_FREQUENCY   │
                       │         │
                       └─────────┘
```

### Structural Equations

```python
# Latent variables (exogenous)
COMPROMISED* ~ Bernoulli(p=0.01)  # 1% of customers compromised

# If compromised, when?
COMPROMISE_DAY* ~ Uniform(0, n_days) if COMPROMISED* else None

DAYS_SINCE_COMPROMISE* = current_day - COMPROMISE_DAY* if COMPROMISED* else ∞

# Phase transitions (meta-causal)
FRAUD_PHASE* = {
    'testing' if DAYS_SINCE_COMPROMISE* < 3,
    'exploitation' if 3 <= DAYS_SINCE_COMPROMISE* < 11,
    'abandonment' if DAYS_SINCE_COMPROMISE* >= 11
}

# Intent
FRAUDSTER_INTENT* = {
    1 if COMPROMISED* and FRAUD_PHASE* != 'abandonment',
    0 otherwise
}

# Observed variables (endogenous)

# Fraud label
TX_FRAUD = FRAUDSTER_INTENT*

# Transaction amount (causal mechanism)
TX_AMOUNT = {
    if FRAUDSTER_INTENT* == 0:
        # Legitimate: customer's normal pattern
        Normal(μ_customer, σ_customer)

    elif FRAUD_PHASE* == 'testing':
        # Small amounts to test
        Normal(μ_customer * 0.5, σ_customer * 0.3) + U_amount

    elif FRAUD_PHASE* == 'exploitation':
        # Large amounts to maximize gain
        Normal(μ_customer * 5.0, σ_customer * 2.0) + U_amount

    else:
        # Abandonment: no transactions
        0
}

# Geographic distance
GEOGRAPHIC_DISTANCE = {
    if FRAUDSTER_INTENT* == 0:
        Normal(0, 5)  # Stay near home

    elif FRAUD_PHASE* == 'testing':
        Normal(20, 10)  # Moderately far (testing remotely)

    elif FRAUD_PHASE* == 'exploitation':
        Normal(50, 20) + U_distance  # Far (fraudster location)
}

# Transaction frequency
TX_FREQUENCY = {
    if FRAUDSTER_INTENT* == 0:
        Poisson(λ_customer)

    elif FRAUD_PHASE* == 'testing':
        Poisson(λ_customer * 2)  # Slightly more frequent

    elif FRAUD_PHASE* == 'exploitation':
        Poisson(λ_customer * 5)  # Rush to cash out
}

# Time of day
TX_TIME_SECONDS = {
    if FRAUDSTER_INTENT* == 0:
        Normal(43200, 20000)  # Normal: around noon

    else:
        Uniform(0, 86400)  # Fraudster: any time (different timezone)
}
```

### Exogenous Noise
- `U_amount ~ Normal(0, 5)`: Random variation in amounts
- `U_distance ~ Normal(0, 10)`: Random variation in location
- `U_time ~ Uniform(-3600, 3600)`: Random time jitter

### Interventions (Do-Calculus)

**Intervention 1**: Early Detection
```
do(FRAUD_PHASE* = 'abandonment') at day 5

Effect: Fraudster stops before exploitation phase
Measurable outcome: Reduced total fraud amount per compromised account
```

**Intervention 2**: Geographic Blocking
```
do(GEOGRAPHIC_DISTANCE > 100 → TX_FRAUD_BLOCKED = 1)

Effect: Block transactions far from customer location
Measurable outcome: Prevent exploitation phase transactions
```

### Counterfactual Queries

**Query 1**: "If we had detected the fraud on day 2, what would total fraud amount be?"
```
TX_AMOUNT_{do(FRAUD_PHASE*='abandonment' at day 2)}(U)

Answer: Sum of amounts in testing phase only (~3 transactions * 0.5 * μ_customer)
vs. Observed: Testing + Exploitation amounts
```

**Query 2**: "Would this transaction be fraud if the customer had not been compromised?"
```
TX_FRAUD_{do(COMPROMISED*=0)}(U)

Answer: 0 (definitionally, no compromise → no fraud in this scenario)
```

### Causal Invariance Property

**Claim**: The *mechanism* "compromised credentials → testing → exploitation → high amounts" is invariant even as specific amount thresholds change.

**Test**: If fraudsters change from 5x multiplier to 3x multiplier (distribution shift), the causal graph structure remains the same. A model that learns the graph will adapt faster than one that memorizes "amount > 220 is fraud".

---

## Causal Scenario 2: Terminal Compromise

### Narrative
A fraudster compromises a payment terminal (e.g., skimming device installed). All transactions at this terminal in the next 28 days have risk of fraud.

### Variables

**Observed**:
- `TERMINAL_ID`: Terminal identifier
- `TX_AMOUNT`: Transaction amount
- `TERMINAL_TRANSACTION_VOLUME`: Daily transaction count at terminal
- `TIME_OF_DAY`: Hour of transaction (0-23)
- `TX_FRAUD`: Fraud label

**Latent**:
- `TERMINAL_COMPROMISED*`: Binary, is terminal compromised?
- `COMPROMISE_DAY*`: Day terminal was compromised
- `TERMINAL_SECURITY_LEVEL*`: Security of terminal (low/medium/high)
- `FRAUDSTER_SKILL*`: Ability to exploit terminal

### Causal Graph

```
TERMINAL_SECURITY_LEVEL* ────┐
           │                 │
           ↓                 ↓
    TERMINAL_COMPROMISED* ───┴→ FRAUDSTER_SKILL*
           │                         │
           ↓                         ↓
    COMPROMISE_DAY*          TX_FRAUD_PROBABILITY
           │                         │
           ↓                         ↓
TERMINAL_TRANSACTION_VOLUME → TX_FRAUD
           │                         │
           ↓                         ↓
      TIME_OF_DAY ──────────────────┘
```

### Structural Equations

```python
# Latent variables
TERMINAL_SECURITY_LEVEL* ~ Categorical(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])

TERMINAL_COMPROMISED* ~ Bernoulli(p = {
    0.1 if TERMINAL_SECURITY_LEVEL* == 'low',
    0.02 if TERMINAL_SECURITY_LEVEL* == 'medium',
    0.001 if TERMINAL_SECURITY_LEVEL* == 'high'
})

COMPROMISE_DAY* ~ Uniform(0, n_days) if TERMINAL_COMPROMISED* else None

FRAUDSTER_SKILL* ~ {
    Normal(0.8, 0.1) if TERMINAL_SECURITY_LEVEL* == 'low',
    Normal(0.5, 0.1) if TERMINAL_SECURITY_LEVEL* == 'medium',
    Normal(0.2, 0.1) if TERMINAL_SECURITY_LEVEL* == 'high'
}

# Observed variables
DAYS_SINCE_TERMINAL_COMPROMISE = current_day - COMPROMISE_DAY* if TERMINAL_COMPROMISED* else ∞

ACTIVE_COMPROMISE = (TERMINAL_COMPROMISED* == 1) and (0 <= DAYS_SINCE_TERMINAL_COMPROMISE < 28)

# Fraud probability depends on terminal volume and time
TX_FRAUD_PROBABILITY = {
    if not ACTIVE_COMPROMISE:
        0

    else:
        base_prob = FRAUDSTER_SKILL*
        volume_factor = min(1.0, TERMINAL_TRANSACTION_VOLUME / 100)  # High volume → easier to hide
        time_factor = 1.5 if 22 <= TIME_OF_DAY or TIME_OF_DAY <= 6 else 1.0  # Night → higher risk

        min(0.95, base_prob * volume_factor * time_factor)
}

TX_FRAUD ~ Bernoulli(TX_FRAUD_PROBABILITY)

# Amount not directly caused by terminal compromise (fraudster uses stolen real cards)
TX_AMOUNT ~ Normal(μ_terminal, σ_terminal)  # Independent of fraud label
```

### Key Causal Property

**Important**: Unlike Scenario 1, `TX_AMOUNT` is NOT a child of `TX_FRAUD`. This is because terminal compromise captures real customer card data, so amounts appear normal.

This creates a **confounding** challenge: amount alone cannot distinguish fraud.

### Interventions

**Intervention 1**: Terminal Security Upgrade
```
do(TERMINAL_SECURITY_LEVEL* = 'high') for all terminals

Effect: Reduces P(TERMINAL_COMPROMISED*)
Measurable: Fraud rate decreases
```

**Intervention 2**: Real-Time Monitoring
```
do(TX_FRAUD_BLOCKED = 1) if TERMINAL_TRANSACTION_VOLUME_ANOMALY detected

Effect: Detect compromised terminals faster
Measurable: Reduce DAYS_SINCE_TERMINAL_COMPROMISE before detection
```

### Counterfactual Queries

**Query**: "Would this transaction be fraud if the terminal had high security?"
```
TX_FRAUD_{do(TERMINAL_SECURITY_LEVEL*='high')}(U)

Likely answer: No, because P(TERMINAL_COMPROMISED*) would be 0.001 instead of 0.1
```

---

## Causal Scenario 3: Adversarial Adaptation (Meta-Causal)

### Narrative
Fraudsters observe detection patterns and adapt their tactics. This is a **meta-causal** scenario where the causal graph itself changes based on interventions (detection).

### Meta-Causal Graph

```
           Detection at day 30
                   ↓
    [Graph A: High Amount] ──(switch)──> [Graph B: High Frequency]
           │                                     │
           ↓                                     ↓
    TX_AMOUNT → TX_FRAUD          TX_FREQUENCY → TX_FRAUD
```

### Timeline

**Days 0-30**: Graph A active
- Fraudsters use high-amount strategy
- `TX_AMOUNT > 5 * μ_customer → TX_FRAUD = 1`

**Day 30**: Detection threshold reached
- System detects 70% of high-amount frauds
- Fraudsters observe this (meta-causal intervention)

**Days 31-60**: Graph B active
- Fraudsters switch to high-frequency, low-amount strategy
- `TX_FREQUENCY > 3 * λ_customer AND TX_AMOUNT < 2 * μ_customer → TX_FRAUD = 1`

### Formal Meta-Causal Model

```python
class MetaCausalState:
    def __init__(self):
        self.current_graph = GraphA()
        self.detection_history = []

    def observe_detection_rate(self, day, rate):
        """Fraudsters observe detection (intervention on their side)"""
        self.detection_history.append((day, rate))

        if rate > 0.7 and day == 30:
            # Meta-causal switch
            self.switch_graph(GraphA(), GraphB())

    def switch_graph(self, old_graph, new_graph):
        """Change causal structure"""
        self.current_graph = new_graph
        # Log for ground truth
        log_meta_causal_transition(old_graph, new_graph)

# Graph A
class GraphA(CausalGraph):
    def mechanism(self, customer):
        if random.random() < 0.01:  # Compromise probability
            return {
                'TX_AMOUNT': customer.mean_amount * 5.0,
                'TX_FRAUD': 1
            }
        return legitimate_transaction(customer)

# Graph B
class GraphB(CausalGraph):
    def mechanism(self, customer):
        if random.random() < 0.01:  # Same compromise rate
            return {
                'TX_AMOUNT': customer.mean_amount * 1.2,  # Slightly higher
                'TX_FREQUENCY': customer.mean_frequency * 4.0,  # Much more frequent
                'TX_FRAUD': 1
            }
        return legitimate_transaction(customer)
```

### Why This Tests Continual Learning

**Correlational Model**:
- Learns: "TX_AMOUNT > 220 → fraud" in days 0-30
- Days 31-60: **Catastrophic forgetting** - forgets old pattern OR fails to learn new pattern
- Performance drops on BOTH old and new periods

**Causal Model**:
- Learns: Graph structure with "Compromise → Abnormal Behavior → Fraud"
- Days 31-60: Recognizes *different abnormality* (frequency vs. amount)
- Maintains knowledge: "Deviation from customer norm indicates fraud"
- **Less forgetting** because causal mechanism is invariant

### Interventions in Meta-Causal Setting

**Intervention**: Adaptive Detection
```
If detect_fraud_rate(GraphA) > 0.7:
    do(alert_manual_review = True)

Effect on fraudsters:
    Observe intervention → switch to GraphB

Effect on detector:
    Must adapt to new graph without forgetting GraphA
```

---

## Summary Table

| Scenario | Key Causal Link | Invariant Property | Drift Type | Challenge for ML |
|----------|----------------|-------------------|------------|------------------|
| Stolen Credentials | COMPROMISED* → FRAUD_PHASE* → TX_AMOUNT | Phase transition logic | Gradual (parameter shift) | Learn temporal patterns |
| Terminal Compromise | TERMINAL_COMPROMISED* → TX_FRAUD_PROBABILITY | Security-risk relationship | Sudden (new terminals) | Confounding (amount independent) |
| Adversarial Adaptation | DETECTION → GRAPH_SWITCH → NEW_MECHANISM | Fraudster rationality | Meta-causal (graph switch) | Catastrophic forgetting |

## Evaluation Metrics for Causal Models

### 1. Causal Graph Recovery
- **Structural Hamming Distance (SHD)**: Distance between learned and true graph
- **Precision/Recall of edges**: How many causal edges correctly identified?

### 2. Counterfactual Accuracy
- Generate counterfactual samples from true SCM
- Test if model predictions match counterfactuals

### 3. Invariance Under Intervention
- Test if model maintains performance when intervening on non-descendants of TX_FRAUD
- Example: Changing TX_TIME_SECONDS should not affect fraud prediction (not a cause)

### 4. Meta-Causal Adaptation
- Speed of adaptation after graph switch
- Retention of performance on old graph

## Next Steps

1. Implement SCMs for each scenario in `simulator/causal/mechanisms.py`
2. Create visualization scripts for causal graphs
3. Generate toy dataset with ground truth causal labels
4. Test causal discovery algorithms (PC, GES, etc.)
5. Compare learned vs. true graphs

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*
- Willig et al. (2025). *Systems with Switching Causal Relations*
