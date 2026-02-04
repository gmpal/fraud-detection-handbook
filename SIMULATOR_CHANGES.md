# Simulator Changes for Causal Continual Learning

## Overview

This document specifies the changes needed to adapt the existing fraud detection simulator for causal continual learning experiments.

## Current Simulator Architecture

### Existing Components (from Chapter 3)
```
generate_customer_profiles_table()
  ↓
generate_terminal_profiles_table()
  ↓
get_list_terminals_within_radius()
  ↓
generate_transactions_table()
  ↓
add_frauds() → Three fraud scenarios:
  1. Amount > 220
  2. Compromised terminals (28 days)
  3. Compromised customers (14 days, 5x amount)
```

### Limitations for Causal Analysis
1. **No explicit causal structure**: Fraud rules are hardcoded conditionals
2. **No mechanistic reasoning**: Cannot model "why" a fraud occurs
3. **No interventional semantics**: Cannot test counterfactuals
4. **Static scenarios**: Fraud patterns don't evolve over time
5. **No latent variables**: Customer intent, fraudster sophistication not modeled

## Proposed New Architecture

### Module Structure
```
simulator/
├── __init__.py
├── core.py                    # Refactored from Chapter 3
│   ├── CustomerProfile
│   ├── TerminalProfile
│   ├── TransactionGenerator
│   └── DatasetGenerator
├── causal/
│   ├── __init__.py
│   ├── scm.py                 # Structural Causal Models
│   ├── graphs.py              # Causal graph definitions
│   ├── interventions.py       # Do-calculus operations
│   └── mechanisms.py          # Causal mechanisms for fraud
├── fraud_scenarios.py         # Original scenarios (legacy)
├── causal_scenarios.py        # NEW: Causal fraud scenarios
├── drift.py                   # NEW: Concept drift mechanisms
├── utils.py                   # Helper functions
└── config.py                  # Configuration management
```

## Key Changes

### 1. Structural Causal Models (SCM) Implementation

**File**: `simulator/causal/scm.py`

```python
class StructuralCausalModel:
    """
    Represents a causal mechanism using functional equations.

    X_i = f_i(PA_i, U_i)
    where PA_i are parents of X_i, U_i is exogenous noise
    """
    def __init__(self, variables, equations, noise_distributions):
        self.variables = variables
        self.equations = equations  # Dict: variable → function
        self.noise = noise_distributions
        self.graph = None  # Causal DAG

    def sample(self, n_samples, interventions=None):
        """Generate samples, optionally with do() interventions"""
        pass

    def counterfactual(self, evidence, intervention, query):
        """Answer counterfactual queries"""
        pass
```

**Example: Stolen Credentials SCM**
```python
Variables:
- CUSTOMER_LEGITIMATE: Binary, is customer making transaction?
- DAYS_SINCE_COMPROMISE: Continuous, time since credentials stolen
- TRANSACTION_AMOUNT: Continuous
- TRANSACTION_FREQUENCY: Continuous (txs per day)
- IS_FRAUD: Binary

Equations:
1. IS_FRAUD = f(CUSTOMER_LEGITIMATE, U_fraud)
   - If CUSTOMER_LEGITIMATE = 0 → IS_FRAUD = 1

2. TRANSACTION_AMOUNT = f(IS_FRAUD, DAYS_SINCE_COMPROMISE, U_amount)
   - If IS_FRAUD = 1 and DAYS_SINCE_COMPROMISE < 3:
       amount ~ Normal(mean_customer * 1.5, high_variance)  # Testing phase
   - If IS_FRAUD = 1 and DAYS_SINCE_COMPROMISE >= 3:
       amount ~ Normal(mean_customer * 5, low_variance)    # Exploitation phase
   - Else:
       amount ~ Normal(mean_customer, std_customer)        # Legitimate

3. TRANSACTION_FREQUENCY = f(IS_FRAUD, DAYS_SINCE_COMPROMISE, U_freq)
   - If IS_FRAUD = 1 and DAYS_SINCE_COMPROMISE < 7:
       freq = customer_mean_freq * 3  # Rush to exploit before detection
   - Else:
       freq = customer_mean_freq
```

### 2. Causal Graph Definitions

**File**: `simulator/causal/graphs.py`

```python
import networkx as nx

class CausalGraph:
    """Represents causal structure as DAG"""
    def __init__(self, name):
        self.name = name
        self.graph = nx.DiGraph()

    def add_edge(self, cause, effect, mechanism=None):
        """Add causal edge with optional mechanism function"""
        self.graph.add_edge(cause, effect, mechanism=mechanism)

    def visualize(self, filepath=None):
        """Plot causal graph"""
        pass

    def to_scm(self):
        """Convert to executable SCM"""
        pass

# Define fraud scenario graphs
def get_stolen_credentials_graph():
    """
    Causal graph for stolen credentials scenario

    CUSTOMER_LEGITIMATE → IS_FRAUD
    DAYS_SINCE_COMPROMISE → TRANSACTION_AMOUNT
    DAYS_SINCE_COMPROMISE → TRANSACTION_FREQUENCY
    IS_FRAUD → TRANSACTION_AMOUNT
    IS_FRAUD → TRANSACTION_FREQUENCY
    TERMINAL_LOCATION → GEOGRAPHIC_RISK
    GEOGRAPHIC_RISK → IS_FRAUD
    """
    graph = CausalGraph("stolen_credentials")
    graph.add_edge("CUSTOMER_LEGITIMATE", "IS_FRAUD")
    graph.add_edge("DAYS_SINCE_COMPROMISE", "TRANSACTION_AMOUNT")
    # ... etc
    return graph

def get_terminal_compromise_graph():
    """
    Causal graph for terminal compromise

    TERMINAL_SECURITY_LEVEL → IS_COMPROMISED
    IS_COMPROMISED → FRAUD_PROBABILITY
    TERMINAL_TRANSACTION_VOLUME → DETECTION_DIFFICULTY
    TIME_OF_DAY → FRAUD_OPPORTUNITY
    """
    pass
```

### 3. Causal Fraud Scenarios

**File**: `simulator/causal_scenarios.py`

```python
class CausalFraudScenario:
    """Base class for causal fraud scenarios"""
    def __init__(self, scm, start_day, duration):
        self.scm = scm
        self.start_day = start_day
        self.duration = duration
        self.active = True

    def apply_to_transaction(self, transaction, current_day):
        """Apply causal mechanism to determine if fraud"""
        if not self.is_active(current_day):
            return transaction

        # Sample from SCM given transaction features
        fraud_label = self.scm.sample(
            evidence={
                'TRANSACTION_AMOUNT': transaction['TX_AMOUNT'],
                'CUSTOMER_ID': transaction['CUSTOMER_ID'],
                # ... etc
            }
        )
        transaction['TX_FRAUD'] = fraud_label
        return transaction

    def is_active(self, day):
        return self.start_day <= day < self.start_day + self.duration

class StolenCredentialsScenario(CausalFraudScenario):
    """
    Scenario: Fraudster steals customer credentials

    Causal mechanism:
    1. Compromise event occurs (latent)
    2. Testing phase: Small transactions to verify
    3. Exploitation phase: Large transactions to maximize gain
    4. Abandonment: Stop before detection

    Parameters evolve over time (meta-causal switching)
    """
    def __init__(self, customer_ids, compromise_day, **kwargs):
        scm = self._build_scm()
        super().__init__(scm, compromise_day, duration=14)
        self.customer_ids = customer_ids
        self.phase = "testing"  # testing → exploitation → abandonment

    def _build_scm(self):
        """Construct SCM for this scenario"""
        # Define functional equations
        pass

    def update_phase(self, days_since_compromise):
        """Meta-causal: Phase transitions"""
        if days_since_compromise < 3:
            self.phase = "testing"
        elif days_since_compromise < 10:
            self.phase = "exploitation"
        else:
            self.phase = "abandonment"
            self.active = False
```

### 4. Concept Drift Mechanisms

**File**: `simulator/drift.py`

```python
class ConceptDriftManager:
    """Manages temporal evolution of causal structures"""
    def __init__(self, scenarios, drift_schedule):
        self.scenarios = scenarios
        self.drift_schedule = drift_schedule  # When to switch graphs
        self.current_graph = None

    def get_active_scenarios(self, day):
        """Return active fraud scenarios for given day"""
        active = []
        for scenario_name, config in self.drift_schedule.items():
            if config['start_day'] <= day < config['end_day']:
                active.append(self.scenarios[scenario_name])
        return active

    def log_graph_switch(self, day, old_graph, new_graph):
        """Record meta-causal transition for ground truth"""
        pass

class MetaCausalDrift:
    """
    Models adversarial adaptation: fraudsters change tactics
    based on detection patterns
    """
    def __init__(self, detection_threshold=0.7):
        self.detection_threshold = detection_threshold
        self.detection_history = []

    def should_switch_tactic(self, recent_detections):
        """
        If detection rate > threshold, switch causal mechanism

        Example: If high-amount fraud is detected often,
        switch to high-frequency low-amount fraud
        """
        detection_rate = np.mean(recent_detections)
        return detection_rate > self.detection_threshold

    def adapt_scm(self, old_scm, detection_pattern):
        """
        Generate new SCM based on what's being detected

        This is the core meta-causal operation:
        The causal graph itself changes based on interventions
        (detection is an intervention from fraudster's perspective)
        """
        pass
```

### 5. Enhanced Transaction Generator

**File**: `simulator/core.py` (modifications)

```python
class CausalTransactionGenerator:
    """
    Extended transaction generator with causal mechanisms
    """
    def __init__(self, customer_profiles, terminal_profiles,
                 causal_scenarios, drift_manager):
        self.customer_profiles = customer_profiles
        self.terminal_profiles = terminal_profiles
        self.causal_scenarios = causal_scenarios
        self.drift_manager = drift_manager
        self.causal_graph_log = []  # Ground truth for evaluation

    def generate_transactions(self, start_date, n_days):
        """Generate transactions with causal fraud labels"""
        transactions = []

        for day in range(n_days):
            # Get active causal scenarios for this day
            active_scenarios = self.drift_manager.get_active_scenarios(day)

            # Log which causal graph is active (ground truth)
            self.causal_graph_log.append({
                'day': day,
                'active_graphs': [s.scm.graph for s in active_scenarios]
            })

            # Generate legitimate transactions
            day_transactions = self._generate_legitimate_transactions(day)

            # Apply causal fraud scenarios
            for transaction in day_transactions:
                for scenario in active_scenarios:
                    transaction = scenario.apply_to_transaction(
                        transaction, day
                    )

            transactions.extend(day_transactions)

        return pd.DataFrame(transactions), self.causal_graph_log
```

## Configuration System

**File**: `simulator/config.py`

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SimulatorConfig:
    """Configuration for reproducible experiments"""
    n_customers: int = 5000
    n_terminals: int = 10000
    n_days: int = 90
    start_date: str = "2018-04-01"
    random_seed: int = 42

@dataclass
class CausalScenarioConfig:
    scenario_type: str  # "stolen_credentials", "terminal_compromise", etc.
    start_day: int
    duration: int
    n_affected_entities: int  # customers or terminals
    scm_parameters: Dict  # Specific to scenario type

@dataclass
class DriftConfig:
    drift_type: str  # "gradual", "sudden", "adversarial"
    drift_schedule: List[Dict]  # When graphs switch
    # Example: [
    #   {'day': 30, 'from': 'graph_A', 'to': 'graph_B'},
    #   {'day': 60, 'from': 'graph_B', 'to': 'graph_C'}
    # ]

@dataclass
class ExperimentConfig:
    name: str
    simulator: SimulatorConfig
    causal_scenarios: List[CausalScenarioConfig]
    drift: DriftConfig

    def to_yaml(self, filepath):
        """Save config for reproducibility"""
        pass

    @classmethod
    def from_yaml(cls, filepath):
        """Load config"""
        pass
```

## Backward Compatibility

To maintain compatibility with existing handbook code:

```python
# simulator/__init__.py
from .core import (
    generate_customer_profiles_table,
    generate_terminal_profiles_table,
    generate_dataset  # Original function preserved
)

# NEW: Causal-aware function
from .core import generate_causal_dataset

# Example usage:
# Old way (still works):
customers, terminals, txs = generate_dataset(n_customers=5000, ...)

# New way (causal):
customers, terminals, txs, causal_log = generate_causal_dataset(
    config=ExperimentConfig(...)
)
```

## Testing Strategy

1. **Unit tests**: Each SCM produces valid samples
2. **Integration tests**: Causal scenarios apply correctly
3. **Validation tests**: Recovered causal structure matches ground truth
4. **Regression tests**: Original simulator behavior preserved

## Implementation Priority

### Phase 1 (Immediate)
1. Extract core simulator to `simulator/core.py`
2. Create simple SCM class
3. Implement one causal scenario (Stolen Credentials)
4. Basic configuration system

### Phase 2 (Week 2)
5. Add causal graph definitions
6. Implement drift manager
7. Create ground truth logging

### Phase 3 (Week 3)
8. Meta-causal switching
9. Intervention operations
10. Counterfactual queries

## Example: Running the New Simulator

```python
from simulator import ExperimentConfig, CausalScenarioConfig, DriftConfig
from simulator import generate_causal_dataset

# Define experiment
config = ExperimentConfig(
    name="toy_causal_drift",
    simulator=SimulatorConfig(
        n_customers=1000,
        n_terminals=100,
        n_days=60,
        random_seed=42
    ),
    causal_scenarios=[
        CausalScenarioConfig(
            scenario_type="stolen_credentials",
            start_day=0,
            duration=30,
            n_affected_entities=50,
            scm_parameters={'exploitation_multiplier': 5.0}
        ),
        CausalScenarioConfig(
            scenario_type="terminal_compromise",
            start_day=30,
            duration=30,
            n_affected_entities=10,
            scm_parameters={'compromise_probability': 0.8}
        )
    ],
    drift=DriftConfig(
        drift_type="sudden",
        drift_schedule=[
            {'day': 30, 'from': 'stolen_credentials', 'to': 'terminal_compromise'}
        ]
    )
)

# Generate data
customers, terminals, transactions, causal_log = generate_causal_dataset(config)

# causal_log contains ground truth:
# - Which causal graph was active each day
# - Which SCM parameters were used
# - Which customers/terminals were affected
# - Counterfactual: what would have happened without fraud

print(causal_log['day_30'])
# Output:
# {
#   'active_graph': 'terminal_compromise',
#   'graph_structure': <networkx.DiGraph>,
#   'affected_terminals': [23, 45, 67, ...],
#   'scm_samples': [...],
#   'interventions': None
# }
```

## Benefits of New Architecture

1. **Explicit Causality**: Can test causal hypotheses
2. **Interpretability**: Understand *why* fraud occurs, not just correlations
3. **Interventions**: Test counterfactuals ("what if we detected earlier?")
4. **Meta-Causal**: Model adversarial adaptation
5. **Ground Truth**: Know true causal structure for evaluation
6. **Reproducibility**: Config-driven experiments
7. **Modularity**: Easy to add new causal scenarios

## Next Steps

1. Review this specification with team
2. Create `simulator/` directory structure
3. Begin Phase 1 implementation
4. Write unit tests for SCM
5. Validate with simple toy example
