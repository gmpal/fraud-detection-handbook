# Causal Continual Learning for Fraud Detection - Roadmap

## Project Overview

**Goal**: Demonstrate that causal models are more robust to concept drift in fraud detection compared to correlation-based models, within a continual learning framework.

**Core Hypothesis**: If a model learns the *causal mechanisms* of fraud (the "why"), it will be more resilient to distribution shifts than models that only recognize correlational patterns, and will suffer less from catastrophic forgetting.

## Background

### The Problem
- **Concept Drift**: Fraudsters constantly change tactics, making old patterns obsolete
- **Catastrophic Forgetting**: Models learning new fraud patterns forget old ones
- **Spurious Correlations**: Traditional ML models overfit to temporary correlations that vanish when fraudsters adapt

### The Solution
- **Causal Invariance**: Causal relationships are more stable across distribution shifts
- **Meta-Causal Models**: Model switching causal relations as fraudster tactics evolve
- **Continual Learning with Causal Structure**: Leverage causal knowledge to prevent forgetting

## Roadmap Phases

### Phase 1: Simulator Extraction & Modularization (Week 1-2)
**Goal**: Extract and refactor the fraud simulator for causal experiments

- [ ] Extract core simulator code from Chapter 3 notebooks
- [ ] Create standalone Python modules:
  - `simulator/core.py` - Customer/terminal/transaction generation
  - `simulator/fraud_scenarios.py` - Fraud scenario definitions
  - `simulator/causal_scenarios.py` - NEW: Causal fraud mechanisms
- [ ] Add configuration system for reproducible experiments
- [ ] Document simulator API and usage

**Deliverable**: `simulator/` package with modular, reusable code

### Phase 2: Causal Structure Design (Week 2-3)
**Goal**: Define explicit causal mechanisms for fraud scenarios

- [ ] Design Causal Graph 1: "Stolen Credentials" mechanism
  - Customer behavior → Transaction pattern
  - Fraudster access → Abnormal amounts/locations
  - Time since compromise → Risk level

- [ ] Design Causal Graph 2: "Terminal Compromise" mechanism
  - Terminal characteristics → Fraud vulnerability
  - Temporal patterns → Detection difficulty
  - Geographic clustering → Spread dynamics

- [ ] Design Causal Graph 3: "Account Takeover Evolution" mechanism
  - Initial test transactions → Confirmation
  - Risk assessment → Transaction limits
  - Detection attempts → Tactic changes (META-CAUSAL)

- [ ] Implement structural causal models (SCMs) in simulator
- [ ] Add intervention capabilities (do-operator)

**Deliverable**: `CAUSAL_STRUCTURES.md` documenting all causal graphs

### Phase 3: Concept Drift Scenarios (Week 3-4)
**Goal**: Create realistic concept drift through meta-causal switching

- [ ] Scenario 1: "Gradual Adaptation"
  - Fraudsters slowly change transaction amounts (50→100→150)
  - Causal mechanism stays same, parameters shift

- [ ] Scenario 2: "Tactic Switch"
  - Fraudsters change from high-amount to high-frequency attacks
  - Causal graph structure changes (meta-causal transition)

- [ ] Scenario 3: "Adversarial Response"
  - Fraudsters adapt based on detection patterns
  - Feedback loop: detection → tactic change → new detection

- [ ] Implement temporal windows for fraud scenario evolution
- [ ] Add ground-truth causal graph switching log

**Deliverable**: `CONCEPT_DRIFT_SCENARIOS.md` with detailed specifications

### Phase 4: Baseline Models (Week 4-5)
**Goal**: Implement correlational and causal baselines

- [ ] **Correlational Baseline**:
  - Standard Random Forest / XGBoost on features
  - No causal structure, pure pattern recognition
  - Continual learning via replay buffer or EWC

- [ ] **Causal Baseline**:
  - Neural Causal Model (following Busch & Seng papers)
  - Learns causal graph structure
  - Intervention-aware training

- [ ] **Hybrid Model**:
  - Causal feature engineering + standard classifier
  - Tests benefit of causal features alone

**Deliverable**: `models/` directory with baseline implementations

### Phase 5: Continual Learning Framework (Week 5-6)
**Goal**: Implement CL evaluation protocol

- [ ] Design temporal evaluation protocol:
  - Train on Period 1 (Causal Graph A)
  - Evaluate on Period 1 (sanity check)
  - Train on Period 2 (Causal Graph B) - CONCEPT DRIFT
  - Evaluate on Period 1 (measure forgetting)
  - Evaluate on Period 2 (measure adaptation)

- [ ] Implement CL strategies:
  - Naive fine-tuning (baseline)
  - Experience Replay
  - Elastic Weight Consolidation (EWC)
  - Causal-aware replay (prioritize causal mechanisms)

- [ ] Metrics:
  - Card Precision@k (CP@k) per period
  - Forward Transfer (FT): Performance on new task
  - Backward Transfer (BT): Performance retention on old task
  - Causal Graph Recovery: Accuracy of learned structure

**Deliverable**: `experiments/continual_learning.py` evaluation harness

### Phase 6: Toy Experiment (Week 6-7)
**Goal**: Small-scale proof of concept

**Setup**:
- 2 time periods (30 days each)
- 1000 customers, 100 terminals
- 2 fraud scenarios with causal graph switch at day 30

**Experiment**:
1. Generate data with known causal structure
2. Train correlational model on Period 1
3. Train causal model on Period 1
4. Introduce concept drift at Period 2
5. Measure:
   - CP@k on Period 1 (before drift)
   - CP@k on Period 2 (after drift)
   - CP@k on Period 1 after retraining (forgetting)

**Expected Result**: Causal model maintains better performance on Period 1 after learning Period 2 (less forgetting), and adapts faster to Period 2 (understands mechanism not just pattern).

**Deliverable**: `experiments/toy_experiment.ipynb` with results

### Phase 7: Analysis & Documentation (Week 7-8)
**Goal**: Interpret results and prepare for paper

- [ ] Generate visualizations:
  - Causal graphs over time
  - Performance degradation curves
  - Forgetting vs. adaptation trade-off

- [ ] Counterfactual analysis:
  - "What if model knew causal structure?"
  - "What if no concept drift occurred?"

- [ ] Write experiment report
- [ ] Update CLAUDE.md with causal extensions

**Deliverable**: `RESULTS.md` with analysis and figures

## Success Criteria

**Minimal Success**:
- Causal model shows <10% performance drop on old task after learning new task
- Correlational model shows >20% performance drop
- Clear visualization of causal graph switching

**Strong Success**:
- Causal model recovers true causal structure with >80% accuracy
- Demonstrates faster adaptation to new fraud tactics
- Counterfactual interventions improve precision by >15%

**Publication-Ready**:
- Statistical significance across multiple random seeds
- Ablation studies on causal structure components
- Real-world dataset validation (if available)

## Technical Dependencies

### New Dependencies to Add
```
pip install causalnex  # For causal graph learning
pip install dowhy      # For causal inference
pip install networkx   # For graph manipulation
pip install pgmpy      # For probabilistic graphical models
```

### Hardware Requirements
- Toy experiment: Standard laptop (CPU sufficient)
- Full experiment: GPU recommended for neural causal models

## Risk Mitigation

**Risk**: Simulator too simple, causal structure obvious
- **Mitigation**: Add noise, confounders, and latent variables

**Risk**: Causal discovery fails to recover structure
- **Mitigation**: Start with known structure (oracle model), show upper bound

**Risk**: No clear benefit over correlational models
- **Mitigation**: Design adversarial drift specifically to break correlational patterns

## Next Steps (Immediate)

1. Create `simulator/` package structure
2. Extract transaction generator to `simulator/core.py`
3. Design first causal graph for "Stolen Credentials" scenario
4. Write `CAUSAL_STRUCTURES.md` documentation
5. Run toy simulator test to validate extraction

## References to Papers

- **Meta-Causal Models**: Willig et al. - Systems with Switching Causal Relations
- **Treatment Effects in CL**: Seng et al. - Treatment Effect Estimation to Guide Model Optimization
- **Neural Causal Models**: Busch & Seng - Continually Updating Neural Causal Models
- **Meta-Causal Interventions**: Kersting et al. - When Causal Dynamics Matter

## Questions for Discussion

1. Should we model fraudster intent as a latent variable?
2. How fine-grained should temporal windows be? (daily vs. weekly)
3. Should causal graphs be fully observable or partially learned?
4. What level of causal complexity is needed for convincing demo?
