# Quick Start Guide: Causal Continual Learning Project

**For**: Quickly understanding and starting the project
**Time to read**: 5 minutes

## What is this project?

Testing if **causal models** are more robust than **correlational models** when fraudsters change tactics (concept drift) in a continual learning setting.

**Key Idea**: Models that understand *why* fraud happens (causal mechanisms) should forget less when learning new fraud patterns.

## The Hypothesis

```
Correlational Model:
  Learns: "Amount > 220 → fraud"
  When fraudsters switch to high-frequency attacks → FORGETS old pattern ❌

Causal Model:
  Learns: "Compromised credentials → abnormal behavior → fraud"
  When fraudsters switch tactics → REMEMBERS both patterns ✓
```

## Project Structure

```
fraud-detection-handbook/
├── ROADMAP.md              ← Full 8-week plan
├── SIMULATOR_CHANGES.md    ← How to modify the simulator
├── CAUSAL_STRUCTURES.md    ← Detailed causal graphs
├── TOY_EXPERIMENT.md       ← Small proof-of-concept
├── PROJECT_STATUS.md       ← Current progress tracking
├── Notes.md                ← Research notes
└── QUICK_START.md          ← This file
```

## Core Concepts

### 1. Structural Causal Model (SCM)
A system of equations that describes how variables cause each other:

```
COMPROMISED* → FRAUD_PHASE* → TX_AMOUNT → TX_FRAUD
                               ↓
                         GEOGRAPHIC_DISTANCE
```

### 2. Concept Drift
Fraudsters change tactics over time:
- **Days 0-29**: High amount attacks (5x normal)
- **Day 30**: DRIFT - Fraudsters switch tactics
- **Days 31-59**: High frequency attacks (4x normal frequency)

### 3. Continual Learning Metrics
- **Forward Transfer (FT)**: How well you learn the new task
- **Backward Transfer (BT)**: How much you forget the old task (KEY METRIC)
  - BT = Performance_after - Performance_before
  - BT ≈ 0 is ideal (no forgetting)
  - BT < -0.20 is catastrophic forgetting

### 4. Meta-Causal Switching
The causal graph *itself* changes (not just parameters):
- Graph A: `COMPROMISED* → TX_AMOUNT → TX_FRAUD`
- Graph B: `COMPROMISED* → TX_FREQUENCY → TX_FRAUD`

## Three Causal Scenarios

### Scenario 1: Stolen Credentials
```
Testing phase (days 0-2)   → Small transactions (0.5x)
Exploitation (days 3-10)   → Large transactions (5x)
Abandonment (days 11+)     → Stop
```

### Scenario 2: Terminal Compromise
```
Low security terminal → High compromise probability → Fraud
High volume terminal  → Easy to hide → Higher fraud probability
```

### Scenario 3: Adversarial Adaptation (Meta-Causal)
```
If detection_rate > 70% at day 30:
  Switch from Graph A (high amount) to Graph B (high frequency)
```

## Toy Experiment (Start Here!)

**Goal**: 10-minute experiment showing causal models forget less

**Setup**:
- 60 days, 1000 customers, 100 terminals
- Period 1 (days 0-29): High amount fraud
- Period 2 (days 30-59): High frequency fraud

**Expected Results**:
| Model | Period 1 Initial | Period 2 FT | Period 1 After CL | Backward Transfer |
|-------|-----------------|-------------|-------------------|-------------------|
| Correlational | 0.85 | 0.65 | 0.45 | **-0.40** (BAD) |
| Causal | 0.82 | 0.70 | 0.75 | **-0.07** (GOOD) |

**Interpretation**: Causal model forgets only 7% vs. 40% for correlational model!

## Getting Started (5 Steps)

### Step 1: Understand the Existing Simulator
Read: Chapter 3 of the handbook, specifically `SimulatedDataset.ipynb`

**Current simulator flow**:
```python
generate_customer_profiles() →
generate_terminal_profiles() →
generate_transactions() →
add_frauds()  # Hardcoded rules
```

### Step 2: Read the Causal Structures
File: `CAUSAL_STRUCTURES.md`

Focus on: Scenario 1 (Stolen Credentials) first - it's the simplest.

### Step 3: Create Simulator Directory
```bash
mkdir -p simulator/causal models experiments/configs results
```

### Step 4: Extract Core Simulator
Copy functions from `Chapter_3_GettingStarted/SimulatedDataset.ipynb` to `simulator/core.py`

**Priority functions**:
- `generate_customer_profiles_table()`
- `generate_terminal_profiles_table()`
- `generate_transactions_table()`

Test: Generate same data as original notebook

### Step 5: Implement First SCM
File: `simulator/causal/scm.py`

Start with simplest version:
```python
class SimpleSCM:
    def sample(self, n_samples):
        # COMPROMISED ~ Bernoulli(0.01)
        compromised = np.random.binomial(1, 0.01, n_samples)

        # TX_AMOUNT = f(COMPROMISED)
        amount = np.where(compromised,
                         np.random.normal(500, 100, n_samples),  # Fraud
                         np.random.normal(100, 50, n_samples))   # Legitimate

        return {'COMPROMISED': compromised, 'TX_AMOUNT': amount}
```

## Common Questions

### Q: Do I need to read all the papers?
**A**: Not initially. Focus on:
- Willig et al. - Meta-Causal Models (for Scenario 3)
- The simulator code (Chapter 3)
- `CAUSAL_STRUCTURES.md` (our design)

### Q: What ML libraries do I need?
**A**:
- Basic: `scikit-learn`, `pandas`, `numpy`
- Causal: `dowhy` or `causalnex` (TBD - research which is better)
- Optional: `pytorch` for neural causal models

### Q: What's the critical path?
**A**:
```
Extract simulator → Implement SCM → Generate toy data →
Train correlational baseline → Train causal baseline →
Compare BT metrics → Visualize results
```

### Q: How do I know if it's working?
**A**:
1. Generated data matches original simulator (validation)
2. SCM produces expected distributions (unit test)
3. Concept drift visible in data visualization (day 30 statistics change)
4. Causal model has BT > -0.15, Correlational has BT < -0.20

### Q: What if causal model doesn't perform better?
**A**:
1. Check: Is drift severe enough? (increase parameter changes)
2. Check: Is causal graph actually different? (visualize)
3. Try: Oracle model (true graph given) as upper bound
4. Debug: Does causal model actually learn structure? (SHD metric)

## File Reading Order

1. **QUICK_START.md** (this file) - Overview
2. **CAUSAL_STRUCTURES.md**, Section on Scenario 1 - Understand one causal graph
3. **TOY_EXPERIMENT.md** - Concrete experiment
4. **SIMULATOR_CHANGES.md**, Phase 1 section - Implementation details
5. **ROADMAP.md** - Full project plan

## Key Terminology

- **SCM**: Structural Causal Model - equations describing cause-effect
- **DAG**: Directed Acyclic Graph - visual representation of causality
- **Do-operator**: do(X=x) - intervention, setting X to value x
- **Counterfactual**: "What if X had been different?" queries
- **SHD**: Structural Hamming Distance - how different are two graphs?
- **BT**: Backward Transfer - measure of forgetting
- **FT**: Forward Transfer - performance on new task
- **Meta-Causal**: Causal graph structure changes over time

## Visual Summary

```
Original Simulator (Correlational)
-----------------------------------
Transaction features → Hardcoded rules → Fraud label
Problem: Rules are brittle, no "why"

Causal Simulator (New)
----------------------
Latent causes → SCM mechanisms → Observable features → Fraud label
                    ↓
            Explicit causal graph
            (can intervene, query counterfactuals)

Concept Drift Test
------------------
Period 1: [Graph A] ──train──> Model_A
                                  ↓
Period 2: [Graph B] ──retrain──> Model_A'
                                  ↓
                         Test on Period 1 again
                                  ↓
                    How much did it forget? ← BT METRIC
```

## Expected Timeline

- **Week 1**: Extract simulator, implement basic SCM
- **Week 2**: Implement Scenario 1 causal graph
- **Week 3**: Generate toy dataset with drift
- **Week 4**: Train correlational baseline
- **Week 5**: Train causal baseline
- **Week 6**: Run toy experiment, collect metrics
- **Week 7**: Visualize, analyze results
- **Week 8**: Document findings, prepare presentation

## Success = Showing This Graph

```
Card Precision@100 Over Time
     ▲
0.9  │         Causal ●─────●──────●  (minimal forgetting)
     │                           /
0.8  │                          /
     │        Correlational    /
0.7  │              ●─────●   /
     │                     \ /
0.6  │                      ●
     │                       \
0.5  │                        ●──────●  (catastrophic forgetting)
     │
     └─────────────────────────────────────────> Time
        Period 1    Train    Period 2   Test
        (Graph A)  on P2   (Graph B)   on P1
                  (Drift)              (BT)
```

**If causal line stays high and correlational drops**: SUCCESS! 🎉

## Next Action

Pick ONE:

**Option A (Implementer)**: Start Step 3 - Create directory structure and extract simulator

**Option B (Researcher)**: Read `CAUSAL_STRUCTURES.md` in detail, sketch improvements

**Option C (Planner)**: Review `ROADMAP.md`, adjust timeline, identify blockers

## Questions? Stuck?

Check:
1. `PROJECT_STATUS.md` - Current progress and open questions
2. `Notes.md` - Research context and arguments for causal approach
3. Original handbook - `Chapter_3_GettingStarted/` for reference implementation

## One-Sentence Summary

**Test if models that learn causal mechanisms (why fraud happens) forget less than models that learn correlations (what fraud looks like) when fraudsters change tactics.**

That's it! Start with the toy experiment and scale up. Good luck! 🚀
