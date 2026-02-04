The discussion between Moritz Willig and Gian Marco Paldino centers on a potential research collaboration at the intersection of **Causality**, **Continual Learning (CL)**, and **Credit Card Fraud Detection**.

Here is a breakdown of the core idea and its components:

### 1. The Domain: Credit Card Fraud Detection
Fraud detection is identified as a "prime application area" because it is inherently dynamic. It suffers from two major challenges:
*   **Concept Drift:** Fraudsters constantly change their tactics, meaning the statistical properties of the "fraud" class change over time.
*   **Data Imbalance/Privacy:** Real-world transaction data is highly sensitive and rarely public. Gian Marco has mitigated this by providing a **fraud simulator** that generates synthetic transaction data.

### 2. The Theoretical Intersection: Causality x Continual Learning
The researchers are looking to combine two advanced machine learning fields:
*   **Continual Learning:** Addressing "Catastrophic Forgetting." When a model learns to detect new types of fraud, it often loses the ability to recognize older patterns. The goal is to create a model that learns incrementally.
*   **Causality:** Instead of just looking at correlations (which can be misleading during concept drift), they want to model the **causal links** between features—such as merchant type, terminal location, and seasonal trends—and the occurrence of fraud.

### 3. The Specific Research Objective
The group aims to determine if **causal modeling** can make **continual learning** more robust. 
*   **The Hypothesis:** If a model understands the *causal mechanism* of a fraud (the "why"), it might be more resilient to changes in the data distribution (concept drift) than a traditional model that only recognizes patterns.
*   **The Methodology:** They plan to use Gian Marco’s existing simulator as a foundation and "tweak it to adapt it to a causal model." This involves creating synthetic scenarios where causal relationships are known and can be tested against the model’s performance over time.

### 4. Current Status
*   **Resource Sharing:** Gian Marco has shared a "Fraud Detection Handbook" and a simulator.
*   **The "Tweaked" Simulator:** Gian Marco is currently developing a variation of the simulator that specifically focuses on **concept drift and causal structures** to provide a concrete starting point for their plan.
*   **The Goal:** They are moving toward drafting a paper or project plan that applies these combined theories to the fraud detection simulator.


That is a very common (and technically interesting) debate in the field! Your supervisor's skepticism likely stems from a traditional view of machine learning and the unique "adversarial" nature of fraud. 

To help you prepare for your next meeting, here is a breakdown of why they might say that, and the counter-arguments you can use to defend the project.

### Why your supervisor might say there is "no causal link"
1.  **Adversarial Reflexivity:** In physics, if $A$ causes $B$, that relationship is usually stable. In fraud, as soon as you detect a "cause" (e.g., a specific transaction pattern), the fraudster *reacts* and changes their behavior. Your supervisor might argue that there are no "stable" causal laws in fraud, only temporary correlations that vanish once the model is deployed.
2.  **Lack of a Physical Mechanism:** Fraud is driven by human intent, not a mechanical process. Unlike medical research where a drug *causes* a biological change, a "transaction amount" doesn't "cause" fraud in a physical sense—it is merely a signal used by a criminal.
3.  **Data Anonymization:** In many real-world datasets (like the famous Kaggle Credit Card Fraud dataset), features are PCA-transformed. It’s impossible to establish causality when you don't even know what the features represent.
4.  **Unobserved Confounders:** There are massive "hidden" variables (e.g., the dark web, global economic shifts, or a specific data breach) that cause both the transaction patterns and the fraud. Since we can't see those, we are only ever looking at correlations.

---

### How to defend the Causal approach
You can argue that **Causality is actually the solution** to the very problems your supervisor is worried about:

1.  **Robustness to Concept Drift (The "Invariance" Argument):**
    *   *The Point:* Standard models fail during "concept drift" because they overfit to spurious correlations that change over time.
    *   *The Causal Defense:* Causal relationships are, by definition, more **invariant**. If we can identify the underlying mechanism of an attack (e.g., "how a stolen credential leads to a sequence of test transactions"), that logic remains true even if the specific dollar amounts or merchant types change.

2.  **Handling "Adversarial Drift" in Continual Learning:**
    *   *The Point:* This is where your CL focus comes in. Fraudsters evolve, so the data distribution shifts.
    *   *The Causal Defense:* If the model only learns *correlations*, it will suffer from "catastrophic forgetting" as it tries to keep up with new patterns. If it learns the **causal structure** of how a user behaves vs. how a fraudster behaves, it can adapt to new tactics without losing its "knowledge" of the fundamental differences between legitimate and fraudulent behavior.

3.  **Counterfactual Reasoning for False Positives:**
    *   *The Point:* One of the biggest costs in banking is blocking legitimate customers (False Positives).
    *   *The Causal Defense:* Using "What-if" scenarios (counterfactuals). For example: *"Would this transaction still look fraudulent if the user had just arrived in a new country?"* Causal models can separate "unusual but legitimate" behavior from fraud better than correlation-based ones.

4.  **The "Simulator" is the Proof:**
    *   Since Gian Marco is giving you a **simulator**, you actually *do* have a ground-truth causal model! In a simulator, the rules are programmed. You can explicitly show that a model which understands the "programmed" causal links outperforms a model that just looks for surface-level patterns.

### Recommendation for your meeting
Instead of arguing whether "True Causality" exists in the real world, frame it as **"Causal Discovery for Robustness."** 

Tell your supervisor: *"Even if we can't find a 'perfect' physical causal link, using causal discovery tools allows us to ignore the 'noise' (spurious correlations) that causes our current models to fail every time a fraudster changes tactics. It's about building a model that is resilient to the data shifts we see in the simulator."*

---

## Project Planning Update (2026-02-04)

### Documentation Created

Comprehensive planning documents have been created for the causal continual learning project:

1. **ROADMAP.md** - 8-week project roadmap with 7 phases:
   - Phase 1: Simulator extraction & modularization
   - Phase 2: Causal structure design (SCMs for 3 scenarios)
   - Phase 3: Concept drift implementation
   - Phase 4: Baseline models (correlational vs. causal)
   - Phase 5: Continual learning framework
   - Phase 6: Toy experiment (60 days, 1000 customers)
   - Phase 7: Analysis & documentation

2. **SIMULATOR_CHANGES.md** - Technical specification for adapting the handbook simulator:
   - New modular architecture: `simulator/core.py`, `simulator/causal/`, etc.
   - SCM implementation with interventions and counterfactuals
   - Configuration system for reproducible experiments
   - Backward compatibility with existing handbook code

3. **CAUSAL_STRUCTURES.md** - Detailed causal graphs and mechanisms:
   - **Scenario 1: Stolen Credentials** - Testing → Exploitation → Abandonment phases
   - **Scenario 2: Terminal Compromise** - Security level → Fraud probability
   - **Scenario 3: Adversarial Adaptation** - Meta-causal graph switching
   - Structural equations, interventions, and counterfactual queries for each

4. **TOY_EXPERIMENT.md** - Small-scale proof-of-concept experiment:
   - Setup: 60 days with concept drift at day 30
   - Compare correlational (Random Forest) vs. causal (Neural Causal Model) vs. oracle
   - Expected results: Causal model BT = -0.07, Correlational BT = -0.40
   - Full implementation code and visualization plans

5. **PROJECT_STATUS.md** - Progress tracking and open questions

### Key Design Decisions

**Three Causal Scenarios** designed to test different aspects:
1. Stolen Credentials: Tests temporal phase transitions and latent variables
2. Terminal Compromise: Tests confounding (amount independent of fraud)
3. Adversarial Adaptation: Tests meta-causal switching (graph structure changes)

**Evaluation Framework**:
- **Forward Transfer (FT)**: Performance on new fraud tactics
- **Backward Transfer (BT)**: Forgetting of old fraud tactics (KEY METRIC)
- **Structural Hamming Distance (SHD)**: Causal graph recovery accuracy

**Core Hypothesis**: Models that learn causal mechanisms will have BT ≈ 0 (minimal forgetting), while correlational models will have BT < -0.20 (catastrophic forgetting).

### Meta-Causal Insight

The Adversarial Adaptation scenario directly implements ideas from Willig et al.'s "Systems with Switching Causal Relations":

```
Detection at day 30 (intervention)
        ↓
[Graph A: High Amount] ──(meta-causal switch)──> [Graph B: High Frequency]
        ↓                                                    ↓
TX_AMOUNT → TX_FRAUD                              TX_FREQUENCY → TX_FRAUD
```

This tests whether causal models can handle **changes in the causal graph itself**, not just parameter shifts.

### Next Immediate Actions

1. Review planning documents with Moritz
2. Start Phase 1: Extract simulator from Chapter 3 notebooks
3. Research Neural Causal Model implementations (dowhy, causalnex, or custom)
4. Create `simulator/` directory structure
5. Set up project Git branch

### Success Criteria Summary

| Level | Backward Transfer | SHD | Sufficient For |
|-------|------------------|-----|----------------|
| Minimal | Causal: BT > -0.15<br>Correlational: BT < -0.20 | Any | Internal demo |
| Strong | Causal: BT > -0.10 | < 5 | Workshop paper |
| Publication | Causal: BT > -0.05 | < 3 | Full conference paper |

The key is showing **statistically significant difference** in forgetting between causal and correlational approaches.