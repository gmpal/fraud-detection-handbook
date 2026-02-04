# Project Status: Causal Continual Learning for Fraud Detection

**Last Updated**: 2026-02-04

## Project Overview

Adapting the fraud detection handbook simulator to test whether causal models are more robust to concept drift in continual learning settings compared to correlational models.

**Core Team**: Moritz Willig, Gian Marco Paldino
**Related Work**: Meta-Causal Models (Willig et al.), Neural Causal Models (Busch & Seng)

## Documentation Created ✅

### Planning Documents
- ✅ `ROADMAP.md` - 8-week project roadmap with phases and deliverables
- ✅ `SIMULATOR_CHANGES.md` - Technical specification for simulator modifications
- ✅ `CAUSAL_STRUCTURES.md` - Detailed causal graphs and SCM definitions
- ✅ `TOY_EXPERIMENT.md` - Small-scale proof-of-concept experiment
- ✅ `CLAUDE.md` - General handbook documentation (for future Claude instances)
- ✅ `PROJECT_STATUS.md` - This file

### Key Design Decisions

1. **Three Causal Scenarios**:
   - Stolen Credentials: Testing → Exploitation → Abandonment phases
   - Terminal Compromise: Security level → Fraud probability
   - Adversarial Adaptation: Meta-causal graph switching

2. **Evaluation Framework**:
   - Forward Transfer (FT): Performance on new task
   - Backward Transfer (BT): Forgetting on old task
   - Causal Graph Recovery: Structural Hamming Distance

3. **Toy Experiment Parameters**:
   - 60 days, 1000 customers, 100 terminals
   - Concept drift at day 30
   - Expected BT: Correlational -0.40, Causal -0.07

## Implementation Progress

### Phase 1: Simulator Extraction (Week 1-2)
**Status**: 🔴 Not Started

**Tasks**:
- [ ] Create `simulator/` directory structure
- [ ] Extract core code from Chapter 3 notebook to `simulator/core.py`
- [ ] Test extraction: Generate same data as original
- [ ] Create `simulator/config.py` for experiment configs
- [ ] Write unit tests for core functions

**Blockers**: None
**Next Action**: Create directory structure and start extraction

### Phase 2: Causal Structure Design (Week 2-3)
**Status**: 🟡 Specification Complete, Implementation Pending

**Completed**:
- ✅ Design Stolen Credentials SCM
- ✅ Design Terminal Compromise SCM
- ✅ Design Adversarial Adaptation meta-causal model
- ✅ Document all causal graphs in detail

**Tasks**:
- [ ] Implement `simulator/causal/scm.py` - Base SCM class
- [ ] Implement `simulator/causal/graphs.py` - Causal graph definitions
- [ ] Implement `simulator/causal/mechanisms.py` - Fraud mechanisms
- [ ] Implement `simulator/causal/interventions.py` - Do-calculus
- [ ] Test: Sample from SCM matches expected distributions

**Blockers**: Depends on Phase 1
**Next Action**: Wait for core simulator extraction

### Phase 3: Concept Drift Scenarios (Week 3-4)
**Status**: 🟡 Specification Complete, Implementation Pending

**Completed**:
- ✅ Define drift scenarios (gradual, sudden, adversarial)
- ✅ Specify meta-causal switching logic

**Tasks**:
- [ ] Implement `simulator/drift.py` - Drift manager
- [ ] Implement `simulator/causal_scenarios.py` - Causal fraud scenarios
- [ ] Create ground truth logging system
- [ ] Test: Verify graph switches at correct times

**Blockers**: Depends on Phase 2
**Next Action**: Wait for causal structures implementation

### Phase 4: Baseline Models (Week 4-5)
**Status**: 🔴 Not Started

**Tasks**:
- [ ] Implement correlational baseline (Random Forest)
- [ ] Research: Find suitable Neural Causal Model library (dowhy? causalnex?)
- [ ] Implement causal baseline
- [ ] Implement causal oracle (uses true graph)
- [ ] Create `models/` directory with all baselines

**Blockers**: Depends on Phase 3 (need data to train on)
**Next Action**: Research NCM libraries

### Phase 5: Continual Learning Framework (Week 5-6)
**Status**: 🔴 Not Started

**Tasks**:
- [ ] Implement temporal evaluation protocol
- [ ] Implement naive fine-tuning
- [ ] Implement Experience Replay
- [ ] Implement EWC (Elastic Weight Consolidation)
- [ ] Implement causal-aware CL strategy
- [ ] Create metrics computation (FT, BT, SHD)

**Blockers**: Depends on Phase 4
**Next Action**: Design CL evaluation harness

### Phase 6: Toy Experiment (Week 6-7)
**Status**: 🟡 Specification Complete, Implementation Pending

**Completed**:
- ✅ Full experiment specification in `TOY_EXPERIMENT.md`
- ✅ Expected results documented
- ✅ Visualization plans created

**Tasks**:
- [ ] Generate toy dataset (60 days, 1000 customers)
- [ ] Run correlational baseline
- [ ] Run causal baseline
- [ ] Run causal oracle
- [ ] Generate visualizations
- [ ] Write results notebook

**Blockers**: Depends on Phases 1-5
**Next Action**: Wait for all components

### Phase 7: Analysis & Documentation (Week 7-8)
**Status**: 🔴 Not Started

**Tasks**:
- [ ] Create performance plots
- [ ] Create causal graph visualizations
- [ ] Perform counterfactual analysis
- [ ] Write `RESULTS.md` with findings
- [ ] Update `CLAUDE.md` with causal extensions
- [ ] Prepare paper outline

**Blockers**: Depends on Phase 6
**Next Action**: Wait for toy experiment results

## Technical Stack

### Confirmed Dependencies
- `jupyter-book` - Documentation
- `pandas`, `numpy` - Data handling
- `scikit-learn` - Correlational baselines
- `matplotlib`, `seaborn` - Visualization
- `networkx` - Graph manipulation

### To Be Decided
- **Causal Inference**: `dowhy` vs. `causalnex` vs. custom implementation?
- **Causal Discovery**: `pcalg`, `GES`, `NOTEARS`?
- **Neural Causal Models**: Existing library or implement from papers?

### Hardware Requirements
- Toy experiment: CPU sufficient
- Full experiment: GPU recommended

## Open Questions

### Technical
1. **SCM Implementation**: Use symbolic equations or neural functional approximations?
   - **Leaning towards**: Neural for flexibility, symbolic for interpretability

2. **Causal Discovery**: Should model learn graph or use oracle?
   - **Decision**: Both - oracle as upper bound, learned for realism

3. **CL Strategy**: Which continual learning method for causal model?
   - **Options**: Parameter isolation, causal replay, structure preservation
   - **Decision**: TBD after literature review

### Experimental
4. **Drift Severity**: How different should Graph A and Graph B be?
   - **Current plan**: Completely different mechanisms (amount vs. frequency)
   - **Risk**: Too easy to distinguish?

5. **Sample Size**: Is 1000 customers enough for toy experiment?
   - **Current plan**: Yes for proof-of-concept, scale up later

6. **Evaluation**: Should we add standard CL benchmarks (MNIST, CIFAR)?
   - **Decision**: No - stay domain-focused on fraud

## Risks and Mitigation

### Risk 1: Causal model shows no benefit
**Probability**: Medium
**Impact**: High (invalidates hypothesis)
**Mitigation**:
- Design adversarial drift specifically to break correlations
- Start with oracle model to show upper bound
- Ablation studies to isolate causal structure benefit

### Risk 2: Simulator too simple
**Probability**: Medium
**Impact**: Medium (reviewers question realism)
**Mitigation**:
- Add confounders and noise to increase complexity
- Validate on real-world dataset if available
- Compare to handbook's existing fraud scenarios

### Risk 3: Implementation complexity
**Probability**: High
**Impact**: Medium (delays timeline)
**Mitigation**:
- Use existing libraries where possible
- Start with simplest SCM implementation
- Prioritize toy experiment over full system

### Risk 4: Causal discovery fails
**Probability**: Medium
**Impact**: Medium (can't recover true graph)
**Mitigation**:
- Provide graph structure hints (semi-supervised)
- Focus on oracle model results
- Compare multiple causal discovery algorithms

## Success Metrics

### Minimal Viable Result
- Toy experiment runs end-to-end
- Correlational model: BT < -0.20
- Causal model: BT > -0.15
- Clear visualization showing difference

**Sufficient for**: Internal presentation, project continuation decision

### Strong Result
- Causal model: BT > -0.10
- Causal graph recovery: SHD < 5
- Statistical significance (p < 0.05) across 5 seeds

**Sufficient for**: Workshop paper, extended abstract

### Publication-Ready Result
- Causal model: BT > -0.05
- Causal graph recovery: SHD < 3
- Ablation studies on all components
- Counterfactual analysis shows 15%+ improvement
- Real-world validation (stretch goal)

**Sufficient for**: Full conference paper (NeurIPS, ICML, ICLR)

## Timeline

**Start Date**: 2026-02-04
**Target Completion**: 2026-04-01 (8 weeks)

### Milestones
- **Week 2 (Feb 18)**: Simulator extracted and tested
- **Week 4 (Mar 4)**: Causal structures implemented
- **Week 6 (Mar 18)**: Baseline models trained
- **Week 7 (Mar 25)**: Toy experiment complete
- **Week 8 (Apr 1)**: Results documented, paper outline ready

### Critical Path
```
Simulator Extraction → Causal SCMs → Drift Manager → Data Generation →
Baseline Models → CL Framework → Toy Experiment → Analysis
```

**Bottleneck**: Causal model implementation (Week 4-5)

## Next Immediate Steps (This Week)

1. **Review documents** with Moritz - get feedback on roadmap
2. **Create directory structure**:
   ```
   mkdir -p simulator/causal models experiments/configs results
   ```
3. **Start Phase 1**: Extract `generate_customer_profiles_table()` to `simulator/core.py`
4. **Research**: Identify best Neural Causal Model library or paper implementation
5. **Set up Git branch**: `feature/causal-continual-learning`

## Meeting Agenda (Next Discussion)

1. Review roadmap - agree on timeline
2. Discuss causal structures - are SCMs realistic enough?
3. Decide on NCM implementation approach
4. Assign responsibilities if team collaboration
5. Set up weekly check-ins

## Resources

### Papers to Reference
- Willig et al. (2025) - Meta-Causal Models
- Busch & Seng - Continually Updating Neural Causal Models
- Seng et al. - Treatment Effect Estimation for CL
- Kersting et al. - Meta-Aware Interventions

### Code Repositories
- Fraud Detection Handbook: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
- DoWhy (Microsoft): https://github.com/microsoft/dowhy
- CausalNex (McKinsey): https://github.com/mckinsey/causalnex

### Related Work
- Catastrophic Forgetting: McCloskey & Cohen (1989)
- Continual Learning Survey: Delange et al. (2021)
- Causal Invariance: Peters et al. (2017)

## Change Log

### 2026-02-04
- Created all initial planning documents
- Defined 3 causal scenarios
- Specified toy experiment
- Set up project tracking
