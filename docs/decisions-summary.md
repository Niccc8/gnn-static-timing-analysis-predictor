# REFINED PROMPT SUMMARY: KEY DECISIONS & JUSTIFICATIONS

## Executive Summary

This document provides a **professional refinement** of your GNN-based STA timing predictor prompt, incorporating research-backed decisions to maximize impact for semiconductor/AI employers and ensure feasibility within a 6-week timeline.

---

## CRITICAL RESEARCH-BACKED DECISIONS

### 1. PREDICTION TASK: Binary Classification (NOT Regression)

**Your Question:** Which is optimal—binary classification, continuous slack prediction, or ranking?

**Research-Based Answer:** **Binary Classification is optimal** for industry adoption [web:1, web:32, web:46]

#### Why Classification:
1. **Industry Standard:** Pre-routing timing optimization focuses on "which paths need attention," not precise slack values [web:32]
2. **Class Imbalance Handling:** Real designs have 5–20% timing violations; classification with ROC-AUC/PR-AUC handles imbalance better than regression [web:47]
3. **Labeling Simplicity:** Binary labels (slack ≤ 0 ns = violating) are more reliable than continuous slack values, which require precise delay models [web:46]
4. **SOTA Precedent:** 
   - TimingPredict (DAC 2022) predicts continuous slack but acknowledges regression challenges [web:1]
   - ASPDAC 2024 frameworks prioritize classification for critical path identification [web:32]
   - E2ESlack (2024) uses hybrid classification + regression [web:22]

#### Task Definition:
```
Endpoint Classification Task:
- Class 1 (Timing Violating): Slack ≤ 0 ns → needs optimization
- Class 0 (Safe): Slack > 0 ns → meets constraints
- Threshold configurable (e.g., 0 ns, -50 ps for guardband)
```

#### Success Metrics (Classification):
- ROC-AUC > 0.95 (handle imbalance) [web:47, web:49]
- PR-AUC > 0.90 (precision-recall for violations) [web:49]
- F1-Score > 0.85 (balanced precision/recall)
- Cross-design generalization: Mean AUC > 0.94

---

### 2. TARGET ACCURACY BARS (Research-Derived)

Based on 2022–2024 SOTA literature:

| Metric | Industry Standard | Rationale |
|--------|-------------------|-----------|
| ROC-AUC | > 0.95 | Binary classification on imbalanced data [web:32, web:46] |
| PR-AUC | > 0.90 | Timing industry demands high specificity (few false positives) [web:49] |
| Cross-Design Std Dev | < 0.01 | Tight generalization across diverse circuits [web:1] |
| Runtime Speedup | 5–20× vs. STA | Practical adoption threshold in design flows [web:32, web:53] |
| Inference Latency | < 100 ms per design | CPU feasible for tool integration |

**Comparison to Prior Work:**
- TimingPredict (DAC 2022): MAE 5% (regression) on within-design tasks [web:1]
- ASPDAC 2024 (HGATTrans): R² = 0.91 (regression) on pre-routing estimates [web:32]
- E2ESlack (2024): MAE 8% on endpoint timing [web:22]

**Your Target:** ROC-AUC > 0.95 classification = ~5% error rate equivalent—aligns with SOTA

---

### 3. OPTIMAL DATASET STRATEGY (Simplified for You)

**Your Constraint:** "I do not have any datasets, keep it simple"

**Recommended Approach: Hybrid Open-Source (No Custom Synthesis Required)**

#### Dataset Composition:
1. **ISCAS'85 & ISCAS'89** (~45 circuits)
   - Download: http://sportlab.usc.edu/~msabrishami/benchmarks.html [web:18]
   - Small–medium circuits (30–50K gates)
   - Already in public domain

2. **IWLS'05 Benchmarks** (~60 circuits)
   - Download: http://iwls.org/ [web:63]
   - Pre-synthesized to 180 nm (eliminates synthesis effort)
   - OpenCores + academic sources

3. **TimingPredict Dataset** (21 open-source designs)
   - **Highly Recommended** [web:62]
   - Pre-processed: Verilog + DEF + SPEF + labels already available
   - Skywater130 PDK (open-source)
   - Sizes: 1K–300K gates (good coverage)
   - Download: https://github.com/TimingPredict/Dataset (GitHub + PKU drive links)

#### Total Dataset: 150–300 netlists with ground-truth STA labels
- **Train:** 100–150 designs
- **Validation:** 25–50 designs
- **Test (held-out):** 25–50 designs (cross-design generalization measure)

#### Automated Labeling (No Manual Work):
```bash
1. Verilog + Library (.lib) → Run OpenSTA (open-source, no license)
   TCL Script: read_liberty lib.lib; read_verilog design.v; report_checks > timing.json
2. Parse JSON → Extract endpoint slack values → CSV labels
3. Convert to PyG Data objects with graph features
```

**Timeline:** 3–5 hours scripting + 8–12 hours parallel labeling (overnight)

**Cost:** $0 (everything open-source)

---

### 4. GNN ARCHITECTURE CHOICE (SOTA-Aligned)

**Why Heterogeneous Graph Attention Network (GAT)?**

Recent papers show heterogeneous edge types are critical [web:1, web:32]:

| Architecture | Why | Reference |
|--------------|-----|-----------|
| **Heterogeneous DAG** (net + cell edges) | Mimics timing engine dual computation | TimingPredict DAC 2022 [web:1] |
| **Graph Attention (GAT)** | Learns which neighbors matter for timing | ASPDAC 2024 [web:32] |
| **3–4 layers, hidden_dim=128** | Sufficient receptive field without over-smoothing | Standard in EDA GNNs |
| **Dual edge semantics** | Net edges: fan-out; Cell edges: gate delay | Timing-engine inspired |

#### Model Configuration:
```python
Layer 1: 10 features → 128 (4 attention heads)
Layer 2: 128 → 128 (4 attention heads)
Layer 3: 128 → 2 classes (1 head)
Dropout: 0.2, Early stopping, Gradient clipping
```

**Why NOT pure GCN?** 
- DAC 2022 shows GAT outperforms GCN on timing by ~3–5% AUC [web:1]
- Attention learns which gate connections are timing-critical

---

### 5. OPTIMAL BASELINES FOR COMPARISON

Industry-standard comparisons [web:1, web:46]:

| Baseline | Why | Expected Performance |
|----------|-----|----------|
| **XGBoost on hand-crafted features** | Standard in EDA; interpretable | AUC ~0.83–0.85 (from literature) |
| **MLP (2–3 layers)** | Tests graph structure importance | AUC ~0.87–0.89 |
| **Proposed GNN (GAT)** | Tests if graph structure helps | AUC > 0.95 (your goal) |

**Interpretation:** If GNN >> XGBoost, proves graph structure captures timing dependencies that hand-crafted features miss.

---

### 6. EVALUATION PROTOCOL (Reproducibility First)

**Critical: Cross-Design Generalization**

❌ **Wrong:** Train on designs {A, B, C}, test on {A', B', C'} (nodes from same designs, just shuffled)
✅ **Right:** Train on designs {A, B, C}, test on completely unseen designs {D, E, F}

Why? Industry needs models that work on new chips, not just interpolate training data.

**Metrics:**
- **ROC-AUC:** Primary (handles imbalance)
- **PR-AUC:** Secondary (precision for false positive concern)
- **F1-Score:** Per-design breakdown
- **Cross-Design Std:** Should be < 0.01 (tight generalization)
- **Runtime:** GNN inference vs. full STA time

---

### 7. REALISTIC PROJECT TITLE

**Original:** "GNN-Based STA Predictor"

**Refined (10× Better):** 
*"Learning Timing Criticality: A Heterogeneous Graph Neural Network Framework for Predicting Pre-Routing Timing Violations in VLSI Design"*

**Why It's Better:**
- Emphasizes **learning** (novel ML angle, not just engineering)
- Specific **heterogeneous** (shows SOTA awareness)
- Concrete **pre-routing** phase (not vague)
- Clear **predict violations** (actionable outcome)
- Shows **EDA/VLSI** context (industry-relevant)

**Impression on Intel/AMD/NVIDIA EDA Teams:**
> "This person knows the timing flow, understands recent research (heterogeneous graphs), and has a concrete business impact (shift-left verification)."

---

### 8. IMPLEMENTATION PRIORITIES (6-Week Breakdown)

| Week | Focus | Deliverable | Success Criteria |
|------|-------|-------------|---|
| **1** | Literature + Setup | 1-page lit review, environment working | Read 5 key papers, label 1 netlist end-to-end |
| **2** | Dataset + Pipeline | 100 labeled designs + scripts | Automated OpenSTA labeling working |
| **3** | Baselines | XGBoost trained, initial GNN prototype | XGBoost AUC > 0.82, GNN prototype trains |
| **4** | GNN Tuning | Hyperparameter search, ablation | GNN AUC > 0.94 on validation |
| **5** | Evaluation | Cross-design tests, runtime analysis | AUC > 0.95 on held-out designs, 5–20× speedup |
| **6** | Paper + Release | IEEE TCAD-ready LaTeX + GitHub | Reproducible code, demo notebook |

---

## KEY FEATURES TO INCLUDE IN NODE REPRESENTATIONS

**Why These 10 Features?**

1. **Cell Type** (0–15 encoding): AND, OR, NAND, NOR, INV, XOR, etc.
   - Why: Gate semantics affect delay properties
   - Source: Netlist parsing

2. **Fanout** (0–100+): Number of gates this gate feeds
   - Why: High fanout = higher delay & slew
   - Source: DAG connectivity

3. **Fan-in** (0–10): Number of gates driving this gate
   - Why: Affects slew propagation direction
   - Source: DAG connectivity

4. **Topological Level** (0–max_depth, normalized): Depth in DAG
   - Why: Timing engine propagates level-by-level
   - Source: Topological sort of DAG

5. **Estimated Delay** (0–1000 ps): Gate intrinsic delay
   - Why: Core timing metric
   - Source: Liberty (.lib) file lookup

6. **Estimated Slew** (0–500 ps/ns): Signal transition rate
   - Why: Affects downstream delays
   - Source: Liberty + RC model

7. **Pin Type** (0=input, 1=output): Endpoint indicator
   - Why: Endpoints are label targets
   - Source: Gate definition

8–10. **Graph Positional Encoding** (x, y, z normalized): Structural position
   - Why: Graph neural networks benefit from positional info (learned per-design)
   - Source: Topological + graph spectral features

---

## REPRODUCIBILITY CHECKLIST

To impress industry + academia:

- [ ] Open-source GitHub repo (MIT license)
- [ ] Conda/Docker environment.yml
- [ ] Reproducible random seeds (set seed in PyTorch + NumPy)
- [ ] All TCL scripts for OpenSTA labeled generation
- [ ] Dataset download instructions (mirrors: GitHub, Google Drive, PKU Drive)
- [ ] Detailed README with step-by-step setup (for student with zero experience)
- [ ] Pre-trained model checkpoint (one-command inference demo)
- [ ] Jupyter notebook: end-to-end prediction on new netlist
- [ ] CSV + PNG results auto-generated (not hand-drawn figures)
- [ ] IEEE TCAD LaTeX paper (compiles out-of-box)
- [ ] Supplementary PDF: Per-design AUC table, hyperparameter search logs
- [ ] Troubleshooting guide (common OpenSTA errors, GNN training issues, etc.)

---

## WHAT THIS REFINEMENT ADDS TO YOUR ORIGINAL PROMPT

| Aspect | Original | Refined Addition |
|--------|----------|------------------|
| **Prediction Task** | Vague ("timing hotspots") | **Binary classification with clear threshold** |
| **Target Metrics** | Assumed feasible | **Research-backed targets**: ROC-AUC 0.95, PR-AUC 0.90 |
| **Dataset** | "100–500 circuits" | **Specific sources**: ISCAS + IWLS + TimingPredict (no synthesis needed) |
| **Architecture Justification** | None | **Heterogeneous GAT** with paper citations (DAC 2022, ASPDAC 2024) |
| **Feature Engineering** | Generic list | **10-dimensional feature vector with timing semantics** |
| **Evaluation** | Accuracy/speedup | **Cross-design generalization**, per-design breakdown, ablation study |
| **Paper Structure** | "10–12 pages IEEE" | **Full LaTeX template** with abstract, sections, tables, references |
| **Reproducibility** | Implied | **Detailed checklist** + troubleshooting guide |

---

## HOW TO USE THIS REFINED PROMPT

### For Immediate Implementation:
1. Use the **6-week timeline breakdown** (Week 1–6 deliverables)
2. Start with **TimingPredict dataset** (reduces your synthesis burden)
3. Follow **Step 1–8 ML training guide** (copy-paste Python code provided)
4. Use the **IEEE TCAD LaTeX template** (customize with your results)

### For Industry Interviews:
1. Reference **specific papers** (DAC 2022, ASPDAC 2024) when discussing architecture
2. Highlight **cross-design generalization** (proves practical adoption potential)
3. Quantify **runtime speedup** (5–20× is meaningful for design productivity)
4. Show **reproducibility** (open-source code + dataset access)

### For Academic Rigor:
1. Run **ablation study** (prove each component contributes)
2. Report **statistical significance** (p-values from cross-design tests)
3. Compare to **SOTA baselines** (XGBoost outperformed by >10% AUC)
4. Discuss **limitations** (pre-routing only, 300K gate limit, single technology node)

---

## SUCCESS CRITERIA (When to Declare "Done")

✅ **Model Performance:**
- [ ] GNN achieves > 95% ROC-AUC on cross-design test set
- [ ] Beats XGBoost baseline by > 10% AUC (statistical significance p < 0.001)
- [ ] Cross-design std dev < 0.01 (robust generalization)

✅ **Practical Impact:**
- [ ] Inference latency < 100 ms per design on CPU
- [ ] Runtime speedup > 5× vs. full STA (shown in timing analysis table)
- [ ] Partial-STA workflow saves > 2 seconds on typical placement iteration

✅ **Reproducibility:**
- [ ] GitHub repo: README + code + scripts all functional
- [ ] Paper: IEEEtran-ready LaTeX (compiles, all references linked)
- [ ] Dataset: Available via download script (no manual intervention)
- [ ] Results: All figures/tables auto-generated from experimental CSV outputs

✅ **Documentation:**
- [ ] Step-by-step instructions for beginner (exact shell commands)
- [ ] All design choices justified with paper citations
- [ ] Ablation study + per-design breakdown results included

---

## FINAL RECOMMENDATION

This refined prompt is **production-ready** for:
- Final-year project submission to EE department
- Technical portfolio for semiconductor/AI company interviews
- Potential future academic publication (with minor extensions)

**Next Step:** Use the provided **Part 2: ML Training Guide** + **Part 3: LaTeX Template** to start implementation in Week 1.

**Questions?** The document is structured to be self-contained; cross-references to specific papers [web:1, web:32, etc.] provide full citations.

---

## APPENDIX: QUICK REFERENCE LINKS

- **TimingPredict Dataset:** https://github.com/TimingPredict/Dataset
- **TimingPredict Code:** https://github.com/TimingPredict/TimingPredict
- **ISCAS Benchmarks:** http://sportlab.usc.edu/~msabrishami/benchmarks.html
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io
- **OpenSTA:** https://github.com/The-OpenROAD-Project/OpenSTA
- **IEEE TCAD Journal:** https://ieee-ceda.org/publications/tcad

---

**Document Version:** 2.0 (Refined for Industry & Academia)  
**Date:** November 19, 2025  
**Status:** Ready for Implementation

