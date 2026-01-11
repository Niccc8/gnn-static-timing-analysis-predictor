# Paper Notes: Ranking Performance & Advanced Deployment Strategies

## FOR PAPER / FUTURE WORK SECTION

### Current Ranking Limitations (IMPORTANT!)

#### 1. Fixed Percentage Assumption
**Problem:** Recommending "top 5%" assumes all circuits have ~0.6% violations.

**Failure Cases:**
- **High-violation circuit (70% violations):** Top 5% catches almost nothing
- **Zero-violation circuit:** Still flags 5% as false positives
- **Varying violation rates:** Optimal K varies per circuit

**Evidence from validation:**
```
Circuit 0: 0.86% violations → Top 5% catches 95% ✅
Circuit 1: 1.11% violations → Top 3% catches 100% ✅ (5% is overkill)
Circuit 2: 0.64% violations → Top 5% catches 22% ❌ (needs 15%)
Circuit 3: 0.08% violations → Top 5% catches 69% ⚠️
```

**Insight:** Fixed percentage is **NOT one-size-fits-all!**

---

### Proposed Solution: Two-Stage Classification

#### Stage 1: Design-Level Triage
**Goal:** Classify entire design into risk categories BEFORE node-level analysis.

**Categories:**
1. **Clean Design** (predicted <0.1% violations) → Skip node analysis or inspect top 1%
2. **Low Risk** (0.1-1% violations) → Inspect top 5%
3. **Moderate Risk** (1-10% violations) → Inspect top 10-15%
4. **High Risk** (>10% violations) → Inspect top 25%+ or full STA

**Benefits:**
- No wasted effort on clean designs
- Adaptive inspection budget
- Better resource allocation

#### Stage 2: Node-Level Ranking (if needed)
Only run GNN node classification if Stage 1 flags the design as risky.

---

### Precision vs Base Rate Explained

#### Base Rate
- **Definition:** Natural occurrence of violations in dataset
- **Your data:** 0.6% average (range: 0.08% to 1.11%)
- **Meaning:** Random guess has 0.6% chance of being correct

#### Precision at Top 5%
- **Definition:** Fraction of flagged nodes that are true violations
- **Your model:** 10.64% precision at top 5%
- **Improvement:** 10.64% / 0.6% = **17.7× better than random!**

**Why still low?**
- Extreme imbalance makes high precision difficult
- Model optimized for AUC (ranking), not precision
- Trade-off: Higher recall → Lower precision

---

### Metrics Breakdown (for Methods Section)

#### Test Set Statistics
```
4 test circuits, 221,877 total nodes
Average violation rate: 0.6% (range: 0.08% - 1.11%)
Extreme class imbalance: 160:1 ratio
```

#### Performance at Different Top-K
| Top % | Avg Recall | Avg Precision | Cost-Benefit |
|-------|------------|---------------|--------------|
| 1% | 27.4% | 21.1% | High precision, miss most violations |
| 3% | 60.8% | 17.1% | Good balance for low-violation circuits |
| 5% | 71.5% | 10.6% | Recommended baseline |
| 10% | 83.9% | 5.8% | High recall, diminishing returns |
| 20% | 96.4% | 3.3% | Near-complete, expensive |

**Recommendation:** Adaptive K based on predicted violation density.

---

### For Discussion Section

#### Why Ranking Outperforms Threshold

**Problem with thresholds:**
- Fixed threshold (0.58) flags **different amounts** per circuit
- No cost control (might flag 1% or 10% depending on circuit)
- Precision varies wildly between circuits

**Advantage of ranking:**
- Fixed inspection budget (always check top K%)
- Predictable workload for manual review
- Better utilizes high AUC (0.95) performance

**But:** Needs adaptive K selection!

---

### Future Work Ideas

1. **Design-level classifier:**
   - Input: Circuit features (size, connectivity, technology node)
   - Output: Predicted violation density
   - Use to select optimal K

2. **Adaptive threshold:**
   - Calibrate per-circuit threshold based on predicted density
   - Combine threshold + ranking for robust performance

3. **Multi-stage filter:**
   - Stage 1: Cheap heuristics to skip clean designs
   - Stage 2: GNN to rank remaining designs
   - Stage 3: Full STA only on top violations

4. **Active learning:**
   - Let user inspect some nodes, update predictions
   - Refine ranking based on user feedback

---

### Visualization Ideas for Paper

#### Figure 1: Ranking Performance Curves
- X-axis: Top K% inspected
- Y-axis: Violations caught (recall)
- Multiple lines for different circuits
- Show trade-off between cost and coverage

#### Figure 2: Precision-Recall at Different K
- Classic PR curve
- Annotate with K values
- Show operating points (1%, 5%, 10%)

#### Figure 3: Two-Stage Classification Flow
```
Circuit → Design Classifier → Risk Category
                             ↓
                         Low Risk? → Top 5%
                         High Risk? → Top 25%
                         Clean? → Skip / Top 1%
```

#### Table: Per-Circuit Performance
```
Circuit | Nodes | Violations | Optimal K | Our K | Recall@5%
--------|-------|------------|-----------|-------|----------
  0     | 59K   | 512 (0.86%)| 3-5%      | 5%    | 95.3%
  1     | 37K   | 414 (1.11%)| 3%        | 5%    | 100%
  2     | 59K   | 380 (0.64%)| 15%       | 5%    | 22.1% ⚠️
  3     | 66K   | 51 (0.08%) | 7-10%     | 5%    | 68.6%
```

Show that **one-size-fits-all fails!**

---

### Empirical Evidence: Probability Distributions (Circuit 1 Case Study)

**Why Thresholds Fail:**
- **True Violations:** Range [0.04, 0.18], Mean **0.08**
- **Clean Nodes:** Range [0.00, 0.69], Mean **0.006**
- **Threshold (0.58):** Misses ALL violations (Max prob 0.18 < 0.58)

**Why Ranking Works:**
- Violations are clustered in the "tail" of the distribution.
- Mean violation probability (0.08) is **12× higher** than clean nodes (0.006).
- **Adaptive Strategy:** Using a sensitivity threshold of **0.01** counts the tail size → suggests K=2062 (5.5%) → Catches **100%** of violations.

### Multi-Circuit Analysis (Validation Set)

Tested adaptive strategy (threshold 0.01) on 5 diverse circuits:

| Circuit | Status | Violations | Adaptive K | Recall | Precision | Notes |
|---------|--------|------------|------------|--------|-----------|-------|
| **Ckt 0** | Violating | 1,827 (0.78%) | 21.7% | **99.2%** | 3.6% | High noise, needs large K |
| **Ckt 1** | Violating | 72 (0.13%) | 5.2% | **95.8%** | 2.3% | Works well |
| **Ckt 2** | Violating | 15 (0.15%) | 4.6% | **100%** | 3.3% | Excellent |
| **Ckt 3** | **CLEAN** | **0 (0.00%)** | **7.8%** | N/A | 0.0% | **False Positive Issue** |
| **Ckt 4** | Violating | 22 (0.03%) | 8.0% | **100%** | 0.4% | Catches rare violations |

**Critical Finding on Clean Designs:**
- The user asked: *"If the design is clean, will it still flag nodes?"*
- **Answer: YES.** Circuit 3 is clean, but the model flags 7.8% of nodes as "suspicious" (prob > 0.01).
- **Implication:** The model is "paranoid" - it prioritizes recall over precision. It ensures safety (catching violations) at the cost of inspecting ~8% of clean designs.
- **Future Work:** Train a separate **Design-Level Classifier** to filter out clean designs *before* node ranking to save this 8% effort.

### Can We Optimize Adaptive K Further? (The "Circuit 3 vs 4" Paradox)
We analyzed if we could reduce the 7.8% false positive rate on the clean circuit (Ckt 3) without hurting recall.

**Comparison:**
- **Circuit 3 (Clean):** Max probability = **0.1655** (False Alarms)
- **Circuit 4 (Violating):** Max violation probability = **0.1016** (True Violations)

**The Problem:**
The "noise" in the clean circuit (0.16) is *louder* than the "signal" in the violating circuit (0.10).
- If we raise the sensitivity threshold to 0.15 to filter out Circuit 3...
- **We would miss 100% of violations in Circuit 4!**

**Conclusion:**
The "paranoid" strategy (threshold 0.01) is mathematically necessary to ensure safety. We must accept the 8% overhead on clean designs to guarantee we catch the subtle violations in hard designs.

---

### Industrial Relevance (Why Ranking Matters)
*See `docs/INDUSTRIAL_USE_CASE.md` for full details.*

**The Value Proposition:**
1.  **Surgical Optimization (ECO):** Instead of running global optimization (20+ hours), use GNN to target only the top 5% risky nodes for surgical fixes (ECOs).
2.  **Path-Based Analysis (PBA):** Replace expensive `report_timing -all` with fast `report_timing -to [list_of_risky_nodes]`.
3.  **Analogy:** Searching for a lost key in a 100-room hotel.
    -   **Full STA:** Search every room (10 hours).
    -   **GNN:** "Check Room 105 & 106" (Search 2 rooms, save 9.8 hours).

---

### Key Takeaways for Paper

1. **AUC 0.95 is excellent** - model ranks violations correctly
2. **F1 0.30 is expected** - extreme imbalance makes high F1 difficult
3. **Ranking >> Threshold** - predictable cost, better resource allocation
4. **BUT: Need adaptive K** - fixed 5% fails for unusual circuits
5. **Two-stage approach** - design triage + node ranking = robust system

---

## Action Items for Paper

- [ ] Create performance curve figure (Recall vs Top K%)
- [ ] Add per-circuit breakdown table
- [ ] Explain why AUC is the right metric (not F1)
- [ ] Discuss two-stage classification in Future Work
- [ ] Show adaptive K selection algorithm (pseudocode)
- [ ] Compare with baseline (random, degree centrality, heuristics)
