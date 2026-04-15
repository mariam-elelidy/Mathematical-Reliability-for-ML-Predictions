# Split Conformal Prediction Intervals as a Reliability Layer

**Author:** Mariam Mohamed Elelidy  
**Topic:** Uncertainty Quantification · Distribution-Free Inference · Reliable ML

---

## TL;DR

Most ML pipelines report a point prediction and a single test score. That hides the question that actually matters in deployment: *how often can you trust this output?*

This artifact adds a small, testable **reliability layer** on top of any regression model — split conformal prediction — that produces prediction intervals with a finite-sample coverage guarantee under an explicit exchangeability assumption. No black-box internals required.

Key outputs:
- Empirical coverage vs interval width, measured per seed and aggregated
- Running mean/std tensors across 10 seeds showing stability
- Per-point preview: `[pred, lo, hi, y_true, covered]`

---

## 1. Problem & Motivation

In applied ML — especially for medical or high-stakes decisions — two models can have the same mean absolute error yet behave very differently when confidence matters.

A point prediction gives no answer to:
- *"How uncertain is this estimate?"*
- *"How often would this output actually contain the truth?"*

Standard practice reports a test metric once, sometimes with a fixed seed. That is not enough. In practical settings, a "good score" can coexist with brittle confidence: the model can be systematically overconfident (too narrow) or uninformative (too wide).

This project documents a reliability layer that does **not depend on model internals**: split conformal prediction. The goal is not to claim perfection — it is to bound what we say about predictions under an explicit, checkable assumption and to measure how stable that behavior is across repeated runs.

---

## 2. Testable Claims

**Primary claim:** Under an exchangeability assumption between calibration and test data, split conformal prediction produces prediction intervals that achieve approximately the target coverage (90%) on held-out test points.

**Secondary claim:** Interval tightness (average width) varies with data difficulty; coverage and width form a trade-off that can be measured, compared across runs, and treated as an output constraint — not a post-hoc justification.

This project does not claim universal correctness. It claims a **measurable reliability behavior under stated conditions** and shows the variability of that behavior across 10 random seeds.

---

## 3. Method

### Data-generating process

Synthetic regression data, designed to be analytically clean:

$$X \in \mathbb{R}^{n \times d} \sim \mathcal{N}(0, I), \quad w^* \sim \mathcal{N}(0, I_d), \quad y = X w^* + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 0.36)$$

### Base model — closed-form ridge regression

$$\hat{w} = (X_{\text{tr}}^\top X_{\text{tr}} + \lambda I)^{-1} X_{\text{tr}}^\top y_{\text{tr}}$$

Ridge with $\lambda = 10^{-3}$ is chosen deliberately: it eliminates gradient-descent variance so the reliability layer is the only source of run-to-run variation.

### Split conformal calibration

Data is partitioned into **60% train / 20% calibration / 20% test** per seeded run.

On the calibration split, compute absolute residuals:

$$r_i = |y_i - \hat{y}_i| \quad \text{for } i \in \text{cal}$$

Choose the conformal quantile $q$ for miscoverage level $\alpha$ using the standard finite-sample rank rule (Vovk et al., 2005):

$$k = \left\lceil (n_{\text{cal}} + 1)(1 - \alpha) \right\rceil, \quad q = r_{(k)}$$

where $r_{(k)}$ is the $k$-th smallest calibration residual.

### Prediction intervals on test

For each test prediction $\hat{y}$, output the symmetric interval:

$$\hat{C}(x) = [\hat{y} - q, \; \hat{y} + q]$$

**Empirical coverage** = fraction of test targets falling inside their interval.  
**Average width** = $2q$ (constant per run, varies across seeds).

---

## 4. Assumptions

| Assumption | What it means | What breaks if violated |
|---|---|---|
| **Exchangeability** | Cal and test points are i.i.d.-like w.r.t. the residual distribution | Coverage guarantee disappears |
| **Representative calibration** | Cal split reflects test-time uncertainty | Intervals are systematically mis-sized |
| **Consistent measurement** | $y$ is measured the same way across splits | Silent distribution shift |
| **Symmetric errors** | Absolute residuals are appropriate nonconformity scores | Skewed distributions need asymmetric scores |
| **Synthetic setting** | Results demonstrate mechanics | External validity requires real-distribution testing |

---

## 5. Evaluation Protocol

| Parameter | Value |
|---|---|
| Data split | 60% train / 20% cal / 20% test |
| Miscoverage $\alpha$ | 0.10 (target coverage = 0.90) |
| Seeds | 10 (`range(10)`) |
| $n$ | 600 |
| $d$ | 8 |

**Metrics reported per seed:**

$$\text{Coverage} = \frac{1}{n_{\text{test}}} \sum_{i \in \text{test}} \mathbf{1}\!\left[y_i \in [\hat{y}_i - q, \hat{y}_i + q]\right]$$

- Average interval width: $\text{mean}(\text{hi} - \text{lo}) = 2q$
- $q$ itself (calibrated half-width)

**Aggregated across seeds:**
- Running mean and std tensors: `[coverage, avg_width, q]`
- Per-seed table
- Preview tensor for first 5 test points per seed: `[pred, lo, hi, y_true, covered]`

---

## 6. Results

Final metrics across 10 seeds:

```
tensor([[0.9333, 1.9711, 0.9856],
        [0.9333, 2.0347, 1.0173],
        [0.8500, 1.8821, 0.9411],
        [0.9000, 1.9047, 0.9524],
        [0.9083, 1.9973, 0.9986],
        [0.8667, 1.9267, 0.9633],
        [0.9000, 1.9746, 0.9873],
        [0.8667, 1.9302, 0.9651],
        [0.9083, 2.2466, 1.1233],
        [0.9250, 2.1458, 1.0729]])

Mean : [0.8992, 2.0014, 1.0007]
Std  : [0.0278, 0.1086, 0.0543]
```

**Key observations:**

- Mean coverage 0.899 ≈ 0.900 target — the finite-sample guarantee holds in expectation.
- 3/10 seeds fell below target (seeds 2, 5, 7 at 0.850, 0.867, 0.867). This is expected: the guarantee is *marginal*, not per-run.
- Width std (0.109) is 4× higher than coverage std (0.028), showing tightness is more sensitive to data variability than whether the target is met.
- Seed 8 produced the widest intervals ($q = 1.123$) — a harder calibration split by chance — yet coverage still held at 0.908, illustrating the width–coverage trade-off.

---

## 7. Failure Modes

Knowing when a method breaks is as important as knowing when it works.

| Failure mode | Mechanism | Signal to watch for |
|---|---|---|
| **Distribution shift** | Cal and test from different regimes | Coverage drops below $1-\alpha$ in deployment |
| **Non-exchangeable sampling** | Time trends, batch effects, selection bias | Systematic over- or under-coverage |
| **Label noise change** | Noise scale differs between cal and test | Intervals too tight (miss) or too wide (useless) |
| **Heavy tails / outliers** | A few calibration points dominate $q$ | Intervals inflate; width std spikes |
| **Small calibration set** | Coarse rank grid | High variance in $q$ across seeds |
| **Wrong score function** | Absolute residuals assume symmetry | Asymmetric errors → one-sided under-coverage |

---

## 8. Reproducibility

```bash
# Clone and install dependencies
pip install numpy torch

# Run with defaults (n=600, d=8, α=0.10, 10 seeds)
python conformal_prediction.py

# Custom run
python conformal_prediction.py --seeds 20 --alpha 0.05 --n 1200
```

Each run is fully seeded — results are deterministic per seed by design.

**Outputs logged:**
- Running mean/std tensors: `[coverage, avg_width, q]`
- Per-seed table with coverage bar
- Preview tensor for first 5 test points: `[pred, lo, hi, y_true, covered]`
- Final summary statistics (mean, std, min, max, seeds below target)

---

## 9. Takeaways

> **I stopped treating uncertainty as an aesthetic add-on. I treat it as a measurable constraint — coverage + width — that belongs alongside accuracy in every evaluation.**

Three shifts in how I think about ML evaluation:

1. **Two separate questions.** "Is the model accurate?" and "Are the outputs reliable enough to act on?" are different questions. Both need to be measured, not just one.

2. **Stability over single-seed scores.** Any reliability claim should survive reruns. Mean/std across seeds is the minimum; a single test metric is not evidence.

3. **Failure modes are first-class citizens.** Documenting when the method breaks (distribution shift, small cal set, wrong score function) is not a disclaimer — it is the most useful part of the writeup for anyone deploying this in practice.

---

## Output Summary

| Output | What it answers | Why it matters |
|---|---|---|
| Coverage per seed + mean/std | "How often do intervals contain the truth?" | Reliability target, not just accuracy score |
| Avg interval width + $q$ | "How informative are the intervals?" | Tightness vs usefulness trade-off |
| Preview tensor `[pred, lo, hi, y, covered]` | "What do intervals look like on real points?" | Sanity check; catches nonsense fast |
| Seeds below target count | "Is the guarantee stable?" | Shows marginal vs per-run coverage behavior |

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
