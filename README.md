# Mathematical Reliability for ML Predictions

> *Most ML pipelines answer "how accurate is this model?" This project answers a harder question: "how often can you trust a specific output?"*

---

## What this is

A minimal, self-contained Python implementation of **split conformal prediction** for regression — a distribution-free method that wraps any point-prediction model in calibrated uncertainty intervals with a provable finite-sample coverage guarantee.

No neural networks. No hyperparameter tuning. No plotting libraries. Just the math, cleanly implemented and measured across 10 random seeds so the reliability claim can be verified, not just asserted.

**Core result:** Across 10 seeds, empirical coverage tracks the 90% target with mean 0.899 and std 0.028 — demonstrating that the finite-sample guarantee holds in expectation while individual runs show expected marginal variation.

---

## Why it matters

A model that achieves 90% test accuracy is not the same as a model that achieves **90% coverage** on its uncertainty intervals. The first is a performance metric; the second is a reliability contract.

Split conformal prediction converts any point predictor into one that provides that contract — without modifying the model and without assuming a parametric noise distribution. The only assumption is exchangeability between calibration and test data, which is explicitly stated and testable.

This is relevant in:
- **Medical / clinical settings** — "how confident is this risk estimate?" has a direct answer
- **Model audits** — coverage is a concrete metric for evaluating reliability, not a vibe
- **Production deployment** — width + coverage can be treated as output constraints in SLA definitions

---

## Quick start

```bash
pip install numpy torch

# Defaults: n=600, d=8, α=0.10, 10 seeds
python conformal_prediction.py

# Custom settings
python conformal_prediction.py --seeds 20 --alpha 0.05 --n 1200 --d 16
```

**CLI arguments:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 600 | Dataset size |
| `--d` | 8 | Feature dimension |
| `--alpha` | 0.10 | Miscoverage level (1-α = target coverage) |
| `--seeds` | 10 | Number of random seeds |
| `--lam` | 0.001 | Ridge regularisation λ |
| `--refresh` | 0.15 | Dashboard refresh delay (seconds) |

---

## How it works

```
Data (n=600, d=8)
       │
       ├── 60% ──► Train ──► Ridge fit: ŵ = (X'X + λI)⁻¹ X'y
       │
       ├── 20% ──► Calibration ──► Residuals: rᵢ = |yᵢ - ŷᵢ|
       │                           Quantile:  q = r₍ₖ₎
       │                           where k = ⌈(n_cal + 1)(1 - α)⌉
       │
       └── 20% ──► Test ──► Intervals: [ŷ - q, ŷ + q]
                            Coverage:  fraction of y inside interval
```

The conformal quantile $q$ is the only free parameter produced by calibration. It does not depend on model internals — only on the empirical residual distribution of the calibration split. That is what makes the coverage guarantee distribution-free.

---

## Results (10 seeds, n=600, d=8, α=0.10)

```
             coverage   avg_width    q
seed  0       0.9333      1.9711   0.9856
seed  1       0.9333      2.0347   1.0173
seed  2       0.8500  *   1.8821   0.9411
seed  3       0.9000      1.9047   0.9524
seed  4       0.9083      1.9973   0.9986
seed  5       0.8667  *   1.9267   0.9633
seed  6       0.9000      1.9746   0.9873
seed  7       0.8667  *   1.9302   0.9651
seed  8       0.9083      2.2466   1.1233
seed  9       0.9250      2.1458   1.0729
─────────────────────────────────────────
Mean          0.8992      2.0014   1.0007
Std           0.0278      0.1086   0.0543
```

`*` = below 0.90 target (3/10 seeds). Expected under a marginal guarantee — the finite-sample guarantee applies on average, not per-run.

---

## Assumptions

This method has a short, explicit assumption list. Violations are documented:

| Assumption | What breaks if violated |
|---|---|
| Exchangeability between cal and test | Coverage guarantee disappears |
| Representative calibration split | Intervals are mis-sized for deployment |
| Consistent label measurement | Silent distribution shift |
| Symmetric absolute-error scores | Skewed errors → one-sided under-coverage |

See [`writeup.md`](writeup.md) for a full failure-mode analysis.

---

## Repository layout

```
├── conformal_prediction.py   # Core implementation + terminal dashboard
├── output.txt                # Annotated run output (10 seeds)
├── writeup.md                # Full technical writeup with math, assumptions, failure modes
└── README.md                 # This file
```

---

## Design decisions worth noting

**Why ridge regression instead of a neural network?**  
To isolate the reliability layer as the variable of interest. Training-loop noise would make it harder to attribute coverage variation to conformal calibration vs model instability.

**Why 10 seeds with mean/std instead of one run?**  
A single test metric is not a reliability claim. Stability across seeds is. The running mean/std dashboard shows when coverage has converged, not just what the final number is.

**Why terminal output instead of plots?**  
Tensor outputs are reproducible, diffable, and embeddable in CI pipelines. Plots are not.

---

## Extending this

| Extension | Change needed |
|---|---|
| Different base model | Replace `ridge_fit()` with any regressor |
| Asymmetric intervals | Replace absolute residuals with a directional score |
| Covariate-shift robustness | Use weighted conformal prediction (Tibshirani et al., 2019) |
| Tighter intervals | Switch to locally-adaptive scores (normalised residuals) |
| Real dataset | Swap the synthetic data generator; keep everything else |

---

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.

---
