"""
Split Conformal Prediction — Finite-Sample Reliability Layer
=============================================================
Purpose: Add measurable uncertainty bounds to point predictions without
         touching model internals. Uses split conformal calibration to
         produce prediction intervals with a finite-sample coverage
         guarantee under an exchangeability assumption.

Key design choices
------------------
- Ridge regression (closed-form) is used *intentionally* so that the
  reliability layer is the focus, not training-loop noise.
- Calibration quantile follows the standard (n+1) finite-sample rule
  (Vovk et al., 2005; Angelopoulos & Bates, 2023).
- All per-seed metrics are stored as PyTorch tensors so downstream
  aggregation is trivial.
- Terminal dashboard prints running mean/std across seeds so coverage
  stability is visible in real time, not only at the end.

Usage
-----
    python conformal_prediction.py               # defaults
    python conformal_prediction.py --seeds 20 --alpha 0.05 --n 1000

Dependencies: numpy, torch  (no plotting libraries required)
"""

from __future__ import annotations

import argparse
import time
from typing import NamedTuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Core mathematical primitives
# ---------------------------------------------------------------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge regression: w = (X'X + λI)^{-1} X'y.

    Closed-form is preferred here because it eliminates gradient-descent
    variance, keeping the reliability layer as the only source of run-to-run
    variation.
    """
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    """Return the (ceil((n+1)(1-alpha)) / n)-th order statistic of residuals.

    This is the standard finite-sample rank rule from split conformal
    prediction. It guarantees marginal coverage ≥ 1-alpha when calibration
    and test data are exchangeable.

    Reference: Angelopoulos & Bates (2023), "A Gentle Introduction to
    Conformal Prediction and Distribution-Free Uncertainty Quantification."
    """
    n = abs_residuals.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)           # clamp to valid index range
    return float(np.sort(abs_residuals)[k - 1])


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

class RunResult(NamedTuple):
    metrics: torch.Tensor   # shape [3]: [coverage, avg_width, q]
    preview: torch.Tensor   # shape [k, 5]: [pred, lo, hi, y_true, covered]
    seed: int


def run_once(
    seed: int,
    n: int = 600,
    d: int = 8,
    alpha: float = 0.10,
    lam: float = 1e-3,
    noise_scale: float = 0.6,
) -> RunResult:
    """Run one full experiment: generate data, fit, calibrate, evaluate.

    Data-generating process
    -----------------------
      X ~ N(0, I_{n×d})
      w* ~ N(0, I_d)
      y  = X w* + ε,  ε ~ N(0, noise_scale²)

    Split: 60% train / 20% calibration / 20% test (random permutation).

    Returns
    -------
    RunResult with:
      metrics : tensor([coverage, avg_width, q])
      preview : tensor of first ≤5 test rows, cols = [pred, lo, hi, y, covered]
      seed    : the seed used
    """
    rng = np.random.default_rng(seed)

    # --- Data generation ---
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=(d,))
    noise = noise_scale * rng.normal(size=(n,))
    y = X @ w_true + noise

    # --- Split ---
    idx = rng.permutation(n)
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    tr  = idx[:n_train]
    cal = idx[n_train : n_train + n_cal]
    te  = idx[n_train + n_cal :]

    X_tr,  y_tr  = X[tr],  y[tr]
    X_cal, y_cal = X[cal], y[cal]
    X_te,  y_te  = X[te],  y[te]

    # --- Fit ---
    w = ridge_fit(X_tr, y_tr, lam=lam)

    # --- Conformal calibration ---
    cal_pred = X_cal @ w
    abs_res  = np.abs(y_cal - cal_pred)
    q        = conformal_quantile(abs_res, alpha)

    # --- Evaluate on test ---
    te_pred = X_te @ w
    lo      = te_pred - q
    hi      = te_pred + q
    covered = (y_te >= lo) & (y_te <= hi)

    coverage  = float(np.mean(covered))
    avg_width = float(np.mean(hi - lo))   # = 2q (symmetric intervals)

    # --- Pack into tensors ---
    metrics = torch.tensor([coverage, avg_width, q], dtype=torch.float32)

    k = min(5, te_pred.shape[0])
    preview = torch.tensor(
        np.stack(
            [te_pred[:k], lo[:k], hi[:k], y_te[:k], covered[:k].astype(float)],
            axis=1,
        ),
        dtype=torch.float32,
    )
    return RunResult(metrics=metrics, preview=preview, seed=seed)


# ---------------------------------------------------------------------------
# Terminal dashboard helpers
# ---------------------------------------------------------------------------

def _clear() -> None:
    print("\033[2J\033[H", end="")


def _bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 20) -> str:
    """Inline ASCII progress bar."""
    if hi <= lo:
        hi = lo + 1e-9
    x    = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    fill = int(round(x * width))
    return "█" * fill + "░" * (width - fill)


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Dashboard render
# ---------------------------------------------------------------------------

def render_dashboard(
    rows: list[torch.Tensor],
    previews: dict[int, torch.Tensor],
    seeds_done: list[int],
    total_seeds: int,
    n: int,
    d: int,
    alpha: float,
) -> None:
    """Re-render the full terminal dashboard in place."""
    target = 1.0 - alpha
    table  = torch.stack(rows)                          # [k, 3]
    mean   = table.mean(dim=0)
    std    = (
        table.std(dim=0, unbiased=False)
        if table.shape[0] > 1
        else torch.zeros(3)
    )

    cov_mean   = mean[0].item()
    width_mean = mean[1].item()
    q_mean     = mean[2].item()

    _clear()
    sep = "─" * 72

    print("┌" + sep + "┐")
    print(f"│  Split Conformal Prediction — Reliability Dashboard{'':>19}│")
    print(f"│  Assumption: exchangeability / i.i.d.{'':>34}│")
    print(f"│  α = {alpha:.2f}  │  target coverage = {target:.2f}  │  "
          f"n = {n}  d = {d}  │  seeds {len(rows)}/{total_seeds}{'':>6}│")
    print("└" + sep + "┘")

    print()
    print("  Running statistics  (tensor: [coverage, avg_width, q])")
    print(f"  mean : {mean}")
    print(f"  std  : {std}")
    print()

    # Coverage bar — highlight if below target
    cov_flag = "  ✓" if cov_mean >= target else "  ← below target"
    print(f"  coverage   {_fmt(cov_mean)}  {_bar(cov_mean, 0.0, 1.0)}  "
          f"(target {target:.3f}){cov_flag}")
    print(f"  avg width  {_fmt(width_mean)}  "
          f"{_bar(width_mean, 0.0, max(0.1, width_mean * 2))}")
    print(f"  q (half)   {_fmt(q_mean)}  "
          f"{_bar(q_mean, 0.0, max(0.1, q_mean * 2))}")
    print()

    # Per-seed table
    print("  Per-seed results")
    print(f"  {'seed':>4}  {'coverage':>8}  {'avg_width':>9}  {'q':>7}  coverage bar")
    print("  " + "─" * 64)
    for j, seed in enumerate(seeds_done):
        r   = table[j]
        cov = r[0].item()
        aw  = r[1].item()
        qq  = r[2].item()
        flag = " *" if cov < target else "  "
        print(
            f"  {seed:>4d}  {_fmt(cov):>8s}  {_fmt(aw):>9s}  "
            f"{_fmt(qq):>7s}  {_bar(cov, 0.0, 1.0)}{flag}"
        )
    print("  (* = below target coverage)")
    print()

    # Preview tensor for the most recent seed
    last = seeds_done[-1]
    print(f"  Preview — seed {last} — first 5 test points")
    print("  cols: [pred, lo, hi, y_true, covered]")
    print(previews[last])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split conformal prediction — reliability dashboard"
    )
    p.add_argument("--n",      type=int,   default=600,  help="dataset size")
    p.add_argument("--d",      type=int,   default=8,    help="feature dimension")
    p.add_argument("--alpha",  type=float, default=0.10, help="miscoverage level (default 0.10 → 90%% target)")
    p.add_argument("--seeds",  type=int,   default=10,   help="number of random seeds")
    p.add_argument("--lam",    type=float, default=1e-3, help="ridge regularisation λ")
    p.add_argument("--refresh",type=float, default=0.15, help="dashboard refresh delay (seconds)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    seeds  = list(range(args.seeds))

    rows:    list[torch.Tensor]      = []
    previews: dict[int, torch.Tensor] = {}
    seeds_done: list[int]            = []

    for s in seeds:
        result = run_once(
            seed=s, n=args.n, d=args.d,
            alpha=args.alpha, lam=args.lam,
        )
        rows.append(result.metrics)
        previews[s]  = result.preview
        seeds_done.append(s)

        render_dashboard(
            rows, previews, seeds_done,
            total_seeds=len(seeds),
            n=args.n, d=args.d, alpha=args.alpha,
        )
        time.sleep(args.refresh)

    # Final summary tensor
    final = torch.stack(rows)
    print("\n" + "═" * 72)
    print("DONE — Final metrics tensor  [coverage, avg_width, q]")
    print("═" * 72)
    print(final)
    print()
    print(f"Mean  : {final.mean(dim=0)}")
    print(f"Std   : {final.std(dim=0, unbiased=False)}")
    print(f"Min   : {final.min(dim=0).values}")
    print(f"Max   : {final.max(dim=0).values}")

    target = 1.0 - args.alpha
    n_below = int((final[:, 0] < target).sum().item())
    print(f"\nSeeds below target coverage ({target:.2f}): {n_below}/{len(seeds)}")


if __name__ == "__main__":
    main()
