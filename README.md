# StochCalc — Statistical Arbitrage via OU Pairs Trading
### An LSTM-Enhanced Framework for Dynamic Hedge Ratios

A complete quantitative pairs trading system for NSE-listed Indian banking stocks. Built around the Ornstein–Uhlenbeck (OU) stochastic process, the system screens for cointegrated pairs, estimates mean-reversion dynamics, and deploys a deep learning model that continuously adapts the hedge ratio to the current market regime — all evaluated on a strictly held-out out-of-sample period with walk-forward validation.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [System Architecture](#3-system-architecture)
4. [Data & Configuration](#4-data--configuration)
5. [Pair Selection Pipeline](#5-pair-selection-pipeline)
6. [The LSTM Model — PairsLSTM](#6-the-lstm-model--pairslstm)
7. [Training Objective](#7-training-objective)
8. [Signal Generation](#8-signal-generation)
9. [Backtesting & Performance Metrics](#9-backtesting--performance-metrics)
10. [Walk-Forward Validation](#10-walk-forward-validation)
11. [Output & Visualisations](#11-output--visualisations)
12. [Dependencies & Usage](#12-dependencies--usage)
13. [Key Parameters Reference](#13-key-parameters-reference)

---

## 1. Background & Motivation

Pairs trading exploits the tendency of two economically linked securities to revert to a stable long-run relationship after temporary divergences. For Indian bank stocks, this relationship is driven by shared macro factors — RBI policy, NPA cycles, credit growth — that affect the entire sector but with varying lags and magnitudes across institutions.

**Why correlation-based pair selection fails.** Stock prices are typically I(1) processes (random walks). Two independent random walks can exhibit near-perfect correlation in finite samples — the phenomenon of *spurious regression* (Granger & Newbold, 1974). Correlation-based pair selection is therefore unreliable; what matters is that the *spread* between two prices is stationary, not that the prices themselves move together.

**Why a static hedge ratio is insufficient.** The Johansen cointegration estimate β\* is computed once on training data and held fixed. In practice, the cointegrating relationship drifts as NPA ratios, capital adequacy requirements, and RBI policy evolve. A fixed β\* progressively mis-hedges the pair, producing a spread that leaks away from stationarity and generating false entry signals precisely when the pair is decoupling.

This system solves both problems: rigorous cointegration-based pair selection, followed by a neural network that learns to track the time-varying cointegrating vector in real time.

---

## 2. Mathematical Foundation

### 2.1 The Ornstein–Uhlenbeck Process

The spread Z_t between two cointegrated stocks is modelled as an OU process (Definition 1.1 in the accompanying paper):

```
dZ_t = θ(µ − Z_t) dt + σ dW_t
```

where:
- **θ > 0** — mean reversion speed. Larger θ means faster pull back to equilibrium.
- **µ** — long-run equilibrium of the spread.
- **σ > 0** — diffusion coefficient (spread volatility).
- **W_t** — standard Brownian motion.

The drift term θ(µ − Z_t) always opposes deviations from µ, making the process self-correcting. Applying Itô's lemma to f(t, Z_t) = e^(θt) Z_t yields the closed-form solution (Theorem 1.1):

```
Z_t = Z_0 · e^(−θt) + µ(1 − e^(−θt)) + σ ∫₀ᵗ e^(−θ(t−s)) dW_s
```

As t → ∞, the distribution converges to the Gaussian stationary distribution:

```
Z_∞ ~ N(µ, σ²/2θ)
```

The stationary standard deviation **σ_∞ = sqrt(σ²/2θ)** — referred to as Theorem 1.2 in the paper — is used to normalise the z-score at inference time rather than a rolling empirical std (see §8).

### 2.2 Half-Life of Mean Reversion

The half-life τ₁/₂ is the expected time for a deviation from µ to shrink by half:

```
τ₁/₂ = ln(2) / θ
```

This is the primary tradability criterion. A pair is considered viable when:

```
2 < τ₁/₂ < 180 days
```

- **Too short (< 2 days):** microstructure noise dominates; transaction costs erode all returns.
- **Too long (> 180 days):** capital is locked up across structural regime changes; the relationship may have permanently shifted.

### 2.3 Cointegration

Two I(1) price series S₁(t) and S₂(t) are cointegrated (Definition 2.1) if there exists a linear combination

```
Z_t = β₁ S₁(t) + β₂ S₂(t)
```

that is I(0) — stationary. The β vector is the *cointegrating vector* and β₁/β₂ gives the hedge ratio that makes the spread mean-reverting. This is fundamentally different from correlation: it is a structural, long-run equilibrium relationship.

### 2.4 Johansen Test

The Johansen procedure works within a VECM framework:

```
ΔX_t = ΠX_{t−1} + Σ Γ_i ΔX_{t−i} + ε_t,    Π = αβᵀ
```

The rank of Π equals the number of cointegrating relationships. Solving the generalised eigenvalue problem λS₁₁v = S₁₀S₀₀⁻¹S₀₁v yields eigenvalues λ̂₁ ≥ ... ≥ λ̂_n. The trace statistic:

```
Λ_trace(r) = −T Σ_{i=r+1}^{n} ln(1 − λ̂_i)
```

tests H₀: rank(Π) ≤ r. A pair passes if Λ_trace > 5% critical value (rank ≥ 1). The static hedge ratio is recovered from the leading eigenvector v̂₁ = (v₁, v₂):

```
β* = −v₁ / v₂
```

### 2.5 OU Parameter Estimation

Discretising the OU SDE (Euler–Maruyama) gives a regression of the form ΔZ_t = a + bZ_{t−1} + ε_t. OLS recovers:

```
θ̂ = −b̂        (mean reversion speed)
µ̂ = −â / b̂    (long-run mean)
σ̂ = std(ε̂_t)  (residual volatility)
τ₁/₂ = ln(2)/θ̂
```

### 2.6 Hurst Exponent

Quantifies long-range dependence via the scaling of lagged standard deviations:

```
σ(τ) ∝ τ^H
```

Estimated by log-log regression over lags 2 to min(100, N/4). Interpretation:
- **H < 0.5** — anti-persistent (mean-reverting). Required for pairs trading.
- **H = 0.5** — random walk (no memory).
- **H > 0.5** — persistent (trending).

The code enforces **H < 0.45** to ensure meaningfully anti-persistent behaviour.

---

## 3. System Architecture

```
Raw Prices (yfinance)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│               PAIR SELECTION PIPELINE               │
│  SSD Pre-screen → ADF Test → Johansen Test →        │
│  OU Parameter Estimation → Hurst Exponent Filter    │
└──────────────────────────┬──────────────────────────┘
                           │  Best pair + static β*, θ, µ, σ
                           ▼
┌─────────────────────────────────────────────────────┐
│                  LSTM TRAINING                      │
│  PairsLSTM: rolling window of log-prices →          │
│  (β̂_t, θ̂_t, µ̂_t) per timestep                     │
│  Loss: L_stat + λ · L_smooth                        │
└──────────────────────────┬──────────────────────────┘
                           │  Trained model weights
                           ▼
┌─────────────────────────────────────────────────────┐
│                   INFERENCE                         │
│  Dynamic spread  Z^dyn_t = S₁ − β̂_t · S₂           │
│  Dynamic z-score z̃_t = (Z_t − µ̂_t) / σ̂_∞,t        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              SIGNAL GENERATION & BACKTEST           │
│  Stateful entry/exit/stop-loss rules (1-period lag) │
│  vs. Static Johansen rolling-window baseline        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
              Walk-Forward Validation (3 folds)
              Performance Dashboard (PNG)
```

---

## 4. Data & Configuration

### Universe

Eight NSE-listed banking stocks covering the full spectrum of Indian public and private sector banks:

| Ticker | Bank |
|---|---|
| HDFCBANK.NS | HDFC Bank |
| ICICIBANK.NS | ICICI Bank |
| AXISBANK.NS | Axis Bank |
| KOTAKBANK.NS | Kotak Mahindra Bank |
| SBIN.NS | State Bank of India |
| BANKBARODA.NS | Bank of Baroda |
| INDUSINDBK.NS | IndusInd Bank |
| FEDERALBNK.NS | Federal Bank |

These stocks are economically linked through shared exposure to RBI policy, credit cycles, and sector-wide regulation — satisfying the fundamental prerequisite that the pairs be driven by common risk factors with idiosyncratic deviations.

### Global Parameters

| Parameter | Value | Notes |
|---|---|---|
| `DATE_RANGE` | 2019-01-01 → 2024-12-31 | Spans pre-COVID, COVID crash, and recovery |
| `TRAIN_FRAC` | 0.80 | ~1,200 training days |
| `TRANSACTION_COST` | 0.0005 (5 bps) | Applied per trade on turnover |
| `SEED` | 42 | NumPy + PyTorch seeds fixed for reproducibility |
| `DEVICE` | CUDA if available, else CPU | Auto-detected |

Price data is fetched with `yfinance` (`auto_adjust=True`). Any ticker with more than 10% missing observations is dropped; remaining gaps are forward-filled.

---

## 5. Pair Selection Pipeline

All filtering is performed exclusively on the **training set** to prevent data leakage into the out-of-sample evaluation.

### Step 1 — SSD Pre-screen

All C(n, 2) candidate pairs are ranked by the sum of squared deviations between normalised price paths:

```
SSD(i, j) = Σ_t  (S̃_i(t) − S̃_j(t))²,    S̃ = S / S(0)
```

This is a fast, model-free proxy for price co-movement, used purely to reduce the number of pairs passed to the expensive statistical tests. Only the top 20 pairs (lowest SSD) proceed.

### Step 2 — ADF Unit Root Test

Each individual price series must be I(1). The Augmented Dickey–Fuller test (with automatic lag selection via AIC) is applied to both series. A pair is retained only if **both** ADF p-values exceed 0.05, confirming that neither series is already stationary on its own. If either series were I(0), the resulting spread would be stationary by construction — which is trivial and not a genuine cointegrating relationship.

### Step 3 — Johansen Cointegration Test

The Johansen procedure tests for a cointegrating vector within a VECM framework (see §2.4). A pair passes if the trace statistic exceeds the 5% critical value, confirming at least one cointegrating relationship. The hedge ratio β\* is extracted from the leading eigenvector.

### Step 4 — OU Parameter Estimation

The cointegrated spread Z_t = S₁ − β\* · S₂ is fitted to a discretised OU process via OLS regression. A pair is retained if:

```
2 < τ₁/₂ < 180 days
```

The 180-day upper bound is a practical extension beyond the theoretical 30-day ceiling suggested in some literature, accommodating slower-reverting but still tradeable relationships in the Indian banking sector.

### Step 5 — Hurst Exponent

Computed via log-log regression over lags 2 to min(100, N/4). A pair is retained if **H < 0.45**. This goes slightly below the random-walk boundary of H = 0.5 to ensure meaningful anti-persistence.

### Pair Selection Output

The pair with the **shortest qualifying half-life** is selected — shortest half-life implies the most reliable and frequent mean-reversion cycles, maximising the number of tradeable opportunities within the test period.

### Filter Summary Table

| Step | Test | Criterion |
|---|---|---|
| 1 | SSD pre-screen | Lowest 20 pairs by normalised price deviation |
| 2 | ADF unit root | Both series: p > 0.05 (confirmed I(1)) |
| 3 | Johansen trace | Λ_trace > 5% critical value (cointegrated) |
| 4 | OU half-life | 2 < τ₁/₂ < 180 days |
| 5 | Hurst exponent | H < 0.45 (anti-persistent) |

---

## 6. The LSTM Model — PairsLSTM

### Motivation

The Johansen β\* is computed once and frozen. For Indian bank stocks, the cointegrating relationship drifts as NPA ratios, Basel III capital requirements, and RBI policy evolve across years. A fixed β\* progressively mis-hedges the pair, generating false signals precisely when the pair is decoupling.

The LSTM learns a mapping (following §3.1 of the paper):

```
(β̂_t, θ̂_t, µ̂_t) = f_ϕ(S_{t−w:t})
```

that reads a rolling window of price history and outputs current OU parameters, adapting to regime changes without any explicit re-fitting.

### Architecture

```
Input: (batch, window, 2)   — locally normalised log-prices of S1, S2
         │
         ▼
    2-layer stacked LSTM
    hidden_dim = 64, dropout = 0.2
         │
         ▼  last hidden state h_T
    ┌──────────────┬───────────────┬──────────────┐
    │  head_beta   │  head_theta   │   head_mu    │
    │  Linear(64,1)│  Linear(64,1) │  Linear(64,1)│
    └──────┬───────┴───────┬───────┴──────┬───────┘
           │               │               │
          β̂_t           exp(·)→θ̂_t       µ̂_t
```

Three independent linear heads project from the final hidden state h_T. The **exponential activation on the θ head** enforces strict positivity consistent with the OU requirement θ > 0, without hard constraints or gradient-blocking clamping.

Total parameter count: ~34,000 (small enough to train within minutes on CPU).

### Input Preprocessing

Each window is locally normalised (Remark 3.1 in the paper):

```
ℓ̃^(i)_t = (log S_i(t) − µ^(i)_w) / σ^(i)_w
```

This ensures the LSTM operates on dimensionless, approximately stationary quantities regardless of absolute price levels or scale differences between the two stocks. The LSTM's persistent cell state then captures slow-moving structural relationships across weeks to months.

### Adaptive Window Size

The window length is set adaptively from the OU half-life estimated during pair selection:

```
window = max(2 × τ₁/₂ + 5, 30)  days
```

Rationale: the OU process loses memory on the timescale of its half-life. A window of approximately 2τ₁/₂ ensures the LSTM sees a full mean-reversion cycle in its context, giving it sufficient history to distinguish regime-driven changes in β from noise.

---

## 7. Training Objective

Training is **fully unsupervised** — no price direction labels are needed or used. The model is rewarded purely for finding parameters that make the spread maximally stationary.

### Stationarity Loss (L_stat)

If the LSTM correctly tracks the cointegrating vector, the OU residuals within each window should be small and approximately i.i.d.:

```
ε^(t)_s = ΔZ^(t)_s − θ̂_t (µ̂_t − Z^(t)_{s−1})
```

The stationarity loss minimises their variance across the batch (Eq. 41–42 in the paper):

```
L_stat = (1/|B|) Σ_{t∈B} Var_s[ε^(t)_s]
```

A lower L_stat means the model's predicted parameters explain the spread dynamics well, leaving residuals that look more like i.i.d. white noise — the hallmark of a correctly specified OU fit.

### Smoothness Loss (L_smooth)

L_stat can be trivially gamed by flipping β̂_t violently between steps. The smoothness regulariser penalises rapid changes in the hedge ratio (Eq. 43):

```
L_smooth = (1/(|B|−1)) Σ_t (β̂_t − β̂_{t−1})²
```

This parallels the process-noise penalty Q⁻¹ in a Kalman filter: λ trades off tracking speed against stability.

### Combined Loss

```
L = L_stat + λ · L_smooth,   λ = 0.05
```

### Optimiser & Schedule

| Component | Setting |
|---|---|
| Optimiser | AdamW, lr = 3e-4, weight_decay = 1e-4 |
| LR Schedule | Cosine annealing, T_max = EPOCHS, eta_min = 1e-5 |
| Gradient clipping | max norm = 1.0 (prevents LSTM gradient explosion) |
| Epochs | 80; best checkpoint (lowest val loss) is restored |
| Validation split | Last 20% of training windows (temporal order, no shuffling) |

---

## 8. Signal Generation

### Dynamic Spread and Z-Score

At inference time, the LSTM produces β̂_t, θ̂_t, µ̂_t for each test-set time step. The dynamic spread and its z-score are computed as (Eq. 40 and 45 in the paper):

```
Z^dyn_t = S₁(t) − β̂_t · S₂(t)

σ̂_∞,t = sqrt(σ̂²_ε,t / (2θ̂_t))       ← OU stationary std (Theorem 1.2)

z̃^dyn_t = (Z^dyn_t − µ̂_t) / σ̂_∞,t
```

Using σ̂_∞,t — derived from the OU stationary variance — rather than a rolling empirical std is a deliberate design choice. It uses the *structural* spread volatility implied by the model's own parameter estimates, which is more stable and theoretically grounded. σ̂_ε,t is estimated from a rolling window of OU residuals of width ≈ `window` days.

### Static Baseline Z-Score

For comparison, the static Johansen strategy uses β\* and a rolling-window empirical z-score (Eq. 46):

```
Z^static_t = S₁(t) − β* · S₂(t)

z̃^static_t = (Z^static_t − µ_roll,t) / σ_roll,t
```

where µ_roll and σ_roll are rolling mean and std over max(2 × τ₁/₂, 5) days.

### Entry / Exit Rules (Eq. 47)

Signals are generated by a stateful rule — position state is carried across time steps:

| Condition | Action |
|---|---|
| z̃_t < −1.5 | Enter **long** spread (+1): spread below equilibrium |
| z̃_t > +1.5 | Enter **short** spread (−1): spread above equilibrium |
| position ≠ 0 and \|z̃_t\| < 0.3 | **Exit** to flat (0): spread has reverted to mean |
| \|z̃_t\| > 3.0 | **Stop-loss**: exit immediately, extreme deviation |
| otherwise | Hold current position |

A one-period lag is applied at backtest time (`signal[t−1]` drives `return[t]`) to eliminate lookahead bias, as required by Eq. 58 in the paper.

---

## 9. Backtesting & Performance Metrics

### Spread Returns (Eq. 48)

```
r^Z_t = (Z_t − Z_{t−1}) / |Z_{t−1}|
```

### Strategy Returns with Transaction Costs (Eq. 49–50)

```
r^strat_t = Signal_{t−1} · r^Z_t

turnover_t = |Signal_t − Signal_{t−1}|

r^net_t = r^strat_t − c · turnover_t,   c = 5 bps
```

### Cumulative Wealth (Eq. 51)

```
W_t = Π_{s=1}^{t} (1 + r^net_s)
```

### Performance Metrics

| Metric | Formula | Eq. |
|---|---|---|
| **Sharpe Ratio** | (mean r^net / std r^net) × √252 | Eq. 52 |
| **Annualised Return** | W_T^(252/T) − 1 | — |
| **Max Drawdown** | max_{s≤t}(W_s − W_t) / W_s | Eq. 53 |
| **Calmar Ratio** | Ann. Return / MDD | Eq. 54 |
| **Win Rate** | fraction of days with r^net > 0 | Eq. 56 |
| **Profit Factor** | Σ(positive r^net) / Σ(\|negative r^net\|) | Eq. 57 |
| **Trade Count** | count of days with turnover > 0 | — |

**Sharpe interpretation used in the printed summary:**

| SR | Grade |
|---|---|
| > 2.0 | Exceptional |
| 1.0 – 2.0 | Good |
| 0.5 – 1.0 | Acceptable |
| < 0.5 | Poor |

---

## 10. Walk-Forward Validation

The main backtest uses a single 80/20 train/test split. Walk-forward validation provides a robustness check by re-estimating parameters on each expanding window and testing on the immediately following out-of-sample period.

**Setup:** 3 folds, each approximately `len(prices) // 4` trading days.

**Per fold:**
1. Re-run Johansen + OU estimation on the fold's training set.
2. Build rolling windows and train a fresh `PairsLSTM` from scratch (40 epochs).
3. Run inference on the fold's test window (prepending the last `window` rows of training data for initial context).
4. Run the full backtest and record OOS Sharpe, annualised return, and MDD.

**Robustness criterion (Eq. 59 in the paper):**

```
Mean OOS Sharpe > 0.5  →  ROBUST
Mean OOS Sharpe ≤ 0.5  →  MARGINAL
```

Walk-forward OOS Sharpe is naturally lower than the main backtest because each fold uses a shorter training history and trains for fewer epochs. What matters most is consistency: low variance in OOS Sharpe across folds indicates the strategy is not overfitted to a specific subperiod.

---

## 11. Output & Visualisations

### Printed Summaries

The script prints at each major stage:

- Data load: tickers retained, date range, total trading days
- Train/test split dates and day counts
- All pairs that passed every filter, with their β, half-life, Hurst, θ, σ, trace statistic
- Selected best pair with full OU parameters and Sharpe interpretation
- Signal distribution (counts of +1, −1, 0) for both strategies on the test set
- Performance table: LSTM Dynamic vs Static Johansen on all metrics
- Walk-forward results per fold and mean OOS Sharpe with ROBUST / MARGINAL verdict

### `backtest_dashboard.png`

A 4×2 figure with six panels:

| Panel | Content |
|---|---|
| Top (full width) | Cumulative wealth curve — LSTM Dynamic vs Static Johansen |
| Middle-left | Time-varying β̂_t (LSTM) vs static β\* over test period |
| Middle-right | Adaptive half-life τ̂₁/₂(t) = ln(2)/θ̂_t vs static baseline, capped at 60 days |
| Lower (full width) | Dynamic z-score with entry/exit/stop thresholds; long/short periods shaded green/red |
| Bottom-left | Drawdown curves (filled area) for both strategies |
| Bottom-right | Bar chart comparing Sharpe, Calmar, Profit Factor, Win Rate×10 |

---

## 12. Dependencies & Usage

### Requirements

All packages are auto-installed at runtime via pip:

```
yfinance        — market data download from Yahoo Finance
statsmodels     — ADF test, Johansen VECM, OLS regression
pandas          — data wrangling and time-series alignment
numpy           — numerical computation
matplotlib      — all plotting and dashboard generation
seaborn         — styling (imported for rcParams compatibility)
torch           — LSTM model definition and training loop
scikit-learn    — (indirect dependency)
tqdm            — training progress bars
```

A CUDA-capable GPU is used automatically if available; the script falls back to CPU. GPU is particularly beneficial during the walk-forward loop, which trains 3 separate models from scratch.

### Running

```bash
python stochcalc.py
```

The script runs linearly from top to bottom and is compatible with both plain Python and Jupyter/Colab environments.

### Customisation

**Change the asset universe:**
```python
TICKERS = ["HDFCBANK.NS", "ICICIBANK.NS", ...]   # any yfinance-compatible tickers
```

**Change the date range:**
```python
DATE_RANGE = ("2019-01-01", "2024-12-31")
```

**If no qualifying pairs are found, try relaxing filters:**
```python
HALF_LIFE_MAX = 180   # extend the upper half-life bound
HURST_MAX     = 0.48  # relax slightly toward random-walk boundary
TOP_N_SSD     = 30    # check more candidates before statistical filtering
```

**Adjust trading thresholds:**
```python
Z_ENTRY = 1.5   # z-score magnitude to open a position
Z_EXIT  = 0.3   # z-score magnitude at which to close
Z_STOP  = 3.0   # stop-loss z-score magnitude
```

**LSTM hyperparameters:**
```python
EPOCHS        = 80      # main model training epochs
LR            = 3e-4    # AdamW learning rate
LAMBDA_SMOOTH = 0.05    # smoothness regularisation weight λ
BATCH_SIZE    = 64      # mini-batch size
```

---

## 13. Key Parameters Reference

### Pair Selection

| Parameter | Default | Description |
|---|---|---|
| `TOP_N_SSD` | 20 | Number of top SSD pairs passed to Johansen |
| `ADF_PVAL_MIN` | 0.05 | Minimum ADF p-value confirming I(1) |
| `HALF_LIFE_MIN` | 2 | Minimum tradeable half-life (days) |
| `HALF_LIFE_MAX` | 180 | Maximum tradeable half-life (days) |
| `HURST_MAX` | 0.45 | Maximum Hurst exponent |

### LSTM Model

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | 64 | LSTM hidden state dimension |
| `num_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Dropout applied between LSTM layers |
| `WINDOW` | max(2×τ₁/₂+5, 30) | Adaptive rolling window (days) |
| `EPOCHS` | 80 | Main model training epochs |
| `WFV_EPOCHS` | 40 | Epochs per walk-forward fold |
| `LR` | 3e-4 | AdamW initial learning rate |
| `LAMBDA_SMOOTH` | 0.05 | Smoothness loss weight λ |
| `BATCH_SIZE` | 64 | Mini-batch size |

### Signal & Backtest

| Parameter | Default | Description |
|---|---|---|
| `Z_ENTRY` | 1.5 | Entry threshold (z-score) |
| `Z_EXIT` | 0.3 | Exit threshold (z-score) |
| `Z_STOP` | 3.0 | Stop-loss threshold (z-score) |
| `TRANSACTION_COST` | 0.0005 | Transaction cost per trade (5 bps) |
| `TRAIN_FRAC` | 0.80 | Proportion of data used for training |
| `N_FOLDS` | 3 | Walk-forward validation folds |

---

## References

- Granger, C.W.J. & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111–120.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.
- Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of the Brownian motion. *Physical Review*, 36(5), 823–841.
- Gatev, E., Goetzmann, W.N. & Rouwenhorst, K.G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.
