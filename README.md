# Genetic Programming Crypto Trading Strategy

A **Genetic Programming (GP)** system that evolves symbolic mathematical formulas
to trade **BTCUSDT** on Binance spot markets using multi-asset context from
5 cryptocurrency pairs on 1-hour candles.

## Overview

The system uses [DEAP](https://github.com/DEAP/deap) (Distributed Evolutionary
Algorithms in Python) to evolve expression trees that take **normalised OHLC
returns** from multiple crypto assets and output a trading signal (desired
portfolio exposure from -100% to +100%).

**Key principle:** Instead of hand-coding technical indicators (RSI, MACD, etc.),
the GP algorithm discovers its own mathematical formulas by combining raw price
data with arithmetic, trigonometric, and comparison operators through evolution.

### Trading Pairs

| Symbol   | Role              | Description                          |
|----------|-------------------|--------------------------------------|
| BTCUSDT  | Primary (traded)  | Bitcoin — the instrument being traded |
| ETHUSDT  | Context feature   | Ethereum — highest BTC correlation   |
| SOLUSDT  | Context feature   | Solana — high-beta Layer-1           |
| BNBUSDT  | Context feature   | Binance Coin — exchange token        |
| LTCUSDT  | Context feature   | Litecoin — oldest altcoin, stable correlation |

The GP tree receives **20 inputs** (5 pairs × 4 OHLC values, normalised to
1-bar percentage returns) and outputs a single number: the desired BTC
exposure percentage.

### Trading Mode

The system uses **margin trading** to support both long and short positions:
- Signal > 0 → **Long** BTC (buy)
- Signal < 0 → **Short** BTC (borrow & sell)
- Signal ≈ 0 → **Flat** (no position)

Default leverage: 5× margin.

## Architecture

```
crypto_data_downloader.py     ← Step 1: Download Binance spot 1h data
        ↓
    data/*.csv                ← Cached OHLCV CSVs
        ↓
gp_crypto_strategy.py        ← Step 2: GP evolution (backtesting.py engine)
gp_crypto_strategy_vectorbt.py  ← Alternative: vectorbt engine (faster)
        ↓
    best_individual.dill      ← Saved best GP tree
        ↓
crypto_trading_system.py      ← Step 3: Inference, backtesting, walk-forward
```

### File Descriptions

| File | Purpose |
|------|---------|
| `crypto_data_downloader.py` | Downloads Binance spot 1h OHLCV data via `ccxt` |
| `gp_crypto_strategy.py` | Main GP evolution using `backtesting.py` engine |
| `gp_crypto_strategy_vectorbt.py` | Alternative GP evolution using `vectorbt` (vectorised, faster) |
| `crypto_trading_system.py` | Model loading, backtesting, signal generation, walk-forward analysis, production bot skeleton |
| `requirements.txt` | Python dependencies |
| `data/` | Directory for cached CSV data (created by downloader) |
| `best_individual.dill` | Saved best GP tree (created after training) |

## Setup

### Prerequisites

- Python 3.10+
- Internet connection (for downloading Binance data)

### Installation

```bash
# Clone or navigate to the project directory
cd Genetics_Trading_Part_1

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `vectorbt` is optional. If you only plan to use the `backtesting.py`
> engine, you can skip it. Remove the `vectorbt` line from `requirements.txt`
> before installing.

## Usage

### Step 1: Download Data

Download historical 1-hour candle data from Binance:

```bash
# Full download (2021-07-01 to now, ~5 minutes)
python crypto_data_downloader.py

# Update existing data (append new candles only)
python crypto_data_downloader.py --update

# Download specific symbols
python crypto_data_downloader.py --symbols BTC/USDT ETH/USDT
```

This creates CSV files in the `data/` directory:
```
data/BTCUSDT_1h.csv
data/ETHUSDT_1h.csv
data/SOLUSDT_1h.csv
data/BNBUSDT_1h.csv
data/LTCUSDT_1h.csv
```

### Step 2: Train the GP Model

Run the genetic programming evolution:

```bash
# Using backtesting.py engine (default, more accurate)
python gp_crypto_strategy.py

# Using vectorbt engine (faster, good for experimentation)
python gp_crypto_strategy_vectorbt.py
```

The evolution pipeline:
1. Loads and merges all 5 pairs into a synchronized dataset
2. Splits into Train (2021-07 → 2024-01), Validation (2024-01 → 2024-07),
   Test (2024-07 → 2025-03)
3. Evolves 7,500 GP trees over 15 generations (backtesting.py) or
   1,000 × 15 (vectorbt)
4. Selects the best tree based on validation Sharpe ratio
5. Runs final out-of-sample test
6. Saves the winner to `best_individual.dill`

**Tuning tip:** For initial experiments, reduce `POP_SIZE` and `N_GEN` in the
script constants. The vectorbt version uses smaller defaults (1000 × 15) and
is significantly faster per evaluation.

### Step 3: Backtest & Analyse

After training, use the trading system module:

```bash
# Backtest the saved model on the test period
python crypto_trading_system.py
```

#### Python API Examples

```python
from pathlib import Path
from crypto_trading_system import (
    GPModelManager, backtest_saved_model,
    generate_signals, walk_forward_analysis, demo_live_trading
)
from gp_crypto_strategy import load_all_pairs

# --- Backtest on a custom date range ---
stats = backtest_saved_model(
    model_path="best_individual.dill",
    start_date="2024-07-01",
    end_date="2025-03-07",
    plot=True,                # generates an interactive chart
)

# --- Generate trading signals ---
mm = GPModelManager("best_individual.dill")
mm.load_model()

data = load_all_pairs()
signals = generate_signals(mm, data, "2025-01-01", "2025-03-01")
print(signals.head())
#                          signal   btc_close
# timestamp
# 2025-01-01 01:00:00      42.3     94250.0
# 2025-01-01 02:00:00     -18.7     93800.0
# ...

# --- Walk-forward analysis ---
results = walk_forward_analysis(
    model_path="best_individual.dill",
    window_months=6,
    step_months=1,
)

# --- Live trading demo (synthetic data) ---
demo_live_trading()
```

#### Using the Model Directly

```python
from gp_crypto_strategy import toolbox
import dill

# Load the saved GP tree
with open("best_individual.dill", "rb") as f:
    best_tree = dill.load(f)

# Compile to a callable function
func = toolbox.compile(expr=best_tree)

# Call with 20 OHLC values (5 pairs × 4)
signal = func(
    # BTCUSDT: Open, High, Low, Close
    95000.0, 96000.0, 94500.0, 95500.0,
    # ETHUSDT
    3200.0, 3250.0, 3150.0, 3220.0,
    # SOLUSDT
    145.0, 148.0, 142.0, 146.0,
    # BNBUSDT
    620.0, 625.0, 615.0, 622.0,
    # LTCUSDT
    85.0, 87.0, 83.0, 86.0,
)
print(f"Signal: {signal:.1f}%")  # e.g. +42.3 = go 42.3% long BTC
```

## Configuration

Key parameters in `gp_crypto_strategy.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POP_SIZE` | 7500 | Population size per generation |
| `N_GEN` | 15 | Number of evolutionary generations |
| `P_CX` | 0.90 | Crossover probability |
| `P_MUT` | 0.15 | Mutation probability |
| `MAX_DEPTH` | 8 | Maximum GP tree depth |
| `MAX_LEN` | 60 | Maximum nodes per tree |
| `INITIAL_CASH` | 100,000 | Starting capital (USD) |
| `COMMISSION_PCT` | 0.001 | Trading fee (0.1% per trade) |
| `MARGIN` | 1/5 | Margin ratio (5× leverage) |
| `NO_TRADE_BAND` | 15 | Dead-band filter (±15 percentage points) |
| `MAX_TRADES` | 500 | Overtrading penalty threshold |

### Data Period

| Split | Period | Bars (~) |
|-------|--------|----------|
| Train | 2021-07-01 → 2024-01-01 | 21,900 |
| Validation | 2024-01-01 → 2024-07-01 | 4,344 |
| Test | 2024-07-01 → 2025-03-07 | 5,800 |

### Input Normalisation

All 20 OHLC inputs are converted from raw prices to **1-bar percentage
returns** before being fed to the GP tree.  This prevents the GP from
exploiting absolute price-scale differences between symbols (e.g. BTC at
~\$90,000 vs SOL at ~\$150) and forces the evolved formulas to learn
meaningful cross-asset relationships rather than trivial price comparisons.

Raw BTCUSDT prices are preserved separately (`Raw_Open/High/Low/Close`
columns) so that the execution engine can still place orders at real prices.

### GP Primitives

The expression trees can use these building blocks:

- **Arithmetic:** `add`, `sub`, `mul`, `protected_div`
- **Trigonometric:** `sin`, `cos`, `tan`, `tanh`
- **Comparison:** `greater(a, b)` → +100 if a > b, else -100
- **Constants:** random float in [-1, 1]

### Fitness Function

The fitness is based on **Sharpe Ratio** (minimise `-sharpe + 0.001 * tree_size`):
- Sharpe Ratio rewards consistent risk-adjusted returns
- Parsimony pressure discourages bloated expression trees
- Penalties applied for: drawdown > 40%, fewer than 20 trades, >500 trades, account blowup

## Project History

This project was originally designed for **FX currency trading** (EURUSD,
GBPUSD, AUDUSD, USDJPY) on 5-minute candles. It was refactored to target
cryptocurrency markets with the following changes:

- FX pairs → Crypto pairs (BTCUSDT + 4 context assets)
- 5-minute → 1-hour timeframe
- FX lot sizing → fractional BTC sizing
- FX commission (0.0015%) → Binance spot fee (0.1%)
- Dukascopy CSV → Binance via ccxt
- Added margin support for long/short trading
- Raw return fitness → Sharpe-based fitness
- Added input normalisation (raw OHLC prices → 1-bar percentage returns)
- Added parsimony pressure and overtrading penalties

The original FX files
`trading_system_Load_Infer.py`) are preserved in the repository for reference.

## License

This project is for educational and research purposes.
