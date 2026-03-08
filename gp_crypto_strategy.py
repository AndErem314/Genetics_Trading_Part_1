"""
Genetic-Programming Crypto Strategy — backtesting.py edition
=============================================================
Evolves symbolic expression trees that take 20 OHLC inputs (5 crypto pairs)
and output a desired exposure percentage (-100 to +100) for BTCUSDT.
Uses margin mode for long/short trading on 1-hour candles.
"""

import math
import operator
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import dill
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from deap import base, creator, gp, tools

# Suppress backtesting.py margin warnings (orders are silently canceled;
# the fitness function already penalises poor individuals)
warnings.filterwarnings("ignore", message=".*insufficient margin.*")

# ─────────────────────────────────────────────────────────────────────────────
# Global random seed for reproducibility
# ─────────────────────────────────────────────────────────────────────────────
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LTCUSDT"]
PRIMARY_SYMBOL = "BTCUSDT"  # the traded instrument

# Canonical input order: 5 symbols × 4 OHLC = 20 names
ARG_NAMES = [f"{s}_{f}" for s in SYMBOLS for f in ("Open", "High", "Low", "Close")]

DATA_DIR = Path(__file__).parent / "data"

# Training / validation / test periods (UTC)
TRAIN_START = "2021-07-01 00:00:00"
TRAIN_END   = "2024-01-01 00:00:00"
VAL_START   = "2024-01-01 00:00:00"
VAL_END     = "2024-07-01 00:00:00"
TEST_START  = "2024-07-01 00:00:00"
TEST_END    = "2025-03-07 00:00:00"

# GP hyper-parameters
POP_SIZE  = 7500
N_GEN     = 15
P_CX      = 0.90
P_MUT     = 0.15
MAX_DEPTH = 8
MAX_LEN   = 60

# Backtest settings
INITIAL_CASH   = 100_000       # USD
COMMISSION_PCT = 0.001         # 0.1% Binance spot taker fee
MARGIN         = 1 / 5         # 5× leverage — gives headroom for long↔short swings
NO_TRADE_BAND  = 15            # ±15 pp dead-band (reduce overtrading)
BTC_TICK       = 0.00001       # minimum BTC position increment
MAX_TRADES     = 500           # penalise strategies exceeding this
RAW_CLOSE_COL  = "Raw_Close"   # raw price column preserved after normalisation

# Progress tracking
_evaluation_count = 0
_current_generation = 0
_start_time = None

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data handling
# ─────────────────────────────────────────────────────────────────────────────

def load_pair_csv(symbol: str, path: Path) -> pd.DataFrame:
    """Load a ccxt-format CSV (timestamp, open, high, low, close, volume)
    and return a DataFrame with columns prefixed by the symbol name.
    """
    print(f"📈 Loading {symbol} from {path.name} ...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Keep only OHLC (drop volume for GP inputs — consistent with FX version)
    df = df[["open", "high", "low", "close"]]
    df.columns = [f"{symbol}_{c.capitalize()}" for c in df.columns]

    print(f"   {symbol}: {len(df):,} bars loaded")
    return df


def load_all_pairs(data_dir: Path = DATA_DIR,
                   symbols: List[str] = None) -> pd.DataFrame:
    """Load all symbol CSVs into a wide DataFrame indexed by timestamp."""
    if symbols is None:
        symbols = SYMBOLS
    print(f"📦 Loading {len(symbols)} crypto pairs from {data_dir} ...")

    dfs = []
    for i, sym in enumerate(symbols):
        csv_path = data_dir / f"{sym}_1h.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run crypto_data_downloader.py first."
            )
        dfs.append(load_pair_csv(sym, csv_path))
        print(f"   Progress: {i + 1}/{len(symbols)} loaded")

    combined = pd.concat(dfs, axis=1).dropna()
    print(f"✅ Merged dataset: {len(combined):,} synchronized 1h bars")
    return combined


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) slices."""
    print("✂  Splitting dataset ...")
    train = df.loc[TRAIN_START:TRAIN_END].copy()
    val   = df.loc[VAL_START:VAL_END].copy()
    test  = df.loc[TEST_START:TEST_END].copy()

    print(f"   Train:      {len(train):>7,} bars  ({TRAIN_START} → {TRAIN_END})")
    print(f"   Validation: {len(val):>7,} bars  ({VAL_START} → {VAL_END})")
    print(f"   Test:       {len(test):>7,} bars  ({TEST_START} → {TEST_END})")
    return train, val, test


def normalize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw OHLC prices to 1-bar percentage returns for GP inputs.

    Prevents the GP from exploiting absolute price-scale differences
    between symbols (e.g. BTC ~90k vs SOL ~150).  All GP inputs become
    small numbers roughly in [-0.05, 0.05], while raw BTCUSDT prices are
    preserved in Raw_* columns for order execution.
    """
    df = df.copy()
    # Preserve raw BTCUSDT OHLC for backtesting.py execution
    prefix = f"{PRIMARY_SYMBOL}_"
    for field in ("Open", "High", "Low", "Close"):
        df[f"Raw_{field}"] = df[f"{prefix}{field}"].copy()

    # Convert all 20 GP input columns to 1-bar percentage returns
    for col in ARG_NAMES:
        df[col] = df[col].pct_change().fillna(0.0)

    print("📐 Inputs normalised to 1-bar percentage returns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def round_btc(x: float, tick: float = BTC_TICK) -> float:
    """Round *x* to nearest multiple of *tick*."""
    return round(x / tick) * tick


def pct_to_units(percent: float, equity: float, price: float) -> float:
    """Convert desired exposure % → raw BTC units at current price."""
    return percent / 100.0 * equity / price


# ─────────────────────────────────────────────────────────────────────────────
# 3. Strategy
# ─────────────────────────────────────────────────────────────────────────────

class GPExposureStrategy(Strategy):
    """Backtesting.py strategy driven by a GP expression tree."""
    commission = COMMISSION_PCT
    expression = None           # injected callable

    def init(self):
        self.current_pct = 0.0

    def next(self):
        # Build input vector (20 normalised-return values)
        inputs = [float(self.data.df[name].iloc[-1]) for name in ARG_NAMES]

        desired = float(self.expression(*inputs))
        # Clamp to safe range so margin is never exceeded
        desired = max(-100.0, min(100.0, desired))

        # Dead-band filter
        if abs(desired - self.current_pct) <= NO_TRADE_BAND:
            return

        broker = self._broker
        equity_now = getattr(broker, "equity", getattr(broker, "_equity"))
        price_now  = self.data.Close[-1]
        target = pct_to_units(desired, equity_now, price_now)
        delta  = target - self.position.size
        order  = round_btc(delta)

        if abs(order) < BTC_TICK:
            return
        if order > 0:
            self.buy(size=abs(order))
        else:
            self.sell(size=abs(order))
        self.current_pct = desired


# ─────────────────────────────────────────────────────────────────────────────
# 4. Genetic-Programming setup (DEAP)
# ─────────────────────────────────────────────────────────────────────────────
print("🧬 Setting up Genetic Programming framework ...")

pset = gp.PrimitiveSet("CRYPTO", len(ARG_NAMES), prefix="inp")

# Arithmetic
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

def protected_div(a, b):
    return a / b if abs(b) > 1e-8 else a
pset.addPrimitive(protected_div, 2)

# Trigonometric & activation
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(math.tanh, 1)

# Comparison → ±100 (long/short)
def greater(a, b):
    return 100.0 if a > b else -100.0
pset.addPrimitive(greater, 2)

# Ephemeral constant
pset.addEphemeralConstant("rand", lambda: random.uniform(-1.0, 1.0))

# Rename arguments for readability
for i, name in enumerate(ARG_NAMES):
    pset.renameArguments(**{f"inp{i}": name})

# Fitness & Individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_LEN))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_LEN))

print("✅ GP framework ready (20 inputs, ±100 long/short)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fitness evaluation via backtesting.py
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_bt_dataframe(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Build backtesting.py DataFrame.

    Raw BTCUSDT prices (from Raw_* columns) become Open/High/Low/Close
    for order execution.  All 20 normalised-return GP input columns
    (ARG_NAMES) are kept alongside for the strategy to read.
    """
    # Raw BTC prices for backtesting.py execution engine
    raw_cols = ["Raw_Open", "Raw_High", "Raw_Low", "Raw_Close"]
    btc = df_slice[raw_cols].copy()
    btc.columns = ["Open", "High", "Low", "Close"]

    # Normalised return columns for GP inputs
    gp_cols = df_slice[ARG_NAMES].copy()
    return btc.join(gp_cols)


def evaluate_individual(individual, df_train: pd.DataFrame) -> Tuple[float]:
    """Backtest on a data slice; return Sharpe-based fitness."""
    global _evaluation_count, _current_generation, _start_time
    _evaluation_count += 1

    if _evaluation_count % 1000 == 0:
        elapsed = time.time() - _start_time if _start_time else 0
        total_evals = POP_SIZE * N_GEN
        pct = _evaluation_count / total_evals * 100
        print(f"  Gen {_current_generation}: {_evaluation_count:,}/{total_evals:,} "
              f"({pct:.1f}%) | {elapsed:.1f}s")

    try:
        func = toolbox.compile(expr=individual)
        GPExposureStrategy.expression = staticmethod(func)

        bt_df = _prepare_bt_dataframe(df_train)
        bt = Backtest(
            bt_df, GPExposureStrategy,
            cash=INITIAL_CASH, commission=COMMISSION_PCT,
            margin=MARGIN,
            exclusive_orders=False, trade_on_close=True,
        )
        stats = bt.run()

        total_return = stats["Return [%]"] / 100.0
        n_trades     = stats["# Trades"]
        sharpe       = stats["Sharpe Ratio"]
        max_dd       = stats["Max. Drawdown [%]"]
        nav_final    = INITIAL_CASH * (1 + total_return)

        # Penalties
        if nav_final <= 0 or n_trades < 20:
            return (1e6,)
        if max_dd < -40:          # drawdown > 40%
            return (1e6,)
        if np.isnan(sharpe):
            return (1e6,)

        # Penalise overtrading: halve Sharpe if too many trades
        if n_trades > MAX_TRADES:
            sharpe *= 0.5

        # Parsimony pressure: slightly penalise large trees
        fitness = -sharpe + 0.001 * len(individual)
        return (fitness,)

    except Exception:
        return (1e6,)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evolutionary loop
# ─────────────────────────────────────────────────────────────────────────────

def custom_ea_simple(population, toolbox, cxpb, mutpb, ngen,
                     stats=None, halloffame=None, verbose=True):
    """EA with detailed progress reporting."""
    global _current_generation, _start_time
    _start_time = time.time()

    print(f"🚀 Starting evolution: {len(population)} individuals × {ngen} generations")
    print(f"   CX={cxpb}, MUT={mutpb}, MaxDepth={MAX_DEPTH}, MaxLen={MAX_LEN}")

    # Evaluate initial population
    print("   Evaluating initial population ...")
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    for gen in range(1, ngen + 1):
        _current_generation = gen
        gen_start = time.time()
        print(f"\n── Generation {gen}/{ngen} ──")

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if random.random() < mutpb:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"   Evaluating {len(invalid)} individuals ...")
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)

        if stats is not None:
            record = stats.compile(population)
            gen_time = time.time() - gen_start
            total    = time.time() - _start_time
            eta      = total * (ngen / gen) - total

            print(f"   Best: {record['fitness']['min']:.6f}  "
                  f"Avg: {record['fitness']['avg']:.6f}  "
                  f"Size: {record['size']['avg']:.1f}")
            print(f"   Time: {gen_time:.1f}s (total {total:.1f}s, ETA {eta:.1f}s)")

            if halloffame:
                best = halloffame[0]
                print(f"   HoF leader: fitness={best.fitness.values[0]:.6f}, "
                      f"size={len(best)}")

    return population


def run_evolution(train_df: pd.DataFrame,
                  n_gen: int = N_GEN,
                  pop_size: int = POP_SIZE) -> tools.HallOfFame:
    """Full evolutionary run; returns Hall-of-Fame (top 10)."""
    toolbox.register("evaluate", evaluate_individual, df_train=train_df)

    print(f"🏗  Creating population of {pop_size:,} ...")
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10, similar=lambda a, b: a == b)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_len = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_len)
    mstats.register("min", np.min)
    mstats.register("avg", np.mean)

    custom_ea_simple(pop, toolbox, P_CX, P_MUT, n_gen,
                     stats=mstats, halloffame=hof, verbose=True)

    print(f"\n🏆 Evolution complete — {len(hof)} individuals in Hall of Fame")
    return hof


# ─────────────────────────────────────────────────────────────────────────────
# 7. Validation & test helpers
# ─────────────────────────────────────────────────────────────────────────────

def backtest_on_slice(individual, df_slice: pd.DataFrame,
                      label: str) -> Dict[str, float]:
    """Run strategy on a data slice and print results."""
    print(f"\n📊 Backtesting on {label} ...")

    func = toolbox.compile(expr=individual)
    GPExposureStrategy.expression = staticmethod(func)

    bt_df = _prepare_bt_dataframe(df_slice)
    bt = Backtest(
        bt_df, GPExposureStrategy,
        cash=INITIAL_CASH, commission=COMMISSION_PCT,
        margin=MARGIN,
        exclusive_orders=False, trade_on_close=True,
    )
    stats = bt.run()

    print(f"=== {label} RESULTS ===")
    print(f"  Return:       {stats['Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  # Trades:     {stats['# Trades']}")
    print(f"  Win Rate:     {stats['Win Rate [%]']:.1f}%")
    print(f"  Final Equity: ${stats['Equity Final [$]']:,.2f}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 8. Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GP Crypto Strategy — Evolution (backtesting.py)")
    print("=" * 60)

    # 1) Load, normalise & split
    print("\n── PHASE 1: Data ──")
    df_all = load_all_pairs()
    df_all = normalize_inputs(df_all)
    train_df, val_df, test_df = split_dataset(df_all)

    # 2) Evolve
    print("\n── PHASE 2: Evolution ──")
    hof = run_evolution(train_df)

    # 3) Validate HoF
    print("\n── PHASE 3: Validation ──")
    val_scores = []
    for i, ind in enumerate(hof):
        score = evaluate_individual(ind, val_df)[0]
        val_scores.append(score)
        print(f"  Individual {i + 1}/{len(hof)}: val_fitness={score:.6f}")

    idx_best = int(np.argmin(val_scores))
    best_ind = hof[idx_best]
    print(f"\n  Best: #{idx_best + 1}  fitness={val_scores[idx_best]:.6f}  "
          f"size={len(best_ind)} nodes")

    # 4) Test
    print("\n── PHASE 4: Test ──")
    backtest_on_slice(best_ind, test_df, "TEST (out-of-sample)")

    # 5) Save
    print("\n── PHASE 5: Save ──")
    with open("best_individual.dill", "wb") as f:
        dill.dump(best_ind, f)
    print("  Saved → best_individual.dill")

    total_time = time.time() - _start_time if _start_time else 0
    print(f"\n🎉 Done in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"   Evaluations: {_evaluation_count:,}")


if __name__ == "__main__":
    main()
