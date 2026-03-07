"""
Genetic-Programming Crypto Strategy — vectorbt edition
=======================================================
Vectorised evaluation using vectorbt for much faster fitness computation.
Same GP tree structure as gp_crypto_strategy.py (20 inputs, ±100 long/short).
"""

import math, operator, random, time
from pathlib import Path
from typing import Tuple, List

import dill
import numpy as np
import pandas as pd
import vectorbt as vbt
from deap import base, creator, gp, tools

# ─────────────────────────────────────────────────────────────────────────────
# 0. Constants (mirror gp_crypto_strategy.py)
# ─────────────────────────────────────────────────────────────────────────────
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

SYMBOLS        = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LTCUSDT"]
PRIMARY_SYMBOL = "BTCUSDT"
ARG_NAMES      = [f"{s}_{f}" for s in SYMBOLS for f in ("Open", "High", "Low", "Close")]
DATA_DIR       = Path(__file__).parent / "data"

TRAIN_START, TRAIN_END = "2021-07-01 00:00:00", "2024-01-01 00:00:00"
VAL_START,   VAL_END   = "2024-01-01 00:00:00", "2024-07-01 00:00:00"
TEST_START,  TEST_END  = "2024-07-01 00:00:00", "2025-03-07 00:00:00"

POP_SIZE, N_GEN = 1000, 15          # smaller default for vectorbt (adjust)
P_CX, P_MUT     = 0.90, 0.15
MAX_DEPTH, MAX_LEN = 8, 60

INITIAL_CASH   = 100_000
COMMISSION_PCT = 0.001               # 0.1 %
NO_TRADE_BAND  = 5                   # ±5 pp dead-band

_eval_count = 0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data utilities (same loader as main version)
# ─────────────────────────────────────────────────────────────────────────────

def load_pair_csv(symbol: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close"]]
    df.columns = [f"{symbol}_{c.capitalize()}" for c in df.columns]
    return df


def load_all_pairs(folder: Path = DATA_DIR, symbols: List[str] = None):
    if symbols is None:
        symbols = SYMBOLS
    dfs = []
    for sym in symbols:
        csv = folder / f"{sym}_1h.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv}. Run crypto_data_downloader.py first.")
        dfs.append(load_pair_csv(sym, csv))
    return pd.concat(dfs, axis=1).dropna()


def split_dataset(df):
    return (df.loc[TRAIN_START:TRAIN_END],
            df.loc[VAL_START:VAL_END],
            df.loc[TEST_START:TEST_END])


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEAP GP primitives (NumPy-vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def vdiv(a, b):
    """Element-wise protected division."""
    return np.divide(a, b, out=np.copy(a), where=np.abs(b) > 1e-8)

def gtpct(a, b):
    """Vectorised a>b ? +100 : -100 (long/short)."""
    return np.where(a > b, 100.0, -100.0)

def rand_uniform():
    return random.uniform(-1, 1)


pset = gp.PrimitiveSet("CRYPTO", len(ARG_NAMES), prefix="inp")

for op in (np.add, np.subtract, np.multiply):
    pset.addPrimitive(op, 2)
pset.addPrimitive(vdiv, 2, name="pdiv")

for f, name in [(np.sin, "sin"), (np.cos, "cos"),
                (np.tan, "tan"), (np.tanh, "tanh")]:
    pset.addPrimitive(f, 1, name=name)

pset.addPrimitive(gtpct, 2, name="gtpct")
pset.addEphemeralConstant("rand", rand_uniform)

for i, n in enumerate(ARG_NAMES):
    pset.renameArguments(**{f"inp{i}": n})

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


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vectorbt-based fitness evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(price: pd.Series, weights: pd.Series):
    """Run vectorbt portfolio simulation from a target-weight series."""
    return vbt.Portfolio.from_orders(
        price,
        size=weights,
        price=price,
        size_type="targetpercent",
        fees=COMMISSION_PCT,
        init_cash=INITIAL_CASH,
        freq="1h",
    )


def evaluate_individual(ind, df_slice: pd.DataFrame) -> Tuple[float]:
    global _eval_count
    _eval_count += 1

    try:
        func = toolbox.compile(expr=ind)
        cols = [df_slice[c].to_numpy(dtype="float64") for c in ARG_NAMES]
        desired_pct = func(*cols)                       # vectorised

        # Sanitise
        desired_pct = np.where(np.isfinite(desired_pct), desired_pct, 0.0)
        desired_pct = np.clip(desired_pct, -100.0, 100.0)
        weights = desired_pct / 100.0                   # -1 to +1

        # Dead-band filter
        delta = np.abs(np.diff(weights, prepend=weights[0]))
        weights[delta < NO_TRADE_BAND / 100] = np.nan
        weights = pd.Series(weights, index=df_slice.index).ffill().fillna(0.0)

        port = _simulate(df_slice[f"{PRIMARY_SYMBOL}_Close"], weights)

        total_ret = port.total_return()
        n_trades  = port.stats()["Total Trades"]
        final_val = port.value().iloc[-1]

        if np.isnan(total_ret) or final_val <= 0 or n_trades < 20:
            return (1e6,)

        # Sharpe-based fitness
        sharpe = port.stats().get("Sharpe Ratio", np.nan)
        if np.isnan(sharpe):
            return (1e6,)

        return (-sharpe,)           # minimise negative Sharpe

    except Exception as e:
        return (1e6,)


# Wrapper for toolbox.evaluate
def evaluate_with_train(ind):
    return evaluate_individual(ind, evaluate_with_train.df)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evolutionary algorithm
# ─────────────────────────────────────────────────────────────────────────────

def run_evolution(train_df):
    evaluate_with_train.df = train_df
    toolbox.register("evaluate", evaluate_with_train)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(10, similar=lambda a, b: a == b)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    global _eval_count
    _eval_count = 0
    start_time = time.time()

    for gen in range(1, N_GEN + 1):
        invalid = [i for i in pop if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        hof.update(pop)

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for m in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(m)
                del m.fitness.values

        pop[:] = offspring

        # Evaluate newly invalidated individuals
        invalid = [i for i in pop if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        record = stats.compile(pop)
        elapsed = time.time() - start_time
        print(f"Gen {gen:02d}/{N_GEN} | min {record['min']:.6f} | "
              f"avg {record['avg']:.6f} | evals {_eval_count} | {elapsed:.1f}s")

    return hof


# ─────────────────────────────────────────────────────────────────────────────
# 5. Back-test helper
# ─────────────────────────────────────────────────────────────────────────────

def backtest_slice(individual, df_slice, label):
    func = toolbox.compile(expr=individual)
    cols = [df_slice[c].values for c in ARG_NAMES]
    weights = func(*cols) / 100.0

    delta = np.abs(np.diff(weights, prepend=weights[0]))
    weights[delta < NO_TRADE_BAND / 100] = np.nan
    weights = pd.Series(weights, index=df_slice.index).ffill().fillna(0.0)

    pf = _simulate(df_slice[f"{PRIMARY_SYMBOL}_Close"], weights)
    stats = pf.stats()

    wanted = ["Total Return [%]", "Sharpe Ratio", "Total Trades", "Win Rate [%]"]
    for label_alt in ("Equity Final [$]", "Final Value [$]"):
        if label_alt in stats:
            wanted.append(label_alt)
            break

    print(f"\n=== {label} ===")
    print(stats.loc[wanted])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("📦 Loading data ...")
    df_all = load_all_pairs()
    train, val, test = split_dataset(df_all)

    print(f"🧬 Running evolution (vectorbt engine) ...")
    hof = run_evolution(train)
    if not hof:
        raise RuntimeError("Hall-of-Fame empty — check logs.")

    # Pick best on validation
    scores = [evaluate_individual(ind, val)[0] for ind in hof]
    best = hof[int(np.argmin(scores))]
    print(f"\n🏆 Validation winner fitness: {min(scores):.6f}")

    backtest_slice(best, test, "TEST (out-of-sample)")

    with open("best_individual.dill", "wb") as f:
        dill.dump(best, f)
    print("💾 Saved → best_individual.dill")


if __name__ == "__main__":
    main()
