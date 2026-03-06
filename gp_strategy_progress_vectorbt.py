"""
Genetic-Programming FX Strategy — vectorbt edition
==================================================
* Uses vectorbt's vectorised engine instead of Backtesting.py.
* Single-process: fast enough that multiprocessing is unnecessary.
"""

import math, operator, random, time, os
from pathlib import Path
from typing import Tuple, List

import dill
import numpy as np
import pandas as pd
import vectorbt as vbt                          # pip install vectorbt
from deap import base, creator, gp, tools

# ─────────────────────────────────────────────
# 0. Globals & constants
# ─────────────────────────────────────────────
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

PAIRS      = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
ARG_NAMES  = [f"{p}_{f}" for p in PAIRS for f in ("Open", "High", "Low", "Close")]
DATA_DIR   = Path(".")

TRAIN_START, TRAIN_END = "2022-07-05 00:00:00", "2024-01-01 23:55:00"
VAL_START,   VAL_END   = "2024-01-02 00:00:00", "2024-07-04 23:55:00"
TEST_START,  TEST_END  = "2024-07-05 00:00:00", "2025-07-05 23:55:00"

POP_SIZE, N_GEN = 1000, 15          # adjust for your CPU
P_CX,   P_MUT   = 0.90, 0.15
MAX_DEPTH, MAX_LEN = 8, 60

INITIAL_CASH   = 1_000_000
COMMISSION_PCT = 0.000015
NO_TRADE_BAND  = 10              # ±10 pp dead-band
POSITION_GRID  = 5_000           # kept only for reference (unused in vbt)

# progress counters
_eval_count, _gen_start_time = 0, None

# ─────────────────────────────────────────────
# 1. Data utilities
# ─────────────────────────────────────────────
def _detect_datetime_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower().replace(" ", "") in {"date", "datetime", "time", "timestamp", "gmttime"}:
            return c
    raise ValueError("Could not find a datetime column")

def load_pair_csv(symbol: str, path: Path, freq="5min") -> pd.DataFrame:
    df = pd.read_csv(path)
    tscol = _detect_datetime_column(df)
    df[tscol] = pd.to_datetime(df[tscol], dayfirst=True, utc=True, errors="coerce")
    df = df.dropna(subset=[tscol]).set_index(tscol)

    # standardise OHLC names & keep only them
    mapper = {}
    for c in df.columns:
        l = c.lower()
        if   l.startswith("open"):   mapper[c] = "Open"
        elif l.startswith("high"):   mapper[c] = "High"
        elif l.startswith("low"):    mapper[c] = "Low"
        elif l.startswith("close"):  mapper[c] = "Close"
    df = df[list(mapper)].rename(columns=mapper)

    df = df.resample(freq, label="right", closed="right").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last"}
    ).ffill()

    df.columns = [f"{symbol}_{c}" for c in df.columns]
    return df

def load_all_pairs(folder: Path, pairs=PAIRS):
    dfs=[]
    for pair in pairs:
        csv = folder / f"{pair}_Candlestick_5_M_BID_05.07.2022-05.07.2025.csv"
        if not csv.exists(): raise FileNotFoundError(csv)
        dfs.append(load_pair_csv(pair, csv))
    return pd.concat(dfs, axis=1).dropna()

def split_dataset(df: pd.DataFrame):
    return (df.loc[TRAIN_START:TRAIN_END],
            df.loc[VAL_START:VAL_END],
            df.loc[TEST_START:TEST_END])

# ─────────────────────────────────────────────
# 2. DEAP GP primitives
# ─────────────────────────────────────────────
def vdiv(a, b):
    """Element-wise protected division."""
    return np.divide(a, b, out=np.copy(a), where=np.abs(b) > 1e-8)

def gtpct(a, b):
    """Vectorised a>b ? 100 : -100."""
    return np.where(a > b, 100.0, -100.0)

def rand_uniform():
    return random.uniform(-1, 1)

pset = gp.PrimitiveSet("FX", 16, prefix="inp")

# basic arithmetic – NumPy ufuncs are already vectorised and fast
for op in (np.add, np.subtract, np.multiply):
    pset.addPrimitive(op, 2)

pset.addPrimitive(vdiv, 2,  name="pdiv")

# trigonometry – use NumPy versions
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

# ─────────────────────────────────────────────
# 3. Vectorbt-based fitness evaluation
# ─────────────────────────────────────────────
def _simulate(price: pd.Series, weights: pd.Series):
    """Helper: run vectorbt simulation from target weight series."""
    # vectorbt wants "size", not weight. Here we pass weights directly
    # and tell it they are TARGET PERCENTS of equity.
    return vbt.Portfolio.from_orders(
        price,
        size=weights,                       # -1 … +1
        price=price,                       # fill @ market close
        size_type='targetpercent',         # interpret as %
        fees=COMMISSION_PCT,
        init_cash=INITIAL_CASH,
        freq="5min"
    )

def evaluate_individual(ind: creator.Individual,
                        df_slice: pd.DataFrame) -> Tuple[float]:
    global _eval_count
    _eval_count += 1

    try:
        func = toolbox.compile(expr=ind)
        cols = [df_slice[c].to_numpy(dtype="float64") for c in ARG_NAMES]
        desired_pct = func(*cols)                       # vectorised

        # sanitise
        desired_pct = np.where(np.isfinite(desired_pct), desired_pct, 0.)
        desired_pct = np.clip(desired_pct, -100., 100.)
        weights = desired_pct / 100.0

        # dead-band
        delta = np.abs(np.diff(weights, prepend=weights[0]))
        weights[delta < NO_TRADE_BAND / 100] = np.nan
        weights = pd.Series(weights, index=df_slice.index).ffill().fillna(0.)

        port = _simulate(df_slice["USDJPY_Close"], weights)

        total_ret = port.total_return()                 # decimal
        n_trades  = port.stats()["Total Trades"]
        final_val = port.value().iloc[-1]

        if np.isnan(total_ret) or final_val <= 0 or n_trades < 20:
            return (1e6,)                         # still penalise hopeless runs
        return (math.exp(-total_ret),)

    except Exception as e:
        print("Evaluation error:", e)         #  remove the env-var check
        return (1e6,)

# wrapper so toolbox.evaluate has fixed signature
def evaluate_with_train(ind):
    return evaluate_individual(ind, evaluate_with_train.df)

# ─────────────────────────────────────────────
# 4. Evolutionary algorithm (single-process)
# ─────────────────────────────────────────────
def run_evolution(train_df):
    evaluate_with_train.df = train_df          # bind slice once
    toolbox.register("evaluate", evaluate_with_train)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(10, similar=lambda a,b: a==b)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min), stats.register("avg", np.mean)

    global _eval_count, _gen_start_time
    _eval_count, _gen_start_time = 0, time.time()

    for gen in range(1, N_GEN+1):
        # evaluate individuals without fitness
        invalid = [i for i in pop if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        hof.update(pop)

        # selection / variation
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for m in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(m)
                del m.fitness.values

        pop[:] = offspring                         # <- new population

        # NEW: evaluate individuals whose fitness was just deleted
        invalid = [i for i in pop if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        record = stats.compile(pop)                # now everyone has a fitness
        print(f"Gen {gen:02d}/{N_GEN} | min {record['min']:.6f} | "
            f"avg {record['avg']:.6f} | evals {_eval_count}")

    return hof

# ─────────────────────────────────────────────
# 5. Quick back-test helper (vectorbt as well)
# ─────────────────────────────────────────────
def backtest_slice(individual, df_slice, label):
    func = toolbox.compile(expr=individual)
    cols = [df_slice[c].values for c in ARG_NAMES]
    weights = func(*cols) / 100.0
    delta = np.abs(np.diff(weights, prepend=weights[0]))
    weights[delta < NO_TRADE_BAND / 100] = np.nan
    weights = pd.Series(weights, index=df_slice.index).ffill().fillna(0.)

    pf = _simulate(df_slice["USDJPY_Close"], weights)
    stats = pf.stats()

    wanted = ["Total Return [%]", "Sharpe Ratio", "Total Trades", "Win Rate [%]"]
    # Either 'Equity Final [$]' (old) or 'Final Value [$]' (new)
    if "Equity Final [$]" in stats:
        wanted.append("Equity Final [$]")
    elif "Final Value [$]" in stats:
        wanted.append("Final Value [$]")

    print(f"\n=== {label} ===")
    print(stats.loc[wanted])

    return stats

# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    print(" Loading data...")
    df_all = load_all_pairs(DATA_DIR)
    train, val, test = split_dataset(df_all)

    print(" Running evolution (vectorbt engine)…")
    hof = run_evolution(train)
    if not hof:
        raise RuntimeError("Hall-of-Fame is empty – check evaluation logs.")

    # pick best on validation
    scores = [evaluate_individual(ind, val)[0] for ind in hof]
    best   = hof[int(np.argmin(scores))]
    print("\n Validation winner fitness:", min(scores))

    backtest_slice(best, test, "TEST (out-of-sample)")

    with open("best_individual.dill","wb") as f:
        dill.dump(best, f)
    print(" Saved best individual → best_individual.dill")

if __name__ == "__main__":
    main()
