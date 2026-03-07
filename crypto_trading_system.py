"""
Crypto Trading System — Load, Infer & Backtest
================================================
Tools for working with trained GP crypto models:
1. Load pre-trained GP models (.dill)
2. Run backtests on any date range
3. Generate trading signals
4. Walk-forward analysis
5. Production trading bot skeleton

Usage:
    python crypto_trading_system.py
"""

import math
import operator
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time

import dill
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from deap import base, creator, gp, tools

# Import constants and utilities from main strategy module
from gp_crypto_strategy import (
    SYMBOLS, PRIMARY_SYMBOL, ARG_NAMES,
    INITIAL_CASH, COMMISSION_PCT, MARGIN,
    NO_TRADE_BAND, BTC_TICK,
    pset, toolbox,
    round_btc, pct_to_units, load_all_pairs, split_dataset,
    _prepare_bt_dataframe,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Loading and Management
# ─────────────────────────────────────────────────────────────────────────────

class GPModelManager:
    """Manages loading, saving, and using GP crypto models."""

    def __init__(self, model_path: str = "best_individual.dill"):
        self.model_path = model_path
        self.model = None
        self.compiled_func = None
        self.loaded_at = None

    def load_model(self) -> bool:
        """Load the trained GP model from disk."""
        try:
            print(f"Loading GP model from {self.model_path} ...")
            with open(self.model_path, "rb") as f:
                self.model = dill.load(f)

            self.compiled_func = toolbox.compile(expr=self.model)
            self.loaded_at = datetime.now()

            print(f"  Tree size: {len(self.model)} nodes")
            print(f"  Fitness:   {self.model.fitness.values[0]:.6f}")
            print(f"  Loaded at: {self.loaded_at}")
            return True

        except FileNotFoundError:
            print(f"Model not found: {self.model_path}")
            print("Run gp_crypto_strategy.py first to train a model.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def save_model(self, individual, filename: str = None):
        """Save a GP individual to disk."""
        fname = filename or self.model_path
        with open(fname, "wb") as f:
            dill.dump(individual, f)
        print(f"  Model saved → {fname}")

    def get_signal(self, market_data: Dict[str, float]) -> float:
        """Get trading signal (-100 to +100) from current market data."""
        if self.compiled_func is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = []
        for arg_name in ARG_NAMES:
            if arg_name not in market_data:
                raise ValueError(f"Missing: {arg_name}")
            inputs.append(market_data[arg_name])

        return float(self.compiled_func(*inputs))

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "tree_size": len(self.model),
            "fitness": self.model.fitness.values[0],
            "loaded_at": self.loaded_at,
            "model_path": self.model_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Live Trading Strategy
# ─────────────────────────────────────────────────────────────────────────────

class LiveGPStrategy(Strategy):
    """Production-ready strategy using a pre-trained GP model."""

    model_manager = None          # injected before backtest

    def init(self):
        if self.model_manager is None or self.model_manager.compiled_func is None:
            self.model_manager = GPModelManager()
            if not self.model_manager.load_model():
                raise RuntimeError("Failed to load GP model")

        self.current_pct = 0.0
        self.signal_history = []
        self.trade_log = []

    def next(self):
        try:
            market_data = {}
            for name in ARG_NAMES:
                if name in self.data.df.columns:
                    market_data[name] = float(self.data.df[name].iloc[-1])
                else:
                    short = name.replace(f"{PRIMARY_SYMBOL}_", "")
                    if short in self.data.df.columns:
                        market_data[name] = float(self.data.df[short].iloc[-1])
                    else:
                        raise RuntimeError(f"Column {name} not found")

            desired_pct = self.model_manager.get_signal(market_data)

            self.signal_history.append({
                "timestamp": self.data.index[-1],
                "signal": desired_pct,
                "price": self.data.Close[-1],
            })

            if abs(desired_pct - self.current_pct) <= NO_TRADE_BAND:
                return

            broker = self._broker
            equity_now = getattr(broker, "equity", getattr(broker, "_equity"))
            price_now = self.data.Close[-1]
            target_units = pct_to_units(desired_pct, equity_now, price_now)
            delta = target_units - self.position.size
            order_size = round_btc(delta)

            if abs(order_size) < BTC_TICK:
                return

            action = "BUY" if order_size > 0 else "SELL"
            if order_size > 0:
                self.buy(size=abs(order_size))
            else:
                self.sell(size=abs(order_size))

            self.trade_log.append({
                "timestamp": self.data.index[-1],
                "action": action,
                "size": abs(order_size),
                "price": price_now,
                "signal": desired_pct,
                "old_pct": self.current_pct,
                "equity": equity_now,
            })
            self.current_pct = desired_pct

        except Exception as e:
            pass  # skip bar on error


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(model_manager: GPModelManager,
                     data: pd.DataFrame,
                     start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
    """Generate trading signals for a date range."""
    if start_date and end_date:
        period = data.loc[start_date:end_date].copy()
    else:
        period = data.copy()

    print(f"Generating signals for {len(period)} bars ...")

    signals = []
    for idx, row in period.iterrows():
        try:
            md = {}
            for name in ARG_NAMES:
                if name in row.index:
                    md[name] = float(row[name])
                else:
                    short = name.replace(f"{PRIMARY_SYMBOL}_", "")
                    md[name] = float(row.get(short, 0.0))

            sig = model_manager.get_signal(md)
            signals.append({"timestamp": idx, "signal": sig,
                            "btc_close": row.get(f"{PRIMARY_SYMBOL}_Close",
                                                 row.get("Close", 0))})
        except Exception:
            signals.append({"timestamp": idx, "signal": 0.0, "btc_close": 0.0})

    return pd.DataFrame(signals).set_index("timestamp")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Walk-Forward Analysis
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_analysis(data_dir: Path = None,
                          model_path: str = "best_individual.dill",
                          window_months: int = 6,
                          step_months: int = 1) -> pd.DataFrame:
    """Sliding-window out-of-sample evaluation."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    print(f"Walk-forward: window={window_months}m, step={step_months}m")

    df_all = load_all_pairs(data_dir)
    mm = GPModelManager(model_path)
    if not mm.load_model():
        raise RuntimeError("Cannot load model")

    LiveGPStrategy.model_manager = mm

    start = df_all.index[0]
    end = df_all.index[-1]
    results = []
    current = start

    while current + timedelta(days=window_months * 30) <= end:
        w_start = current
        w_end = current + timedelta(days=window_months * 30)
        window = df_all.loc[w_start:w_end]

        if len(window) < 100:
            current += timedelta(days=step_months * 30)
            continue

        try:
            bt_df = _prepare_bt_dataframe(window)
            bt = Backtest(bt_df, LiveGPStrategy,
                          cash=INITIAL_CASH, commission=COMMISSION_PCT,
                          margin=MARGIN,
                          exclusive_orders=False, trade_on_close=True)
            stats = bt.run()

            results.append({
                "start": w_start, "end": w_end,
                "return_pct": stats["Return [%]"],
                "sharpe": stats["Sharpe Ratio"],
                "max_dd": stats["Max. Drawdown [%]"],
                "trades": stats["# Trades"],
                "win_rate": stats["Win Rate [%]"],
                "equity": stats["Equity Final [$]"],
            })
            print(f"  {w_start.date()} → {w_end.date()}: "
                  f"ret={stats['Return [%]']:.2f}% sharpe={stats['Sharpe Ratio']:.3f}")

        except Exception as e:
            print(f"  {w_start.date()} → {w_end.date()}: ERROR {e}")

        current += timedelta(days=step_months * 30)

    df_res = pd.DataFrame(results)
    df_res.to_csv("walk_forward_results.csv", index=False)
    print(f"\nSaved → walk_forward_results.csv ({len(df_res)} windows)")
    return df_res


# ─────────────────────────────────────────────────────────────────────────────
# 5. Production Trading Bot
# ─────────────────────────────────────────────────────────────────────────────

class ProductionTradingBot:
    """Skeleton for a live-trading bot using a GP model."""

    def __init__(self, model_path: str = "best_individual.dill"):
        self.model_manager = GPModelManager(model_path)
        self.position = 0.0
        self.equity = INITIAL_CASH
        self.trade_history = []
        self.last_signal = 0.0

    def initialize(self) -> bool:
        print("Initializing Production Trading Bot ...")
        if not self.model_manager.load_model():
            return False
        print("Bot ready.")
        return True

    def process_market_data(self, market_data: Dict[str, float]) -> Dict:
        """Process a new bar and return trading decision."""
        try:
            signal = self.model_manager.get_signal(market_data)
            position_change = 0.0

            if abs(signal - self.last_signal) > NO_TRADE_BAND:
                price = market_data.get(f"{PRIMARY_SYMBOL}_Close", 1.0)
                target_units = pct_to_units(signal, self.equity, price)
                position_change = target_units - self.position

                if abs(position_change) >= BTC_TICK:
                    position_change = round_btc(position_change)
                    self.position += position_change
                    self.last_signal = signal

                    trade = {
                        "timestamp": datetime.now(),
                        "action": "BUY" if position_change > 0 else "SELL",
                        "size": abs(position_change),
                        "price": price,
                        "signal": signal,
                        "position": self.position,
                    }
                    self.trade_history.append(trade)
                    print(f"  {trade['action']} {trade['size']:.5f} BTC "
                          f"@ ${trade['price']:,.2f}")

            return {
                "signal": signal,
                "position_change": position_change,
                "position": self.position,
                "should_trade": abs(position_change) > 0,
            }

        except Exception as e:
            return {"signal": 0.0, "position_change": 0.0,
                    "position": self.position, "should_trade": False,
                    "error": str(e)}

    def get_status(self) -> Dict:
        return {
            "position": self.position,
            "equity": self.equity,
            "last_signal": self.last_signal,
            "total_trades": len(self.trade_history),
            "model_info": self.model_manager.get_model_info(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convenience backtest function
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")       # non-interactive backend
import matplotlib.pyplot as plt


def backtest_saved_model(data_dir: Path = None,
                         model_path: str = "best_individual.dill",
                         start_date: str = None,
                         end_date: str = None,
                         plot: bool = False):
    """Load a saved model and backtest it on a date range."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    print(f"Backtesting model: {model_path}")

    df_all = load_all_pairs(data_dir)

    if start_date and end_date:
        df_test = df_all.loc[start_date:end_date]
        print(f"Period: {start_date} → {end_date} ({len(df_test)} bars)")
    else:
        df_test = df_all
        print(f"Full dataset ({len(df_test)} bars)")

    mm = GPModelManager(model_path)
    if not mm.load_model():
        return None

    LiveGPStrategy.model_manager = mm
    bt_df = _prepare_bt_dataframe(df_test)

    bt = Backtest(bt_df, LiveGPStrategy,
                  cash=INITIAL_CASH, commission=COMMISSION_PCT,
                  margin=MARGIN,
                  exclusive_orders=False, trade_on_close=True)

    print("Running backtest ...")
    stats = bt.run()

    print("\n=== BACKTEST RESULTS ===")
    print(f"  Return:       {stats['Return [%]']:.2f}%")
    print(f"  Sharpe Ratio: {stats['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  # Trades:     {stats['# Trades']}")
    print(f"  Win Rate:     {stats['Win Rate [%]']:.1f}%")
    print(f"  Final Equity: ${stats['Equity Final [$]']:,.2f}")

    if plot:
        try:
            bt.plot()
        except Exception as e:
            print(f"Could not plot: {e}")

    return stats


def demo_live_trading():
    """Demo the production bot with synthetic BTC-range data."""
    print("=== Live Trading Demo ===")
    bot = ProductionTradingBot()
    if not bot.initialize():
        return

    for i in range(10):
        market_data = {}
        for name in ARG_NAMES:
            sym = name.split("_")[0]
            # Rough realistic price ranges
            ranges = {
                "BTCUSDT": (80_000, 100_000),
                "ETHUSDT": (2_500, 4_000),
                "SOLUSDT": (80, 200),
                "BNBUSDT": (500, 700),
                "LTCUSDT": (60, 120),
            }
            lo, hi = ranges.get(sym, (1, 100))
            market_data[name] = np.random.uniform(lo, hi)

        decision = bot.process_market_data(market_data)
        print(f"  Step {i + 1}: signal={decision['signal']:+.1f}  "
              f"pos={decision['position']:.5f} BTC  "
              f"trade={decision['should_trade']}")
        time.sleep(0.1)

    print(f"\nFinal: position={bot.position:.5f}, "
          f"trades={len(bot.trade_history)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GP Crypto Trading System")
    print("=" * 60)

    # Backtest saved model on the test period
    print("\n1. Backtesting saved model ...")
    backtest_saved_model(start_date="2024-07-01", end_date="2025-03-07")
