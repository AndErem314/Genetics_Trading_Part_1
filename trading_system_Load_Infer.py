"""
Complete Trading System for GP FX Strategy
==========================================
This module provides tools to:
1. Load pre-trained GP models
2. Run live trading simulations
3. Generate trading signals
4. Perform walk-forward analysis
5. Create production-ready trading bots
"""

import math
import operator
import pickle
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

# Import the same constants and setup from your main file
from gp_strategy_progress import (
    PAIRS, ARG_NAMES, INITIAL_CASH, COMMISSION_PCT, 
    POSITION_GRID, NO_TRADE_BAND, pset, toolbox,
    round_grid, pct_to_units, load_all_pairs, split_dataset
)

# Override the initial cash amount
#INITIAL_CASH = 1000000

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Loading and Management
# ─────────────────────────────────────────────────────────────────────────────

class GPModelManager:
    """Manages loading, saving, and using GP models for trading."""
    
    def __init__(self, model_path: str = "best_individual.dill"):
        self.model_path = model_path
        self.model = None
        self.compiled_func = None
        self.loaded_at = None
        
    def load_model(self) -> bool:
        """Load the trained GP model from disk."""
        try:
            print(f"Loading GP model from {self.model_path}...")
            with open(self.model_path, "rb") as f:
                self.model = dill.load(f)
            
            # Compile the model into a callable function
            self.compiled_func = toolbox.compile(expr=self.model)
            self.loaded_at = datetime.now()
            
            print(f"Model loaded successfully!")
            print(f"Tree size: {len(self.model)} nodes")
            print(f"Fitness: {self.model.fitness.values[0]:.6f}")
            print(f"Loaded at: {self.loaded_at}")
            
            return True
            
        except FileNotFoundError:
            print(f"Model file not found: {self.model_path}")
            print("Run the training script first to generate the model")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, individual, filename: str = None):
        """Save a GP individual to disk."""
        if filename is None:
            filename = self.model_path
        
        with open(filename, "wb") as f:
            dill.dump(individual, f)
        print(f" Model saved to {filename}")
    
    def get_signal(self, market_data: Dict[str, float]) -> float:
        """Get trading signal from current market data."""
        if self.compiled_func is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Build input vector in canonical order
        inputs = []
        for arg_name in ARG_NAMES:
            if arg_name not in market_data:
                raise ValueError(f"Missing market data for {arg_name}")
            inputs.append(market_data[arg_name])
        
        # Get signal from GP model
        signal = float(self.compiled_func(*inputs))
        return signal
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "tree_size": len(self.model),
            "fitness": self.model.fitness.values[0] if self.model.fitness.values else None,
            "loaded_at": self.loaded_at,
            "model_path": self.model_path
        }

# ─────────────────────────────────────────────────────────────────────────────
# 2. Live Trading Strategy
# ─────────────────────────────────────────────────────────────────────────────

class LiveGPStrategy(Strategy):
    """Production-ready strategy using pre-trained GP model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_manager = GPModelManager()
        self.current_pct = 0.0
        self.signal_history = []
        self.trade_log = []
        
    def init(self):
        """Initialize the strategy."""
        print("Initializing Live GP Strategy...")
        
        # Load the trained model
        if not self.model_manager.load_model():
            raise RuntimeError("Failed to load GP model")
        
        # Initialize position
        broker = self._broker
        equity_start = getattr(broker, "equity", getattr(broker, "_equity"))
        cash_half = equity_start / 2
        price0 = self.data.Close[0]
        qty = round_grid(cash_half / price0)
        
        if qty:
            self.buy(size=qty)
            self.sell(size=qty)
        
        print(f"Strategy initialized with ${equity_start:,.2f}")
        print(f"Initial position: {qty:,} units")
        
    def next(self):
        """Execute trading logic for each time step."""
        try:
            # Build current market data
            market_data = {}
            for name in ARG_NAMES:
                if name in self.data.df.columns:
                    market_data[name] = float(self.data.df[name].iloc[-1])
                else:
                    # Handle USDJPY special case
                    alt_name = name.replace("USDJPY_", "")
                    if alt_name in self.data.df.columns:
                        market_data[name] = float(self.data.df[alt_name].iloc[-1])
                    else:
                        raise RuntimeError(f"Column {name} not found")
            
            # Get signal from GP model
            desired_pct = self.model_manager.get_signal(market_data)
            
            # Store signal history
            self.signal_history.append({
                'timestamp': self.data.index[-1],
                'signal': desired_pct,
                'price': self.data.Close[-1]
            })
            
            # Check if we need to trade (dead-band filter)
            if abs(desired_pct - self.current_pct) <= NO_TRADE_BAND:
                return
            
            # Calculate position size
            broker = self._broker
            equity_now = getattr(broker, "equity", getattr(broker, "_equity"))
            price_now = self.data.Close[-1]
            target_units = pct_to_units(desired_pct, equity_now, price_now)
            delta = target_units - self.position.size
            order_size = round_grid(delta)
            
            if order_size == 0:
                return
            
            # Execute trade
            if order_size > 0:
                self.buy(size=order_size)
                action = "BUY"
            else:
                self.sell(size=-order_size)
                action = "SELL"
            
            # Log trade
            trade_info = {
                'timestamp': self.data.index[-1],
                'action': action,
                'size': abs(order_size),
                'price': price_now,
                'signal': desired_pct,
                'old_pct': self.current_pct,
                'equity': equity_now
            }
            self.trade_log.append(trade_info)
            
            self.current_pct = desired_pct
            
        except Exception as e:
            print(f"Error in trading step: {e}")
            # Continue without trading on error

# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal Generation and Analysis
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals_for_period(model_manager: GPModelManager, 
                              data: pd.DataFrame, 
                              start_date: str = None, 
                              end_date: str = None) -> pd.DataFrame:
    """Generate trading signals for a specific period."""
    
    print(f"Generating signals for period {start_date} to {end_date}...")
    
    # Filter data for the period
    if start_date and end_date:
        period_data = data.loc[start_date:end_date].copy()
    else:
        period_data = data.copy()
    
    signals = []
    
    for idx, row in period_data.iterrows():
        try:
            # Build market data dictionary
            market_data = {}
            for arg_name in ARG_NAMES:
                if arg_name in row.index:
                    market_data[arg_name] = float(row[arg_name])
                else:
                    # Handle USDJPY special case
                    alt_name = arg_name.replace("USDJPY_", "")
                    if alt_name in row.index:
                        market_data[arg_name] = float(row[alt_name])
                    else:
                        market_data[arg_name] = 0.0  # Default value
            
            # Get signal
            signal = model_manager.get_signal(market_data)
            
            signals.append({
                'timestamp': idx,
                'signal': signal,
                'usdjpy_close': row.get('USDJPY_Close', row.get('Close', 0))
            })
            
        except Exception as e:
            print(f"Error generating signal for {idx}: {e}")
            signals.append({
                'timestamp': idx,
                'signal': 0.0,
                'usdjpy_close': row.get('USDJPY_Close', row.get('Close', 0))
            })
    
    signals_df = pd.DataFrame(signals).set_index('timestamp')
    print(f"Generated {len(signals_df)} signals")
    
    return signals_df

# ─────────────────────────────────────────────────────────────────────────────
# 4. Walk-Forward Analysis
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_analysis(data_dir: Path, 
                         model_path: str = "best_individual.dill",
                         window_months: int = 6,
                         step_months: int = 1) -> pd.DataFrame:
    """Perform walk-forward analysis on the trained model."""
    
    print(f"Starting walk-forward analysis...")
    print(f"Window: {window_months} months, Step: {step_months} months")
    
    # Load data and model
    df_all = load_all_pairs(data_dir)
    model_manager = GPModelManager(model_path)
    
    if not model_manager.load_model():
        raise RuntimeError("Failed to load model for walk-forward analysis")
    
    # Generate date windows
    start_date = df_all.index[0]
    end_date = df_all.index[-1]
    
    results = []
    current_date = start_date
    
    while current_date + timedelta(days=window_months*30) <= end_date:
        window_start = current_date
        window_end = current_date + timedelta(days=window_months*30)
        
        print(f"\nTesting window: {window_start.date()} to {window_end.date()}")
        
        # Get data for this window
        window_data = df_all.loc[window_start:window_end]
        
        if len(window_data) < 100:  # Skip if not enough data
            current_date += timedelta(days=step_months*30)
            continue
        
        # Run backtest
        try:
            # Prepare data for backtesting
            usdjpy = window_data[[c for c in window_data.columns if c.startswith("USDJPY_")]].copy()
            usdjpy.columns = [c.replace("USDJPY_", "") for c in usdjpy.columns]
            usdjpy = usdjpy.join(window_data[[c for c in window_data.columns if not c.startswith("USDJPY_")]])
            
            # Set up strategy with model
            LiveGPStrategy.model_manager = model_manager
            
            bt = Backtest(usdjpy, LiveGPStrategy,
                         cash=INITIAL_CASH, commission=COMMISSION_PCT,
                         exclusive_orders=False, trade_on_close=True)
            
            stats = bt.run()
            
            results.append({
                'start_date': window_start,
                'end_date': window_end,
                'return_pct': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'],
                'final_equity': stats['Equity Final [$]']
            })
            
            print(f"Return: {stats['Return [%]']:.2f}%, Sharpe: {stats['Sharpe Ratio']:.3f}")
            
        except Exception as e:
            print(f"Error in window: {e}")
            results.append({
                'start_date': window_start,
                'end_date': window_end,
                'return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'final_equity': INITIAL_CASH
            })
        
        current_date += timedelta(days=step_months*30)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("walk_forward_results.csv", index=False)
    print(f"\nWalk-forward analysis complete!")
    print(f"Results saved to walk_forward_results.csv")
    
    return results_df

# ─────────────────────────────────────────────────────────────────────────────
# 5. Production Trading Bot
# ─────────────────────────────────────────────────────────────────────────────

class ProductionTradingBot:
    """A production-ready trading bot using the GP model."""
    
    def __init__(self, model_path: str = "best_individual.dill"):
        self.model_manager = GPModelManager(model_path)
        self.position = 0.0
        self.equity = INITIAL_CASH
        self.trade_history = []
        self.signal_history = []
        self.last_signal = 0.0
        
    def initialize(self) -> bool:
        """Initialize the trading bot."""
        print("Initializing Production Trading Bot...")
        
        if not self.model_manager.load_model():
            return False
        
        print("Trading bot ready!")
        return True
    
    def process_market_data(self, market_data: Dict[str, float]) -> Dict:
        """Process new market data and generate trading decision."""
        
        try:
            # Get signal from GP model
            signal = self.model_manager.get_signal(market_data)
            
            # Store signal
            self.signal_history.append({
                'timestamp': datetime.now(),
                'signal': signal,
                'price': market_data.get('USDJPY_Close', 0)
            })
            
            # Calculate position change
            position_change = 0.0
            if abs(signal - self.last_signal) > NO_TRADE_BAND:
                # Calculate target position in units
                current_price = market_data.get('USDJPY_Close', 1.0)
                target_units = pct_to_units(signal, self.equity, current_price)
                position_change = target_units - self.position
                
                if abs(position_change) > POSITION_GRID:
                    position_change = round_grid(position_change)
                    self.position += position_change
                    self.last_signal = signal
                    
                    # Log trade
                    trade = {
                        'timestamp': datetime.now(),
                        'action': 'BUY' if position_change > 0 else 'SELL',
                        'size': abs(position_change),
                        'price': current_price,
                        'signal': signal,
                        'new_position': self.position
                    }
                    self.trade_history.append(trade)
                    
                    print(f"{trade['action']} {trade['size']:,} units at {trade['price']:.5f}")
            
            return {
                'signal': signal,
                'position_change': position_change,
                'current_position': self.position,
                'should_trade': abs(position_change) > 0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error processing market data: {e}")
            return {
                'signal': 0.0,
                'position_change': 0.0,
                'current_position': self.position,
                'should_trade': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            'position': self.position,
            'equity': self.equity,
            'last_signal': self.last_signal,
            'total_trades': len(self.trade_history),
            'model_info': self.model_manager.get_model_info(),
            'last_update': datetime.now()
        }

# ─────────────────────────────────────────────────────────────────────────────
# 6. Example Usage Functions
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
def backtest_saved_model(
    data_dir: Path,
    model_path: str = "best_individual.dill",
    start_date: str = None,
    end_date: str = None,
    plot: bool = False,      # <-- add this
):
    print(f"Backtesting saved model: {model_path}", flush=True)

    # Load data once
    print("Loading data once...", flush=True)
    df_all = load_all_pairs(data_dir)

    if start_date and end_date:
        df_test = df_all.loc[start_date:end_date]
        print(f"Testing period: {start_date} to {end_date} | bars={len(df_test)}", flush=True)
    else:
        df_test = df_all
        print(f"Testing on full dataset | bars={len(df_test)}", flush=True)

    # Load model once
    model_manager = GPModelManager(model_path)
    if not model_manager.load_model():
        return None

    # Inject model into strategy to avoid double loading
    LiveGPStrategy.model_manager = model_manager

    # Prepare USDJPY base + extras
    usdjpy = df_test[[c for c in df_test.columns if c.startswith("USDJPY_")]].copy()
    usdjpy.columns = [c.replace("USDJPY_", "") for c in usdjpy.columns]
    extras = df_test[[c for c in df_test.columns if not c.startswith("USDJPY_")]]
    usdjpy = usdjpy.join(extras)

    bt = Backtest(
        usdjpy,
        LiveGPStrategy,
        margin=1/5,
        cash=INITIAL_CASH,
        commission=COMMISSION_PCT,
        exclusive_orders=False,
        trade_on_close=True,
    )

    print("Running backtest (this can take time)...", flush=True)
    stats = bt.run()

    print("\nBACKTEST RESULTS")
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
    print(f"Final Equity: ${stats['Equity Final [$]']:,.2f}")

    if plot:
        try:
            print("\nPlotting…", flush=True)
            bt.plot()
        except Exception as e:
            print(f"Could not plot: {e}")

    return stats

def demo_live_trading():
    """Demonstrate live trading with the production bot."""
    
    print("Live Trading Demo")
    print("=" * 50)
    
    # Initialize bot
    bot = ProductionTradingBot()
    if not bot.initialize():
        print("Failed to initialize trading bot")
        return
    
    # Simulate some market data updates
    print("\nSimulating market data updates...")
    
    for i in range(10):
        # Generate random market data (in practice, this comes from your data feed)
        market_data = {}
        for arg_name in ARG_NAMES:
            market_data[arg_name] = np.random.uniform(1.0, 1.5)  # Random prices
        
        # Process market data
        decision = bot.process_market_data(market_data)
        
        print(f"Step {i+1}: Signal={decision['signal']:.2f}, "
              f"Position={decision['current_position']:.0f}, "
              f"Trade={decision['should_trade']}")
        
        time.sleep(0.1)  # Simulate time between updates
    
    # Show final status
    status = bot.get_status()
    print(f"\nFinal Status:")
    print(f"Position: {status['position']:.0f}")
    print(f"Total Trades: {status['total_trades']}")

if __name__ == "__main__":
    # Example usage
    print("GP Trading System - Model Usage Examples")
    print("=" * 60)
    
    # Example 1: Backtest a saved model
    print("\n1. Backtesting saved model...")
    backtest_saved_model(Path("."), start_date="2024-07-05", end_date="2025-07-05")
    
    # Example 2: Generate signals for a period
    #print("\n2. Generating signals...")
    # model = GPModelManager()
    # if model.load_model():
    #     data = load_all_pairs(Path("."))
    #     signals = generate_signals_for_period(model, data, "2025-01-01", "2025-02-01")
    #     print(f"Generated {len(signals)} signals")
    
    # Example 3: Live trading demo
    #print("\n3. Live trading demo...")
    #demo_live_trading()
