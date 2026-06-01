"""
Integrated FinRL Configuration - Combines Stock and Crypto Trading
"""

from datetime import datetime, timedelta
import numpy as np
import operator as op
from functools import reduce

def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

# General Training Settings
#######################################################################################################
#######################################################################################################

# Trading Period Settings
trade_start_date = '2022-04-30 00:00:00'
trade_end_date = '2022-06-27 00:00:00'

# Training Configuration
SEED_CFG = 2390408
TIMEFRAME = '5m'
H_TRIALS = 50
KCV_groups = 5
K_TEST_GROUPS = 2
NUM_PATHS = 4
N_GROUPS = NUM_PATHS + 1
NUMBER_OF_SPLITS = nCr(N_GROUPS, N_GROUPS - K_TEST_GROUPS)

print(NUMBER_OF_SPLITS)

no_candles_for_train = 20000
no_candles_for_val = 5000

# Asset Lists
CRYPTO_TICKER_LIST = [
    'BTCUSDT',
    'ETHUSDT',
]

STOCK_TICKER_LIST = [
    'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'GOOGL', 'TSLA',
    'XOM', 'CVX', 'COP', 'FCX', 'BHP', 'GLD', 'SLV',
    'TLT', 'IEF', 'XLU', 'XLV', 'IAU', 'SHY', 'UUP'
]

# Trading Limits
CRYPTO_LIMITS = np.array([0.0001, 0.001, 0.01, 0.1, 0.1, 1.0, 0.1, 0.01, 0.1, 0.001])
STOCK_LIMITS = np.array([0.01, 0.10, 0.0001, 0.1, 0.1, 0.001, 0.01, 10, 0.1, 0.01])

# Exchange Configuration
CCXT_CONFIG = {
    'exchange_name': 'binance',
    'sandbox': False,
    'enable_rate_limit': True,
    'timeout': 30000,
    'verbose': False,
}

# Environment Parameters
ENV_PARAMS_CCXT = {
    'lookback': 50,
    'norm_cash': 1e-6,
    'norm_stocks': 100,
    'norm_tech': 1,
    'norm_reward': 1,
    'norm_action': 1,
}

ENV_PARAMS_ALPACA = {
    'lookback': 50,
    'norm_cash': 1e-6,
    'norm_stocks': 100,
    'norm_tech': 1,
    'norm_reward': 1,
    'norm_action': 1,
}

# Trading Parameters
TRADING_PARAMS = {
    'initial_capital': 1000000,
    'buy_cost_pct': 0.001,
    'sell_cost_pct': 0.001,
    'gamma': 0.99,
    'safety_factor': 0.95,
    'cooldown_periods': 24,
    'forced_sell_pct': 0.05,
}

# Technical Indicators
TECHNICAL_INDICATORS_LIST = [
    'open', 'high', 'low', 'close', 'volume',
    'macd', 'macd_signal', 'macd_hist',
    'rsi', 'cci', 'dx'
]

# Auto compute dates
def calculate_start_end_dates(candlewidth):
    no_minutes = int
    candle_to_no_minutes = {
        '1m': 1, '5m': 5, '10m': 10, '30m': 30, 
        '1h': 60, '2h': 2*60, '4h': 4*60, '12h': 12*60
    }
    no_minutes = candle_to_no_minutes[candlewidth]
    
    trade_start_date_datetimeObj = datetime.strptime(trade_start_date, "%Y-%m-%d %H:%M:%S")
    
    train_start_date = (trade_start_date_datetimeObj
                        - timedelta(minutes=no_minutes * (no_candles_for_train + no_candles_for_val))).strftime("%Y-%m-%d %H:%M:%S")
    
    train_end_date = (trade_start_date_datetimeObj
                      - timedelta(minutes=no_minutes * (no_candles_for_val + 1))).strftime("%Y-%m-%d %H:%M:%S")
    
    val_start_date = (trade_start_date_datetimeObj
                      - timedelta(minutes=no_minutes * no_candles_for_val)).strftime("%Y-%m-%d %H:%M:%S")
    
    val_end_date = (trade_start_date_datetimeObj
                    - timedelta(minutes=no_minutes * 1)).strftime("%Y-%m-%d %H:%M:%S")
    
    return train_start_date, train_end_date, val_start_date, val_end_date

TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE = calculate_start_end_dates(TIMEFRAME)
print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("VAL_END_DATE: ", VAL_END_DATE)