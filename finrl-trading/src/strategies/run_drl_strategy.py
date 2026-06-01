#!/usr/bin/env python3
"""
DRL Strategy Runner - Integrated version for FinRL-Integrated
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config_main import *
from src.strategies.drl_strategy import create_drl_strategy, DRL_STRATEGY_PRESETS
from src.data.integrated_data_fetcher import get_data_fetcher
from src.trading.integrated_trading_environment import EnvironmentFactory, ENVIRONMENT_PRESETS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DRLStrategyRunner:
    """DRL Strategy Runner for integrated FinRL system."""
    
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.strategy = None
        self.data_fetcher = None
        self.environment = None
        self.trained = False
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Use default configuration
            return {
                'drl_algorithm': 'PPO',
                'state_dim': 50,
                'action_dim': 3,
                'net_dim': 256,
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'env_params': {
                    'lookback': 50,
                    'norm_cash': 1e-6,
                    'norm_stocks': 100,
                    'norm_tech': 1,
                    'norm_reward': 1,
                    'norm_action': 1
                },
                'trading_params': {
                    'episodes': 100,
                    'initial_capital': 1000000,
                    'buy_cost_pct': 0.001,
                    'sell_cost_pct': 0.001
                }
            }
    
    def prepare_data(self, start_date, end_date, crypto_tickers=None, stock_tickers=None):
        """Prepare training data for DRL strategy."""
        logger.info(f"Preparing data from {start_date} to {end_date}")
        
        # Initialize data fetcher
        self.data_fetcher = get_data_fetcher('integrated')
        
        # Prepare data based on asset type
        if crypto_tickers:
            logger.info(f"Downloading crypto data for: {crypto_tickers}")
            crypto_ticker_list = [t.strip() for t in crypto_tickers.split(',')]
            
            # Download crypto data with technical indicators
            crypto_data = self.data_fetcher.get_crypto_data_with_indicators(
                ticker_list=crypto_ticker_list,
                start_date=start_date,
                end_date=end_date,
                time_interval='5m',
                technical_indicators=['macd', 'rsi', 'cci', 'dx']
            )
            
            if not crypto_data.empty:
                logger.info(f"Crypto data shape: {crypto_data.shape}")
            else:
                logger.warning("No crypto data downloaded")
        
        if stock_tickers:
            logger.info(f"Downloading stock data for: {stock_tickers}")
            stock_ticker_list = [t.strip() for t in stock_tickers.split(',')]
            
            # Download stock data
            stock_data = self.data_fetcher.get_stock_data(
                tickers=stock_ticker_list,
                start_date=start_date,
                end_date=end_date
            )
            
            if not stock_data.empty:
                logger.info(f"Stock data shape: {stock_data.shape}")
            else:
                logger.warning("No stock data downloaded")
        
        return crypto_data, stock_data
    
    def prepare_environment(self, crypto_data=None, stock_data=None, asset_type='both'):
        """Prepare trading environment."""
        logger.info(f"Preparing {asset_type} trading environment")
        
        # Build environment configuration
        env_config = ENVIRONMENT_PRESETS['integrated_trading'].copy()
        
        if crypto_data is not None:
            env_config['crypto_price_array'] = crypto_data[['open', 'high', 'low', 'close']].values
            env_config['crypto_tech_array'] = crypto_data[['macd', 'rsi', 'cci', 'dx']].values
        
        if stock_data is not None:
            env_config['stock_tickers'] = [t.strip() for t in self.config.get('stock_tickers', 'AAPL,MSFT,GOOGL').split(',')]
        
        # Create environment
        self.environment = EnvironmentFactory.create_environment(asset_type, env_config)
        
        logger.info(f"Environment created - State dim: {self.environment.get_state_dim()}, Action dim: {self.environment.get_action_dim()}")
        
        return self.environment
    
    def create_strategy(self, algorithm='PPO'):
        """Create DRL strategy."""
        logger.info(f"Creating {algorithm} strategy")
        
        # Create strategy configuration
        strategy_config = DRL_STRATEGY_PRESETS[f'crypto_{algorithm.lower()}'].copy()
        strategy_config.trading_params = self.config.get('trading_params', {})
        strategy_config.env_params = self.config.get('env_params', {})
        
        # Create strategy
        self.strategy = create_drl_strategy(algorithm, strategy_config)
        
        return self.strategy
    
    def train(self, data, validation_data=None, episodes=None):
        """Train DRL strategy."""
        if not self.strategy:
            raise ValueError("Strategy must be created before training")
        
        episodes = episodes or self.config.get('trading_params', {}).get('episodes', 100)
        
        logger.info(f"Training strategy for {episodes} episodes")
        
        # Train strategy
        results = self.strategy.train(data, validation_data)
        
        self.trained = True
        
        logger.info("Training completed")
        logger.info(f"Training results: {results}")
        
        return results
    
    def backtest(self, data, start_date=None, end_date=None):
        """Run backtest on trained strategy."""
        if not self.trained:
            raise ValueError("Strategy must be trained before backtesting")
        
        logger.info("Running backtest")
        
        # Generate weights
        result = self.strategy.generate_weights(data)
        
        # Calculate portfolio performance
        portfolio_value = self._calculate_portfolio_performance(data, result.weights)
        
        backtest_results = {
            'final_value': portfolio_value[-1] if portfolio_value else 0,
            'total_return': (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] if portfolio_value else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_value) if portfolio_value else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_value) if portfolio_value else 0,
            'weights': result.weights,
            'metadata': result.metadata
        }
        
        logger.info(f"Backtest results: {backtest_results}")
        
        return backtest_results
    
    def _calculate_portfolio_performance(self, data, weights):
        """Calculate portfolio performance over time."""
        portfolio_values = []
        
        for idx, row in data.iterrows():
            portfolio_value = 0
            
            for asset, weight in weights.items():
                if asset in data.columns:
                    portfolio_value += weight * row[asset]
            
            portfolio_values.append(portfolio_value)
        
        return np.array(portfolio_values)
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown."""
        if len(portfolio_values) == 0:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.02):
        """Calculate Sharpe ratio."""
        if len(portfolio_values) < 2:
            return 0
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) == 0:
            return 0
        
        return np.sqrt(252) * (np.mean(returns) - risk_free_rate/252) / np.std(returns)
    
    def run_single_prediction(self, data, date):
        """Run single date prediction."""
        logger.info(f"Running prediction for {date}")
        
        if not self.trained:
            raise ValueError("Strategy must be trained before prediction")
        
        # Filter data up to the prediction date
        prediction_data = data[data.index <= date]
        
        if prediction_data.empty:
            raise ValueError(f"No data available for date {date}")
        
        # Generate weights
        result = self.strategy.generate_weights(prediction_data)
        
        return result
    
    def save_model(self, path):
        """Save trained model."""
        if not self.trained:
            raise ValueError("No trained model to save")
        
        # Save strategy configuration and weights
        model_data = {
            'config': self.config,
            'strategy_type': self.strategy.drl_algorithm,
            'trained': self.trained,
            'metadata': getattr(self.strategy, 'metadata', {})
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        self.config = model_data['config']
        self.trained = model_data['trained']
        
        # Recreate strategy
        self.create_strategy(model_data['strategy_type'])
        
        logger.info(f"Model loaded from {path}")

def main():
    parser = argparse.ArgumentParser(description='DRL Strategy Runner for FinRL-Integrated')
    parser.add_argument('--mode', choices=['train', 'backtest', 'single'], required=True,
                       help='Execution mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--drl-algorithm', choices=['PPO', 'SAC', 'DDPG', 'ENSEMBLE'], default='PPO',
                       help='DRL algorithm to use')
    parser.add_argument('--crypto-tickers', type=str, default='BTCUSDT,ETHUSDT',
                       help='Comma-separated crypto tickers')
    parser.add_argument('--stock-tickers', type=str, default='AAPL,MSFT,GOOGL',
                       help='Comma-separated stock tickers')
    parser.add_argument('--start-date', type=str, default=TRAIN_START_DATE,
                       help='Start date for data')
    parser.add_argument('--end-date', type=str, default=VAL_END_DATE,
                       help='End date for data')
    parser.add_argument('--single-date', type=str, help='Single date for single mode')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize runner
    runner = DRLStrategyRunner(args.config)
    
    try:
        if args.mode == 'train':
            # Prepare data
            crypto_data, stock_data = runner.prepare_data(
                args.start_date, args.end_date, args.crypto_tickers, args.stock_tickers
            )
            
            # Determine asset type and prepare environment
            asset_type = 'both' if (crypto_data is not None and stock_data is not None) else \
                         'crypto' if crypto_data is not None else 'stock'
            
            runner.prepare_environment(crypto_data, stock_data, asset_type)
            
            # Create and train strategy
            strategy = runner.create_strategy(args.drl_algorithm)
            results = runner.train(crypto_data or stock_data, episodes=args.episodes)
            
            # Save results
            results_path = os.path.join(args.output, 'training_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save model
            model_path = os.path.join(args.output, 'drl_model.json')
            runner.save_model(model_path)
            
            logger.info(f"Training completed. Results saved to {results_path}")
        
        elif args.mode == 'backtest':
            # Load model
            model_path = os.path.join(args.output, 'drl_model.json')
            runner.load_model(model_path)
            
            # Prepare data
            crypto_data, stock_data = runner.prepare_data(
                args.start_date, args.end_date, args.crypto_tickers, args.stock_tickers
            )
            
            # Run backtest
            test_data = crypto_data if crypto_data is not None else stock_data
            results = runner.backtest(test_data, args.start_date, args.end_date)
            
            # Save results
            results_path = os.path.join(args.output, 'backtest_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Backtest completed. Results saved to {results_path}")
        
        elif args.mode == 'single':
            if not args.single_date:
                raise ValueError("Single date is required for single mode")
            
            # Load model
            model_path = os.path.join(args.output, 'drl_model.json')
            runner.load_model(model_path)
            
            # Prepare data
            crypto_data, stock_data = runner.prepare_data(
                args.start_date, args.end_date, args.crypto_tickers, args.stock_tickers
            )
            
            # Run prediction
            test_data = crypto_data if crypto_data is not None else stock_data
            result = runner.run_single_prediction(test_data, args.single_date)
            
            # Save results
            results_path = os.path.join(args.output, 'prediction_results.json')
            with open(results_path, 'w') as f:
                json.dump(result._asdict(), f, indent=2)
            
            logger.info(f"Prediction completed. Results saved to {results_path}")
    
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        raise

if __name__ == '__main__':
    main()