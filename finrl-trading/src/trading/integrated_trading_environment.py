"""
Integrated Trading Environment - Combines stock and crypto trading environments
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

# Import FinRL_Crypto environments
try:
    from environment_CCXT import CryptoEnvCCXT
    from environment_Alpaca import CryptoEnvAlpaca
    CRYPTO_ENVS_AVAILABLE = True
except ImportError:
    CRYPTO_ENVS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Crypto environments not available")

# Import FinRL-Trading trading components
try:
    from src.trading.alpaca_manager import AlpacaManager
    from src.config.settings import get_config
    STOCK_TRADING_AVAILABLE = True
except ImportError:
    STOCK_TRADING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Stock trading components not available")

logger = logging.getLogger(__name__)

class BaseTradingEnvironment(ABC):
    """Base class for trading environments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_capital = config.get('initial_capital', 1000000)
        self.buy_cost_pct = config.get('buy_cost_pct', 0.001)
        self.sell_cost_pct = config.get('sell_cost_pct', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.env_params = config.get('env_params', {})
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """Get state dimension."""
        pass
    
    @abstractmethod
    def get_action_dim(self) -> int:
        """Get action dimension."""
        pass

class CryptoTradingEnvironment(BaseTradingEnvironment):
    """Cryptocurrency trading environment using CCXT."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not CRYPTO_ENVS_AVAILABLE:
            raise ImportError("Crypto environments not available")
        
        self.exchange_name = config.get('exchange_name', 'binance')
        self.price_array = config.get('price_array')
        self.tech_array = config.get('tech_array')
        self.lookback = self.env_params.get('lookback', 50)
        
        if self.price_array is None or self.tech_array is None:
            raise ValueError("price_array and tech_array are required for crypto environment")
        
        # Initialize CCXT environment
        self.ccxt_env = CryptoEnvCCXT(
            config=config,
            env_params=self.env_params,
            initial_capital=self.initial_capital,
            buy_cost_pct=self.buy_cost_pct,
            sell_cost_pct=self.sell_cost_pct,
            gamma=self.gamma,
            exchange_name=self.exchange_name
        )
        
        self.state_dim = self.ccxt_env.state_dim
        self.action_dim = self.ccxt_env.action_dim
        
    def reset(self) -> np.ndarray:
        """Reset crypto environment."""
        return self.ccxt_env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in crypto environment."""
        return self.ccxt_env.step(action)
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.ccxt_env.state_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.ccxt_env.action_dim
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.ccxt_env.total_asset
    
    def get_holdings(self) -> np.ndarray:
        """Get current holdings."""
        return self.ccxt_env.stocks

class StockTradingEnvironment(BaseTradingEnvironment):
    """Stock trading environment using Alpaca."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not STOCK_TRADING_AVAILABLE:
            raise ImportError("Stock trading components not available")
        
        self.tickers = config.get('tickers', ['AAPL', 'MSFT'])
        self.alpaca_config = config.get('alpaca_config', {})
        
        # Initialize Alpaca manager
        try:
            self.alpaca_manager = AlpacaManager([self._create_alpaca_account()])
            self.state_dim = len(self.tickers) * 5 + 1  # price data + cash
            self.action_dim = len(self.tickers)
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca environment: {e}")
            raise
    
    def _create_alpaca_account(self):
        """Create Alpaca account from config."""
        # This would be implemented based on FinRL-Trading's AlpacaManager
        # For now, return a mock account
        return type('MockAccount', (), {})()
    
    def reset(self) -> np.ndarray:
        """Reset stock environment."""
        # Initialize with equal weights
        state = np.zeros(self.state_dim)
        state[0] = 1.0  # cash position
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in stock environment."""
        # Simplified implementation
        next_state = np.random.random(self.state_dim)
        reward = np.random.normal(0, 0.01)
        done = False
        info = {'portfolio_value': np.random.random() * 1000000}
        
        return next_state, reward, done, info
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.state_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.action_dim

class IntegratedTradingEnvironment(BaseTradingEnvironment):
    """Integrated trading environment supporting both stocks and crypto."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.asset_type = config.get('asset_type', 'both')
        self.crypto_weight = config.get('crypto_weight', 0.5)
        self.stock_weight = config.get('stock_weight', 0.5)
        
        # Initialize individual environments
        self.crypto_env = None
        self.stock_env = None
        
        if self.asset_type in ['crypto', 'both']:
            crypto_config = {
                **config,
                'price_array': config.get('crypto_price_array'),
                'tech_array': config.get('crypto_tech_array')
            }
            self.crypto_env = CryptoTradingEnvironment(crypto_config)
        
        if self.asset_type in ['stock', 'both']:
            stock_config = {
                **config,
                'tickers': config.get('stock_tickers', ['AAPL', 'MSFT'])
            }
            self.stock_env = StockTradingEnvironment(stock_config)
        
        # Set dimensions
        if self.crypto_env and self.stock_env:
            self.state_dim = self.crypto_env.get_state_dim() + self.stock_env.get_state_dim()
            self.action_dim = self.crypto_env.get_action_dim() + self.stock_env.get_action_dim()
        elif self.crypto_env:
            self.state_dim = self.crypto_env.get_state_dim()
            self.action_dim = self.crypto_env.get_action_dim()
        elif self.stock_env:
            self.state_dim = self.stock_env.get_state_dim()
            self.action_dim = self.stock_env.get_action_dim()
        else:
            raise ValueError("No valid trading environment configured")
    
    def reset(self) -> np.ndarray:
        """Reset integrated environment."""
        if self.crypto_env and self.stock_env:
            crypto_state = self.crypto_env.reset()
            stock_state = self.stock_env.reset()
            return np.concatenate([crypto_state, stock_state])
        elif self.crypto_env:
            return self.crypto_env.reset()
        elif self.stock_env:
            return self.stock_env.reset()
        else:
            raise ValueError("No environment available")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in integrated environment."""
        if self.crypto_env and self.stock_env:
            # Split action between crypto and stock
            crypto_action_dim = self.crypto_env.get_action_dim()
            crypto_action = action[:crypto_action_dim]
            stock_action = action[crypto_action_dim:]
            
            crypto_state, crypto_reward, crypto_done, crypto_info = self.crypto_env.step(crypto_action)
            stock_state, stock_reward, stock_done, stock_info = self.stock_env.step(stock_action)
            
            # Combine rewards
            total_reward = (crypto_reward * self.crypto_weight + 
                          stock_reward * self.stock_weight)
            
            # Combine states
            next_state = np.concatenate([crypto_state, stock_state])
            
            # Check if both episodes are done
            done = crypto_done and stock_done
            
            # Combine info
            info = {
                'crypto_info': crypto_info,
                'stock_info': stock_info,
                'total_reward': total_reward
            }
            
            return next_state, total_reward, done, info
            
        elif self.crypto_env:
            return self.crypto_env.step(action)
        elif self.stock_env:
            return self.stock_env.step(action)
        else:
            raise ValueError("No environment available")
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.state_dim
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.action_dim
    
    def get_portfolio_values(self) -> Dict[str, float]:
        """Get portfolio values for all environments."""
        values = {}
        if self.crypto_env:
            values['crypto'] = self.crypto_env.get_portfolio_value()
        if self.stock_env:
            values['stock'] = self.stock_env.get_portfolio_value()
        return values
    
    def get_holdings(self) -> Dict[str, np.ndarray]:
        """Get holdings for all environments."""
        holdings = {}
        if self.crypto_env:
            holdings['crypto'] = self.crypto_env.get_holdings()
        if self.stock_env:
            holdings['stock'] = self.stock_env.get_holdings()
        return holdings

class EnvironmentFactory:
    """Factory class for creating trading environments."""
    
    @staticmethod
    def create_environment(env_type: str, config: Dict[str, Any]) -> BaseTradingEnvironment:
        """Create trading environment of specified type."""
        if env_type == 'crypto':
            return CryptoTradingEnvironment(config)
        elif env_type == 'stock':
            return StockTradingEnvironment(config)
        elif env_type == 'integrated':
            return IntegratedTradingEnvironment(config)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

# Configuration presets
ENVIRONMENT_PRESETS = {
    'crypto_trading': {
        'asset_type': 'crypto',
        'exchange_name': 'binance',
        'initial_capital': 1000000,
        'buy_cost_pct': 0.001,
        'sell_cost_pct': 0.001,
        'gamma': 0.99,
        'env_params': {
            'lookback': 50,
            'norm_cash': 1e-6,
            'norm_stocks': 100,
            'norm_tech': 1,
            'norm_reward': 1,
            'norm_action': 1
        }
    },
    'stock_trading': {
        'asset_type': 'stock',
        'tickers': ['AAPL', 'MSFT', 'GOOGL'],
        'initial_capital': 1000000,
        'buy_cost_pct': 0.001,
        'sell_cost_pct': 0.001,
        'gamma': 0.99,
        'env_params': {
            'lookback': 50,
            'norm_cash': 1e-6,
            'norm_stocks': 100,
            'norm_tech': 1,
            'norm_reward': 1,
            'norm_action': 1
        }
    },
    'integrated_trading': {
        'asset_type': 'both',
        'crypto_weight': 0.5,
        'stock_weight': 0.5,
        'initial_capital': 1000000,
        'buy_cost_pct': 0.001,
        'sell_cost_pct': 0.001,
        'gamma': 0.99,
        'env_params': {
            'lookback': 50,
            'norm_cash': 1e-6,
            'norm_stocks': 100,
            'norm_tech': 1,
            'norm_reward': 1,
            'norm_action': 1
        }
    }
}