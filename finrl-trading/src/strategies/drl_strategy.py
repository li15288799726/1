"""
Integrated DRL Strategy Module - Combines FinRL-Trading and FinRL_Crypto strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path

# Import DRL agents from FinRL_Crypto
try:
    from drl_agents.agents.AgentBase import AgentBase
    from drl_agents.agents.AgentPPO import AgentPPO
    from drl_agents.agents.AgentA2C import AgentA2C
    from drl_agents.agents.AgentDDPG import AgentDDPG
    from drl_agents.agents.AgentTD3 import AgentTD3
    from drl_agents.agents.AgentSAC import AgentSAC
    DRL_AVAILABLE = True
except ImportError:
    DRL_AVAILABLE = False
    logger.warning("DRL agents not available")

# Import FinRL-Trading base strategy
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

logger = logging.getLogger(__name__)

class DRLStrategyConfig(StrategyConfig):
    """Configuration for DRL-based strategies."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drl_algorithm = kwargs.get('drl_algorithm', 'PPO')
        self.state_dim = kwargs.get('state_dim', 50)
        self.action_dim = kwargs.get('action_dim', 3)  # buy/sell/hold
        self.net_dim = kwargs.get('net_dim', 256)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gpu_id = kwargs.get('gpu_id', 0)
        self.env_params = kwargs.get('env_params', {})
        self.trading_params = kwargs.get('trading_params', {})

class BaseDRLStrategy(BaseStrategy):
    """Base class for DRL strategies."""
    
    def __init__(self, config: DRLStrategyConfig):
        super().__init__(config)
        self.drl_algorithm = config.drl_algorithm
        self.agent = None
        self.trained = False
        
        if not DRL_AVAILABLE:
            raise ImportError("DRL agents not available. Please install required dependencies.")

    @abstractmethod
    def _create_agent(self, state_dim: int, action_dim: int) -> AgentBase:
        """Create and return the DRL agent."""
        pass

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for DRL training."""
        # Extract price and technical indicator data
        price_data = data[['open', 'high', 'low', 'close', 'volume']].values
        tech_data = data[['macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']].values
        time_data = data.index.values
        
        # Handle missing values
        tech_data = np.nan_to_num(tech_data, nan=0.0)
        
        return price_data, tech_data, time_data

    def _create_state(self, price_data: np.ndarray, tech_data: np.ndarray, t: int) -> np.ndarray:
        """Create state vector for time t."""
        lookback = self.config.env_params.get('lookback', 50)
        
        if t < lookback:
            # Pad with zeros if not enough history
            state = np.zeros(lookback * 2)  # price + tech indicators
            state[-(lookback - t):lookback] = price_data[:t].flatten()
            state[lookback + (lookback - t):2*lookback] = tech_data[:t].flatten()
        else:
            state = np.concatenate([
                price_data[t-lookback:t].flatten(),
                tech_data[t-lookback:t].flatten()
            ])
        
        return state

    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the DRL agent."""
        logger.info(f"Training {self.drl_algorithm} strategy...")
        
        # Prepare data
        price_data, tech_data, time_data = self._prepare_data(data)
        
        # Create agent
        state_dim = self.config.env_params.get('lookback', 50) * 2  # price + tech indicators
        action_dim = self.config.action_dim
        
        self.agent = self._create_agent(state_dim, action_dim)
        
        # Training loop
        episodes = self.config.trading_params.get('episodes', 100)
        for episode in range(episodes):
            logger.info(f"Training episode {episode + 1}/{episodes}")
            
            # Reset environment
            state = self._create_state(price_data, tech_data, 0)
            done = False
            total_reward = 0
            
            while not done:
                # Get action from agent
                action = self.agent.act(state)
                
                # Execute action (simplified)
                next_state, reward, done = self._execute_action(state, action, price_data, tech_data, time_data)
                
                # Store experience
                self.agent.store(state, action, reward, next_state, done)
                
                # Learn
                self.agent.learn()
                
                total_reward += reward
                state = next_state
            
            logger.info(f"Episode {episode + 1} completed with total reward: {total_reward}")
        
        self.trained = True
        return {'episodes': episodes, 'total_reward': total_reward}

    def _execute_action(self, state: np.ndarray, action: int, price_data: np.ndarray, 
                       tech_data: np.ndarray, time_data: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute trading action and return next state, reward, and done flag."""
        # Simplified action execution
        # In real implementation, this would interact with a trading environment
        
        reward = 0.0
        done = False
        
        # Simple reward function
        if action == 0:  # buy
            reward = np.random.normal(0.001, 0.01)
        elif action == 1:  # sell
            reward = np.random.normal(-0.001, 0.01)
        else:  # hold
            reward = np.random.normal(0, 0.005)
        
        # Move to next time step
        # This is simplified - in real implementation would track portfolio state
        
        return state, reward, done

    def generate_weights(self, data: pd.DataFrame, **kwargs) -> StrategyResult:
        """Generate portfolio weights using trained DRL agent."""
        if not self.trained:
            raise ValueError("Strategy must be trained before generating weights")
        
        # Prepare data
        price_data, tech_data, time_data = self._prepare_data(data)
        
        # Generate actions/weights
        weights = {}
        actions = []
        
        for t in range(len(price_data)):
            state = self._create_state(price_data, tech_data, t)
            action = self.agent.act(state)
            actions.append(action)
            
            # Convert action to weight (simplified)
            if action == 0:  # buy
                weights[f'asset_{t}'] = 0.1
            elif action == 1:  # sell
                weights[f'asset_{t}'] = -0.1
            else:  # hold
                weights[f'asset_{t}'] = 0.0
        
        return StrategyResult(
            weights=weights,
            metadata={'actions': actions, 'algorithm': self.drl_algorithm}
        )

class PPOCryptoStrategy(BaseDRLStrategy):
    """PPO strategy for cryptocurrency trading."""
    
    def _create_agent(self, state_dim: int, action_dim: int) -> AgentBase:
        """Create PPO agent."""
        return AgentPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            net_dim=self.config.net_dim,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gpu_id=self.config.gpu_id
        )

class SACCryptoStrategy(BaseDRLStrategy):
    """SAC strategy for cryptocurrency trading."""
    
    def _create_agent(self, state_dim: int, action_dim: int) -> AgentBase:
        """Create SAC agent."""
        return AgentSAC(
            state_dim=state_dim,
            action_dim=action_dim,
            net_dim=self.config.net_dim,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gpu_id=self.config.gpu_id
        )

class DDPGCryptoStrategy(BaseDRLStrategy):
    """DDPG strategy for cryptocurrency trading."""
    
    def _create_agent(self, state_dim: int, action_dim: int) -> AgentBase:
        """Create DDPG agent."""
        return AgentDDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            net_dim=self.config.net_dim,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gpu_id=self.config.gpu_id
        )

class IntegratedDRLStrategy(BaseStrategy):
    """Strategy that combines multiple DRL algorithms."""
    
    def __init__(self, config: DRLStrategyConfig):
        super().__init__(config)
        self.strategies = {}
        self.weights = {}
        
        # Initialize individual strategies
        if 'PPO' in config.drl_algorithm:
            self.strategies['PPO'] = PPOCryptoStrategy(config)
        if 'SAC' in config.drl_algorithm:
            self.strategies['SAC'] = SACCryptoStrategy(config)
        if 'DDPG' in config.drl_algorithm:
            self.strategies['DDPG'] = DDPGCryptoStrategy(config)

    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train all DRL strategies."""
        results = {}
        
        for name, strategy in self.strategies.items():
            logger.info(f"Training {name} strategy...")
            results[name] = strategy.train(data, validation_data)
        
        return results

    def generate_weights(self, data: pd.DataFrame, **kwargs) -> StrategyResult:
        """Generate weights using ensemble of DRL strategies."""
        all_weights = {}
        all_metadata = {}
        
        for name, strategy in self.strategies.items():
            result = strategy.generate_weights(data, **kwargs)
            all_weights[name] = result.weights
            all_metadata[name] = result.metadata
        
        # Combine weights (simple average)
        combined_weights = {}
        for name, weights in all_weights.items():
            for asset, weight in weights.items():
                if asset not in combined_weights:
                    combined_weights[asset] = 0.0
                combined_weights[asset] += weight / len(all_weights)
        
        return StrategyResult(
            weights=combined_weights,
            metadata={'ensemble': all_metadata}
        )

# Factory function
def create_drl_strategy(algorithm: str, config: DRLStrategyConfig) -> BaseStrategy:
    """Factory function to create DRL strategies."""
    
    if algorithm == 'PPO':
        return PPOCryptoStrategy(config)
    elif algorithm == 'SAC':
        return SACCryptoStrategy(config)
    elif algorithm == 'DDPG':
        return DDPGCryptoStrategy(config)
    elif algorithm == 'ENSEMBLE':
        return IntegratedDRLStrategy(config)
    else:
        raise ValueError(f"Unknown DRL algorithm: {algorithm}")

# Configuration presets
DRL_STRATEGY_PRESETS = {
    'crypto_ppo': DRLStrategyConfig(
        drl_algorithm='PPO',
        state_dim=50,
        action_dim=3,
        net_dim=256,
        learning_rate=0.0001,
        gamma=0.99,
        env_params={'lookback': 50, 'norm_cash': 1e-6, 'norm_stocks': 100},
        trading_params={'episodes': 100, 'initial_capital': 1000000}
    ),
    'crypto_sac': DRLStrategyConfig(
        drl_algorithm='SAC',
        state_dim=50,
        action_dim=3,
        net_dim=256,
        learning_rate=0.0001,
        gamma=0.99,
        env_params={'lookback': 50, 'norm_cash': 1e-6, 'norm_stocks': 100},
        trading_params={'episodes': 100, 'initial_capital': 1000000}
    ),
    'crypto_ensemble': DRLStrategyConfig(
        drl_algorithm='ENSEMBLE',
        state_dim=50,
        action_dim=3,
        net_dim=256,
        learning_rate=0.0001,
        gamma=0.99,
        env_params={'lookback': 50, 'norm_cash': 1e-6, 'norm_stocks': 100},
        trading_params={'episodes': 100, 'initial_capital': 1000000}
    )
}