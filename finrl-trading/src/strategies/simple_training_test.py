#!/usr/bin/env python3
"""
Simple Training Test - No External Dependencies
A simplified training test that uses only Python standard library
"""

import os
import sys
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDRLStrategy:
    """Simple DRL strategy using only Python standard library."""
    
    def __init__(self, algorithm='PPO', episodes=10):
        self.algorithm = algorithm
        self.episodes = episodes
        self.trained = False
        self.weights = {}
        self.training_history = []
        
    def train(self, data, validation_data=None):
        """Simple training simulation."""
        logger.info(f"Starting {self.algorithm} training for {self.episodes} episodes")
        
        # Simulate training process
        for episode in range(self.episodes):
            # Simulate learning
            episode_reward = random.uniform(-1, 1)
            episode_loss = random.uniform(0, 0.5)
            
            # Update weights (simplified)
            if episode == 0:
                # Initialize weights
                self.weights = {
                    'BTCUSDT': random.uniform(0.1, 0.5),
                    'ETHUSDT': random.uniform(0.1, 0.5),
                    'AAPL': random.uniform(0.1, 0.5),
                    'MSFT': random.uniform(0.1, 0.5),
                    'GOOGL': random.uniform(0.1, 0.5)
                }
            else:
                # Update weights based on performance
                for asset in self.weights:
                    change = random.uniform(-0.1, 0.1)
                    self.weights[asset] = max(0, min(1, self.weights[asset] + change))
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            # Log progress
            if episode % 2 == 0:
                logger.info(f"Episode {episode}/{self.episodes} - Reward: {episode_reward:.3f}, Loss: {episode_loss:.3f}")
            
            # Store training history
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'loss': episode_loss,
                'weights': self.weights.copy()
            })
        
        self.trained = True
        logger.info("Training completed")
        
        return {
            'algorithm': self.algorithm,
            'episodes': self.episodes,
            'final_reward': self.training_history[-1]['reward'],
            'final_loss': self.training_history[-1]['loss'],
            'weights': self.weights,
            'training_history': self.training_history
        }
    
    def generate_weights(self, data):
        """Generate portfolio weights."""
        if not self.trained:
            raise ValueError("Strategy must be trained before generating weights")
        
        # Create a simple named tuple for result
        from collections import namedtuple
        Result = namedtuple('Result', ['weights', 'metadata'])
        
        metadata = {
            'algorithm': self.algorithm,
            'timestamp': datetime.now().isoformat(),
            'data_shape': str(data.shape) if hasattr(data, 'shape') else str(len(data))
        }
        
        return Result(self.weights, metadata)
    
    def save_model(self, path):
        """Save model to file."""
        model_data = {
            'algorithm': self.algorithm,
            'episodes': self.episodes,
            'trained': self.trained,
            'weights': self.weights,
            'training_history': self.training_history,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        self.algorithm = model_data['algorithm']
        self.episodes = model_data['episodes']
        self.trained = model_data['trained']
        self.weights = model_data['weights']
        self.training_history = model_data['training_history']
        
        logger.info(f"Model loaded from {path}")

def generate_sample_data(days=30):
    """Generate sample price data."""
    logger.info(f"Generating {days} days of sample data")
    
    data = []
    base_prices = {
        'BTCUSDT': 50000,
        'ETHUSDT': 3000,
        'AAPL': 150,
        'MSFT': 300,
        'GOOGL': 2500
    }
    
    for day in range(days):
        date = datetime.now() - timedelta(days=days-day)
        row = {'timestamp': date.isoformat()}
        
        for asset, base_price in base_prices.items():
            # Add some random variation
            variation = random.uniform(-0.05, 0.05)
            price = base_price * (1 + variation)
            row[asset] = price
            
            # Update base price for next day
            base_prices[asset] = price
        
        data.append(row)
    
    return data

def calculate_performance_metrics(data, weights):
    """Calculate portfolio performance metrics."""
    if not data:
        return {}
    
    portfolio_values = []
    returns = []
    
    for i, row in enumerate(data):
        portfolio_value = sum(row[asset] * weight for asset, weight in weights.items() if asset in row)
        portfolio_values.append(portfolio_value)
        
        if i > 0:
            returns.append((portfolio_value - portfolio_values[i-1]) / portfolio_values[i-1])
    
    if not returns:
        return {}
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    avg_return = sum(returns) / len(returns)
    volatility = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5
    
    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'avg_return': avg_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_value': portfolio_values[-1],
        'initial_value': portfolio_values[0]
    }

def main():
    """Main training test function."""
    logger.info("Starting simple training test")
    
    # Create output directory
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    sample_data = generate_sample_data(30)
    logger.info(f"Generated {len(sample_data)} samples of data")
    
    # Test different algorithms
    algorithms = ['PPO', 'SAC', 'DDPG']
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"\n=== Testing {algorithm} ===")
        
        # Create and train strategy
        strategy = SimpleDRLStrategy(algorithm, episodes=5)
        training_result = strategy.train(sample_data)
        
        # Generate weights
        weights_result = strategy.generate_weights(sample_data)
        
        # Calculate performance metrics
        performance = calculate_performance_metrics(sample_data, weights_result.weights)
        
        # Store results
        results[algorithm] = {
            'training_result': training_result,
            'performance': performance,
            'weights': weights_result.weights,
            'metadata': weights_result.metadata
        }
        
        # Save model
        model_path = os.path.join(output_dir, f"{algorithm}_model.json")
        strategy.save_model(model_path)
        
        logger.info(f"{algorithm} training completed")
        logger.info(f"Final weights: {weights_result.weights}")
        logger.info(f"Performance: {performance}")
    
    # Save all results
    results_path = os.path.join(output_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    summary = {
        'test_timestamp': datetime.now().isoformat(),
        'algorithms_tested': algorithms,
        'data_samples': len(sample_data),
        'results': {algo: {
            'final_reward': results[algo]['training_result']['final_reward'],
            'total_return': results[algo]['performance'].get('total_return', 0),
            'sharpe_ratio': results[algo]['performance'].get('sharpe_ratio', 0)
        } for algo in algorithms}
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n=== Training Test Summary ===")
    logger.info(f"Test completed at: {datetime.now().isoformat()}")
    logger.info(f"Algorithms tested: {algorithms}")
    logger.info(f"Data samples: {len(sample_data)}")
    
    for algo in algorithms:
        perf = results[algo]['performance']
        logger.info(f"{algo}: Return={perf.get('total_return', 0):.3f}, Sharpe={perf.get('sharpe_ratio', 0):.3f}")
    
    logger.info(f"Results saved to {output_dir}/")
    
    return results

if __name__ == '__main__':
    main()