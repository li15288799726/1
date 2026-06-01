#!/usr/bin/env python3
"""
Data Download Script for FinRL-Integrated
Downloads both crypto and stock data for training and backtesting
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config_main import *
from src.data.integrated_data_fetcher import get_data_fetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDownloader:
    """Data downloader for integrated FinRL system."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        self.data_fetcher = get_data_fetcher('integrated', cache_dir)
        
    def download_crypto_data(self, tickers: List[str], start_date: str, end_date: str,
                           time_interval: str = '5m', technical_indicators: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Download cryptocurrency data with technical indicators."""
        logger.info(f"Downloading crypto data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        if technical_indicators is None:
            technical_indicators = ['macd', 'rsi', 'cci', 'dx']
        
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Downloading data for {ticker}...")
                
                # Download data with technical indicators
                data = self.data_fetcher.get_crypto_data_with_indicators(
                    ticker_list=[ticker],
                    start_date=start_date,
                    end_date=end_date,
                    time_interval=time_interval,
                    technical_indicators=technical_indicators
                )
                
                if not data.empty:
                    # Save to CSV
                    os.makedirs('data/crypto', exist_ok=True)
                    filename = f"data/crypto/{ticker}_{time_interval}.csv"
                    data.to_csv(filename)
                    logger.info(f"Saved {ticker} data to {filename}")
                    
                    results[ticker] = data
                else:
                    logger.warning(f"No data downloaded for {ticker}")
                    
            except Exception as e:
                logger.error(f"Failed to download data for {ticker}: {e}")
                continue
        
        logger.info(f"Successfully downloaded data for {len(results)} tickers")
        return results
    
    def download_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download stock data."""
        logger.info(f"Downloading stock data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Downloading data for {ticker}...")
                
                # Download stock data
                data = self.data_fetcher.get_stock_data(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not data.empty:
                    # Save to CSV
                    os.makedirs('data/stock', exist_ok=True)
                    filename = f"data/stock/{ticker}.csv"
                    data.to_csv(filename)
                    logger.info(f"Saved {ticker} data to {filename}")
                    
                    results[ticker] = data
                else:
                    logger.warning(f"No data downloaded for {ticker}")
                    
            except Exception as e:
                logger.error(f"Failed to download data for {ticker}: {e}")
                continue
        
        logger.info(f"Successfully downloaded data for {len(results)} tickers")
        return results
    
    def download_training_data(self, crypto_tickers: List[str], stock_tickers: List[str],
                              start_date: str, end_date: str) -> Dict[str, Any]:
        """Download both crypto and stock training data."""
        logger.info("Downloading training data...")
        
        results = {
            'crypto': {},
            'stock': {},
            'metadata': {
                'crypto_tickers': crypto_tickers,
                'stock_tickers': stock_tickers,
                'start_date': start_date,
                'end_date': end_date,
                'download_time': datetime.now().isoformat()
            }
        }
        
        # Download crypto data
        if crypto_tickers:
            results['crypto'] = self.download_crypto_data(crypto_tickers, start_date, end_date)
        
        # Download stock data
        if stock_tickers:
            results['stock'] = self.download_stock_data(stock_tickers, start_date, end_date)
        
        # Save metadata
        metadata_file = "data/metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(results['metadata'], f, indent=2)
        
        logger.info(f"Training data downloaded. Metadata saved to {metadata_file}")
        return results
    
    def validate_data(self, data: pd.DataFrame, min_samples: int = 1000) -> bool:
        """Validate downloaded data quality."""
        if data.empty:
            return False
        
        if len(data) < min_samples:
            logger.warning(f"Data has only {len(data)} samples (minimum: {min_samples})")
            return False
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.1:
            logger.warning(f"High missing value ratio: {missing_ratio:.2%}")
            return False
        
        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in price_cols):
            for col in price_cols:
                if (data[col] <= 0).any():
                    logger.warning(f"Found non-positive values in {col}")
                    return False
        
        return True
    
    def preprocess_data(self, data: pd.DataFrame, asset_type: str = 'crypto') -> pd.DataFrame:
        """Preprocess downloaded data."""
        logger.info(f"Preprocessing {asset_type} data...")
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Handle missing values
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        processed_data = processed_data.drop_duplicates()
        
        # Sort by timestamp
        if 'timestamp' in processed_data.columns:
            processed_data = processed_data.sort_values('timestamp')
        elif hasattr(processed_data.index, 'to_pydatetime'):
            processed_data = processed_data.sort_index()
        
        # Calculate additional technical indicators if not present
        if asset_type == 'crypto' and 'returns' not in processed_data.columns:
            processed_data['returns'] = processed_data['close'].pct_change()
        
        logger.info(f"Data preprocessed. Shape: {processed_data.shape}")
        return processed_data

def main():
    parser = argparse.ArgumentParser(description='Data Download Script for FinRL-Integrated')
    parser.add_argument('--tickers', type=str, help='Comma-separated tickers')
    parser.add_argument('--crypto-tickers', type=str, default=','.join(CRYPTO_TICKER_LIST),
                       help='Comma-separated crypto tickers')
    parser.add_argument('--stock-tickers', type=str, default=','.join(STOCK_TICKER_LIST),
                       help='Comma-separated stock tickers')
    parser.add_argument('--start-date', type=str, default=TRAIN_START_DATE,
                       help='Start date for data download')
    parser.add_argument('--end-date', type=str, default=VAL_END_DATE,
                       help='End date for data download')
    parser.add_argument('--time-interval', type=str, default='5m',
                       help='Time interval for crypto data (1m, 5m, 15m, 1h, etc.)')
    parser.add_argument('--technical-indicators', type=str, default='macd,rsi,cci,dx',
                       help='Comma-separated technical indicators')
    parser.add_argument('--asset-type', choices=['crypto', 'stock', 'both'], default='both',
                       help='Type of assets to download')
    parser.add_argument('--validate', action='store_true',
                       help='Validate downloaded data')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess downloaded data')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for downloaded data')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(args.output_dir)
    
    # Parse tickers
    crypto_tickers = [t.strip() for t in args.crypto_tickers.split(',')]
    stock_tickers = [t.strip() for t in args.stock_tickers.split(',')]
    
    logger.info("Starting data download...")
    logger.info(f"Crypto tickers: {crypto_tickers}")
    logger.info(f"Stock tickers: {stock_tickers}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        # Download data based on asset type
        if args.asset_type in ['crypto', 'both']:
            logger.info("Downloading cryptocurrency data...")
            crypto_data = downloader.download_crypto_data(
                crypto_tickers, args.start_date, args.end_date, args.time_interval,
                args.technical_indicators.split(',')
            )
            
            # Validate and preprocess crypto data
            if args.validate:
                for ticker, data in crypto_data.items():
                    if not downloader.validate_data(data):
                        logger.warning(f"Data validation failed for {ticker}")
            
            if args.preprocess:
                crypto_data = {ticker: downloader.preprocess_data(data, 'crypto') 
                             for ticker, data in crypto_data.items()}
        
        if args.asset_type in ['stock', 'both']:
            logger.info("Downloading stock data...")
            stock_data = downloader.download_stock_data(
                stock_tickers, args.start_date, args.end_date
            )
            
            # Validate and preprocess stock data
            if args.validate:
                for ticker, data in stock_data.items():
                    if not downloader.validate_data(data):
                        logger.warning(f"Data validation failed for {ticker}")
            
            if args.preprocess:
                stock_data = {ticker: downloader.preprocess_data(data, 'stock') 
                            for ticker, data in stock_data.items()}
        
        logger.info("Data download completed successfully!")
        
        # Print summary
        if args.asset_type in ['crypto', 'both']:
            logger.info(f"Crypto data downloaded for {len(crypto_data)} tickers")
        if args.asset_type in ['stock', 'both']:
            logger.info(f"Stock data downloaded for {len(stock_data)} tickers")
    
    except Exception as e:
        logger.error(f"Error during data download: {e}")
        raise

if __name__ == '__main__':
    main()