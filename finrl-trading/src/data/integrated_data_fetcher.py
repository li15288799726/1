"""
Integrated Data Fetcher - Combines Stock and Crypto Data Sources
"""

import os
import logging
import json
from typing import List, Optional, Dict, Any, Protocol, Tuple
from datetime import datetime
from abc import ABC
import pandas as pd
from pathlib import Path
import yfinance as yf
import pandas_market_calendars as mcal
import requests
import numpy as np
import concurrent.futures
from tqdm import tqdm
import pandas_market_calendars as mcal

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Configure logging
logger = logging.getLogger(__name__)

# Import crypto processors
try:
    from processor_Binance import BinanceProcessor
    from processor_Yahoo import Yahoofinance
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("Crypto processors not available")

class DataSource(Protocol):
    """Protocol for data source implementations."""

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components."""
        ...

    def get_fundamental_data(self, tickers: List[str],
                           start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data for tickers."""
        ...

    def get_price_data(self, tickers: pd.DataFrame,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for tickers."""
        ...

    def is_available(self) -> bool:
        """Check if data source is available."""
        ...

    def get_news(self, ticker: str, from_date: str, to_date: str,
                 analyze_sentiment: bool = False,
                 sentiment_model: Optional[str] = None,
                 force_refresh: bool = False) -> pd.DataFrame:
        """Get news articles for a ticker."""
        ...

class BaseDataFetcher(ABC):
    """Base class for data fetchers with common functionality."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize base data fetcher.
        
        Args:
            cache_dir: Deprecated, kept for backward compatibility. Uses DATA_BASE_DIR env var instead.
        """
        # Import here to avoid circular dependency
        from src.data.data_store import get_data_store
        self.data_store = get_data_store(base_dir=cache_dir)

    def _standardize_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize fundamental data format."""
        required_columns = ['gvkey', 'datadate', 'tic', 'prccd', 'ajexdi']
        df = df.copy()

        # Ensure required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col == 'gvkey':
                    df['gvkey'] = df.get('tic', df.index)
                elif col == 'datadate':
                    df['datadate'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('date', df.index))
                elif col == 'tic':
                    df['tic'] = df.get('gvkey', df.index)
                elif col == 'prccd':
                    df['prccd'] = df.get('close', df.get('adj_close', 100))
                elif col == 'ajexdi':
                    df['ajexdi'] = df.get('adj_factor', 1.0)

        # Add adjusted price
        if 'adj_close' not in df.columns and 'prccd' in df.columns and 'ajexdi' in df.columns:
            df['adj_close'] = df['prccd'] / df['ajexdi']

        return df[required_columns + ['adj_close']]

    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize price data format."""
        df = df.copy()

        # Rename columns to match expected format
        column_mapping = {
            'Open': 'prcod',
            'High': 'prchd',
            'Low': 'prcld',
            'Close': 'prccd',
            'Adj Close': 'adj_close',
            'Volume': 'cshtrd'
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ['datadate', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'adj_close']
        for col in required_columns:
            if col not in df.columns:
                if col == 'datadate':
                    df['datadate'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
                elif col == 'prccd':
                    df['prccd'] = df.get('Close', df.get('close', 100))
                elif col == 'prcod':
                    df['prcod'] = df.get('Open', df.get('open', df['prccd']))
                elif col == 'prchd':
                    df['prchd'] = df.get('High', df.get('high', df['prccd']))
                elif col == 'prcld':
                    df['prcld'] = df.get('Low', df.get('low', df['prccd']))
                elif col == 'cshtrd':
                    df['cshtrd'] = df.get('Volume', df.get('volume', 1000000))
                elif col == 'adj_close':
                    df['adj_close'] = df.get('Adj Close', df.get('adj_close', df['prccd']))

        # Add gvkey column if missing
        if 'gvkey' not in df.columns:
            if 'tic' in df.columns:
                df['gvkey'] = df['tic']
            else:
                df['gvkey'] = 'UNKNOWN'

        # Add tic column if missing
        if 'tic' not in df.columns:
            if 'gvkey' in df.columns:
                df['tic'] = df['gvkey']
            else:
                df['tic'] = 'UNKNOWN'

        return df[['gvkey', 'datadate', 'tic', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'adj_close']]

class YahooDataFetcher(BaseDataFetcher, DataSource):
    """Yahoo Finance data fetcher for stocks."""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        self.base_url = "https://query1.finance.yahoo.com"

    def is_available(self) -> bool:
        """Check if data source is usable."""
        return True

    def get_price_data(self, tickers: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from Yahoo Finance."""
        all_data = []
        
        for _, row in tickers.iterrows():
            ticker = row['tic']
            try:
                # Check cache first
                cached_data = self.data_store.get_price_data(ticker, start_date, end_date, source='Yahoo')
                if not cached_data.empty:
                    logger.info(f"Loading cached data for {ticker}")
                    all_data.append(cached_data)
                    continue
                
                # Fetch from Yahoo
                yf_ticker = yf.Ticker(ticker)
                hist_data = yf_ticker.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    # Standardize the data
                    standardized = self._standardize_price_data(hist_data)
                    standardized['tic'] = ticker
                    
                    # Cache the data
                    self.data_store.save_price_data(ticker, start_date, end_date, standardized, source='Yahoo')
                    all_data.append(standardized)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components."""
        # This could be implemented using Yahoo Finance or other sources
        return pd.DataFrame({'tickers': [], 'sectors': [], 'dateFirstAdded': []})

    def get_fundamental_data(self, tickers: List[str], start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data from Yahoo Finance."""
        # Yahoo Finance has limited fundamental data
        return pd.DataFrame()

    def get_news(self, ticker: str, from_date: str, to_date: str, analyze_sentiment: bool = False, sentiment_model: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Get news from Yahoo Finance."""
        # Yahoo Finance news is not easily accessible via API
        return pd.DataFrame()

class CryptoDataFetcher(BaseDataFetcher):
    """Crypto data fetcher combining Binance and Yahoo Finance."""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        
        if not CRYPTO_AVAILABLE:
            raise ImportError("Crypto processors not available. Please install required dependencies.")
        
        self.binance_processor = BinanceProcessor()
        self.yahoo_processor = Yahoofinance()

    def get_crypto_data(self, ticker_list: List[str], start_date: str, end_date: str, 
                       time_interval: str = '5m', technical_indicators: List[str] = None) -> pd.DataFrame:
        """Get cryptocurrency data from Binance."""
        try:
            technical_indicators = technical_indicators or ['macd', 'rsi', 'cci', 'dx']
            
            data = self.binance_processor.run(
                ticker_list=ticker_list,
                start_date=start_date,
                end_date=end_date,
                time_interval=time_interval,
                technical_indicator_list=technical_indicators,
                if_vix=False
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch crypto data: {e}")
            return pd.DataFrame()

    def get_crypto_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get simple price data for cryptocurrencies."""
        all_data = []
        
        for ticker in tickers:
            try:
                # Try Yahoo Finance first
                yf_ticker = yf.Ticker(ticker.replace('USDT', '-USD'))
                hist_data = yf_ticker.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    standardized = self._standardize_price_data(hist_data)
                    standardized['tic'] = ticker
                    all_data.append(standardized)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch crypto data for {ticker} from Yahoo: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

class IntegratedDataFetcher:
    """Main data fetcher that combines stock and crypto data sources."""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.yahoo_fetcher = YahooDataFetcher(cache_dir)
        self.crypto_fetcher = CryptoDataFetcher(cache_dir) if CRYPTO_AVAILABLE else None

    def get_price_data(self, tickers: pd.DataFrame, start_date: str, end_date: str, 
                      asset_type: str = 'stock') -> pd.DataFrame:
        """Get price data for specified asset type."""
        if asset_type == 'crypto' and self.crypto_fetcher:
            crypto_tickers = tickers['tic'].tolist()
            return self.crypto_fetcher.get_crypto_price_data(crypto_tickers, start_date, end_date)
        else:
            return self.yahoo_fetcher.get_price_data(tickers, start_date, end_date)

    def get_crypto_data_with_indicators(self, ticker_list: List[str], start_date: str, end_date: str,
                                       time_interval: str = '5m', technical_indicators: List[str] = None) -> pd.DataFrame:
        """Get cryptocurrency data with technical indicators."""
        if not self.crypto_fetcher:
            raise ValueError("Crypto data fetcher not available")
        
        return self.crypto_fetcher.get_crypto_data(
            ticker_list, start_date, end_date, time_interval, technical_indicators
        )

    def get_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get stock data."""
        tickers_df = pd.DataFrame({'tic': tickers})
        return self.get_price_data(tickers_df, start_date, end_date, 'stock')

# Factory function
def get_data_fetcher(asset_type: str = 'integrated', cache_dir: str = "./data/cache") -> BaseDataFetcher:
    """Factory function to get appropriate data fetcher."""
    if asset_type == 'crypto' and CRYPTO_AVAILABLE:
        return CryptoDataFetcher(cache_dir)
    elif asset_type == 'stock':
        return YahooDataFetcher(cache_dir)
    else:
        return IntegratedDataFetcher(cache_dir)