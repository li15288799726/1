#!/usr/bin/env python3
"""
FinRL-X 真正训练系统 v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
从 Binance 拉真实 ETH/USDT 数据 → 交易环境 → PPO 训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys, json, time, logging, numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/home/administrator/FinRL-Integrated')
sys.path.insert(0, '/home/administrator/FinRL_Crypto')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── 0. 环境配置 ──────────────────────────────────────────
PROJECT_ROOT = '/home/administrator/FinRL-Integrated'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_v2')
os.makedirs(RESULTS_DIR, exist_ok=True)
BINANCE_API_KEY = ''
BINANCE_SECRET = ''


# ─── 1. 数据层 ────────────────────────────────────────────

class DataFetcher:
    """从 Gate.io API 获取多维度加密数据"""

    BASE = "https://api.gateio.ws/api/v4"

    def __init__(self):
        import requests
        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json"})

    def fetch_klines(self, symbol="ETH_USDT", interval="1h", limit=1000):
        """OHLCV K线"""
        resp = self.http.get(
            f"{self.BASE}/spot/candlesticks",
            params={"currency_pair": symbol, "interval": interval, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        raw = resp.json()
        import pandas as pd
        df = pd.DataFrame(raw, columns=['timestamp', 'volume', 'close', 'high', 'low', 'open', 'amount', 'dummy'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        df.set_index('timestamp', inplace=True)
        df = df.iloc[::-1]  # Gate返回逆序
        logger.info(f"✅ 获取 {len(df)} 条 {symbol} 现货K线")
        return df[['open','high','low','close','volume']]

    def fetch_funding_rates(self, contract="ETH_USDT", limit=100):
        """资金费率历史"""
        resp = self.http.get(
            f"{self.BASE}/futures/usdt/funding_rate",
            params={"contract": contract, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        import pandas as pd
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'].astype(int), unit='s')
        df.set_index('t', inplace=True)
        df = df.iloc[::-1]
        logger.info(f"✅ 获取 {len(df)} 条资金费率记录")
        return df.rename(columns={'r': 'funding_rate'})

    def fetch_contract_info(self, contract="ETH_USDT"):
        """合约完整信息（含OI、多空比等）"""
        resp = self.http.get(
            f"{self.BASE}/futures/usdt/contracts/{contract}",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_all_features(self, kline_limit=1000, funding_limit=300):
        """一次性拉取所有数据并合并为特征矩阵"""
        import pandas as pd

        # 1. 现货K线
        df = self.fetch_klines(limit=kline_limit)

        # 2. 资金费率历史
        fr = self.fetch_funding_rates(limit=funding_limit)
        # 重采样到小时，填充间隙
        fr_hourly = fr['funding_rate'].astype(float).resample('1h').ffill()
        df['funding_rate'] = fr_hourly.reindex(df.index, method='ffill').fillna(0)

        # 3. 合约信息（当前快照）
        ci = self.fetch_contract_info()
        df['open_interest'] = float(ci.get('position_size', 0)) / 1e6  # 百万单位
        df['trade_size'] = float(ci.get('trade_size', 0)) / 1e6
        df['mark_price'] = float(ci.get('mark_price', 0))
        df['funding_rate_current'] = float(ci.get('funding_rate', 0))
        df['long_short_ratio'] = ci.get('long_users', 1) / max(ci.get('short_users', 1), 1)
        df['index_price'] = float(ci.get('index_price', 0))

        # 计算OI变化率
        df['oi_change_24h'] = df['open_interest'].pct_change(24).fillna(0)

        logger.info(f"📊 特征矩阵: {df.shape[0]}行 × {df.shape[1]}列")
        return df


class FeatureEngine:
    """从价格数据计算技术指标"""

    @staticmethod
    def compute(df):
        import pandas as pd
        # 收益率
        df['returns'] = df['close'].pct_change()

        # 移动均线
        df['ma7']   = df['close'].rolling(7).mean()
        df['ma25']  = df['close'].rolling(25).mean()
        df['ma99']  = df['close'].rolling(99).mean()

        # 价格相对于均线的位置
        df['price_to_ma7']  = df['close'] / df['ma7'] - 1
        df['price_to_ma25'] = df['close'] / df['ma25'] - 1

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid

        # ATR
        df['atr'] = pd.concat([
            (df['high'] - df['low']),
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1).rolling(14).mean()

        # 成交量指标
        df['volume_ma7'] = df['volume'].rolling(7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma7']
        df['volume_change'] = df['volume'].pct_change(5)

        # 波动率
        df['volatility'] = df['returns'].rolling(20).std() * 100

        # 价格模式
        df['highest_20'] = df['high'].rolling(20).max()
        df['lowest_20']  = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['lowest_20']) / (df['highest_20'] - df['lowest_20'] + 1e-8)

        # 填充
        return df.bfill().ffill().fillna(0)


    FEATURE_COLS = [
        'open','high','low','close','volume',
        'returns','ma7','ma25','ma99',
        'price_to_ma7','price_to_ma25',
        'macd','macd_signal','macd_hist',
        'rsi',
        'bb_upper','bb_lower','bb_width',
        'atr',
        'volume_ma7','volume_ratio','volume_change',
        'volatility',
        'price_position',
        # 新增: 资金费率 + 未平仓量 + 多空
        'funding_rate','open_interest','trade_size',
        'long_short_ratio','oi_change_24h',
    ]

    @classmethod
    def build_state(cls, df, lookback=50):
        """构建归一化的滑动窗口状态"""
        arr = df[cls.FEATURE_COLS].values.astype(np.float32)
        m, s = arr.mean(0), arr.std(0)
        s[s == 0] = 1
        norm = (arr - m) / s
        states = np.array([
            norm[i-lookback:i].flatten()
            for i in range(lookback, len(norm))
        ], dtype=np.float32)
        logger.info(f"状态维度: {states.shape[1]} ({len(cls.FEATURE_COLS)}特征 × {lookback}窗口)")
        return states


# ─── 2. 交易环境 ──────────────────────────────────────────

class TradingEnv:
    """基于真实价格的模拟交易环境 — 使用完整特征状态"""

    def __init__(self, prices, feature_states, initial_capital=10000, fee=0.001):
        self.prices = prices
        self.feature_states = feature_states  # 预计算的特征状态
        self.initial_capital = initial_capital
        self.fee = fee
        self.n = len(feature_states)
        self.reset()

    def reset(self):
        self.idx = 0
        self.cash = self.initial_capital
        self.position = 0.0
        return self.feature_states[self.idx]

    def step(self, action):
        """
        action: 0=卖出, 1=持有, 2=买入
        返回: next_state, reward, done, info
        """
        price = self.prices[self.idx]
        prev_value = self.cash + self.position * price

        # 执行动作
        if action == 0 and self.position > 0:  # 卖出
            self.cash += self.position * price * (1 - self.fee)
            self.position = 0
        elif action == 2 and self.cash > 10:    # 买入（95%资金）
            buy_amount = self.cash * 0.95 / price
            self.position += buy_amount * (1 - self.fee)
            self.cash *= 0.05

        # 进入下一时间步
        self.idx += 1
        done = self.idx >= self.n - 1

        if not done:
            new_price = self.prices[self.idx]
            new_value = self.cash + self.position * new_price
            reward = (new_value - prev_value) / prev_value * 100  # 收益百分比
        else:
            final_value = self.cash + self.position * price
            reward = (final_value - self.initial_capital) / self.initial_capital * 100
            # 训练结束给一个大奖励/惩罚
            reward += 10 if reward > 0 else -10

        return self.feature_states[min(self.idx, self.n-1)], reward, done, {
            'value': prev_value if done else self.cash + self.position * (self.prices[self.idx] if not done else price),
            'position': self.position,
            'price': price
        }


# ─── 3. PPO 训练 ──────────────────────────────────────────

def train_ppo_with_env(env, agent, n_episodes=50, steps_per_ep=128):
    """在真实交易环境中训练 PPO"""
    import torch

    device = agent.device
    log = []

    for ep in range(n_episodes):
        state = env.reset()
        traj = []

        for _ in range(steps_per_ep):
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_t, n_t = agent.act.get_action(s_t)

            action = a_t.cpu().numpy()[0, 0].round().clip(0, 2).astype(int)
            next_s, reward, done, info = env.step(action)

            traj.append((s_t.cpu(), reward, done, a_t.cpu(), n_t.cpu()))
            state = next_s
            if done:
                break

        if not traj:
            continue

        # 构建 buffer
        buf = (
            torch.cat([t[0] for t in traj]),
            torch.tensor([t[1] for t in traj], dtype=torch.float32),
            torch.tensor([1 - t[2] for t in traj], dtype=torch.float32),
            torch.cat([t[3] for t in traj]),
            torch.cat([t[4] for t in traj]),
        )

        c_loss, a_loss, _ = agent.update_net(buf)
        log.append({'episode': ep, 'critic_loss': c_loss, 'actor_loss': a_loss})

        if ep % 10 == 0:
            logger.info(f"  Episode {ep:2d}/{n_episodes} | Critic={c_loss:.4f} | Actor={a_loss:.4f}")

    return log


# ─── 4. 主程序 ─────────────────────────────────────────────

def main():
    import torch
    from drl_agents.agents.AgentPPO import AgentPPO

    logger.info("=" * 60)
    logger.info("🚀 FinRL-X v2 — 真实 ETH 数据 PPO 训练")
    logger.info("=" * 60)

    # ── 4a. 数据（多维度：现货K线 + 资金费率 + 未平仓量 + 多空比）──
    fetcher = DataFetcher()
    df = fetcher.fetch_all_features(kline_limit=1000, funding_limit=300)

    logger.info(f"价格范围: ${df['low'].min():.2f} ~ ${df['high'].max():.2f}")
    logger.info(f"当前ETH:  ${df['close'].iloc[-1]:.2f}")

    # ── 4b. 特征工程 ──
    df_feat = FeatureEngine.compute(df.copy())
    states = FeatureEngine.build_state(df_feat)
    logger.info(f"可用训练样本: {len(states)}")

    # ── 4c. PPO 智能体 ──
    state_dim = states.shape[1]
    action_dim = 3

    agent = AgentPPO(
        net_dim=256,
        state_dim=state_dim,
        action_dim=action_dim,
        gpu_id=0,
    )
    logger.info(f"🧠 PPO 初始化完成 | 输入={state_dim} → 策略网络 → 输出={action_dim}")

    # ── 4e. 创建交易环境 ──
    prices = df['close'].values
    env = TradingEnv(prices, states, initial_capital=10000)
    logger.info(f"📊 交易环境: 初始资金 $10,000 | 手续费 0.1%")

    # ── 4e. 训练 ──
    n_episodes = 200
    logger.info(f"🎯 开始训练 {n_episodes} 轮...")
    t0 = time.time()

    log = train_ppo_with_env(env, agent, n_episodes=n_episodes, steps_per_ep=128)

    elapsed = time.time() - t0
    logger.info(f"✅ 训练完成! 耗时 {elapsed:.1f}s ({elapsed/n_episodes*1000:.0f}ms/ep)")

    # ── 4f. 回测 ──
    logger.info("📈 运行回测...")
    env.reset()
    portfolio_values = []
    actions_taken = []
    state_idx = 0

    for i in range(len(states)):
        state = states[i]
        s_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            a_t, _ = agent.act.get_action(s_t)
        action = a_t.cpu().numpy()[0, 0].round().clip(0, 2).astype(int)

        _, _, done, info = env.step(action)
        portfolio_values.append(info['value'])
        actions_taken.append(action)

        if done:
            break

    # ── 4g. 结果 ──
    final_value = portfolio_values[-1] if portfolio_values else 10000
    total_return = (final_value - 10000) / 10000 * 100
    buy_hold_return = (prices[-1] / prices[0] - 1) * 100

    results = {
        'algorithm': 'PPO',
        'data_source': 'binance(data-api.binance.vision)',
        'symbol': 'ETH/USDT',
        'data_points': len(df),
        'price_range': f"${df['low'].min():.2f} ~ ${df['high'].max():.2f}",
        'current_price': float(df['close'].iloc[-1]),
        'state_dim': state_dim,
        'n_features': len(FeatureEngine.FEATURE_COLS),
        'n_episodes': n_episodes,
        'training_time_s': round(elapsed, 1),
        'initial_capital': 10000,
        'final_portfolio_value': round(final_value, 2),
        'strategy_return_pct': round(total_return, 2),
        'buy_hold_return_pct': round(buy_hold_return, 2),
        'action_counts': {0: actions_taken.count(0), 1: actions_taken.count(1), 2: actions_taken.count(2)},
        'features': FeatureEngine.FEATURE_COLS,
        'training_log': log[-10:],
    }

    path = os.path.join(RESULTS_DIR, 'training_v2_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    # 保存策略权重
    torch.save(agent.act.state_dict(), os.path.join(RESULTS_DIR, 'ppo_eth_actor.pth'))

    print()
    print("=" * 60)
    print("📊 训练结果报告")
    print("=" * 60)
    print(f"  数据:           ETH/USDT, {len(df)} 条 1h K线")
    print(f"  价格区间:       ${df['low'].min():.2f} ~ ${df['high'].max():.2f}")
    print(f"  特征维度:       {state_dim} ({len(FeatureEngine.FEATURE_COLS)} 个指标)")
    print(f"  训练轮数:       {n_episodes} 轮")
    print(f"  训练耗时:       {elapsed:.1f}s")
    print()
    print(f"  ┌──────────────────┬──────────────┬──────────────┐")
    print(f"  │                   │   PPO策略    │   Buy & Hold │")
    print(f"  ├──────────────────┼──────────────┼──────────────┤")
    print(f"  │   最终资产        │  ${final_value:>8.2f}   │  ${10000*(1+buy_hold_return/100):>8.2f}   │")
    print(f"  │   收益率          │  {total_return:>+8.2f}%   │  {buy_hold_return:>+8.2f}%   │")
    print(f"  └──────────────────┴──────────────┴──────────────┘")
    print()
    print(f"  动作分布: 卖出={results['action_counts'][0]} | "
          f"持有={results['action_counts'][1]} | 买入={results['action_counts'][2]}")
    print(f"  特征列表: {', '.join(FeatureEngine.FEATURE_COLS)}")
    print(f"  结果已保存: {path}")
    print()

    return results


if __name__ == '__main__':
    main()
