#!/usr/bin/env python3
"""
FinRL-X 合约交易训练系统 v3 — Gate.io · 做多/做空 · 资金费率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys, json, time, logging, numpy as np
from datetime import datetime

sys.path.insert(0, '/home/administrator/FinRL-Integrated')
sys.path.insert(0, '/home/administrator/FinRL_Crypto')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = '/home/administrator/FinRL-Integrated'
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_v3')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── 1. 数据层 — Gate.io ──────────────────────────────────

class DataFetcher:
    BASE = "https://api.gateio.ws/api/v4"

    def __init__(self):
        import requests
        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json"})

    def fetch_klines(self, contract="ETH_USDT", interval="1h", limit=1000):
        """永续合约K线"""
        import pandas as pd
        resp = self.http.get(f"{self.BASE}/futures/usdt/candlesticks",
            params={"contract": contract, "interval": interval, "limit": limit}, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        df = pd.DataFrame(raw)
        df['open'] = df['o'].astype(float)
        df['high'] = df['h'].astype(float)
        df['low'] = df['l'].astype(float)
        df['close'] = df['c'].astype(float)
        df['volume'] = df['v'].astype(float)           # 合约张数
        df['timestamp'] = pd.to_datetime(df['t'].astype(int), unit='s')
        df.set_index('timestamp', inplace=True)
        df = df.iloc[::-1]
        logger.info(f"✅ 获取 {len(df)} 条 {contract} 合约K线")
        return df[['open','high','low','close','volume']]

    def fetch_funding_rates(self, contract="ETH_USDT", limit=300):
        import pandas as pd
        resp = self.http.get(f"{self.BASE}/futures/usdt/funding_rate",
            params={"contract": contract, "limit": limit}, timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df['t'] = pd.to_datetime(df['t'].astype(int), unit='s')
        df.set_index('t', inplace=True)
        df = df.iloc[::-1]
        return df.rename(columns={'r': 'funding_rate'})

    def fetch_contract_info(self, contract="ETH_USDT"):
        resp = self.http.get(f"{self.BASE}/futures/usdt/contracts/{contract}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def fetch_all(self, kline_limit=1000):
        import pandas as pd
        df = self.fetch_klines(limit=kline_limit)

        # 资金费率历史 → 重采样到小时
        fr = self.fetch_funding_rates(limit=300)
        fr_h = fr['funding_rate'].astype(float).resample('1h').ffill()
        df['funding_rate'] = fr_h.reindex(df.index, method='ffill').fillna(0)

        # 合约完整快照（最新值，整列填充）
        ci = self.fetch_contract_info()
        df['open_interest'] = float(ci.get('position_size', 0)) / 1e6
        df['long_users'] = float(ci.get('long_users', 1))
        df['short_users'] = float(ci.get('short_users', 1))
        df['mark_price'] = float(ci.get('mark_price', 0))
        df['index_price'] = float(ci.get('index_price', 0))
        df['funding_rate_indicative'] = float(ci.get('funding_rate_indicative', 0))
        df['funding_next_apply'] = float(ci.get('funding_next_apply', 0))
        df['maintenance_rate'] = float(ci.get('maintenance_rate', 0))
        df['leverage_max'] = float(ci.get('leverage_max', 200))
        df['trade_size_cum'] = float(ci.get('trade_size', 0)) / 1e6

        # OI变化率
        df['oi_change_24h'] = df['open_interest'].pct_change(24).fillna(0)

        # ═══ 额外API调用 ═══

        # 现货行情 (24h统计数据)
        try:
            st = self.http.get(f"{self.BASE}/spot/tickers?currency_pair=ETH_USDT", timeout=10).json()[0]
            df['change_pct_24h'] = float(st.get('change_percentage', 0))
            df['high_24h'] = float(st.get('high_24h', 0))
            df['low_24h'] = float(st.get('low_24h', 0))
            df['quote_volume_24h'] = float(st.get('quote_volume', 0))
        except: pass

        # 订单簿价差
        try:
            ob = self.http.get(f"{self.BASE}/spot/order_book?currency_pair=ETH_USDT&limit=1", timeout=10).json()
            if 'asks' in ob and 'bids' in ob:
                ask = float(ob['asks'][0][0]) if ob['asks'] else 0
                bid = float(ob['bids'][0][0]) if ob['bids'] else 0
                df['spread'] = (ask - bid) / ((ask + bid) / 2) * 100 if (ask + bid) > 0 else 0
                df['bid_size'] = float(ob['bids'][0][1]) if ob['bids'] else 0
                df['ask_size'] = float(ob['asks'][0][1]) if ob['asks'] else 0
        except: pass

        logger.info(f"📊 数据: {df.shape[0]}行 × {df.shape[1]}列")
        return df


# ─── 2. 特征工程 ──────────────────────────────────────────

class FeatureEngine:
    @staticmethod
    def compute(df):
        import pandas as pd
        # ── 基础 ──
        df['returns'] = df['close'].pct_change()
        # ── 移动均线 ──
        df['ma7'] = df['close'].rolling(7).mean()
        df['ma25'] = df['close'].rolling(25).mean()
        df['ma99'] = df['close'].rolling(99).mean()
        df['price_to_ma7'] = df['close'] / df['ma7'] - 1
        df['price_to_ma25'] = df['close'] / df['ma25'] - 1
        # ── MACD ──
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        # ── RSI ──
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        # ── 布林带 ──
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_width'] = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / bb_mid
        # ── ATR ──
        df['atr'] = pd.concat([
            (df['high']-df['low']),
            (df['high']-df['close'].shift(1)).abs(),
            (df['low']-df['close'].shift(1)).abs()
        ], axis=1).max(axis=1).rolling(14).mean()
        # ── 成交量 ──
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(7).mean()
        df['volatility'] = df['returns'].rolling(20).std() * 100
        # ── 价格位置 ──
        df['price_position'] = ((df['close'] - df['low'].rolling(20).min())
                                / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-8))

        # ═══ 新增: Gate.io 合约衍生特征 ═══

        # 基差 = 标记价 - 指数价 (contango/backwardation)
        df['basis'] = df['mark_price'] - df['index_price']
        df['basis_pct'] = df['basis'] / df['index_price'] * 100

        # 资金费率差值 = 预测费率 - 当前费率 (趋势方向)
        df['funding_rate_diff'] = df['funding_rate_indicative'] - df['funding_rate']
        # 费率变化率
        df['funding_rate_change'] = df['funding_rate'].diff().fillna(0)

        # 多空人数比例派生
        df['long_short_ratio'] = df['long_users'] / df['short_users'].replace(0, 1)
        df['total_users'] = df['long_users'] + df['short_users']
        df['long_pct'] = df['long_users'] / df['total_users'].replace(0, 1) * 100

        # 交易量变化
        df['trade_size_change'] = df['trade_size_cum'].diff().fillna(0)

        # 24h涨跌幅
        if 'change_pct_24h' in df.columns:
            df['change_pct_24h_smooth'] = df['change_pct_24h'].rolling(3).mean()
        if 'quote_volume_24h' in df.columns:
            df['volume_24h_change'] = df['quote_volume_24h'].pct_change().fillna(0)

        # 价差变化率
        if 'spread' in df.columns:
            df['spread_change'] = df['spread'].diff().fillna(0)

        return df.bfill().ffill().fillna(0)

    FEATURE_COLS = [
        # ── 价格核心 ──
        'close','volume','returns',
        # ── 趋势 ──
        'ma7','ma25','ma99','price_to_ma7','price_to_ma25',
        'macd','macd_signal','macd_hist',
        # ── 震荡 ──
        'rsi','bb_width','atr',
        # ── 量价 ──
        'volume_ratio','volatility','price_position',
        # ── 合约核心 ──
        'funding_rate','open_interest',
        # ── 基差 ──
        'basis','basis_pct',
        # ── 资金费率深度 ──
        'funding_rate_indicative','funding_rate_diff','funding_rate_change',
        # ── 多空 ──
        'long_short_ratio','long_pct','total_users',
        # ── OI ──
        'oi_change_24h','trade_size_change',
        # ── 市场宽度 ──
        'spread','spread_change',
        # ── 24h统计 ──
        'change_pct_24h_smooth','volume_24h_change',
    ]

    @classmethod
    def build_state(cls, df, lookback=50):
        arr = df[cls.FEATURE_COLS].values.astype(np.float32)
        m, s = arr.mean(0), arr.std(0)
        s[s == 0] = 1
        norm = (arr - m) / s
        states = np.array([norm[i-lookback:i].flatten() for i in range(lookback, len(norm))], dtype=np.float32)
        return states


# ─── 3. 合约交易环境 ──────────────────────────────────────

class FuturesEnv:
    """
    永续合约模拟环境（正确资金核算）
    - 做多/做空/平仓
    - 保证金模式：margin = capital × leverage
    - 资金费率每8小时结算
    - 爆仓检测
    """
    FLAT = 0; LONG = 1; SHORT = -1

    def __init__(self, prices, funding_rates, feature_states,
                 initial_capital=1000, leverage=10, fee=0.0005):
        self.prices = prices
        self.funding_rates = funding_rates
        self.feature_states = feature_states
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee = fee
        self.n = len(feature_states)
        self.reset()

    def reset(self):
        self.idx = 0
        self.equity = self.initial_capital     # 总权益
        self.margin = 0.0                      # 占用保证金
        self.position_value = 0.0              # 仓位名义价值
        self.direction = self.FLAT
        self.entry_price = 0.0
        self.last_funding_idx = 0
        return self.feature_states[self.idx]

    def _get_state(self):
        return self.feature_states[min(self.idx, self.n-1)]

    def step(self, action):
        """
        action: 0=做空, 1=平仓, 2=做多
        """
        price = self.prices[self.idx]
        prev_equity = self.equity

        # ── 浮盈结算 —— 更新 equity ──
        if self.direction != self.FLAT and self.position_value > 0:
            price_change = (price - self.entry_price) / self.entry_price
            unrealized = self.position_value * price_change * self.direction
            self.equity = max(self.margin + self.equity - self.margin + unrealized, 0)
            # 爆仓
            if self.equity <= self.margin * 0.3:  # 维持保证金率 30%
                self.equity = self.margin * 0.3
                self.margin = 0
                self.position_value = 0
                self.direction = self.FLAT

        # ── 资金费率结算 ──
        if self.direction != self.FLAT and self.position_value > 0:
            if self.idx - self.last_funding_idx >= 8:
                fr = self.funding_rates[self.idx] if self.idx < len(self.funding_rates) else 0
                funding_cost = self.position_value * fr * self.direction
                self.equity -= funding_cost
                self.last_funding_idx = self.idx

        # ── 动作执行 ──
        if action == 0 and self.direction != self.SHORT:   # 开空
            self._close_position(price)
            self.direction = self.SHORT
            self.margin = self.equity * 0.95
            self.position_value = self.margin * self.leverage
            self.entry_price = price
            self.equity -= self.position_value * self.fee

        elif action == 2 and self.direction != self.LONG:  # 开多
            self._close_position(price)
            self.direction = self.LONG
            self.margin = self.equity * 0.95
            self.position_value = self.margin * self.leverage
            self.entry_price = price
            self.equity -= self.position_value * self.fee

        elif action == 1:  # 平仓
            self._close_position(price)

        # ── 进入下一时间步 ──
        self.idx += 1
        done = self.idx >= self.n - 1

        # ── 奖励 ──
        if self.equity <= 0:
            reward = -100
            self.equity = 0
        elif not done:
            reward = (self.equity - prev_equity) / prev_equity * 100
        else:
            reward = (self.equity - self.initial_capital) / self.initial_capital * 100

        return self._get_state(), reward, done, {
            'equity': round(self.equity, 2),
            'direction': self.direction,
            'position_value': round(self.position_value, 2),
        }

    def _close_position(self, price):
        if self.direction != self.FLAT and self.position_value > 0:
            price_change = (price - self.entry_price) / self.entry_price
            pnl = self.position_value * price_change * self.direction
            self.equity += pnl - abs(self.position_value) * self.fee
            self.margin = 0
            self.position_value = 0
            self.direction = self.FLAT
            self.entry_price = 0


# ─── 4. PPO 训练 ──────────────────────────────────────────

def train_ppo(env, agent, n_episodes=100, steps_per_ep=128):
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
            logger.info(f"  Episode {ep:3d}/{n_episodes} | C={c_loss:.4f} | A={a_loss:.4f} | "
                        f"Equity=${info['equity']:.0f} | Dir={info['direction']:+d}")

    return log


# ─── 5. 主程序 ─────────────────────────────────────────────

def main():
    import torch
    from drl_agents.agents.AgentPPO import AgentPPO

    logger.info("=" * 60)
    logger.info("🚀 FinRL-X v3 — ETH 永续合约 PPO 训练")
    logger.info("=" * 60)

    # ── 数据 ──
    fetcher = DataFetcher()
    df = fetcher.fetch_all(kline_limit=1000)
    logger.info(f"📈 ETH: ${df['close'].iloc[-1]:.2f} | "
                f"资金费率: {df['funding_rate'].iloc[-1]:+.6f} | "
                f"多头: {df['long_users'].iloc[-1]:.0f} | 空头: {df['short_users'].iloc[-1]:.0f}")

    # ── 特征 ──
    df_feat = FeatureEngine.compute(df.copy())
    states = FeatureEngine.build_state(df_feat)
    state_dim = states.shape[1]
    logger.info(f"🧮 特征: {len(FeatureEngine.FEATURE_COLS)}个 | 状态维度: {state_dim}")

    # ── PPO ──
    agent = AgentPPO(net_dim=256, state_dim=state_dim, action_dim=3, gpu_id=0)

    # ── 合约环境 ──
    env = FuturesEnv(
        prices=df['close'].values,
        funding_rates=df['funding_rate'].values,
        feature_states=states,
        initial_capital=1000,
        leverage=10,
        fee=0.0005,
    )
    logger.info(f"📊 合约环境: $1000, {10}x杠杆, 手续费0.05%")

    # ── 训练 ──
    n_episodes = 300
    t0 = time.time()
    log = train_ppo(env, agent, n_episodes=n_episodes, steps_per_ep=128)
    elapsed = time.time() - t0
    logger.info(f"✅ 训练完成! {elapsed:.1f}s ({elapsed/n_episodes*1000:.0f}ms/ep)")

    # ── 回测 ──
    logger.info("📈 运行回测...")
    env.reset()
    equities = []
    actions = []
    for i in range(len(states)):
        s_t = torch.tensor(states[i], dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            a_t, _ = agent.act.get_action(s_t)
        action = a_t.cpu().numpy()[0, 0].round().clip(0, 2).astype(int)
        _, _, done, info = env.step(action)
        equities.append(info['equity'])
        actions.append(action)
        if done:
            break

    final = equities[-1] if equities else 1000
    ret = (final - 1000) / 1000 * 100
    bh_ret = (df['close'].iloc[-1] / df['close'].iloc[50] - 1) * 100

    results = {
        'version': 'v3-futures',
        'algorithm': 'PPO',
        'exchange': 'Gate.io',
        'symbol': 'ETH/USDT永续合约',
        'data_points': len(df),
        'features': FeatureEngine.FEATURE_COLS,
        'state_dim': state_dim,
        'initial_capital': 1000,
        'leverage': 10,
        'n_episodes': n_episodes,
        'training_time_s': round(elapsed, 1),
        'final_equity': round(final, 2),
        'strategy_return_pct': round(ret, 2),
        'buy_hold_return_pct': round(bh_ret, 2),
        'action_distribution': {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)},
        'price_range': f"${df['low'].min():.2f} ~ ${df['high'].max():.2f}",
        'current_price': float(df['close'].iloc[-1]),
    }

    path = os.path.join(RESULTS_DIR, 'futures_training_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(agent.act.state_dict(), os.path.join(RESULTS_DIR, 'ppo_futures_actor.pth'))

    print()
    print("=" * 60)
    print("📊 合约训练结果报告")
    print("=" * 60)
    print(f"  数据:    ETH/USDT 永续合约 | Gate.io")
    print(f"  特征:    {len(FeatureEngine.FEATURE_COLS)}个 (含资金费率/OI/多空比)")
    print(f"  杠杆:    {10}x | 初始资金 $1,000")
    print(f"  训练:    {n_episodes}轮 | {elapsed:.1f}s")
    print()
    print(f"  ┌──────────────────┬──────────────┬──────────────┐")
    print(f"  │                   │   PPO合约策略  │  Buy & Hold │")
    print(f"  ├──────────────────┼──────────────┼──────────────┤")
    print(f"  │   最终权益        │  ${final:>8.2f}   │  ${1000*(1+bh_ret/100):>8.2f}   │")
    print(f"  │   收益率          │  {ret:>+8.2f}%   │  {bh_ret:>+8.2f}%   │")
    print(f"  └──────────────────┴──────────────┴──────────────┘")
    print()
    print(f"  动作分布: 做空={results['action_distribution'][0]} | "
          f"平仓={results['action_distribution'][1]} | 做多={results['action_distribution'][2]}")
    print(f"  特征: {', '.join(FeatureEngine.FEATURE_COLS)}")
    print(f"  结果: {path}")
    print()

    return results


if __name__ == '__main__':
    main()
