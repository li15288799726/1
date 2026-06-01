#!/usr/bin/env python3
"""
FinRL-X 多时间框架持续训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━
5m / 15m / 1h 分别训练 → 集成决策
每5分钟迭代一次，各自在对应K线闭合时更新
"""
import os, sys, json, time, logging, numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, '/home/administrator/FinRL-Integrated')
sys.path.insert(0, '/home/administrator/FinRL_Crypto')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/home/administrator/FinRL-Integrated')
DATA_DIR = ROOT/'live_data'; MODEL_DIR = ROOT/'live_model'
DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)

import requests, pandas as pd
import torch
from drl_agents.agents.AgentPPO import AgentPPO

BASE = "https://api.gateio.ws/api/v4"
http = requests.Session(); http.headers.update({"Accept":"application/json"})

# ─── 配置 ────────────────────────────────────────────────
TIMEFRAMES = {
    '5m':  {'interval':'5m',  'train_every':1,   'n_steps':32,  'lookback':50},
    '15m': {'interval':'15m', 'train_every':3,   'n_steps':48,  'lookback':50},
    '1h':  {'interval':'1h',  'train_every':12,  'n_steps':64,  'lookback':50},
}
FEATURE_COLS = ['close','volume','returns','ma7','ma25','ma99','price_to_ma7','price_to_ma25',
    'macd','macd_signal','macd_hist','rsi','bb_width','atr','volume_ratio','volatility',
    'price_position','funding_rate','open_interest','basis','basis_pct','funding_rate_indicative',
    'funding_rate_diff','funding_rate_change','long_short_ratio','long_pct','oi_change_24h',
    'trade_size_change','spread','spread_change','change_pct_24h_smooth']
LEVERAGE = 10; CAPITAL = 1000; FEE = 0.0005

# ─── 数据 ────────────────────────────────────────────────
def fetch_klines(interval, limit=500):
    r = http.get(f"{BASE}/futures/usdt/candlesticks",
        params={"contract":"ETH_USDT","interval":interval,"limit":limit}, timeout=10)
    raw = r.json()
    df = pd.DataFrame(raw)
    df['open']=df['o'].astype(float); df['high']=df['h'].astype(float)
    df['low']=df['l'].astype(float); df['close']=df['c'].astype(float)
    df['volume']=df['v'].astype(float)
    df['timestamp']=pd.to_datetime(df['t'].astype(int),unit='s')
    df.set_index('timestamp',inplace=True)
    return df.iloc[::-1]

def fetch_shared():
    """资金费率 + 合约快照 + 订单簿（所有TF共享）"""
    fr = http.get(f"{BASE}/futures/usdt/funding_rate",
        params={"contract":"ETH_USDT","limit":300}, timeout=10).json()
    fr_df = pd.DataFrame(fr)
    fr_df['t']=pd.to_datetime(fr_df['t'].astype(int),unit='s')
    fr_df.set_index('t',inplace=True)
    fr_df=fr_df.iloc[::-1]
    fr_h = fr_df['r'].astype(float).resample('1h').ffill()

    ci = http.get(f"{BASE}/futures/usdt/contracts/ETH_USDT", timeout=10).json()
    data = {
        'funding_rate_series': fr_h,
        'open_interest': float(ci.get('position_size',0))/1e6,
        'long_users': float(ci.get('long_users',1)),
        'short_users': float(ci.get('short_users',1)),
        'mark_price': float(ci.get('mark_price',0)),
        'index_price': float(ci.get('index_price',0)),
        'funding_rate_indicative': float(ci.get('funding_rate_indicative',0)),
    }
    try:
        st = http.get(f"{BASE}/spot/tickers?currency_pair=ETH_USDT", timeout=10).json()[0]
        data['change_pct_24h'] = float(st.get('change_percentage',0))
    except: data['change_pct_24h'] = 0
    try:
        ob = http.get(f"{BASE}/spot/order_book?currency_pair=ETH_USDT&limit=1", timeout=10).json()
        if 'asks' in ob and 'bids' in ob:
            a=float(ob['asks'][0][0]) if ob['asks'] else 0
            b=float(ob['bids'][0][0]) if ob['bids'] else 0
            data['spread'] = (a-b)/((a+b)/2)*100 if (a+b)>0 else 0
        else: data['spread'] = 0
    except: data['spread'] = 0
    return data

def build_features(df, shared):
    """33特征计算"""
    df['returns'] = df['close'].pct_change()
    df['ma7']=df['close'].rolling(7).mean(); df['ma25']=df['close'].rolling(25).mean(); df['ma99']=df['close'].rolling(99).mean()
    df['price_to_ma7']=df['close']/df['ma7']-1; df['price_to_ma25']=df['close']/df['ma25']-1
    e12=df['close'].ewm(span=12).mean(); e26=df['close'].ewm(span=26).mean()
    df['macd']=e12-e26; df['macd_signal']=df['macd'].ewm(span=9).mean(); df['macd_hist']=df['macd']-df['macd_signal']
    d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l_=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/l_.replace(0,np.nan)))
    bm=df['close'].rolling(20).mean(); bs=df['close'].rolling(20).std()
    df['bb_width']=((bm+2*bs)-(bm-2*bs))/bm
    df['atr']=pd.concat([(df['high']-df['low']),(df['high']-df['close'].shift(1)).abs(),(df['low']-df['close'].shift(1)).abs()],axis=1).max(axis=1).rolling(14).mean()
    df['volume_ratio']=df['volume']/df['volume'].rolling(7).mean()
    df['volatility']=df['returns'].rolling(20).std()*100
    df['price_position']=((df['close']-df['low'].rolling(20).min())/(df['high'].rolling(20).max()-df['low'].rolling(20).min()+1e-8))

    # 共享数据对齐
    df['funding_rate'] = shared['funding_rate_series'].reindex(df.index, method='ffill').fillna(0)
    df['open_interest'] = shared['open_interest']
    df['long_users'] = shared['long_users']; df['short_users'] = shared['short_users']
    df['mark_price'] = shared['mark_price']; df['index_price'] = shared['index_price']
    df['funding_rate_indicative'] = shared['funding_rate_indicative']
    df['oi_change_24h'] = df['open_interest'].pct_change(min(24, len(df)-1)).fillna(0)

    df['basis']=df['mark_price']-df['index_price']
    df['basis_pct']=df['basis']/df['index_price']*100
    df['funding_rate_diff']=df['funding_rate_indicative']-df['funding_rate']
    df['funding_rate_change']=df['funding_rate'].diff().fillna(0)
    df['long_short_ratio']=df['long_users']/df['short_users'].replace(0,1)
    df['long_pct']=df['long_users']/(df['long_users']+df['short_users']+1)*100
    df['trade_size_change']=df['open_interest'].diff().fillna(0)
    df['spread']=shared['spread']
    df['spread_change']=df['spread'].diff().fillna(0)
    df['change_pct_24h']=shared['change_pct_24h']
    df['change_pct_24h_smooth']=df['change_pct_24h'].rolling(3).mean()
    return df.bfill().ffill().fillna(0)

# ─── 合约环境 ────────────────────────────────────────────
class FuturesEnv:
    FLAT=0; LONG=1; SHORT=-1
    def __init__(self, prices, funding_rates, states, cap=CAPITAL, lev=LEVERAGE):
        self.p=prices; self.fr=funding_rates; self.s=states
        self.c0=cap; self.lev=lev; self.n=len(states); self.reset()
    def reset(self):
        self.i=0; self.eq=self.c0; self.mg=0; self.pv=0; self.d=self.FLAT; self.en=0; self.lf=0
        return self.s[0]
    def step(self, a):
        p=self.p[self.i]; pe=self.eq
        if self.d!=self.FLAT and self.pv>0:
            ch=(p-self.en)/self.en
            self.eq=max(self.mg+self.eq-self.mg+self.pv*ch*self.d,0)
            if self.eq<=self.mg*0.3: self.eq=self.mg*0.3; self.mg=0; self.pv=0; self.d=self.FLAT
        if self.i-self.lf>=8 and self.d!=self.FLAT and self.pv>0:
            fr=self.fr[self.i] if self.i<len(self.fr) else 0
            self.eq-=self.pv*fr*self.d; self.lf=self.i
        if a==0 and self.d!=self.SHORT:
            self._c(p); self.d=self.SHORT; self.mg=self.eq*0.95; self.pv=self.mg*self.lev; self.en=p; self.eq-=self.pv*0.0005
        elif a==2 and self.d!=self.LONG:
            self._c(p); self.d=self.LONG; self.mg=self.eq*0.95; self.pv=self.mg*self.lev; self.en=p; self.eq-=self.pv*0.0005
        elif a==1: self._c(p)
        self.i+=1; dn=self.i>=self.n-1
        r=-100 if self.eq<=0 else ((self.eq-pe)/pe*100 if not dn else (self.eq-self.c0)/self.c0*100)
        if self.eq<=0: self.eq=0
        return self.s[min(self.i,self.n-1)], r, dn, {'eq':self.eq,'d':self.d}
    def _c(self,p):
        if self.d!=self.FLAT and self.pv>0:
            ch=(p-self.en)/self.en; self.eq+=self.pv*ch*self.d-abs(self.pv)*0.0005
            self.mg=0; self.pv=0; self.d=self.FLAT; self.en=0

# ─── 模型管理 ────────────────────────────────────────────
def get_model(tf_name, state_dim):
    path = MODEL_DIR/f'ppo_{tf_name}.pth'
    agent = AgentPPO(net_dim=256, state_dim=state_dim, action_dim=3, gpu_id=0)
    if path.exists():
        agent.act.load_state_dict(torch.load(str(path)))
        logger.info(f"  🔄 {tf_name}: 加载已有模型")
    else:
        logger.info(f"  🆕 {tf_name}: 创建新模型")
    return agent

def train_one_tf(tf_name, cfg, shared, tick_count):
    """训练单个时间框架"""
    interval = cfg['interval']; n_steps = cfg['n_steps']; lookback = cfg['lookback']

    df = fetch_klines(interval, limit=500)
    df = build_features(df, shared)
    arr = df[FEATURE_COLS].values.astype(np.float32)
    m,s=arr.mean(0),arr.std(0); s[s==0]=1
    norm = (arr - m) / s
    states = np.array([norm[i-lookback:i].flatten() for i in range(lookback,len(norm))], dtype=np.float32)

    state_dim = states.shape[1]
    agent = get_model(tf_name, state_dim)
    env = FuturesEnv(df['close'].values, df['funding_rate'].values, states)
    device = agent.device; t0=time.time()

    state=env.reset(); traj=[]
    for _ in range(n_steps):
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad(): a_t,n_t=agent.act.get_action(s_t)
        a=a_t.cpu().numpy()[0,0].round().clip(0,2).astype(int)
        ns,r,d,info=env.step(a)
        traj.append((s_t.cpu(),r,d,a_t.cpu(),n_t.cpu()))
        state=ns
        if d: break
    if traj:
        buf=(torch.cat([t[0] for t in traj]), torch.tensor([t[1] for t in traj], dtype=torch.float32),
             torch.tensor([1-t[2] for t in traj], dtype=torch.float32), torch.cat([t[3] for t in traj]),
             torch.cat([t[4] for t in traj]))
        c_loss,a_loss,_=agent.update_net(buf)
    else: c_loss=0

    torch.save(agent.act.state_dict(), str(MODEL_DIR/f'ppo_{tf_name}.pth'))
    eq=info['eq']; di=info['d']
    return tf_name, round(df['close'].iloc[-1],2), round(eq,2), di, round(c_loss,2), time.time()-t0

# ─── 集成决策 ────────────────────────────────────────────
def ensemble(models_signal):
    """
    models_signal: [(tf, price, equity, direction, loss, train_time), ...]
    集成规则: 加权投票 (equity越高权重大)
    """
    if not models_signal: return "N/A"
    total_w = 0; weighted = 0
    for _,_,eq,di,_,_ in models_signal:
        w = max(eq, 1)  # equity 作为权重
        weighted += di * w
        total_w += w
    avg = weighted / total_w if total_w else 0
    if avg > 0.3: return "🟢做多"
    elif avg < -0.3: return "🔴做空"
    else: return "⚪观望"

# ─── 主循环 ──────────────────────────────────────────────
def main():
    logger.info("="*70)
    logger.info("🚀 FinRL-X 多时间框架持续训练 (5m/15m/1h)")
    logger.info("⏰ 每5分钟迭代 — Gate.io ETH永续合约")
    logger.info("="*70)

    tick_count = 0
    while True:
        tick_count += 1
        now = datetime.now()
        logger.info(f"\n{'─'*60}")
        logger.info(f"📡 Tick #{tick_count} | {now.strftime('%H:%M:%S')}")

        t_all = time.time()
        try:
            shared = fetch_shared()

            results = []
            for tf_name, cfg in TIMEFRAMES.items():
                if tick_count % cfg['train_every'] == 0:
                    res = train_one_tf(tf_name, cfg, shared, tick_count)
                    results.append(res)
                    dir_s = {0:'空仓',1:'🟢做多',-1:'🔴做空'}
                    logger.info(f"  {res[0]:>3s} | ETH=${res[1]} | 权益=${res[2]:.0f} | "
                                f"{dir_s.get(res[3],'?')} | Loss={res[4]} | {res[5]:.1f}s")

            # 集成
            if results:
                ens = ensemble(results)
                logger.info(f"  ─── 集成决策: {ens} ───")

            # 统一日志
            latest = results[-1] if results else ('N/A',0,0,0,0,0)
            log_path = DATA_DIR/'multi_log.csv'
            row = pd.DataFrame([{
                'time':now.isoformat(), 'tick':tick_count,
                'eth_price':latest[1],
                f'eq_5m': results[0][2] if len(results)>0 else 0,
                f'dir_5m': results[0][3] if len(results)>0 else 0,
                f'eq_15m':results[1][2] if len(results)>1 else 0,
                f'dir_15m':results[1][3] if len(results)>1 else 0,
                f'eq_1h': results[2][2] if len(results)>2 else 0,
                f'dir_1h': results[2][3] if len(results)>2 else 0,
                'ensemble':ens,
                'funding_rate':round(float(shared.get('funding_rate_series',pd.Series([0])).iloc[-1]),6),
                'oi':round(shared['open_interest'],2),
                'lr':round(shared['long_users']/max(shared['short_users'],1),2),
            }])
            if log_path.exists():
                row.to_csv(log_path, mode='a', header=False, index=False)
            else:
                row.to_csv(log_path, index=False)

            elapsed = time.time()-t_all
            logger.info(f"  ✅ 本轮完成 | {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"❌ 出错: {e}")
            import traceback; traceback.print_exc()

        # 睡到下一轮
        now2 = datetime.now()
        next_5m = (now2 + timedelta(minutes=5)).replace(second=0, microsecond=0)
        wait = (next_5m - now2).total_seconds()
        logger.info(f"  ⏳ 等待 {wait:.0f}s → {next_5m.strftime('%H:%M')}")
        time.sleep(max(wait, 1))

if __name__ == '__main__':
    main()
