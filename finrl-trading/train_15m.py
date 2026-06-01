#!/usr/bin/env python3
"""FinRL-X 15m 独立训练 — 每15分钟一次"""
import os,sys,time,logging
sys.path.insert(0,'/home/administrator/FinRL-Integrated')
sys.path.insert(0,'/home/administrator/FinRL_Crypto')
import numpy as np,requests,pandas as pd
import torch; from drl_agents.agents.AgentPPO import AgentPPO
from pathlib import Path; ROOT=Path('/home/administrator/FinRL-Integrated')
MODEL=ROOT/'live_model'; STATE=ROOT/'live_state'
MODEL.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s [15m] %(message)s')
logger=logging.getLogger(__name__)

BASE="https://api.gateio.ws/api/v4"; http=requests.Session(); http.headers.update({"Accept":"application/json"})
FEATURES=['close','volume','returns','ma7','ma25','ma99','price_to_ma7','price_to_ma25',
    'macd','macd_signal','macd_hist','rsi','bb_width','atr','volume_ratio','volatility',
    'price_position','funding_rate','open_interest','basis','basis_pct','funding_rate_indicative',
    'funding_rate_diff','funding_rate_change','long_short_ratio','long_pct','oi_change_24h',
    'trade_size_change','spread','spread_change','change_pct_24h_smooth']
LOOKBACK=50
def klines(): return pd.DataFrame(
    http.get(f"{BASE}/futures/usdt/candlesticks",params={"contract":"ETH_USDT","interval":"15m","limit":500},timeout=10).json()
).pipe(lambda d:d.assign(**{c:d[c].astype(float) for c in ['o','h','l','c','v']})
       .assign(timestamp=pd.to_datetime(d['t'].astype(int),unit='s'))
       .set_index('timestamp').iloc[::-1])[['o','h','l','c','v']].rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
def shared():
    fr=pd.DataFrame(http.get(f"{BASE}/futures/usdt/funding_rate",params={"contract":"ETH_USDT","limit":300},timeout=10).json())
    fr['t']=pd.to_datetime(fr['t'].astype(int),unit='s'); fr.set_index('t',inplace=True); fr=fr.iloc[::-1]
    fr_h=fr['r'].astype(float).resample('15min').ffill()
    ci=http.get(f"{BASE}/futures/usdt/contracts/ETH_USDT",timeout=10).json()
    s={'fr':fr_h,'oi':float(ci.get('position_size',0))/1e6,'lu':float(ci.get('long_users',1)),'su':float(ci.get('short_users',1)),
       'mp':float(ci.get('mark_price',0)),'ip':float(ci.get('index_price',0)),'fi':float(ci.get('funding_rate_indicative',0))}
    try: st=http.get(f"{BASE}/spot/tickers?currency_pair=ETH_USDT",timeout=10).json()[0]; s['chg']=float(st.get('change_percentage',0))
    except: s['chg']=0
    try:
        ob=http.get(f"{BASE}/spot/order_book?currency_pair=ETH_USDT&limit=1",timeout=10).json()
        if 'asks' in ob and 'bids' in ob:
            a=float(ob['asks'][0][0]);b=float(ob['bids'][0][0])
            s['sp']=(a-b)/((a+b)/2)*100 if (a+b)>0 else 0
        else: s['sp']=0
    except: s['sp']=0
    return s
def build(df,s):
    df['returns']=df['close'].pct_change()
    df['ma7']=df['close'].rolling(7).mean();df['ma25']=df['close'].rolling(25).mean();df['ma99']=df['close'].rolling(99).mean()
    df['price_to_ma7']=df['close']/df['ma7']-1;df['price_to_ma25']=df['close']/df['ma25']-1
    e12=df['close'].ewm(span=12).mean();e26=df['close'].ewm(span=26).mean()
    df['macd']=e12-e26;df['macd_signal']=df['macd'].ewm(span=9).mean();df['macd_hist']=df['macd']-df['macd_signal']
    d=df['close'].diff();g=d.where(d>0,0).rolling(14).mean();l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/l.replace(0,np.nan)))
    bm=df['close'].rolling(20).mean();bs=df['close'].rolling(20).std()
    df['bb_width']=((bm+2*bs)-(bm-2*bs))/bm
    df['atr']=pd.concat([(df['high']-df['low']),(df['high']-df['close'].shift(1)).abs(),(df['low']-df['close'].shift(1)).abs()],axis=1).max(axis=1).rolling(14).mean()
    df['volume_ratio']=df['volume']/df['volume'].rolling(7).mean();df['volatility']=df['returns'].rolling(20).std()*100
    df['price_position']=((df['close']-df['low'].rolling(20).min())/(df['high'].rolling(20).max()-df['low'].rolling(20).min()+1e-8))
    df['funding_rate']=s['fr'].reindex(df.index,method='ffill').fillna(0)
    df['open_interest']=s['oi'];df['long_users']=s['lu'];df['short_users']=s['su']
    df['mark_price']=s['mp'];df['index_price']=s['ip'];df['funding_rate_indicative']=s['fi']
    df['oi_change_24h']=df['open_interest'].pct_change(min(24,len(df)-1)).fillna(0)
    df['basis']=df['mark_price']-df['index_price'];df['basis_pct']=df['basis']/df['index_price']*100
    df['funding_rate_diff']=df['funding_rate_indicative']-df['funding_rate']
    df['funding_rate_change']=df['funding_rate'].diff().fillna(0)
    df['long_short_ratio']=df['long_users']/df['short_users'].replace(0,1)
    df['long_pct']=df['long_users']/(df['long_users']+df['short_users']+1)*100
    df['trade_size_change']=df['open_interest'].diff().fillna(0)
    df['spread']=s['sp'];df['spread_change']=df['spread'].diff().fillna(0);df['change_pct_24h']=s['chg']
    df['change_pct_24h_smooth']=df['change_pct_24h'].rolling(3).mean()
    return df.bfill().ffill().fillna(0)

class Env:
    F=0;L=1;S=-1
    def __init__(self,p,fr,st):self.p=p;self.fr=fr;self.st=st;self.n=len(st);self.reset()
    def reset(self):self.i=0;self.e=1000;self.m=0;self.pv=0;self.d=self.F;self.en=0;self.lf=0;return self.st[0]
    def step(self,a):
        p=self.p[self.i];pe=self.e
        if self.d!=self.F and self.pv>0:
            ch=(p-self.en)/self.en;self.e=max(self.m+self.e-self.m+self.pv*ch*self.d,0)
            if self.e<=self.m*0.3:self.e=self.m*0.3;self.m=0;self.pv=0;self.d=self.F
        if self.i-self.lf>=8 and self.d!=self.F and self.pv>0:
            fr=self.fr[self.i] if self.i<len(self.fr) else 0;self.e-=self.pv*fr*self.d;self.lf=self.i
        if a==0 and self.d!=self.S:self._c(p);self.d=self.S;self.m=self.e*0.95;self.pv=self.m*10;self.en=p;self.e-=self.pv*0.0005
        elif a==2 and self.d!=self.L:self._c(p);self.d=self.L;self.m=self.e*0.95;self.pv=self.m*10;self.en=p;self.e-=self.pv*0.0005
        elif a==1:self._c(p)
        self.i+=1;dn=self.i>=self.n-1
        r=-100 if self.e<=0 else ((self.e-pe)/pe*100 if not dn else (self.e-1000)/1000*100)
        if self.e<=0:self.e=0
        return self.st[min(self.i,self.n-1)],r,dn,{'eq':self.e,'d':self.d}
    def _c(self,p):
        if self.d!=self.F and self.pv>0:ch=(p-self.en)/self.en;self.e+=self.pv*ch*self.d-abs(self.pv)*0.0005;self.m=0;self.pv=0;self.d=self.F;self.en=0

def main():
    logger.info("🚀 15m 训练启动 — 每15分钟")
    while True:
        try:
            t0=time.time()
            df=klines();s=shared();df=build(df,s)
            arr=df[FEATURES].values.astype(np.float32);m,arr_m=arr.mean(0),arr.std(0);arr_m[arr_m==0]=1
            states=np.array([((arr-m)/arr_m)[i-LOOKBACK:i].flatten() for i in range(LOOKBACK,len(arr))],dtype=np.float32)
            pt=MODEL/'ppo_15m.pth'
            ag=AgentPPO(net_dim=256,state_dim=states.shape[1],action_dim=3,gpu_id=0)
            if pt.exists():ag.act.load_state_dict(torch.load(str(pt)))
            env=Env(df['close'].values,df['funding_rate'].values,states);device=ag.device
            st=env.reset();tj=[]
            for _ in range(48):
                s_t=torch.tensor(st,dtype=torch.float32,device=device).unsqueeze(0)
                with torch.no_grad():a_t,n_t=ag.act.get_action(s_t)
                ns,r,d,info=env.step(a_t.cpu().numpy()[0,0].round().clip(0,2).astype(int))
                tj.append((s_t.cpu(),r,d,a_t.cpu(),n_t.cpu()));st=ns
                if d:break
            if tj:
                buf=(torch.cat([t[0] for t in tj]),torch.tensor([t[1] for t in tj],dtype=torch.float32),
                     torch.tensor([1-t[2] for t in tj],dtype=torch.float32),torch.cat([t[3] for t in tj]),
                     torch.cat([t[4] for t in tj]))
                cl,al,_=ag.update_net(buf)
            torch.save(ag.act.state_dict(),str(pt))
            # 神经进化：时间戳备份
            ts=time.strftime('%Y%m%d_%H%M')
            bk=MODEL/f'ppo_15m_{ts}.pth'
            if not bk.exists(): torch.save(ag.act.state_dict(),str(bk))
            eq=round(info['eq'],2);dir_=info['d']
            price=round(float(df['close'].iloc[-1]),2)
            import json
            with open(STATE/'15m.json','w') as f:json.dump({'time':time.strftime('%H:%M'),'price':price,'equity':eq,'direction':dir_,'loss':round(float(cl),2) if 'cl' in dir() else 0},f)
            dir_s={0:'空仓',1:'做多',-1:'做空'}
            logger.info(f"ETH=${price} | 权益=${eq} | {dir_s.get(dir_,'?')} | {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"出错: {e}")
            import traceback;traceback.print_exc()
        time.sleep(900-(time.time()%900))

if __name__=='__main__':main()
