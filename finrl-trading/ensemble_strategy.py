#!/usr/bin/env python3
"""FinRL-X 集成策略 — 每小时分析三个模型，输出综合信号"""
import os,sys,time,json,logging
from pathlib import Path
ROOT=Path('/home/administrator/FinRL-Integrated'); STATE=ROOT/'live_state'
logging.basicConfig(level=logging.INFO,format='%(asctime)s [ensemble] %(message)s')
logger=logging.getLogger(__name__)

TF_NAMES = ['5m','15m','1h']
TF_WEIGHTS = {'5m':0.2,'15m':0.3,'1h':0.5}  # 长周期权重更大

def analyze():
    states = {}
    for tf in TF_NAMES:
        p = STATE/f'{tf}.json'
        if p.exists():
            with open(p) as f: states[tf] = json.load(f)
    
    if not states:
        return {'signal':'等待数据','detail':'模型尚未产生信号','color':'flat'}

    # 加权投票
    total_w = 0; weighted_dir = 0
    details = []
    for tf in TF_NAMES:
        s = states.get(tf)
        if s:
            w = TF_WEIGHTS.get(tf, 0.33)
            weighted_dir += s['direction'] * w * (s['equity']/1000)
            total_w += w * (s['equity']/1000)
            dir_s = {0:'⚪',1:'🟢',-1:'🔴'}.get(s['direction'],'?')
            details.append(f"{tf}:${s['equity']:.0f}{dir_s}(w={w:.1f})")

    if total_w == 0: return {'signal':'⚪观望','detail':' | '.join(details),'color':'flat'}
    avg = weighted_dir / total_w

    if avg > 0.25: signal, color = '🟢做多', 'green'
    elif avg < -0.25: signal, color = '🔴做空', 'red'
    else: signal, color = '⚪观望', 'flat'

    decision = {
        'signal': signal,
        'detail': ' | '.join(details),
        'color': color,
        'avg_weighted': round(avg, 3),
        'timestamp': time.strftime('%H:%M'),
    }
    
    with open(STATE/'ensemble.json','w') as f:
        json.dump(decision, f, indent=2)
    
    logger.info(f"集成: {signal} (avg={avg:.3f}) | {decision['detail']}")
    return decision

def main():
    logger.info("🚀 集成策略启动 — 每小时分析")
    while True:
        try:
            analyze()
        except Exception as e:
            logger.error(f"出错: {e}")
        time.sleep(3600-(time.time()%3600))

if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        print(json.dumps(analyze(), ensure_ascii=False, indent=2))
    else:
        main()
