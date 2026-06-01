#!/usr/bin/env python3
"""FinRL-X Web面板 — 读取独立模型状态"""
import os,json
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template_string

ROOT = Path('/home/administrator/FinRL-Integrated')
STATE = ROOT/'live_state'
MODEL = ROOT/'live_model'

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinRL-X 多TF合约面板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI',sans-serif;background:#0f0f0f;color:#e0e0e0;padding:20px}
h1{font-size:24px;margin-bottom:20px;color:#00d4aa}
h1 small{font-size:14px;color:#888;margin-left:12px}
h2{font-size:16px;margin:16px 0 8px;color:#aaa}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:20px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:16px;position:relative}
.card .label{color:#888;font-size:12px;margin-bottom:4px}
.card .value{font-size:22px;font-weight:700}
.card .sub{font-size:13px;color:#666;margin-top:4px}
.card .badge{position:absolute;top:12px;right:12px;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600}
.bg-green{background:#00d4aa22;color:#00d4aa}.bg-red{background:#ff6b6b22;color:#ff6b6b}.bg-flat{background:#555;color:#ccc}
.green{color:#00d4aa}.red{color:#ff6b6b}.yellow{color:#ffd93d}.flat{color:#888}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
.chart-wrap{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:16px;margin-bottom:20px}
#status{color:#555;font-size:12px;margin-top:8px;text-align:right}
.signal-box{text-align:center;padding:24px;background:#1a1a2e;border:2px solid #2a2a4a;border-radius:12px;margin-bottom:20px}
.signal-box .sig{font-size:48px;font-weight:800;margin:8px 0}
.signal-box .weight{font-size:13px;color:#666}
</style>
</head>
<body>

<h1>FinRL-X ETH 永续合约 <small>独立模型 · 每小时集成</small></h1>

<div class="grid" id="market-bar"></div>

<div class="signal-box" id="signal-box">
  <div class="label">🧠 集成策略（每小时）</div>
  <div class="sig" id="ensemble-sig">--</div>
  <div class="weight" id="ensemble-detail"></div>
</div>

<h2>📊 独立模型</h2>
<div class="grid-3" id="tf-cards"></div>

<h2>📈 权益曲线</h2>
<div class="chart-wrap"><canvas id="eqChart" height="180"></canvas></div>

<div id="status">等待数据...</div>

<script>
let eqChart=null;

function fetchData(){
  fetch('/api/data').then(r=>r.json()).then(d=>{
    // 市场
    const m=d.market;
    document.getElementById('market-bar').innerHTML=
      `<div class="card"><div class="label">ETH</div><div class="value green">$${m.price}</div><div class="sub">${m.change} 24h</div></div>
       <div class="card"><div class="label">资金费率</div><div class="value">${(m.fr*100).toFixed(4)}%</div><div class="sub">8h结算</div></div>
       <div class="card"><div class="label">未平仓量</div><div class="value">$${m.oi.toFixed(0)}M</div></div>
       <div class="card"><div class="label">多空比</div><div class="value">${m.lr.toFixed(2)}</div></div>`;

    // 集成
    document.getElementById('ensemble-sig').textContent=d.ensemble.signal;
    document.getElementById('ensemble-sig').className='sig '+d.ensemble.color;
    document.getElementById('ensemble-detail').textContent=d.ensemble.detail;

    // TF卡片
    document.getElementById('tf-cards').innerHTML=d.timeframes.map(t=>{
      const c=t.direction==1?'green':t.direction==-1?'red':'flat';
      const l=t.direction==1?'做多':t.direction==-1?'做空':'空仓';
      const bc=t.direction==1?'bg-green':t.direction==-1?'bg-red':'bg-flat';
      return `<div class="card"><span class="badge ${bc}">${l}</span>
        <div class="label">${t.name}</div>
        <div class="value ${c}">$${t.equity}</div>
        <div class="sub">更新: ${t.time} | 模型 ${t.model}</div></div>`;
    }).join('');

    // 图表
    if(eqChart){
      eqChart.data.labels=d.chart.labels;
      eqChart.data.datasets[0].data=d.chart.eq_5m;
      eqChart.data.datasets[1].data=d.chart.eq_15m;
      eqChart.data.datasets[2].data=d.chart.eq_1h;
      eqChart.update();
    }else{initChart(d);}

    document.getElementById('status').textContent='🟢 运行中 | '+d.timestamp;
  }).catch(()=>document.getElementById('status').textContent='🔴 等待...');
}

function initChart(d){
  const ctx=document.getElementById('eqChart').getContext('2d');
  eqChart=new Chart(ctx,{type:'line',data:{labels:d.chart.labels,datasets:[
    {label:'5m',data:d.chart.eq_5m,borderColor:'#00d4aa',borderWidth:2,pointRadius:0},
    {label:'15m',data:d.chart.eq_15m,borderColor:'#ffd93d',borderWidth:2,pointRadius:0},
    {label:'1h',data:d.chart.eq_1h,borderColor:'#ff6b6b',borderWidth:2,pointRadius:0}
  ]},options:{
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#888',boxWidth:12}}},
    scales:{x:{ticks:{color:'#555',maxTicksLimit:8}},y:{ticks:{color:'#555',callback:v=>'$'+v}}},
    animation:{duration:300}
  }});
}

fetchData();setInterval(fetchData,10000);
</script>
</body></html>
"""

def read_state(name):
    p = STATE/f'{name}'
    if p.exists():
        with open(p) as f: return json.load(f)
    return {}

def model_size(name):
    p = MODEL/f'ppo_{name}.pth'
    if p.exists(): return f"{p.stat().st_size/1024:.0f}KB"
    return '无'

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/data')
def api():
    s5 = read_state('5m.json')
    s15 = read_state('15m.json')
    s1h = read_state('1h.json')
    ens = read_state('ensemble.json')

    market = {
        'price': s5.get('price',0) or s15.get('price',0) or s1h.get('price',0),
        'change': '--','fr': 0,'oi': 0,'lr': 0,
    }
    # 从latest market data尝试获取
    import requests as r
    try:
        st=r.get("https://api.gateio.ws/api/v4/spot/tickers?currency_pair=ETH_USDT",timeout=5).json()[0]
        market['change']=st.get('change_percentage','0')+'%'
    except: pass
    try:
        ci=r.get("https://api.gateio.ws/api/v4/futures/usdt/contracts/ETH_USDT",timeout=5).json()
        fr_v=float(ci.get('funding_rate',0))
        oi_v=float(ci.get('position_size',0))/1e6
        lu=float(ci.get('long_users',1));su=float(ci.get('short_users',1))
        market['fr']=fr_v;market['oi']=oi_v;market['lr']=lu/su if su>0 else 0
    except: pass

    timeframes=[]
    for name,st in [('5m',s5),('15m',s15),('1h',s1h)]:
        if st:
            timeframes.append({
                'name':name,'equity':st.get('equity',0),'direction':st.get('direction',0),
                'time':st.get('time','--'),'model':model_size(name),
            })

    ensemble = {
        'signal': ens.get('signal','等待数据'),
        'color': ens.get('color','flat'),
        'detail': ens.get('detail',''),
    }

    # 简单轮询历史
    eq_5m=[s5.get('equity',0)]; eq_15m=[s15.get('equity',0)]; eq_1h=[s1h.get('equity',0)]
    labels=[s5.get('time','')]

    chart={'labels':labels,'eq_5m':eq_5m,'eq_15m':eq_15m,'eq_1h':eq_1h}

    return jsonify({
        'market':market,'timeframes':timeframes,'ensemble':ensemble,
        'chart':chart,'timestamp':datetime.now().strftime('%H:%M:%S'),
    })

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
