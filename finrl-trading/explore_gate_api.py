#!/usr/bin/env python3
"""Gate.io API — 全数据目录"""
import requests, json

BASE = "https://api.gateio.ws/api/v4"
H = {"Accept": "application/json"}
h = requests.Session(); h.headers.update(H)

cols = ["端点", "返回字段", "用途"]

items = []

def add(ep, desc, fields, note=""):
    items.append({"端点": ep, "返回字段": fields, "用途": desc})

# ─── 现货 ───
r = h.get(f"{BASE}/spot/tickers?currency_pair=ETH_USDT", timeout=10).json()
add("spot/tickers", "现货实时行情",
    list(r[0].keys()) if r else [],
    "当前价、24h涨跌、最高最低、成交量")

r = h.get(f"{BASE}/spot/candlesticks?currency_pair=ETH_USDT&interval=1h&limit=3", timeout=10).json()
add("spot/candlesticks", "现货K线",
    ["timestamp","volume(USDT)","close","high","low","open","amount(币)","complete"],
    "1m/5m/15m/30m/1h/4h/8h/1d/7d/30d")

r = h.get(f"{BASE}/spot/order_book?currency_pair=ETH_USDT&limit=5", timeout=10).json()
if r:
    add("spot/order_book", "订单簿深度",
        list(r.keys()),
        "前N档买卖挂单 + 更新时间")

r = h.get(f"{BASE}/spot/trades?currency_pair=ETH_USDT&limit=5", timeout=10).json()
add("spot/trades", "最近成交",
    list(r[0].keys()) if r else [],
    "成交价、量、方向、时间")

r = h.get(f"{BASE}/spot/currency_pairs", timeout=10).json()
eth_pair = [p for p in r if p['id'] == 'ETH_USDT']
if eth_pair:
    add("spot/currency_pairs", "交易对信息",
        list(eth_pair[0].keys()) if eth_pair else [],
        "最小/最大交易量、精度、最低价格")

# ─── 永续合约 ───
r = h.get(f"{BASE}/futures/usdt/contracts/ETH_USDT", timeout=10).json()
add("futures/contracts", "合约完整信息",
    list(r.keys()),
    "资金费率、标记价、指数价、持仓量、多空人数、杠杆范围")

r = h.get(f"{BASE}/futures/usdt/tickers?contract=ETH_USDT", timeout=10).json()
add("futures/tickers", "合约行情",
    list(r[0].keys()) if r else [],
    "24h涨跌、成交量、最高最低、标记价")

r = h.get(f"{BASE}/futures/usdt/funding_rate?contract=ETH_USDT&limit=5", timeout=10).json()
add("futures/funding_rate", "资金费率历史",
    list(r[0].keys()) if r else [],
    "每8小时费率记录")

r = h.get(f"{BASE}/futures/usdt/order_book?contract=ETH_USDT&limit=5", timeout=10).json()
if r:
    add("futures/order_book", "合约订单簿",
        list(r.keys()),
        "合约买卖挂单")

r = h.get(f"{BASE}/futures/usdt/trades?contract=ETH_USDT&limit=5", timeout=10).json()
add("futures/trades", "合约最近成交",
    list(r[0].keys()) if r else [],
    "合约逐笔成交")

# ─── BBO（最优买卖价）───
r = h.get(f"{BASE}/spot/bbo_ticker?currency_pair=ETH_USDT", timeout=10).json()
if r:
    add("spot/bbo_ticker", "最优买卖报价",
        list(r.keys()) if r else [],
        "实时买一卖一价和量")

r = h.get(f"{BASE}/futures/usdt/bbo_ticker?contract=ETH_USDT", timeout=10).json()
if r:
    add("futures/bbo_ticker", "合约最优报价",
        list(r.keys()) if r else [],
        "合约买一卖一")

print("=" * 130)
print(f"{'📡 Gate.io API 数据目录 — 全部可用数据源（ETH/USDT）':^130}")
print("=" * 130)

print(f"\n{'#':>2} {'端点':<35} {'用途':<25} {'可用字段':<70}")
print("-" * 130)
for i, item in enumerate(items, 1):
    fields = ", ".join(item['返回字段'][:8])
    if len(item['返回字段']) > 8:
        fields += f"... (+{len(item['返回字段'])-8})"
    print(f"{i:>2} {item['端点']:<35} {item['用途']:<25} {fields:<70}")

# ─── 合约关键字段详解 ───
print("\n")
print("=" * 130)
print(f"{'🔬 永续合约关键字段详解 (futures/contracts/ETH_USDT)':^130}")
print("=" * 130)

contract_info = h.get(f"{BASE}/futures/usdt/contracts/ETH_USDT", timeout=10).json()
for k, v in contract_info.items():
    if isinstance(v, dict):
        print(f"  {k:35s} = ...(dict)")
    elif isinstance(v, list):
        print(f"  {k:35s} = [{', '.join(str(x)[:30] for x in v[:3])}, ...]")
    else:
        print(f"  {k:35s} = {v}")

# ─── 最后总结可用的特征维度 ──
print("\n")
print("=" * 130)
print(f"{'✅ 可直接用于训练的特征维度':^130}")
print("=" * 130)

features_from_api = {
    "现货K线(5个)":      ["open", "high", "low", "close", "volume"],
    "资金费率(1个)":      ["funding_rate"],
    "合约行情(3个)":      ["mark_price", "index_price", "last_price"],
    "未平仓量(2个)":      ["position_size(OI)", "trade_size(累计交易量)"],
    "多空比(2个)":        ["long_users", "short_users"],
    "杠杆信息(3个)":      ["leverage_min", "leverage_max", "cross_leverage_default"],
    "费率信息(4个)":      ["funding_rate", "funding_rate_indicative", "funding_next_apply", "funding_cap_ratio"],
    "风险参数(5个)":      ["mark_price_round", "order_price_round", "maintenance_rate",
                           "risk_limit_base", "risk_limit_max"],
}

for cat, feats in features_from_api.items():
    print(f"  {cat:<25} {', '.join(feats)}")
