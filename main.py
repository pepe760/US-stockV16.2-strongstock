#@title US Quant Master V16.2 (Stage Colors + AI Prompts + Rec Chart + Robust Dashboard) good
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import json
import warnings
import math
import requests
from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 系統參數設定
# ==============================================================================
@dataclass
class Config:
    vix_max_threshold: float = 22.0
    cup_min_bars:       int   = 35
    cup_max_bars:       int   = 250
    cup_min_depth:      float = 0.12
    cup_max_depth:      float = 0.35
    handle_max_bars:    int   = 20
    handle_max_retrace: float = 0.12
    vol_breakout_ratio: float = 1.20
    vcp_min_contractions:  int   = 2
    vcp_contraction_ratio: float = 0.75
    vcp_final_max_range:   float = 0.10
    min_close_range:  float = 0.5
    pullback_enabled:       bool  = True
    pullback_max_wait:      int   = 15
    pullback_ma_period:     int   = 20
    pullback_ma_tolerance:  float = 0.015
    pullback_bounce_ratio:  float = 0.50
    pullback_vol_confirm:   float = 0.80
    rs_anchor_enabled: bool  = True
    rs_min_excess:     float = 0.0
    stage2_only: bool = True
    sl_mult:        float = 2.2
    ts_pct:         float = 0.15
    breakeven_trigger_pct: float = 0.15
    breakeven_lock_pct:    float = 0.01
    max_portfolio_size: int = 10
    ftd_correction_pct:  float = 0.07
    ftd_rally_min_day:   int   = 4
    ftd_rally_max_day:   int   = 7
    ftd_min_gain_pct:    float = 0.012
    dist_min_drop_pct:   float = 0.002
    dist_rolling_window: int   = 25

CFG = Config()
START_DATE = "1995-01-01"
END_DATE   = "2027-01-01"

# ==============================================================================
# 2. 數據獲取與技術指標計算
# ==============================================================================
def get_sp500_tickers() -> List[str]:
    print("🌐 1/7 正在獲取 S&P 500 最新成分股名單...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
        df = pd.read_html(html)[0]
        return df['Symbol'].str.replace('.', '-', regex=False).tolist()
    except:
        return ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","NFLX","JPM"]

WATCHLIST = get_sp500_tickers()[:100]

print(f"⏳ 2/7 下載大盤與 {len(WATCHLIST)} 檔個股數據...")
spy_df = yf.download(["SPY","^VIX"], start=START_DATE, end=END_DATE, progress=False, threads=True)
spy_c  = spy_df['Close']['SPY'].ffill()
spy_v  = spy_df['Volume']['SPY'].fillna(0)
vix_c  = spy_df['Close']['^VIX'].ffill()

data = yf.download(WATCHLIST, start=START_DATE, end=END_DATE, progress=False, threads=True, group_by='ticker')
opens, highs, lows, closes, volumes = {},{},{},{},{}
for ticker in WATCHLIST:
    try:
        df_t = data if len(WATCHLIST)==1 else data[ticker]
        if not df_t.empty and not df_t.isna().all().all():
            opens[ticker]   = df_t['Open']
            highs[ticker]   = df_t['High']
            lows[ticker]    = df_t['Low']
            closes[ticker]  = df_t['Close']
            volumes[ticker] = df_t['Volume']
    except: pass

opens   = pd.DataFrame(opens).ffill()
highs   = pd.DataFrame(highs).ffill()
lows    = pd.DataFrame(lows).ffill()
closes  = pd.DataFrame(closes).ffill()
volumes = pd.DataFrame(volumes).fillna(0)
dates_array = closes.index.strftime('%Y-%m-%d').tolist()

spy_50ma  = spy_c.rolling(50).mean()
spy_200ma = spy_c.rolling(200).mean()
spy_pqr   = (spy_c - spy_c.shift(50)) / spy_c.shift(50)

sma20_all  = closes.rolling(20).mean()
sma50_all  = closes.rolling(50).mean()
sma200_all = closes.rolling(200).mean()
vol20_all  = volumes.rolling(20).mean()

atrs, pqrs = {}, {}
for ticker in closes.columns:
    tr = pd.concat([
        highs[ticker]-lows[ticker],
        (highs[ticker]-closes[ticker].shift(1)).abs(),
        (lows[ticker]-closes[ticker].shift(1)).abs()
    ], axis=1).max(axis=1)
    atrs[ticker] = tr.rolling(14).mean()
    pqrs[ticker] = (closes[ticker]-closes[ticker].shift(50))/closes[ticker].shift(50)
atrs = pd.DataFrame(atrs)
pqrs = pd.DataFrame(pqrs)

# ==============================================================================
# 2b. SPX Market Health & Stages
# ==============================================================================
print("⏳ 3/7 偵測大盤階段與健康度...")
spy_dates = spy_c.index.strftime('%Y-%m-%d').tolist()
spy_sma200_vals = spy_200ma.values
spy_sma50_vals  = spy_50ma.values
spy_close_vals  = spy_c.values
spy_vol_vals    = spy_v.values

dist_days, rolling_dist, dist_count_series = [], [], []
for i in range(1, len(spy_close_vals)):
    pct_chg = (spy_close_vals[i] - spy_close_vals[i-1]) / spy_close_vals[i-1]
    is_dist = (pct_chg <= -CFG.dist_min_drop_pct) and (spy_vol_vals[i] > spy_vol_vals[i-1])
    if is_dist:
        rolling_dist.append(i)
        dist_days.append({'idx': i, 'date': spy_dates[i], 'chg': round(float(pct_chg*100), 2)})
    rolling_dist = [d for d in rolling_dist if i - d <= CFG.dist_rolling_window]
    dist_count_series.append({'idx': i, 'date': spy_dates[i], 'count': len(rolling_dist)})

ftd_days = []
i = 250
while i < len(spy_close_vals):
    recent_high = max(spy_close_vals[max(0,i-100):i+1])
    drawdown = (recent_high - spy_close_vals[i]) / recent_high
    if drawdown >= CFG.ftd_correction_pct:
        rally_start = next((j for j in range(i, min(i+60, len(spy_close_vals))) if spy_close_vals[j] > spy_close_vals[j-1]), None)
        if rally_start is None: i += 1; continue
        ftd_found = False
        for day_num in range(CFG.ftd_rally_min_day, CFG.ftd_rally_max_day + 1):
            k = rally_start + day_num - 1
            if k >= len(spy_close_vals): break
            day_gain = (spy_close_vals[k] - spy_close_vals[k-1]) / spy_close_vals[k-1]
            if day_gain >= CFG.ftd_min_gain_pct and spy_vol_vals[k] > spy_vol_vals[k-1]:
                ftd_days.append({'idx':k,'date':spy_dates[k],'gain':round(float(day_gain*100),2),'correction_depth':round(float(drawdown*100),1),'rally_day':day_num})
                ftd_found = True; i = k + 20; break
        if not ftd_found: i += 1
    else: i += 1

spx_chart_start = max(0, len(spy_dates) - 252*5)
spx_chart_dates  = spy_dates[spx_chart_start:]
spx_chart_close  = [round(float(x),2) if not pd.isna(x) else None for x in spy_close_vals[spx_chart_start:]]
spx_chart_sma200 = [round(float(x),2) if not pd.isna(x) else None for x in spy_sma200_vals[spx_chart_start:]]
spx_chart_sma50  = [round(float(x),2) if not pd.isna(x) else None for x in spy_50ma.values[spx_chart_start:]]

# 計算 SPX 階段陣列 (用於著色)
spx_chart_stages = []
for idx in range(spx_chart_start, len(spy_close_vals)):
    c, m50, m200 = spy_close_vals[idx], spy_sma50_vals[idx], spy_sma200_vals[idx]
    if pd.isna(c) or pd.isna(m50) or pd.isna(m200): stage = 1
    elif c > m50 and m50 > m200: stage = 2
    elif c < m50 and m50 < m200: stage = 4
    else: stage = 1
    spx_chart_stages.append(stage)

ftd_chart_points = [{'ci':spx_chart_dates.index(f['date']),'date':f['date'],'gain':f['gain'],'depth':f['correction_depth'],'day':f['rally_day']} for f in ftd_days if f['date'] in spx_chart_dates]
dist_chart_points = [{'ci':spx_chart_dates.index(d['date']),'date':d['date'],'chg':d['chg']} for d in dist_days if d['date'] in spx_chart_dates]
dist_count_map = {dc['date']: dc['count'] for dc in dist_count_series if dc['date'] in spx_chart_dates}

# ==============================================================================
# 3. & 4. 預計算突破信號與拉回引擎 (簡化略過詳細註解)
# ==============================================================================
print("⏳ 4/7 計算交易信號與回測...")
raw_breakout_signals = {}
start_idx = 250
for i in tqdm(range(start_idx, len(closes)), desc="掃描"):
    current_vix = vix_c.iloc[i]
    if current_vix > CFG.vix_max_threshold: continue
    cur_spy, cur_s50, cur_s200 = spy_c.iloc[i], spy_50ma.iloc[i], spy_200ma.iloc[i]
    spy_stage = 2 if cur_spy>cur_s50>cur_s200 else (4 if cur_spy<cur_s50<cur_s200 else 1)
    if spy_stage == 4 or (CFG.stage2_only and spy_stage != 2): continue
    daily_signals = []
    for ticker in closes.columns:
        c, h, l = closes[ticker].iloc[i], highs[ticker].iloc[i], lows[ticker].iloc[i]
        if pd.isna(c) or c < 10.0 or c < sma200_all[ticker].iloc[i] or pqrs[ticker].iloc[i] <= 0: continue
        if CFG.rs_anchor_enabled and (pqrs[ticker].iloc[i] <= (spy_pqr.iloc[i] + CFG.rs_min_excess)): continue
        signal_type = "VCP" if (h-l)>0 and (c-l)/(h-l)>=CFG.min_close_range else None # 簡化觸發條件展示
        if signal_type:
            daily_signals.append({"ticker":ticker,"idx":i,"date":dates_array[i],"type":signal_type,"atr":float(atrs[ticker].iloc[i]),"pqr":float(pqrs[ticker].iloc[i]),"excess_rs":round((float(pqrs[ticker].iloc[i])-float(spy_pqr.iloc[i]))*100,2),"breakout_price":float(c),"vix":float(current_vix),"stage":spy_stage})
    if daily_signals: raw_breakout_signals[i]={'stage':spy_stage,'sigs':daily_signals}

signals_by_idx = raw_breakout_signals
for idx,entry in signals_by_idx.items():
    for sig in entry['sigs']:
        t1=opens[sig['ticker']].iloc[idx+1] if idx+1<len(closes) else sig['breakout_price']
        sig.update({'entry_mode':'Breakout','t1_open':float(t1) if not pd.isna(t1) else sig['breakout_price']})

# ==============================================================================
# 5. 回測引擎
# ==============================================================================
def run_simulation(sim_start,sim_end):
    active_pos, completed_trades, equity_curve = {}, [], []
    eq = 1.0
    for i in range(sim_start, sim_end+1):
        ds = dates_array[i]; to_remove = []
        for ticker, trade in active_pos.items():
            h, l, c = highs[ticker].iloc[i], lows[ticker].iloc[i], closes[ticker].iloc[i]
            trade['Highest_Price'] = max(trade['Highest_Price'], h)
            if (trade['Highest_Price']-trade['Entry_Price'])/trade['Entry_Price'] >= CFG.breakeven_trigger_pct:
                trade['Hard_SL'] = max(trade['Hard_SL'], trade['Entry_Price']*(1+CFG.breakeven_lock_pct))
                trade['BE_Locked'] = True
            csl = max(trade['Hard_SL'], trade['Highest_Price']*(1-CFG.ts_pct))
            if l <= csl:
                er = 'BE Lock' if trade.get('BE_Locked') and csl<=trade['Entry_Price']*(1.05) else ('Trailing SL' if csl>trade['Original_Hard_SL'] else 'Hard SL')
                trade.update({'Exit_Date':ds,'Exit_Price':csl,'Status':'Win' if csl>trade['Entry_Price'] else 'Loss','Exit_Reason':er, 'Exit_Idx':i, 'Holding_Days':i-trade['Entry_Idx'], 'PnL_%':round(((csl-trade['Entry_Price'])/trade['Entry_Price'])*100,2)})
                completed_trades.append(trade); to_remove.append(ticker)
        for ticker in to_remove: del active_pos[ticker]
        
        if i in signals_by_idx:
            avail = CFG.max_portfolio_size - len(active_pos)
            if avail > 0:
                dsigs = sorted([s for s in signals_by_idx[i]['sigs'] if s['ticker'] not in active_pos], key=lambda x:x.get('excess_rs',0), reverse=True)
                for sig in dsigs[:avail]:
                    tk, ep, atr = sig['ticker'], sig['t1_open'], sig['atr']
                    hsl = ep - (atr*CFG.sl_mult)
                    active_pos[tk] = {"Trade_ID":f"{tk}_{ds}","Ticker":tk,"Type":sig['type'],"Entry_Date":ds,"Entry_Price":ep,"Entry_Idx":i,"Entry_VIX":round(sig['vix'],2),"Highest_Price":ep,"Hard_SL":hsl,"Original_Hard_SL":hsl,"TS_Pct":CFG.ts_pct,"Status":"Active","BE_Locked":False}
        deq = eq
        for td in [t for t in completed_trades if t['Exit_Date']==ds]: deq *= (1+(td['PnL_%']*0.10/100))
        eq = deq; equity_curve.append({'date':ds,'eq':eq})
    return completed_trades, equity_curve

main_start_idx = next(i for i,d in enumerate(dates_array) if d.startswith('2000'))
all_trades, main_eq_curve = run_simulation(main_start_idx, len(dates_array)-1)

# ==============================================================================
# 6. 生成推薦清單與基本面數據
# ==============================================================================
print("⏳ 6/7 生成推薦清單並抓取基本面數據...")
last_idx = len(closes)-1
cur_spy_pqr = float(spy_pqr.iloc[last_idx]) if not pd.isna(spy_pqr.iloc[last_idx]) else 0
current_vix_val = float(vix_c.iloc[last_idx])
current_date_str = dates_array[last_idx]
cv, c50v, c200v = spy_c.iloc[last_idx], spy_50ma.iloc[last_idx], spy_200ma.iloc[last_idx]
current_stage = 2 if cv>c50v>c200v else (4 if cv<c50v<c200v else 1)

raw_recommended = []
for ticker in closes.columns:
    c = closes[ticker].iloc[last_idx]
    s200 = sma200_all[ticker].iloc[last_idx]
    spqr = pqrs[ticker].iloc[last_idx]
    if pd.isna(c) or c<10 or pd.isna(s200) or c<s200 or spqr<=cur_spy_pqr or spqr<=0: continue
    h52 = float(highs[ticker].iloc[max(0,last_idx-252):last_idx+1].max())
    raw_recommended.append({
        'ticker': ticker, 'price': round(float(c),2), 'pqr': round(float(spqr)*100,2),
        'excess_rs': round((spqr-cur_spy_pqr)*100,2), 'dist_from_high': round((c-h52)/h52*100,1)
    })

raw_recommended.sort(key=lambda x: x['excess_rs'], reverse=True)
top_rec = raw_recommended[:30]

# 抓取基本面
rec_chart_data = {}
chart_len = 252 # 1 Year
spy_1y = spy_c.iloc[-chart_len:].values.tolist()

for rec in tqdm(top_rec, desc="基本面與圖表數據"):
    tk = rec['ticker']
    try:
        info = yf.Ticker(tk).info
        rec['div'] = round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 0.0
        rec['pe'] = round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 0.0
        rec['pb'] = round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else 0.0
        rec['roe'] = round(info.get('returnOnEquity', 0) * 100, 2) if info.get('returnOnEquity') else 0.0
    except:
        rec['div'], rec['pe'], rec['pb'], rec['roe'] = 0.0, 0.0, 0.0, 0.0
    
    # 1y momentum
    hist_1m = closes[tk].iloc[-21] if len(closes[tk])>21 else closes[tk].iloc[0]
    rec['mom'] = round((rec['price'] - hist_1m)/hist_1m * 100, 1) if hist_1m else 0.0
    
    # Chart Data
    tk_prices = closes[tk].iloc[-chart_len:].values.tolist()
    spy_p = spy_1y
    # Normalize to percentage change from start
    t0_tk = tk_prices[0] if tk_prices[0] else 1
    t0_spy = spy_p[0] if spy_p[0] else 1
    
    rec_chart_data[tk] = {
        'dates': dates_array[-chart_len:],
        'tk_pct': [round((p-t0_tk)/t0_tk*100, 2) if not pd.isna(p) else 0 for p in tk_prices],
        'spy_pct': [round((p-t0_spy)/t0_spy*100, 2) if not pd.isna(p) else 0 for p in spy_p]
    }

# ==============================================================================
# 7. HTML 儀表板生成
# ==============================================================================
print("⏳ 7/7 封裝 HTML...")
def clean_nans(obj):
    if isinstance(obj,dict): return {k:clean_nans(v) for k,v in obj.items()}
    if isinstance(obj,list): return [clean_nans(v) for v in obj]
    if isinstance(obj,float) and (pd.isna(obj) or math.isnan(obj) or math.isinf(obj)): return None
    return obj

json_rec_chart_data = json.dumps(clean_nans(rec_chart_data), ensure_ascii=False)
json_rec_list = json.dumps(clean_nans(top_rec), ensure_ascii=False)
spxStagesJson = json.dumps(spx_chart_stages)

# HTML Prompt Template
base_prompt = f"""請扮演一位頂尖的美股量化分析師。我的一個量化交易系統剛剛為【[TICKER]】發出了觀察訊號。
請務必【全程使用繁體中文】回答，並執行以下驗證與分析：

[系統當前偵測數據]
- 觸發策略：US Quant Master V16 突破
- 最新股價：$[PRICE]
- 50日動能超額報酬：+[EXCESS]%
- 距52週高點：[DIST]%
- 基本面：股息: [DIV]%, 動能(近1月): [MOM]%, P/E: [PE] | P/B: [PB] | ROE: [ROE]%

[該策略的技術觸發條件]
1. VIX 恐慌指數 <= 22 (當前: {round(current_vix_val,1)})
2. SPY 位於 50與200 日均線之上 (當前大盤階段: Stage {current_stage})
3. 該股價位於 200 日均線之上，相對強度處於市場前列

請根據你的最新網路搜尋能力，完成任務：
1. 【技術條件雙重驗證】：上述條件是否真實反映在最新圖表中？
2. 【基本面與籌碼/消息面】：公司近期基本面是否穩健？有無隱藏地雷（如大行降評、政策打壓）導致假突破？
3. 【最終實戰建議】：值得跟單買進嗎？給出強烈買進(BUY)、觀望(HOLD)或避開(AVOID)結論及停損建議。"""

rec_rows = ""
for i, r in enumerate(top_rec):
    rc="color:var(--cyan400)" if r['excess_rs']>=10 else "color:var(--slate300)"
    p_str = base_prompt.replace('[TICKER]', r['ticker']).replace('[PRICE]', str(r['price'])).replace('[EXCESS]', str(r['excess_rs'])).replace('[DIST]', str(r['dist_from_high'])).replace('[DIV]', str(r['div'])).replace('[MOM]', str(r['mom'])).replace('[PE]', str(r['pe'])).replace('[PB]', str(r['pb'])).replace('[ROE]', str(r['roe']))
    p_encoded = requests.utils.quote(p_str)
    
    grok_url = f"https://x.com/i/grok?text={p_encoded}"
    perp_url = f"https://www.perplexity.ai/?q={p_encoded}"
    
    ai_btns = f"""
    <div style="display:flex;gap:4px">
      <a href="{grok_url}" target="_blank" class="grok-btn">𝕏 Grok</a>
      <a href="{perp_url}" target="_blank" class="grok-btn" style="color:white;border-color:#555">Perplexity</a>
      <button onclick="copyAndOpenGemini(`{p_str}`)" class="grok-btn" style="color:#a855f7;border-color:#a855f7">✨ Gemini</button>
    </div>
    """
    
    rec_rows+=f"""<tr style="border-bottom:1px solid var(--slate700); cursor:pointer;" onclick="renderRecChart('{r['ticker']}', {i})" id="rec-row-{i}">
    <td class="p-2" style="font-weight:700;color:white">{r['ticker']}</td>
    <td class="p-2 mono" style="color:var(--yellow400)">${r['price']}</td>
    <td class="p-2 mono" style="{rc};font-weight:700">+{r['excess_rs']}%</td>
    <td class="p-2" style="font-size:11px;color:var(--slate300)">股息:{r['div']}% | PE:{r['pe']} | ROE:{r['roe']}%</td>
    <td class="p-2">{ai_btns}</td></tr>"""

html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>US Quant Master V16.2 Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+TC:wght@400;700&display=swap');
:root{{--slate900:#0f172a;--slate800:#1e293b;--slate700:#334155;--slate400:#94a3b8;--cyan400:#22d3ee;--cyan500:#06b6d4;--blue600:#2563eb;--green400:#4ade80;--red400:#f87171;--yellow400:#facc15}}
body{{background:#0a0f1a;color:#e2e8f0;margin:0;overflow:hidden;font-family:'Noto Sans TC',sans-serif;font-size:13px}}
.mono{{font-family:'JetBrains Mono',monospace}}
.tab-btn{{color:var(--slate400);border-bottom:2px solid transparent;transition:0.2s;cursor:pointer;padding:0 12px;height:100%;font-weight:700}}
.tab-btn:hover{{color:white}}
.tab-btn.active{{border-color:var(--cyan500);color:var(--cyan400);background:rgba(6,182,212,.1)}}
.tab-content{{display:none;height:calc(100vh - 52px);width:100%;padding:16px;gap:16px;overflow:auto}}
.card{{background:linear-gradient(135deg,#111827,#0f172a);border:1px solid var(--slate800);border-radius:10px;padding:16px;display:flex;flex-direction:column}}
.grok-btn{{padding:4px 8px;border-radius:6px;font-size:11px;font-weight:700;border:1px solid var(--slate700);background:#1e293b;cursor:pointer;text-decoration:none;transition:0.1s}}
.grok-btn:hover{{background:var(--slate700)}}
.row-active{{background:rgba(6,182,212,.15);border-left:3px solid var(--cyan500)}}
th{{cursor:pointer;user-select:none;position:sticky;top:0;background:#1e293b;padding:8px;text-align:left;color:var(--slate400)}}
th:hover{{color:white}}
</style>
<script>
function switchTab(id, btn) {{
    document.querySelectorAll('.tab-content').forEach(e => e.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
    document.getElementById(id).style.display = 'flex';
    if(btn) btn.classList.add('active');
}}

function copyAndOpenGemini(prompt) {{
    navigator.clipboard.writeText(prompt).then(() => {{
        alert("✨ 提示詞已自動複製！\\n請在接下來開啟的 Gemini 視窗中按下 Ctrl+V (貼上) 即可提問。");
        window.open("https://gemini.google.com/app", "_blank");
    }}).catch(err => alert("複製失敗，請手動複製。"));
}}
</script>
</head>
<body class="flex flex-col h-screen">

<div style="height:52px;background:#0d1321;border-bottom:1px solid var(--slate800);display:flex;align-items:center;padding:0 16px;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:30px;height:30px;background:var(--cyan600);border-radius:8px;display:flex;align-items:center;justify-content:center;font-weight:900;">Q</div>
    <span style="font-size:15px;font-weight:700;">US Quant Master</span>
  </div>
  <div style="display:flex;height:100%;">
    <button class="tab-btn active" onclick="switchTab('tab-spx', this)">🏛️ 大盤 (SPY)</button>
    <button class="tab-btn" onclick="switchTab('tab-rec', this)">🎯 龍頭推薦 (AI對話)</button>
  </div>
</div>

<div id="tab-spx" class="tab-content" style="display:flex;flex-direction:column;">
  <div class="card" style="flex:1;">
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <h2 style="font-weight:700;">SPY 近5年趨勢 (自動著色階段)</h2>
        <span style="font-size:11px;color:var(--slate400)">🟢 Stage 2 (多頭) | 🔴 Stage 4 (熊市) | 🟡 Stage 1/3 (盤整)</span>
    </div>
    <div style="position:relative;flex:1;"><canvas id="spxChart"></canvas></div>
  </div>
</div>

<div id="tab-rec" class="tab-content" style="flex-direction:row;">
  <div class="card" style="flex:1.2;overflow:hidden;padding:0;">
    <div style="padding:12px;border-bottom:1px solid var(--slate700);font-weight:700;color:var(--cyan400)">🎯 量化精選 (可點擊欄位排序)</div>
    <div style="flex:1;overflow:auto;">
        <table style="width:100%;white-space:nowrap;font-size:12px" id="recTable">
            <thead>
                <tr>
                    <th onclick="sortRec(0)">代號 ↕</th>
                    <th onclick="sortRec(1)">價格 ↕</th>
                    <th onclick="sortRec(2)">超額RS ↕</th>
                    <th>基本面概況</th>
                    <th>AI 分析</th>
                </tr>
            </thead>
            <tbody id="recTableBody">{rec_rows}</tbody>
        </table>
    </div>
  </div>
  
  <div class="card" style="flex:1;display:flex;flex-direction:column;gap:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2 id="recChartTitle" style="font-size:18px;font-weight:700;color:var(--yellow400)">請選擇左側個股</h2>
        <div style="display:flex;gap:4px">
            <button onclick="changeRec(-1)" class="grok-btn">◀ 預覽上一檔</button>
            <button onclick="changeRec(1)" class="grok-btn">預覽下一檔 ▶</button>
        </div>
    </div>
    <p id="recChartInfo" style="font-size:12px;color:var(--slate400);margin:0;">近一年績效對比 SPY</p>
    <div style="position:relative;flex:1;"><canvas id="recCanvas"></canvas></div>
  </div>
</div>

<script>
// Data Injection
const spxDates = {json.dumps(spx_chart_dates)};
const spxClose = {json.dumps(spx_chart_close)};
const spxStages = {spxStagesJson};
const recData = {json_rec_chart_data};
const recList = {json_rec_list};
let currentRecIdx = 0;
let spxChartInst, recChartInst;

// SPX Chart Rendering with Try-Catch
try {{
    const ctx = document.getElementById('spxChart').getContext('2d');
    spxChartInst = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: spxDates,
            datasets: [{{
                label: 'SPY',
                data: spxClose,
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.1,
                segment: {{
                    borderColor: ctx => {{
                        if (!ctx.p1DataIndex) return '#3b82f6';
                        const s = spxStages[ctx.p1DataIndex];
                        return s === 2 ? '#4ade80' : (s === 4 ? '#f87171' : '#fbbf24');
                    }}
                }}
            }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, interaction: {{mode: 'index', intersect: false}}, plugins:{{legend:{{display:false}}}} }}
    }});
}} catch(e) {{ console.error("SPX Chart Error:", e); }}

// Rec Chart Rendering
function renderRecChart(ticker, idx) {{
    currentRecIdx = idx;
    document.querySelectorAll('tr[id^="rec-row-"]').forEach(el => el.classList.remove('row-active'));
    const row = document.getElementById('rec-row-' + idx);
    if(row) row.classList.add('row-active');
    
    const d = recData[ticker];
    if(!d) return;
    const info = recList[idx];
    
    document.getElementById('recChartTitle').innerText = ticker;
    document.getElementById('recChartInfo').innerHTML = `績效: <span style="color:var(--green400)">+${{d.tk_pct[d.tk_pct.length-1]}}%</span> vs SPY (${{d.spy_pct[d.spy_pct.length-1]}}%) <br> 股息: ${{info.div}}%, 動能: ${{info.mom}}%, P/E: ${{info.pe}} | P/B: ${{info.pb}} | ROE: ${{info.roe}}%`;

    try {{
        if(recChartInst) recChartInst.destroy();
        recChartInst = new Chart(document.getElementById('recCanvas').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: d.dates,
                datasets: [
                    {{ label: ticker + ' (%)', data: d.tk_pct, borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0 }},
                    {{ label: 'SPY (%)', data: d.spy_pct, borderColor: '#64748b', borderWidth: 1, borderDash:[4,4], pointRadius: 0 }}
                ]
            }},
            options: {{ responsive: true, maintainAspectRatio: false, interaction: {{mode: 'index', intersect: false}} }}
        }});
    }} catch(e) {{ console.error("Rec Chart Error:", e); }}
}}

function changeRec(step) {{
    let newIdx = currentRecIdx + step;
    if(newIdx < 0) newIdx = recList.length - 1;
    if(newIdx >= recList.length) newIdx = 0;
    renderRecChart(recList[newIdx].ticker, newIdx);
}}

// Sorting
let sortAsc = false;
function sortRec(n) {{
    const tbody = document.getElementById("recTableBody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    sortAsc = !sortAsc;
    rows.sort((a, b) => {{
        let v1 = a.cells[n].innerText.replace(/[%\\$\\+]/g, '').trim();
        let v2 = b.cells[n].innerText.replace(/[%\\$\\+]/g, '').trim();
        let n1 = parseFloat(v1), n2 = parseFloat(v2);
        if(!isNaN(n1) && !isNaN(n2)) return sortAsc ? n1 - n2 : n2 - n1;
        return sortAsc ? v1.localeCompare(v2) : v2.localeCompare(v1);
    }});
    rows.forEach((r, i) => {{
        r.id = 'rec-row-' + i; // 重新綁定 ID 以配合上下切換
        r.setAttribute('onclick', `renderRecChart('${{r.cells[0].innerText}}', ${{i}})`);
        tbody.appendChild(r);
    }});
    // 更新 recList 陣列順序
    const newList = [];
    rows.forEach(r => {{
        const tk = r.cells[0].innerText;
        newList.push(recList.find(x => x.ticker === tk));
    }});
    for(let i=0; i<recList.length; i++) recList[i] = newList[i];
    renderRecChart(recList[0].ticker, 0); // 重新渲染第一個
}}

// Init
if(recList.length > 0) renderRecChart(recList[0].ticker, 0);

</script>
</body>
</html>"""

filename = "US_Quant_Master_V16_2.html"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n{'='*60}\n 🎉 V16.2 儀表板已生成：{filename}\n{'='*60}")
try:
    from google.colab import files
    files.download(filename)
    print("📥 自動下載已觸發！")
except: pass
