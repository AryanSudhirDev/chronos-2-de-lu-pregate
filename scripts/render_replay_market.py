"""Visualize retained forecasts, realized prices and the fixed naive benchmark."""
import json
import pandas as pd
from epf.data import ROOT

out=ROOT/'runs/replay'
s=json.loads((out/'market_summary.json').read_text())
c=s['comparison']
d=pd.read_csv(out/'market_daily.csv').fillna(0)
h=pd.read_csv(out/'hourly.csv')
b=pd.read_csv(out/'market_comparison.csv')
daily=d[['day','actual_mean','forecast_mean','margin_eur']].values.tolist()
hourly=h[['day','berlin_hour','actual_eur_mwh','forecast_eur_mwh']].values.tolist()
comparison=b[['day','model_margin_eur','benchmark_margin_eur']].values.tolist()
down=next(x for x in s['groups'] if x['regime']=='falling')
table='\n'.join(f"| {g['regime']} | {g['n_days']} | {g['margin_eur']:,.2f} | {g['net_eur']:,.2f} |" for g in s['groups'])
md=f"""# Forecasts versus the electricity market

The first 30 scored-day average actual price was EUR {s['first30_mean']:.2f}/MWh
({s['first30_dates'][0]}..{s['first30_dates'][1]}). The last 30 scored-day average was EUR
{s['last30_mean']:.2f}/MWh ({s['last30_dates'][0]}..{s['last30_dates'][1]}), up {s['market_level_pct']:.1f}%.
Single-day endpoints were EUR {s['market_first_day']:.2f} and {s['market_last_day']:.2f}; the last day
was unusually expensive. Spot price change is not an investable buy-and-hold return or battery ROI.

The battery starts and ends every day empty. It cannot earn from holding electricity from November
2025 to November 2026. It earns from within-day prices and is exposed to conversion losses and costs.
Rising price levels do not mechanically create profit; opportunities depend on the hourly price pattern.
This does not prove independence from market conditions: predictable daily spreads help simple policies too.

## Descriptive rising/falling-day check (370 original replay days)

Correction: this isolated-day grouping does not establish resilience through sustained weak
periods. The November–February simulation lost EUR 6,560.84 after assumed costs; even the
constrained perfect-foresight schedule lost EUR 5,669.45. See diagnosis.md and diagnosis.html
for all monthly outcomes, drawdown/cost attribution, and clarification of fixed versus adaptive models.

A falling day means its actual daily average is below the preceding calendar day's daily average.
Groups are defined after outcomes, for description only; they never control decisions.
Four days cannot be classified because the preceding calendar day is outside the sample or missing.

| Daily average price direction | Days | Margin after variable costs EUR | After allocated fixed costs EUR |
|---|---:|---:|---:|
{table}

The {down['n_days']} falling days still earned EUR {down['net_eur']:,.2f} after allocated fixed costs;
{down['positive_variable_margin_days']} had positive margin before fixed costs. Four entirely skipped
calendar days add EUR 328.54 fixed costs outside this table, as in the original replay.
This grouping is a retrospective association, not a causal market-adjusted alpha estimate.

## Causal simple forecast comparison ({c['n_common_days']} identical days)

The benchmark uses the study's existing naive rule: forecast yesterday's hourly prices, except
Monday/Saturday/Sunday use the same weekday last week. This rule predates the original heldout study.
All those price curves were cleared before the simulated decision deadline. No parameters were fitted.
The same battery dispatch, fees, wear and fixed costs apply. Every benchmark schedule was saved
before its target-day settlement; its OS worker sees only lagged known prices and assumptions.

| Policy | Price MAE EUR/MWh | Margin after variable costs EUR | After assumed fixed costs EUR |
|---|---:|---:|---:|
| Model | {c['model_mae']:.2f} | {c['model_margin_eur']:,.2f} | {c['model_net_eur']:,.2f} |
| Naive | {c['benchmark_mae']:.2f} | {c['benchmark_margin_eur']:,.2f} | {c['benchmark_net_eur']:,.2f} |

Incremental model margin: EUR {c['incremental_margin_eur']:,.2f},
{100*c['incremental_margin_eur']/c['benchmark_margin_eur']:.1f}% over the naive variable-cost margin.
Much of the simulated opportunity is also captured by the simple benchmark. The model adds a smaller
increment; profitability alone is not evidence that all its earnings came from forecasting skill.
This benchmark was added after holdout inspection as a diagnostic; no new untouched test is claimed.
It is not a comparison against the strongest equally informed LEAR/LightGBM baselines.

The common comparison excludes three extra target days (2025-11-02, 2026-04-05, 2026-11-01),
whose naive lagged source days were 25/23/25-hour DST days. Both policies are assumed idle on all
seven excluded target days and both bear EUR {c['fixed_cost_eur']:,.2f} fixed costs across all 374 days.
Consequently the model comparison total differs from the earlier 370-day result.

## Evidence limits

These remain retrospective simulated battery margins, not real returns. Historical publication
versions, original model pretraining, quarter-hour execution, real asset costs, outages, taxes and
imbalance exposure remain unverified or excluded. The original forecasts/ledger/paper policy are
unchanged. No retraining, strategy tuning, live orders, trades or bids were performed.

The interactive graph defaults to daily average forecasts versus daily average actual prices.
Daily averages hide hourly errors; select hourly and a shorter date range to inspect those.
The lower graph compares cash margins in EUR on identical days, on a separate axis from EUR/MWh prices.
Files: market_report.html, market_daily.csv, market_comparison.csv, market_benchmark_decisions.json,
market_protocol.json and market_summary.json. Run scripts/replay_market.py then scripts/render_replay_market.py.
"""
(out/'market_report.md').write_text(md)
html='''<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}p{line-height:1.5}.muted,small{color:var(--muted-foreground)}h3{font-size:15px;margin:20px 0 8px}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{flex:1;min-width:160px;background:var(--card);padding:16px;border:1px solid var(--border);border-radius:var(--radius)}b.value{display:block;font-size:23px;margin:6px 0}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}input,select,button{font:inherit;background:var(--secondary);color:var(--foreground);border:1px solid var(--border);padding:7px;border-radius:6px;max-width:100%}svg{height:230px;width:100%;display:block}.axis{font-size:11px;fill:var(--muted-foreground)}.legend{font-size:12px;margin:8px 0}.note{color:var(--warning);font-size:12px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--border)}td:first-child,th:first-child{text-align:left}#hover{min-height:20px;font-size:12px}.explain{background:var(--card);padding:16px;border-radius:var(--radius);border:1px solid var(--border)}
</style>
<p><b>Forecasts versus actual electricity prices</b><br><small>Pre-gate model · Germany–Luxembourg · 1 Nov 2025 – 9 Nov 2026</small></p>
<div class="cards"><div class="card">Market price level<b class="value">+MARKET%</b><small>First vs last 30 evaluated-day average. Not an investment return.</small></div><div class="card">Model advantage over naive<b class="value">€EDGE</b><small>Same 367 days, same battery and costs.</small></div><div class="card">On falling-price days<b class="value">€DOWN</b><small>198 days, after allocated variable and fixed costs.</small></div></div>
<h3>Predicted and actual price · EUR/MWh</h3>
<div class="controls"><label>View <select id="frequency"><option value="daily">Daily average</option><option value="hourly">Every hour</option></select></label><label>From <input id="start" type="date" min="2025-11-01" max="2026-11-09" value="2025-11-01"></label><label>To <input id="end" type="date" min="2025-11-01" max="2026-11-09" value="2026-11-09"></label><button id="all">Full period</button></div>
<p class="legend"><span style="color:var(--chart-1)">● Forecast</span> &nbsp; <span style="color:var(--chart-2)">● Actual market price</span> · Hover for values</p>
<svg id="prices" viewBox="0 0 680 230" role="img" aria-label="Overall predicted versus actual electricity prices"></svg><div id="hover" class="muted"></div>
<p class="muted" id="range"></p>
<h3>Did the model add value beyond a simple forecast?</h3>
<p class="legend"><span style="color:var(--chart-1)">● Model</span> &nbsp; <span style="color:var(--chart-3)">● Naive: repeat earlier cleared prices</span> · Cumulative EUR after assumed costs</p>
<svg id="cash" viewBox="0 0 680 230" role="img" aria-label="Model versus naive cumulative simulated margin"></svg>
<p class="muted">Like-for-like: 367 evaluated days; both idle on 7 excluded days. Both pay €30,000/year fixed costs across all 374 calendar days. Model €MODELNET; naive €NAIVENET. The earlier model-only total used 370 evaluated days.</p>
<div class="explain"><b>Falling-day totals do not prove resilience through a losing period.</b><p>The battery is empty overnight: it buys and sells within each day. On 198 scattered days when average prices fell, it earned €2,153 after allocated costs. But across November–February it lost €6,561. Even the constrained perfect-foresight schedule lost €5,669 in those months. The isolated-day result was too reassuring about sustained performance.</p><p>The model added €EDGE above the naive forecast—about 9.7% more margin after variable costs. Much of the available opportunity came from daily price swings that the naive rule also captured. The follow-up diagnosis separates market levels, daily spreads and costs.</p></div>
<p class="note">Retrospective simulation, not proven live returns. Historical input revisions, model pretraining and actual execution remain unverified. Daily averages hide hourly forecast errors: select “Every hour” and shorten the date range. All market trends use the 370 retained replay days.</p>
<script>
const daily=DAILY,hourly=HOURLY,comp=COMPARISON,byDay=Object.fromEntries(comp.map(r=>[r[0],r]));
const stamp=day=>Date.parse(day+'T00:00:00Z'),fmt=v=>v.toLocaleString(undefined,{maximumFractionDigits:1});
function chart(id,series,gap,interactive=false){const svg=document.getElementById(id),left=59,right=15,top=12,bottom=30,w=680,h=230;const points=series.flatMap(s=>s.points);if(!points.length){svg.innerHTML='<text x="60" y="90" class="axis">No evaluated hours in this range</text>';return}let minX=Math.min(...points.map(p=>p[0])),maxX=Math.max(...points.map(p=>p[0])),minY=Math.min(0,...points.map(p=>p[1])),maxY=Math.max(0,...points.map(p=>p[1]));if(minX===maxX)maxX+=86400000;if(minY===maxY)maxY=minY+1;let x=t=>left+(t-minX)*(w-left-right)/(maxX-minX),y=v=>h-bottom-(v-minY)*(h-top-bottom)/(maxY-minY),html='';for(let i=0;i<4;i++){let v=minY+i*(maxY-minY)/3;html+=`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"/><text class="axis" x="${left-5}" y="${y(v)+4}" text-anchor="end">${Math.round(v).toLocaleString()}</text>`}for(const s of series){let prev=null,path='';for(const p of s.points){path+=(prev&&p[0]-prev[0]<=gap?'L':'M')+x(p[0])+','+y(p[1]);prev=p}html+=`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="1.7"/>`}for(let i=0;i<3;i++){const t=minX+i*(maxX-minX)/2;html+=`<text class="axis" x="${x(t)}" y="225" text-anchor="${i===0?'start':i===2?'end':'middle'}">${new Date(t).toISOString().slice(0,10)}</text>`}svg.innerHTML=html;if(interactive)svg.onmousemove=e=>{let t=minX+((e.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width*w-left)/(w-left-right)*(maxX-minX),items=series.map(s=>s.points.reduce((a,p)=>Math.abs(p[0]-t)<Math.abs(a[0]-t)?p:a)),p=items[0];document.getElementById('hover').textContent=p[2]+' · forecast €'+fmt(p[1])+' · actual €'+fmt(items[1][1])+'/MWh'};}
function draw(){const from=document.getElementById('start').value,to=document.getElementById('end').value,isHourly=document.getElementById('frequency').value==='hourly';const selected=(isHourly?hourly:daily).filter(r=>r[0]>=from&&r[0]<=to);const get=(col)=>selected.map(r=>[stamp(r[0])+(isHourly?r[1]*3600000:0),r[col],r[0]+(isHourly?' '+String(r[1]).padStart(2,'0')+':00 Berlin':' daily average')]);chart('prices',[{points:get(isHourly?3:2),color:'var(--chart-1)'},{points:get(isHourly?2:1),color:'var(--chart-2)'}],isHourly?5400000:129600000,true);const mae=selected.length?selected.reduce((sum,r)=>sum+Math.abs(r[isHourly?3:2]-r[isHourly?2:1]),0)/selected.length:0;document.getElementById('range').textContent=selected.length+' '+(isHourly?'hourly':'daily-average')+' pairs · mean absolute difference €'+fmt(mae)+'/MWh'+(isHourly?'':' (daily averaging can hide hourly errors)');document.getElementById('hover').textContent='';}
for(const id of ['start','end','frequency'])document.getElementById(id).onchange=draw;document.getElementById('all').onclick=()=>{document.getElementById('start').value='2025-11-01';document.getElementById('end').value='2026-11-09';draw()};draw();
let model=0,naive=0,mp=[],np=[];for(let t=stamp('2025-11-01');t<=stamp('2026-11-09');t+=86400000){let day=new Date(t).toISOString().slice(0,10),r=byDay[day];model+=(r?r[1]:0)-30000/365.25;naive+=(r?r[2]:0)-30000/365.25;mp.push([t,model]);np.push([t,naive])}chart('cash',[{points:mp,color:'var(--chart-1)'},{points:np,color:'var(--chart-3)'}],129600000);
</script>'''
replacements={'MARKET':f"{s['market_level_pct']:.1f}",'EDGE':f"{c['incremental_margin_eur']:,.0f}",'DOWN':f"{down['net_eur']:,.0f}",'MODELNET':f"{c['model_net_eur']:,.0f}",'NAIVENET':f"{c['benchmark_net_eur']:,.0f}",'DAILY':json.dumps(daily,separators=(',',':')),'HOURLY':json.dumps(hourly,separators=(',',':')),'COMPARISON':json.dumps(comparison,separators=(',',':'))}
for key,value in replacements.items():html=html.replace(key,value)
(out/'market_report.html').write_text(html)
print(out/'market_report.html')
