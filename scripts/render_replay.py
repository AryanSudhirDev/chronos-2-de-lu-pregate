"""Render the existing replay artifacts; no outcome collection or model calls."""
import json
from pathlib import Path
from epf.data import ROOT

out = ROOT/'runs/replay'
s = json.loads((out/'summary.json').read_text())
p = json.loads((out/'protocol.json').read_text())
hourly = json.loads((out/'hourly.json').read_text())
t = s['totals']
grouped = {}
for row in hourly:
    grouped.setdefault(row['day'], []).append(row)
data = [dict(day=d['day'], margin=d['margin_eur'], mae=d['mae'],
             f=[round(r['forecast_eur_mwh'],3) for r in grouped[d['day']]],
             a=[round(r['actual_eur_mwh'],3) for r in grouped[d['day']]],
             g=[round(r['grid_mwh'],5) for r in grouped[d['day']]]) for d in s['days']]
report = f"""# Historical pre-gate decision replay

This is a retrospective diagnostic of forecasts already produced during the one-time holdout opening.
It is not a fresh unseen experiment or evidence of live execution. No model was retrained or selected.

## Results

Window: {s['window'][0]} through {s['window'][1]}; {s['n_days']} scored days out of {s['calendar_days']} calendar days.
Forecast MAE: {s['mae']:.2f} EUR/MWh. Physical battery: 1 MW grid power / 1 MWh capacity,
95% efficiency each way, empty at start/end of each day, maximum two full discharge cycles/day,
SOC grid 0.1 MWh. Positive grid energy is a purchase; negative is a sale.

| Item | EUR |
|---|---:|
| Energy cashflow at actual historical hourly average prices | {t['gross_eur']:,.2f} |
| Assumed variable fees/slippage and wear | -{t['variable_cost_eur']:,.2f} |
| Margin after variable costs | {t['margin_eur']:,.2f} |
| Assumed fixed costs for all {s['calendar_days']} calendar days | -{t['fixed_cost_eur']:,.2f} |
| Illustrative remaining margin | {t['scenario_net_eur']:,.2f} |

Fees/slippage: EUR 3/grid MWh bought or sold; wear: EUR 20/discharged battery MWh.
Fixed costs: EUR 30,000/year, prorated at 365.25 days/year, including skipped days.
These are assumptions, not asset quotes or actual net trading returns.
{s['profitable_days']} days had positive margin after variable costs, {s['losing_days']} negative,
and {s['n_days']-s['profitable_days']-s['losing_days']} zero. These day counts exclude fixed costs.
No-action energy margin is zero before the same fixed-cost assumption.
Scorer-only perfect-foresight margin after variable costs: EUR {t['oracle_margin_eur']:,.2f}; never supplied to the decision worker.

## What prevents looking ahead

1. Reused forecast 15830042bf. Every one of its 372 daily forecasts exactly matches the saved
   median of the hashed original issued-forecast archive. No model rerun, tuning or ledger append.
2. Separate `issue` invocation has an audit hook rejecting all data/ and ledger reads.
   An OS-sandboxed decision worker cannot read raw outcomes, ledger, forecast files or network;
   its pipe accepts only that day's 24 forecast prices and the fixed battery assumptions.
3. Persisted all 370 decisions with exclusive creation and fsync before invoking `score`.
   Protocol, source forecast, original archive, original manifest and dispatch-code hashes are checked.
4. The scorer alone reads the actual price column. It requires exactly the 24 hourly timestamps
   for each Berlin delivery day, checks battery power/SOC/cycles, and settles the unchanged schedule.
   A UTC/Berlin representation mismatch was caught by this validation and fixed; missing/duplicate
   timestamp tests now cover it. No outcome rows were filled or shifted to bypass the check.
5. Synthetic poisoning checks alter current/future targets and current post-gate wind/solar;
   pre-gate prediction inputs stay exactly unchanged. Actual prices are rejected by the dispatch API.

## Exclusions and evidence limits

Missing forecasts: 2026-04-10, 2026-04-11. DST days excluded: 2026-03-29, 2026-10-25.
Skipped days are assumed idle, while fixed costs still accrue.
The original holdout normalized DST days; this execution replay deliberately excludes them.
Only pre-gate LoRA is replayed here. The better-scoring post-gate blend uses late TSO inputs
and cannot support an auction-time claim. Equally informed pre-gate baseline forecasts were
not retained for this full holdout; their dev-only comparison is in runs/readiness/report.md.

""" + '\n'.join('- '+x for x in p['limits']) + """

The archive hashes prove consistency with retained artifacts, not their historical live publication.
The trusted controller remains outside the OS sandbox. The original model may have been pretrained
on data whose exact provenance is unknown. Historical input archives may include later revisions.
These limits prevent calling this a fully certified leakage-free historical trading test.
The scheduled forward paper stream collects timestamped inputs before future auctions to address
the historical publication uncertainty. It does not place trades or bids.

## Inspect or reproduce

- `hourly.csv`: every buy/sell/hold, forecast, actual price, grid MWh, SOC, cashflow and variable cost.
- `daily.csv`: daily accuracy and economics. Fixed costs for the four skipped days are additionally
  accounted for in summary.json; summing the scored-day CSV alone excludes that EUR 328.54.
- `decisions.json`: frozen forecast-only schedules with source/code hashes.
- `summary.json`: totals, actual-source hash and exclusions.
- `report.html`: interactive day selector, forecast/actual chart and actions.

Run `.venv/bin/python scripts/replay_holdout.py issue`, then a separate
`.venv/bin/python scripts/replay_holdout.py score`. Existing differing artifacts cause refusal.
Run `.venv/bin/python scripts/render_replay.py` to rebuild this presentation.
"""
(out/'report.md').write_text(report)
html = '''<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}p{line-height:1.5}small,.muted{color:var(--muted-foreground)}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{flex:1;min-width:170px;background:var(--card);border:1px solid var(--border);border-radius:var(--radius,10px);padding:16px}.card b{display:block;font-size:23px;margin-top:8px}select,button{background:var(--secondary);color:var(--foreground);border:1px solid var(--border);padding:8px;border-radius:6px}svg{width:100%;display:block;height:175px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;border-bottom:1px solid var(--border);padding:6px}th:first-child,td:first-child{text-align:left}.wrap{max-height:225px;overflow:auto}h3{font-size:15px;margin:18px 0 8px}.note{color:var(--warning);font-size:12px} .axis{fill:var(--muted-foreground);font-size:11px}
</style>
<p><b>Historical battery replay · pre-gate forecasts</b><br><small>1 Nov 2025 – 9 Nov 2026 · 370 scored days · illustrative 1 MW / 1 MWh battery</small></p>
<div class="cards"><div class="card">Energy cashflow<b>€GROSS</b></div><div class="card">After variable costs<b>€MARGIN</b></div><div class="card">After assumed fixed costs<b>€NET</b></div></div>
<p class="note">Retrospective simulation. Archive publication vintages and original model pretraining are unverified. Hourly fills and costs are assumptions; these figures are not actual returns.</p>
<h3>Cumulative margin after all assumed costs</h3><svg id="cum" viewBox="0 0 650 175" role="img" aria-label="Cumulative simulated margin"></svg>
<h3>Inspect a delivery day</h3><div style="display:flex;gap:8px;align-items:center"><button id="prev" aria-label="Previous day">←</button><select id="day" aria-label="Delivery day"></select><button id="next" aria-label="Next day">→</button></div>
<p id="details"></p><small><span style="color:var(--chart-1)">● Forecast</span> &nbsp; <span style="color:var(--chart-2)">● Actual price</span> · EUR/MWh · Berlin hours</small>
<svg id="prices" viewBox="0 0 650 175" role="img" aria-label="Forecast and actual hourly prices"></svg>
<div class="wrap"><table><thead><tr><th>Hour</th><th>Action</th><th>MWh</th><th>Forecast €</th><th>Actual €</th></tr></thead><tbody id="actions"></tbody></table></div>
<p><small>Buy/sell schedules use forecasts only. 95% efficiency each way, at most 2 cycles/day. Fees/slippage €3/grid MWh; wear €20/discharged MWh; fixed costs €30,000/year across all 374 calendar days. Four skipped days assumed idle. No live trades.</small></p>
<script>
const days=DATA, fixed=30000/365.25;
function chart(id,lines,labels){const svg=document.getElementById(id),w=650,h=175,l=58,r=12,t=12,b=26;let vals=lines.flatMap(x=>x.v),lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(hi===lo)hi=lo+1;const x=(i,n)=>l+i*(w-l-r)/Math.max(1,n-1),y=v=>h-b-(v-lo)*(h-t-b)/(hi-lo);let out='';for(let i=0;i<3;i++){let v=lo+(hi-lo)*i/2;out+=`<line x1="${l}" x2="${w-r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"/><text class="axis" x="${l-6}" y="${y(v)+4}" text-anchor="end">${Math.round(v).toLocaleString()}</text>`}for(const line of lines)out+=`<polyline fill="none" stroke="${line.color}" stroke-width="2" points="${line.v.map((v,i)=>`${x(i,line.v.length)},${y(v)}`).join(' ')}"/>`;for(let i=0;i<labels.length;i++)out+=`<text class="axis" x="${l+i*(w-l-r)/Math.max(1,labels.length-1)}" y="172" text-anchor="${i===0?'start':i===labels.length-1?'end':'middle'}">${labels[i]}</text>`;svg.innerHTML=out}
let curve=[],balance=0,byDay=Object.fromEntries(days.map(d=>[d.day,d]));for(let date=new Date('2025-11-01T00:00:00Z');date<=new Date('2026-11-09T00:00:00Z');date.setUTCDate(date.getUTCDate()+1)){balance+=(byDay[date.toISOString().slice(0,10)]?.margin||0)-fixed;curve.push(balance)}chart('cum',[{v:curve,color:'var(--chart-1)'}],['Nov 2025','May 2026','Nov 2026']);
const select=document.getElementById('day');for(const d of days)select.add(new Option(d.day,d.day));
function draw(){const d=days[select.selectedIndex];document.getElementById('details').textContent=`Margin after variable costs: €${d.margin.toFixed(2)} · MAE: €${d.mae.toFixed(2)}/MWh`;chart('prices',[{v:d.f,color:'var(--chart-1)'},{v:d.a,color:'var(--chart-2)'}],['00:00','23:00']);document.getElementById('actions').innerHTML=d.g.map((g,h)=>`<tr><td>${String(h).padStart(2,'0')}:00</td><td>${g>0?'Buy':g<0?'Sell':'Hold'}</td><td>${Math.abs(g).toFixed(3)}</td><td>${d.f[h].toFixed(2)}</td><td>${d.a[h].toFixed(2)}</td></tr>`).join('')}
select.onchange=draw;document.getElementById('prev').onclick=()=>{select.selectedIndex=Math.max(0,select.selectedIndex-1);draw()};document.getElementById('next').onclick=()=>{select.selectedIndex=Math.min(days.length-1,select.selectedIndex+1);draw()};draw();
</script>'''
html = html.replace('GROSS',f"{t['gross_eur']:,.0f}").replace('MARGIN',f"{t['margin_eur']:,.0f}").replace('NET',f"{t['scenario_net_eur']:,.0f}").replace('DATA',json.dumps(data,separators=(',',':')))
(out/'report.html').write_text(html)
print(out/'report.html')
