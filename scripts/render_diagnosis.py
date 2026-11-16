"""Present the cost/drawdown diagnosis without any model or policy changes."""
import json
import pandas as pd
from epf.data import ROOT

out=ROOT/'runs/replay'
s=json.loads((out/'diagnosis.json').read_text())
d=pd.read_csv(out/'diagnosis_daily.csv')
m=pd.read_csv(out/'diagnosis_monthly.csv')
daily=d[['date','cumulative_net','cumulative_oracle_net','cumulative_variable_margin']].values.tolist()
monthly=m[['month','mean_price','mean_spread','net','oracle_net']].values.tolist()
rows=''.join(f"<tr><td>{r.month[:7]}</td><td>{r.mean_price:.0f}</td><td>{r.mean_spread:.0f}</td><td style='color:var({'--destructive' if r.net<0 else '--success'})'>€{r.net:,.0f}</td></tr>" for r in m.itertuples())
html='''<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}p{line-height:1.5}h3{font-size:15px;margin:20px 0 8px}small,.muted{color:var(--muted-foreground)}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:16px;flex:1;min-width:165px}.card b{display:block;font-size:23px;margin-top:6px}.negative{color:var(--destructive)}svg{height:220px;width:100%;display:block}.axis{font-size:11px;fill:var(--muted-foreground)}.legend{font-size:12px;margin:8px 0}.note{font-size:12px;color:var(--warning)}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;text-align:right;border-bottom:1px solid var(--border)}td:first-child,th:first-child{text-align:left}.table{max-height:225px;overflow:auto}.box{background:var(--card);padding:16px;border:1px solid var(--border);border-radius:var(--radius);margin-top:18px}label{font-size:12px}
</style>
<p><b>The simulation had a substantial losing period</b><br><small>Existing pre-gate forecasts and decisions · no retraining or new strategy</small></p>
<div class="cards"><div class="card">Nov–Feb after assumed costs<b class="negative">−€6,561</b></div><div class="card">Deepest cumulative loss<b class="negative">−€6,564</b><small>1 March 2026</small></div><div class="card">Back above starting zero<b>26 June</b><small>2026 · cash margin, not investment ROI</small></div></div>
<h3>Cash margin after variable and fixed costs · EUR</h3>
<p class="legend"><span style="color:var(--chart-1)">● Model</span> &nbsp; <span style="color:var(--chart-3)">● Perfect foresight, same battery</span></p>
<label><input id="before" type="checkbox"> Also show model margin before fixed costs</label>
<svg id="cash" viewBox="0 0 680 220" role="img" aria-label="Model and perfect foresight cumulative cash margins"></svg>
<p class="muted">The first four months earned €3,295 after variable costs, below €9,856 assumed fixed costs. Even perfect forecasts lost €5,669 in those months. Later earnings recovered the loss; the full-period model total was +€8,105.</p>
<h3>Average price and daily price range · EUR/MWh</h3>
<p class="legend"><span style="color:var(--chart-2)">● Monthly mean price</span> &nbsp; <span style="color:var(--chart-4)">● Monthly mean of each day's high–low range</span></p>
<svg id="market" viewBox="0 0 680 220" role="img" aria-label="Monthly electricity price level and within-day high-low spread"></svg>
<p class="muted">The within-day range was much smaller in the early losing months. April prices fell but the model earned +€1,329 after assumed costs. January prices rose but the model lost €1,597. Price direction alone does not explain profit.</p>
<div class="table"><table><thead><tr><th>Month</th><th>Mean price</th><th>Daily range</th><th>Model after costs</th></tr></thead><tbody>ROWS</tbody></table></div>
<div class="box"><b>What “learning” means in this experiment</b><p>The main Chronos model is fixed. Each forecast receives fresh recent inputs (85 days), so it adapts its prediction without updating its trained weights. This graph combines separate day-ahead forecasts, not a year-ahead forecast.</p><p>The separate adaptive blend changes model weights using earlier errors. It is not the pre-gate model replayed here, and updating does not guarantee improvement.</p><small>An online training test must follow: predict → save → reveal outcome when available → update for a later day. No training rule was changed in this diagnosis.</small></div>
<p class="note">Correction to the earlier interpretation: profits on scattered falling days do not establish resilience through a sustained weak period. These are retrospective diagnostics with illustrative costs and hourly execution. Historical input versions and original pretraining remain unverified. No robust live profit claim is established.</p>
<script>
const daily=DAILY,months=MONTHLY;const t=d=>Date.parse(d.slice(0,10)+'T00:00:00Z');
function chart(id,series){const svg=document.getElementById(id),w=680,h=220,l=63,r=12,top=12,b=27,points=series.flatMap(s=>s.p),lo=Math.min(0,...points.map(p=>p[1])),hi=Math.max(0,...points.map(p=>p[1])),x0=points[0][0],x1=Math.max(...points.map(p=>p[0])),x=v=>l+(v-x0)*(w-l-r)/(x1-x0),y=v=>h-b-(v-lo)*(h-top-b)/(hi-lo);let html='';for(let i=0;i<4;i++){let v=lo+(hi-lo)*i/3;html+=`<line x1="${l}" x2="${w-r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"/><text class="axis" x="${l-5}" y="${y(v)+4}" text-anchor="end">${Math.round(v).toLocaleString()}</text>`}if(lo<0&&hi>0)html+=`<line x1="${l}" x2="${w-r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--muted-foreground)" stroke-dasharray="4 4"/>`;for(const s of series)html+=`<polyline fill="none" stroke="${s.c}" stroke-width="2" points="${s.p.map(p=>x(p[0])+','+y(p[1])).join(' ')}"/>`;for(let i=0;i<3;i++){let v=x0+(x1-x0)*i/2;html+=`<text class="axis" x="${x(v)}" y="218" text-anchor="${i===0?'start':i===2?'end':'middle'}">${new Date(v).toISOString().slice(0,10)}</text>`}svg.innerHTML=html}
function cash(){let series=[{c:'var(--chart-1)',p:daily.map(d=>[t(d[0]),d[1]])},{c:'var(--chart-3)',p:daily.map(d=>[t(d[0]),d[2]])}];if(document.getElementById('before').checked)series.push({c:'var(--chart-5)',p:daily.map(d=>[t(d[0]),d[3]])});chart('cash',series)}cash();document.getElementById('before').onchange=cash;chart('market',[{c:'var(--chart-2)',p:months.map(d=>[t(d[0]),d[1]])},{c:'var(--chart-4)',p:months.map(d=>[t(d[0]),d[2]])}]);
</script>'''
html=html.replace('ROWS',rows).replace('DAILY',json.dumps(daily,separators=(',',':'))).replace('MONTHLY',json.dumps(monthly,separators=(',',':')))
(out/'diagnosis.html').write_text(html)
print(out/'diagnosis.html')
