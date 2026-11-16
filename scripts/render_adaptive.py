"""Render the single predeclared adaptive experiment; no model calls."""
import json
import pandas as pd
from epf.data import ROOT

out=ROOT/'runs/adaptive'
s=json.loads((out/'summary.json').read_text())
d=pd.read_csv(out/'daily.csv')
d[['base_mae','adaptive_mae']]=d[['base_mae','adaptive_mae']].rolling(7,min_periods=7).mean()
d=d.dropna()
points=d[['day','base_mae','adaptive_mae']].values.tolist()
html='''<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box}body{font:14px var(--font-sans,system-ui);color:var(--foreground);margin:0}p{line-height:1.5}small,.muted{color:var(--muted-foreground)}.cards{display:flex;gap:10px;flex-wrap:wrap}.card{flex:1;min-width:170px;border:1px solid var(--border);border-radius:var(--radius);background:var(--card);padding:16px}.card b{display:block;font-size:23px;margin:7px 0}svg{height:220px;width:100%;display:block}.axis{fill:var(--muted-foreground);font-size:11px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:right;padding:9px;border-bottom:1px solid var(--border)}td:first-child,th:first-child{text-align:left}.note{color:var(--warning);font-size:12px}
</style>
<p><b>One adaptive correction test · dev data only</b><br><small>July–October 2025 · 123 days · learned from prior errors, never current/future outcomes</small></p>
<div class="cards"><div class="card">Forecast error reduction<b>3.25%</b><small>A small correction layer updated daily. Core Chronos weights stayed fixed.</small></div><div class="card">Change in simulated margin<b>−€4.62</b><small>After variable costs across 123 days; essentially unchanged.</small></div><div class="card">Decision<b>Keep control</b><small>Did not pass the rule set before the run.</small></div></div>
<p>Daily forecast error, trailing 7-day average · EUR/MWh · lower is better</p><p><small><span style="color:var(--chart-1)">● Original pre-gate Chronos</span> &nbsp; <span style="color:var(--chart-2)">● Daily adaptive correction</span></small></p>
<svg id="chart" viewBox="0 0 680 220" role="img" aria-label="Original versus adaptive seven-day average forecast error"></svg>
<table><thead><tr><th>Method</th><th>MAE €/MWh</th><th>Margin after variable costs</th></tr></thead><tbody><tr><td>Original</td><td>BASEMAE</td><td>€BASEMARGIN</td></tr><tr><td>Adaptive correction</td><td>ADAPTMAE</td><td>€ADAPTMARGIN</td></tr></tbody></table>
<p>Promotion required at least 1% lower forecast error <b>and</b> no lower simulated battery margin. Accuracy passed; the margin condition failed. No further settings were searched to force a pass.</p>
<p class="note">This is a lightweight 24-weight correction layer, not continual neural-network retraining. It was evaluated on already-used dev data, not a new unseen test. Costs and hourly fills are assumptions. Frozen paper model and completed holdout remain unchanged.</p>
<script>
const points=DATA,w=680,h=220,l=48,r=10,t=12,b=26;let max=Math.max(...points.flatMap(p=>[p[1],p[2]])),x0=Date.parse(points[0][0]),x1=Date.parse(points.at(-1)[0]),x=d=>l+(Date.parse(d)-x0)*(w-l-r)/(x1-x0),y=v=>h-b-v*(h-t-b)/max,svg='';for(let i=0;i<4;i++){let v=max*i/3;svg+=`<line x1="${l}" x2="${w-r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"/><text class="axis" x="${l-5}" y="${y(v)+4}" text-anchor="end">${v.toFixed(0)}</text>`}for(const [col,color] of [[1,'var(--chart-1)'],[2,'var(--chart-2)']])svg+=`<polyline fill="none" stroke="${color}" stroke-width="2" points="${points.map(p=>x(p[0])+','+y(p[col])).join(' ')}"/>`;for(let i=0;i<3;i++){const time=x0+(x1-x0)*i/2;svg+=`<text class="axis" x="${l+(w-l-r)*i/2}" y="217" text-anchor="${i===0?'start':i===2?'end':'middle'}">${new Date(time).toISOString().slice(0,10)}</text>`}document.getElementById('chart').innerHTML=svg;
</script>'''
for key,value in dict(BASEMAE=f"{s['base']['mae']:.4f}",ADAPTMAE=f"{s['adaptive']['mae']:.4f}",BASEMARGIN=f"{s['base']['economics']['margin_eur']:,.2f}",ADAPTMARGIN=f"{s['adaptive']['economics']['margin_eur']:,.2f}",DATA=json.dumps(points,separators=(',',':'))).items():
    html=html.replace(key,value)
(out/'report.html').write_text(html)
print(out/'report.html')
