"""Naive baselines on the dev window."""
from epf.data import eval_days, load_panel
from epf.ledger import naive_forecast, record

panel = load_panel()
days = eval_days(panel)
record("naive", {"rule": "D-7 Mon/Sat/Sun, else D-1"}, days, naive_forecast(panel, days), panel=panel)
p = panel["price_de"]
record("naive_weekly", {"rule": "D-7"}, days, p[[panel.idx(d) - 7 for d in days]], panel=panel)
