"""Dev-only calibration and constrained battery economics for prospective validation."""
from dataclasses import dataclass, asdict

import numpy as np


def interval_adjustment(y, lower, upper, alpha=0.2):
    """Per-hour split conformal expansion; fit only on earlier, out-of-sample dev forecasts.

    Serial dependence and distribution shifts prevent a coverage guarantee here.
    """
    y, lower, upper = map(np.asarray, (y, lower, upper))
    if y.shape != lower.shape or y.shape != upper.shape or y.ndim != 2 or len(y) < 30:
        raise ValueError("need >=30 aligned calibration days")
    if not 0 < alpha < 1 or not np.isfinite([y, lower, upper]).all() or (lower > upper).any():
        raise ValueError("invalid calibration inputs")
    scores = np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])
    k = int(np.ceil((len(y) + 1) * (1 - alpha)))
    if k > len(y):
        raise ValueError("too few calibration days for requested alpha")
    return np.sort(scores, axis=0)[k - 1]


@dataclass(frozen=True)
class Battery:
    capacity_mwh: float = 1.0
    power_mw: float = 1.0
    eta: float = 0.95
    steps: int = 10
    max_cycles: int = 2
    fee_slippage_eur_mwh: float = 3.0  # illustrative, per MWh bought or sold
    wear_eur_mwh: float = 20.0  # illustrative, per discharged battery MWh

    def __post_init__(self):
        if not np.isfinite([self.capacity_mwh,self.power_mw,self.eta,self.fee_slippage_eur_mwh,self.wear_eur_mwh]).all() or not (self.capacity_mwh > 0 and self.power_mw > 0 and 0 < self.eta <= 1
                and self.steps > 0 and self.max_cycles > 0
                and self.fee_slippage_eur_mwh >= 0 and self.wear_eur_mwh >= 0):
            raise ValueError("invalid battery assumptions")


def dispatch(prices, b=Battery()):
    """Hourly DP with grid power limits, SOC, cycle budget and empty start/end.

    Positive grid MWh = purchase. At most one direction per hour. SOC discretized.
    """
    prices = np.asarray(prices, float)
    if prices.shape != (24,) or not np.isfinite(prices).all():
        raise ValueError("need 24 finite hourly prices")
    n, budget = b.steps, b.steps * b.max_cycles
    change = np.arange(n + 1)[None, :] - np.arange(n + 1)[:, None]
    energy = change * b.capacity_mwh / n
    grid = np.where(energy > 0, energy / b.eta, energy * b.eta)
    used = np.maximum(-change, 0)
    remaining = np.arange(budget + 1)[None, None, :] - used[:, :, None]
    feasible = (remaining >= 0) & (np.abs(grid[:, :, None]) <= b.power_mw + 1e-10)
    value = np.full((n + 1, budget + 1), -np.inf)
    value[0] = 0
    choices = []
    for price in prices[::-1]:
        cash = -price * grid - b.fee_slippage_eur_mwh * np.abs(grid) - b.wear_eur_mwh * np.maximum(-energy, 0)
        future = value[np.arange(n + 1)[None, :, None], np.maximum(remaining, 0)]
        options = np.where(feasible, cash[:, :, None] + future, -np.inf)
        choices.append(np.argmax(options, axis=1))
        value = options.max(axis=1)
    soc, left, out, discharged = 0, budget, [], []
    for policy in choices[::-1]:
        nxt = int(policy[soc, left])
        out.append(grid[soc, nxt])
        discharged.append(max(-energy[soc, nxt], 0))
        left -= used[soc, nxt]
        soc = nxt
    assert soc == 0 and left >= 0
    return np.asarray(out), np.asarray(discharged)


def economics(y, forecast, b=Battery(), annual_fixed_eur=0.0):
    y, forecast = np.asarray(y), np.asarray(forecast)
    if y.shape != forecast.shape or y.ndim != 2 or y.shape[1] != 24 or not np.isfinite(y).all():
        raise ValueError("need aligned daily prices/forecasts")
    if not np.isfinite(annual_fixed_eur) or annual_fixed_eur < 0:
        raise ValueError("annual fixed costs must be finite and nonnegative")
    gross = variable = oracle = discharged = turnover = 0.0
    for actual, prediction in zip(y, forecast):
        grid, wear = dispatch(prediction, b)
        og, ow = dispatch(actual, b)
        gross += float(-actual @ grid)
        variable += float(b.fee_slippage_eur_mwh * abs(grid).sum() + b.wear_eur_mwh * wear.sum())
        oracle += float(-actual @ og - b.fee_slippage_eur_mwh * abs(og).sum() - b.wear_eur_mwh * ow.sum())
        turnover += float(abs(grid).sum())
        discharged += float(wear.sum())
    fixed = annual_fixed_eur * len(y) / 365.25
    return dict(gross_eur=gross, variable_cost_eur=variable, margin_eur=gross-variable,
                fixed_cost_eur=fixed, scenario_net_eur=gross-variable-fixed,
                oracle_margin_eur=oracle, grid_turnover_mwh=turnover, discharged_mwh=discharged,
                assumptions=asdict(b), annual_fixed_eur=annual_fixed_eur)
