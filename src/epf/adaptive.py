"""Small causal online correction, learned from observed errors rather than self-labels."""
from dataclasses import dataclass, field
from datetime import date
import numpy as np


@dataclass
class AdaptiveBias:
    halflife: float = 30.
    bias: np.ndarray = field(default_factory=lambda: np.zeros(24))
    last_observed: str | None = None
    pending: tuple | None = None

    def __post_init__(self):
        self.bias=np.asarray(self.bias,float).copy()
        if not np.isfinite(self.halflife) or self.halflife<=0 or self.bias.shape!=(24,) or not np.isfinite(self.bias).all():
            raise ValueError('invalid adaptive state')
        if self.last_observed is not None:date.fromisoformat(self.last_observed)

    def predict(self, day, forecast):
        if self.pending is not None:raise ValueError('previous prediction must be observed before the next request')
        target=date.fromisoformat(day)
        forecast=np.asarray(forecast,float)
        if forecast.shape!=(24,) or not np.isfinite(forecast).all():raise ValueError('need 24 finite original forecasts')
        gap=0 if self.last_observed is None else (target-date.fromisoformat(self.last_observed)).days-1
        if gap<0:raise ValueError('predictions must follow the last observed day')
        weights=self.bias * .5**(gap/self.halflife)
        self.pending=(day,forecast.copy(),weights.copy())
        return dict(point=forecast+weights,weights=weights.copy(),last_observed=self.last_observed)

    def observe(self, day, actual):
        if self.pending is None or day!=self.pending[0]:raise ValueError('observation must match the outstanding prediction')
        actual=np.asarray(actual,float)
        if actual.shape!=(24,) or not np.isfinite(actual).all():raise ValueError('need 24 finite released outcomes')
        _,forecast,weights=self.pending
        decay=.5**(1/self.halflife)
        self.bias=decay*weights+(1-decay)*(actual-forecast)
        self.last_observed=day
        self.pending=None
        return dict(bias=self.bias.tolist(),last_observed=self.last_observed,halflife=self.halflife)
