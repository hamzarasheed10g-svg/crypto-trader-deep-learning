"""LSTM forecasting network (Section 1.5 of the methodology).

Architecture (per Section 1.5.1):
    Input sequence layer -> stacked LSTM(s) -> dropout -> dense layers -> output

The LSTM cell internally computes Equations 1.3–1.6 (forget, input, candidate,
cell state). We use ``torch.nn.LSTM``, which implements the same gated update.

This module supports two head configurations:

- **Single-head** (legacy, ``prediction_target ∈ {log_return, close, direction}``):
  one output neuron, MSE or BCE loss. Kept for backward compatibility with
  existing checkpoints.

- **Dual-head** (``prediction_target = "dual"``): the shared LSTM backbone feeds
  *two* heads in parallel — a regression head that predicts next-bar log-return
  (MSE loss) and a classification head that predicts P(up) (BCE-with-logits
  loss). The two losses are weighted by ``cfg.lstm.loss.alpha_regression`` and
  ``cfg.lstm.loss.alpha_direction``. This gives the trader both magnitude and
  direction signals in a single forward pass, and the classification head
  typically achieves better directional accuracy than a single regression head
  pushed through ``sign()``.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single-head model (kept for backward compatibility)
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Stacked LSTM -> dropout -> dense -> single output.

    Used when ``prediction_target`` is ``log_return``, ``close``, or
    ``direction``. For ``dual``, see ``DualHeadLSTMForecaster``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * directions, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias = 1 (Jozefowicz et al., 2015 trick)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape: (batch, seq_len, input_size). Returns (batch, output_size)."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # take the last timestep
        last = self.dropout(last)
        h = self.act(self.fc1(last))
        h = self.dropout(h)
        return self.fc2(h)

    @torch.no_grad()
    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper. Returns shape (batch,) for scalar output."""
        self.eval()
        y = self.forward(x)
        if y.dim() == 2 and y.size(-1) == 1:
            return y.squeeze(-1)
        return y


# ---------------------------------------------------------------------------
# Dual-head model
# ---------------------------------------------------------------------------

class DualHeadLSTMForecaster(nn.Module):
    """Shared LSTM backbone + two parallel heads.

    Forward returns a tuple ``(regression_out, direction_logit)`` where:

    - ``regression_out``: shape ``(batch, 1)``, predicted next-bar log return.
    - ``direction_logit``: shape ``(batch, 1)``, raw logit for P(up). Apply
      sigmoid at inference time to get a probability.

    The classification head gives an explicit directional probability that is
    well-calibrated for use as a confidence score in the live trader: e.g. only
    open a position when ``P(up) > 0.55`` AND the regression head agrees.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        feat_dim = hidden_size * directions

        # Shared trunk
        self.dropout = nn.Dropout(dropout)
        self.fc_shared = nn.Linear(feat_dim, hidden_size)
        self.act = nn.ReLU()

        # Regression head: next-bar log return
        self.head_reg = nn.Linear(hidden_size, 1)
        # Direction head: logit for P(up)
        self.head_dir = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        for layer in (self.fc_shared, self.head_reg, self.head_dir):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        h = self.act(self.fc_shared(last))
        h = self.dropout(h)
        reg = self.head_reg(h)
        dir_logit = self.head_dir(h)
        return reg, dir_logit

    @torch.no_grad()
    def predict_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(predicted_log_return, prob_up)`` each shape ``(batch,)``."""
        self.eval()
        reg, dir_logit = self.forward(x)
        return reg.squeeze(-1), torch.sigmoid(dir_logit).squeeze(-1)


# ---------------------------------------------------------------------------
# Compatibility shim — the trading env and live trader call ``model(x)`` and
# expect a single tensor of shape (batch, 1) (the log-return prediction).
# DualHeadLSTMForecaster returns a tuple, which would break those call sites.
# DualHeadShim wraps a dual-head model and forwards only the regression output
# for that path, while still exposing the direction probability via predict_both.
# ---------------------------------------------------------------------------

class DualHeadShim(nn.Module):
    """Makes a ``DualHeadLSTMForecaster`` look single-headed to legacy callers.

    ``forward(x)`` returns only the regression prediction (shape ``(batch, 1)``).
    Use ``predict_both(x)`` to access both heads.
    """

    def __init__(self, dual: DualHeadLSTMForecaster):
        super().__init__()
        self.dual = dual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reg, _ = self.dual(x)
        return reg

    @torch.no_grad()
    def predict_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(log_return_prediction, prob_up)`` each shape ``(batch,)``."""
        return self.dual.predict_step(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model_from_config(
    cfg, input_size: int
) -> Tuple[Union[LSTMForecaster, DualHeadLSTMForecaster], str]:
    """Construct the appropriate model and return ``(model, loss_kind)``.

    ``loss_kind`` is one of:
        - ``"mse"``   : single-head regression (log_return / close)
        - ``"bce"``   : single-head binary classification (direction)
        - ``"dual"``  : dual-head (MSE + BCE jointly)
    """
    target = cfg.lstm.prediction_target

    if target == "dual":
        model = DualHeadLSTMForecaster(
            input_size=input_size,
            hidden_size=cfg.lstm.hidden_size,
            num_layers=cfg.lstm.num_layers,
            dropout=cfg.lstm.dropout,
            bidirectional=cfg.lstm.bidirectional,
        )
        return model, "dual"

    if target == "direction":
        loss_kind = "bce"
    elif target in ("log_return", "close"):
        loss_kind = "mse"
    else:
        raise ValueError(f"Unknown prediction_target: {target!r}")

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=cfg.lstm.hidden_size,
        num_layers=cfg.lstm.num_layers,
        dropout=cfg.lstm.dropout,
        bidirectional=cfg.lstm.bidirectional,
        output_size=1,
    )
    return model, loss_kind
