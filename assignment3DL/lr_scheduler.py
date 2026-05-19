from __future__ import annotations
from typing import Optional
import torch


class NoamScheduler:

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
    ) -> None:
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self._step_num    = 0

    def step(self) -> float:
        self._step_num += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_lr(self, step: Optional[int] = None) -> float:
        s = step if step is not None else self._step_num
        if s == 0:
            return 0.0
        return (self.d_model ** -0.5) * min(s ** -0.5, s * self.warmup_steps ** -1.5)

    @property
    def last_lr(self) -> float:
        return self.get_lr()

    @property
    def step_num(self) -> int:
        return self._step_num
