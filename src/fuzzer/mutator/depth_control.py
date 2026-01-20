from __future__ import annotations

import random

from fuzzer.mutator.config import DepthConfig


class DepthControlMixin:
    """Mixin providing unified depth control with exponential decay.

    Classes using this mixin must have:
    - self.rng: random.Random
    - self.depth_cfg: DepthConfig
    """

    rng: random.Random
    depth_cfg: DepthConfig

    def should_continue(self, depth: int) -> bool:
        """Decide whether to continue recursing at the given depth.

        Uses exponential decay: P(continue) = decay_base ^ depth
        Always returns False at max_depth (hard cap).
        """
        if depth >= self.depth_cfg.max_depth:
            return False
        return self.rng.random() < self.depth_cfg.decay_base ** depth
