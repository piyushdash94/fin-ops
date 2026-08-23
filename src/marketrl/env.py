"""A trading environment for reinforcement learning.

Gym-like but dependency-free. The design decision that matters is the reward:

    reward_t = position_t * forward_return_t - cost * |position_t - position_{t-1}|

Transaction costs are charged inside the reward, not subtracted afterwards.
Without them an RL agent learns to flip position every single day -- the
resulting equity curve looks superb and is unreachable in reality. Charging the
cost in the reward makes the agent's own objective include the friction it
creates, which is the only way it learns to hold a position.

The state includes the current position, which makes the problem properly
Markovian: what to do next genuinely depends on what you already hold, because
changing your mind costs money.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TradingEnv", "ACTIONS", "ACTION_NAMES"]

#: Action index -> target position. Short, flat, long.
ACTIONS = np.array([-1.0, 0.0, 1.0])
ACTION_NAMES = ("short", "flat", "long")


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    position: float
    gross_return: float
    cost: float


class TradingEnv:
    """Single-asset long/flat/short environment over a fixed window of history.

    Parameters
    ----------
    features:
        ``(n_steps, n_features)`` matrix. Row ``t`` is what is known at the
        close of day ``t``.
    forward_returns:
        ``(n_steps,)`` simple returns earned by a unit long position held from
        the close of day ``t`` to the close of day ``t+1``.
    cost_bps:
        Round-trip-equivalent cost in basis points per unit of position change.
        10 bps is a reasonable retail estimate for a liquid US equity including
        spread and slippage; institutional execution is lower.
    allow_short:
        When False the action set is restricted to flat/long.
    risk_aversion:
        Optional penalty on squared reward, which discourages the agent from
        earning its return through a handful of violent days.
    """

    def __init__(
        self,
        features: np.ndarray,
        forward_returns: np.ndarray,
        *,
        cost_bps: float = 10.0,
        allow_short: bool = True,
        risk_aversion: float = 0.0,
    ) -> None:
        features = np.asarray(features, dtype=float)
        forward_returns = np.asarray(forward_returns, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be a 2-D (n_steps, n_features) array")
        if len(features) != len(forward_returns):
            raise ValueError(
                f"features has {len(features)} rows but forward_returns has "
                f"{len(forward_returns)}; they must align"
            )
        if len(features) == 0:
            raise ValueError("environment needs at least one step")

        self.features = features
        self.forward_returns = forward_returns
        self.cost = cost_bps / 10_000.0
        self.risk_aversion = risk_aversion
        self.actions = ACTIONS if allow_short else ACTIONS[1:]
        self.n_actions = len(self.actions)
        self.n_steps = len(features)
        self.state_dim = features.shape[1] + 1  # + current position
        self.reset()

    def reset(self, position: float = 0.0) -> np.ndarray:
        self.t = 0
        self.position = float(position)
        return self._state()

    def _state(self) -> np.ndarray:
        return np.append(self.features[self.t], self.position)

    def step(self, action: int) -> StepResult:
        """Take ``action``, earn the next day's return on the new position."""
        if self.t >= self.n_steps:
            raise RuntimeError("environment is exhausted; call reset()")

        target = float(self.actions[int(action)])
        cost = self.cost * abs(target - self.position)
        gross = target * self.forward_returns[self.t]
        reward = gross - cost
        if self.risk_aversion:
            reward -= self.risk_aversion * reward * reward

        self.position = target
        self.t += 1
        done = self.t >= self.n_steps
        state = self._state() if not done else np.append(self.features[-1], self.position)
        return StepResult(state, reward, done, target, gross, cost)

    def run(self, policy) -> dict[str, np.ndarray]:
        """Play one full pass with ``policy(state) -> action`` and record it."""
        state = self.reset()
        positions, rewards, gross, costs, actions = [], [], [], [], []
        while True:
            action = int(policy(state))
            result = self.step(action)
            actions.append(action)
            positions.append(result.position)
            rewards.append(result.reward)
            gross.append(result.gross_return)
            costs.append(result.cost)
            state = result.state
            if result.done:
                break
        return {
            "actions": np.array(actions),
            "positions": np.array(positions),
            "net_returns": np.array(rewards),
            "gross_returns": np.array(gross),
            "costs": np.array(costs),
        }
