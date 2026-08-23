"""Reinforcement-learning agents and the baseline policies they must beat.

Two agents, both pure NumPy so the whole thing trains in seconds on a laptop
and has no deep-learning dependency:

* :class:`QLearningAgent` -- tabular Q-learning over a discretized state. Small,
  fast, and inspectable: you can print the learned policy and read what it
  believes. The discretization is fitted on training data only.
* :class:`ReinforceAgent` -- policy gradient (REINFORCE) with a learned value
  baseline, entropy regularization and Adam. Handles continuous features
  directly and can express a stochastic policy, which matters when the signal
  is weak enough that always committing is the wrong move.

Both are trained and evaluated walk-forward like every other model here. An RL
agent evaluated on its own training window is a random number generator with
extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import ACTIONS, TradingEnv

__all__ = [
    "QLearningAgent",
    "ReinforceAgent",
    "BuyAndHoldPolicy",
    "AlwaysFlatPolicy",
    "RandomPolicy",
    "SupervisedPolicy",
]


# --------------------------------------------------------------------------
# Baseline policies
# --------------------------------------------------------------------------
class BuyAndHoldPolicy:
    """Always long. The benchmark an equity strategy actually has to beat."""

    name = "buy_and_hold"

    def __call__(self, state) -> int:
        return 2

    def fit(self, env):  # noqa: D102 - baselines need no training
        return self


class AlwaysFlatPolicy:
    """Never hold a position. Earns zero, loses nothing -- the zero-risk floor."""

    name = "always_flat"

    def __call__(self, state) -> int:
        return 1

    def fit(self, env):
        return self


class RandomPolicy:
    """Uniformly random actions. Exists to show what noise plus costs looks like."""

    name = "random"

    def __init__(self, n_actions: int = 3, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, state) -> int:
        return int(self.rng.integers(self.n_actions))

    def fit(self, env):
        return self


class SupervisedPolicy:
    """Turn a return forecast into positions: long if predicted up, short if down.

    ``threshold`` suppresses trading on weak forecasts, which is how a
    supervised model is made cost-aware after the fact -- the honest comparison
    against an RL agent that learned the cost directly.
    """

    name = "supervised"

    def __init__(self, predictions: np.ndarray, threshold: float = 0.0) -> None:
        self.predictions = np.asarray(predictions, dtype=float)
        self.threshold = threshold
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def __call__(self, state) -> int:
        prediction = self.predictions[min(self._i, len(self.predictions) - 1)]
        self._i += 1
        if prediction > self.threshold:
            return 2
        if prediction < -self.threshold:
            return 0
        return 1

    def fit(self, env):
        return self


# --------------------------------------------------------------------------
# Tabular Q-learning
# --------------------------------------------------------------------------
@dataclass
class QLearningAgent:
    """Tabular Q-learning over a quantile-discretized state.

    The state space is kept deliberately tiny -- ``n_features`` columns binned
    into ``n_bins`` buckets, times the current position. With a few thousand
    training days, a larger table would have a handful of visits per cell and
    learn nothing but noise.
    """

    n_actions: int = 3
    n_bins: int = 5
    n_features: int = 4
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995
    episodes: int = 60
    seed: int = 0
    name: str = "q_learning"

    edges_: np.ndarray | None = field(default=None, init=False)
    columns_: np.ndarray | None = field(default=None, init=False)
    q_: np.ndarray | None = field(default=None, init=False)
    rewards_: list[float] = field(default_factory=list, init=False)

    def _select_columns(self, features: np.ndarray, forward_returns: np.ndarray) -> np.ndarray:
        """Pick the features most correlated with the outcome, on train data only."""
        n_available = features.shape[1]
        k = min(self.n_features, n_available)
        scores = np.zeros(n_available)
        for j in range(n_available):
            column = features[:, j]
            if np.std(column) > 0 and np.std(forward_returns) > 0:
                scores[j] = abs(np.corrcoef(column, forward_returns)[0, 1])
        return np.argsort(scores)[::-1][:k]

    def _discretize(self, row: np.ndarray) -> tuple:
        selected = row[self.columns_]
        bins = tuple(
            int(np.digitize(value, self.edges_[i])) for i, value in enumerate(selected)
        )
        position = int(np.searchsorted(ACTIONS, row[-1]))
        return bins + (position,)

    def _index(self, state: np.ndarray) -> int:
        key = self._discretize(state)
        index = 0
        for value, size in zip(key, self._shape):
            index = index * size + min(value, size - 1)
        return index

    def fit(self, env: TradingEnv) -> "QLearningAgent":
        rng = np.random.default_rng(self.seed)
        features = env.features
        self.columns_ = self._select_columns(features, env.forward_returns)

        # Quantile edges from the training window only.
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        self.edges_ = np.array(
            [np.percentile(features[:, c], quantiles) for c in self.columns_]
        )
        self._shape = tuple([self.n_bins + 1] * len(self.columns_) + [len(ACTIONS) + 1])
        n_states = int(np.prod(self._shape))
        self.q_ = np.zeros((n_states, env.n_actions))

        epsilon = self.epsilon
        for _ in range(self.episodes):
            state = env.reset()
            s = self._index(state)
            total = 0.0
            while True:
                if rng.random() < epsilon:
                    action = int(rng.integers(env.n_actions))
                else:
                    action = int(np.argmax(self.q_[s]))

                result = env.step(action)
                s_next = self._index(result.state)
                best_next = 0.0 if result.done else float(np.max(self.q_[s_next]))
                target = result.reward + self.gamma * best_next
                self.q_[s, action] += self.alpha * (target - self.q_[s, action])

                total += result.reward
                s = s_next
                if result.done:
                    break
            self.rewards_.append(total)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
        return self

    def __call__(self, state: np.ndarray) -> int:
        if self.q_ is None:
            raise RuntimeError("agent is not fitted; call fit(env) first")
        return int(np.argmax(self.q_[self._index(state)]))


# --------------------------------------------------------------------------
# REINFORCE policy gradient
# --------------------------------------------------------------------------
class _Adam:
    """Minimal Adam optimizer over a list of parameter arrays."""

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grad**2
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            param += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # gradient ASCENT
        return params


@dataclass
class ReinforceAgent:
    """REINFORCE with a value baseline, entropy bonus and Adam.

    Gradient ascent on expected discounted reward. The entropy bonus is not
    decoration: without it the policy collapses onto a single action within a
    few episodes, because on noisy financial data any early lucky streak looks
    decisive.
    """

    hidden: int = 24
    lr: float = 0.01
    gamma: float = 0.95
    entropy_beta: float = 0.01
    episodes: int = 150
    seed: int = 0
    name: str = "reinforce"

    W1: np.ndarray | None = field(default=None, init=False)
    rewards_: list[float] = field(default_factory=list, init=False)

    def _init_params(self, state_dim: int, n_actions: int) -> None:
        rng = np.random.default_rng(self.seed)
        # He-style init for the tanh layer; near-zero output layer so the initial
        # policy is close to uniform rather than arbitrarily opinionated.
        self.W1 = rng.standard_normal((state_dim, self.hidden)) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((self.hidden, n_actions)) * 0.01
        self.b2 = np.zeros(n_actions)
        self.rng = rng

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.mu_) / self.sigma_

    def _forward(self, state: np.ndarray):
        hidden = np.tanh(state @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        logits -= logits.max()
        exp = np.exp(logits)
        return hidden, exp / exp.sum()

    def fit(self, env: TradingEnv) -> "ReinforceAgent":
        # Feature standardization fitted on the training window only.
        states = np.column_stack([env.features, np.zeros(len(env.features))])
        self.mu_ = states.mean(axis=0)
        self.sigma_ = np.where(states.std(axis=0) > 1e-9, states.std(axis=0), 1.0)

        self._init_params(env.state_dim, env.n_actions)
        params = [self.W1, self.b1, self.W2, self.b2]
        optimizer = _Adam(params, lr=self.lr)
        baseline = 0.0

        for episode in range(self.episodes):
            state = self._normalize(env.reset())
            trajectory = []
            while True:
                hidden, probs = self._forward(state)
                action = int(self.rng.choice(len(probs), p=probs))
                result = env.step(action)
                trajectory.append((state, hidden, probs, action, result.reward))
                state = self._normalize(result.state)
                if result.done:
                    break

            rewards = np.array([step[4] for step in trajectory])
            returns = np.zeros_like(rewards)
            running = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                running = rewards[t] + self.gamma * running
                returns[t] = running

            # Whiten the returns, then centre on a slow-moving baseline. Both
            # matter: raw discounted PnL has a tiny scale and huge variance.
            baseline = 0.9 * baseline + 0.1 * returns.mean()
            advantages = returns - baseline
            scale = advantages.std()
            if scale > 1e-12:
                advantages = advantages / scale

            grads = [np.zeros_like(p) for p in params]
            for (state_t, hidden_t, probs_t, action_t, _), advantage in zip(trajectory, advantages):
                d_logits = -probs_t.copy()
                d_logits[action_t] += 1.0
                d_logits *= advantage

                # Entropy gradient: dH/dz_j = -p_j (log p_j + H).
                log_probs = np.log(np.clip(probs_t, 1e-12, None))
                entropy = -float(probs_t @ log_probs)
                d_logits += self.entropy_beta * (-probs_t * (log_probs + entropy))

                grads[2] += np.outer(hidden_t, d_logits)
                grads[3] += d_logits
                d_hidden = (self.W2 @ d_logits) * (1 - hidden_t**2)
                grads[0] += np.outer(state_t, d_hidden)
                grads[1] += d_hidden

            n = max(len(trajectory), 1)
            optimizer.step(params, [g / n for g in grads])
            self.W1, self.b1, self.W2, self.b2 = params
            self.rewards_.append(float(rewards.sum()))
        return self

    def __call__(self, state: np.ndarray) -> int:
        if self.W1 is None:
            raise RuntimeError("agent is not fitted; call fit(env) first")
        _, probs = self._forward(self._normalize(np.asarray(state, dtype=float)))
        return int(np.argmax(probs))  # greedy at evaluation time
