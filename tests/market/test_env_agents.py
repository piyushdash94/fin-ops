import numpy as np
import pytest

from marketrl.agents import (
    AlwaysFlatPolicy,
    BuyAndHoldPolicy,
    QLearningAgent,
    RandomPolicy,
    ReinforceAgent,
    SupervisedPolicy,
)
from marketrl.env import ACTIONS, TradingEnv


@pytest.fixture
def solvable():
    """A market where feature 0 tells you tomorrow's sign exactly."""
    rng = np.random.default_rng(0)
    n = 500
    signal = rng.standard_normal(n)
    forward = np.sign(signal) * 0.01
    features = np.column_stack([signal, rng.standard_normal(n)])
    return features, forward


def test_reward_is_gross_minus_cost(solvable):
    features, forward = solvable
    env = TradingEnv(features, forward, cost_bps=10)
    out = env.run(BuyAndHoldPolicy())
    np.testing.assert_allclose(out["net_returns"], out["gross_returns"] - out["costs"])


def test_holding_pays_the_entry_cost_once(solvable):
    features, forward = solvable
    env = TradingEnv(features, forward, cost_bps=10)
    out = env.run(BuyAndHoldPolicy())
    assert out["costs"].sum() == pytest.approx(0.001)  # one 10bp entry
    assert (out["costs"][1:] == 0).all()


def test_churning_is_punished(solvable):
    features, forward = solvable
    env = TradingEnv(features, forward, cost_bps=10)
    flat = env.run(AlwaysFlatPolicy())["costs"].sum()
    churn = env.run(lambda s, c=iter(range(10**9)): next(c) % 2 * 2)["costs"].sum()
    assert flat == 0.0
    assert churn > 100 * 0.001


def test_flat_policy_earns_and_loses_nothing(solvable):
    features, forward = solvable
    out = TradingEnv(features, forward).run(AlwaysFlatPolicy())
    assert out["net_returns"].sum() == 0.0


def test_state_includes_current_position(solvable):
    features, forward = solvable
    env = TradingEnv(features, forward)
    state = env.reset()
    assert state.shape == (features.shape[1] + 1,)
    assert state[-1] == 0.0
    assert env.step(2).state[-1] == 1.0


def test_long_only_removes_the_short_action(solvable):
    features, forward = solvable
    env = TradingEnv(features, forward, allow_short=False)
    assert env.n_actions == 2
    assert env.actions.min() == 0.0


def test_misaligned_inputs_are_rejected():
    with pytest.raises(ValueError, match="must align"):
        TradingEnv(np.zeros((10, 2)), np.zeros(9))
    with pytest.raises(ValueError, match="at least one step"):
        TradingEnv(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError, match="2-D"):
        TradingEnv(np.zeros(10), np.zeros(10))


def test_exhausted_env_raises(solvable):
    features, forward = solvable
    env = TradingEnv(features[:2], forward[:2])
    env.step(1)
    env.step(1)
    with pytest.raises(RuntimeError, match="exhausted"):
        env.step(1)


def test_unfitted_agents_raise():
    with pytest.raises(RuntimeError, match="not fitted"):
        QLearningAgent()(np.zeros(3))
    with pytest.raises(RuntimeError, match="not fitted"):
        ReinforceAgent()(np.zeros(3))


@pytest.mark.parametrize(
    "factory", [lambda: QLearningAgent(episodes=40), lambda: ReinforceAgent(episodes=120)]
)
def test_agents_learn_a_solvable_market(solvable, factory):
    features, forward = solvable
    env = TradingEnv(features, forward, cost_bps=1.0)
    agent = factory().fit(env)

    learned = env.run(agent)["net_returns"].sum()
    baseline = env.run(RandomPolicy(seed=1))["net_returns"].sum()
    perfect = float(np.abs(forward).sum())

    assert learned > baseline
    assert learned > 0.5 * perfect  # recovers most of the available profit


@pytest.mark.parametrize(
    "factory", [lambda: QLearningAgent(episodes=30), lambda: ReinforceAgent(episodes=80)]
)
def test_agents_improve_over_training(solvable, factory):
    features, forward = solvable
    agent = factory().fit(TradingEnv(features, forward, cost_bps=1.0))
    curve = np.array(agent.rewards_)
    first, last = curve[: len(curve) // 4].mean(), curve[-len(curve) // 4 :].mean()
    assert last > first


def test_agents_are_deterministic_given_a_seed(solvable):
    features, forward = solvable
    env = TradingEnv(features, forward)
    a = QLearningAgent(episodes=10, seed=3).fit(env)
    b = QLearningAgent(episodes=10, seed=3).fit(env)
    np.testing.assert_allclose(a.q_, b.q_)


def test_agents_cannot_profit_from_pure_noise():
    """No edge exists, so a trained agent must not beat flat after costs."""
    rng = np.random.default_rng(7)
    n = 600
    features = rng.standard_normal((n, 3))
    forward = rng.standard_normal(n) * 0.01  # unrelated to features
    env = TradingEnv(features, forward, cost_bps=10)

    agent = QLearningAgent(episodes=40, seed=0).fit(env)
    # Evaluate on *fresh* noise: whatever it memorized cannot generalize.
    holdout = TradingEnv(
        rng.standard_normal((n, 3)), rng.standard_normal(n) * 0.01, cost_bps=10
    )
    assert holdout.run(agent)["net_returns"].sum() < 0.5


def test_supervised_policy_maps_predictions_to_positions():
    policy = SupervisedPolicy(np.array([0.01, -0.01, 0.0]), threshold=0.0)
    assert [policy(None) for _ in range(3)] == [2, 0, 1]


def test_supervised_policy_threshold_creates_a_dead_zone():
    policy = SupervisedPolicy(np.array([0.001, -0.001]), threshold=0.005)
    assert [policy(None) for _ in range(2)] == [1, 1]


def test_action_table_is_short_flat_long():
    np.testing.assert_allclose(ACTIONS, [-1.0, 0.0, 1.0])
