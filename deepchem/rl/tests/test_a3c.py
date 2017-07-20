from flaky import flaky

import deepchem as dc
from deepchem.models.tensorgraph.layers import Reshape, Variable, SoftMax, GRU
import numpy as np
import tensorflow as tf
import unittest


class TestA3C(unittest.TestCase):

  @flaky
  def test_roulette(self):
    """Test training a policy for the roulette environment."""

    # This is modeled after the Roulette-v0 environment from OpenAI Gym.
    # The player can bet on any number from 0 to 36, or walk away (which ends the
    # game).  The average reward for any bet is slightly negative, so the best
    # strategy is to walk away.

    class RouletteEnvironment(dc.rl.Environment):

      def __init__(self):
        super(RouletteEnvironment, self).__init__([(1,)], 38)
        self._state = [np.array([0])]

      def step(self, action):
        if action == 37:
          self._terminated = True  # Walk away.
          return 0.0
        wheel = np.random.randint(37)
        if wheel == 0:
          if action == 0:
            return 35.0
          return -1.0
        if action != 0 and wheel % 2 == action % 2:
          return 1.0
        return -1.0

      def reset(self):
        self._terminated = False

    env = RouletteEnvironment()

    # This policy just learns a constant probability for each action, and a constant for the value.

    class TestPolicy(dc.rl.Policy):

      def create_layers(self, state, **kwargs):
        action = Variable(np.ones(env.n_actions))
        output = SoftMax(
            in_layers=[Reshape(in_layers=[action], shape=(-1, env.n_actions))])
        value = Variable([0.0])
        return {'action_prob': output, 'value': value}

    # Optimize it.

    a3c = dc.rl.A3C(
        env,
        TestPolicy(),
        max_rollout_length=20,
        optimizer=dc.models.tensorgraph.TFWrapper(
            tf.train.AdamOptimizer, learning_rate=0.001))
    a3c.fit(100000)

    # It should have learned that the expected value is very close to zero, and that the best
    # action is to walk away.

    action_prob, value = a3c.predict([[0]])
    assert -0.5 < value[0] < 0.5
    assert action_prob.argmax() == 37
    assert a3c.select_action([[0]], deterministic=True) == 37

    # Verify that we can create a new A3C object, reload the parameters from the first one, and
    # get the same result.

    new_a3c = dc.rl.A3C(env, TestPolicy(), model_dir=a3c._graph.model_dir)
    new_a3c.restore()
    action_prob2, value2 = new_a3c.predict([[0]])
    assert value2 == value

    # Do the same thing, only using the "restore" argument to fit().

    new_a3c = dc.rl.A3C(env, TestPolicy(), model_dir=a3c._graph.model_dir)
    new_a3c.fit(0, restore=True)
    action_prob2, value2 = new_a3c.predict([[0]])
    assert value2 == value

  def test_recurrent_states(self):
    """Test a policy that involves recurrent layers."""

    # The environment just has a constant state.

    class TestEnvironment(dc.rl.Environment):

      def __init__(self):
        super(TestEnvironment, self).__init__([(10,)], 10)
        self._state = [np.random.random(10)]

      def step(self, action):
        self._state = [np.random.random(10)]
        return 0.0

      def reset(self):
        pass

    # The policy includes a single recurrent layer.

    class TestPolicy(dc.rl.Policy):

      def create_layers(self, state, **kwargs):

        reshaped = Reshape(shape=(1, -1, 10), in_layers=state)
        gru = GRU(n_hidden=10, batch_size=1, in_layers=reshaped)
        output = SoftMax(
            in_layers=[Reshape(in_layers=[gru], shape=(-1, env.n_actions))])
        value = Variable([0.0])
        return {'action_prob': output, 'value': value}

    # We don't care about actually optimizing it, so just run a few rollouts to make
    # sure fit() doesn't crash, then check the behavior of the GRU state.

    env = TestEnvironment()
    a3c = dc.rl.A3C(env, TestPolicy())
    a3c.fit(100)
    # On the first call, the initial state should be all zeros.
    prob1, value1 = a3c.predict(
        env.state, use_saved_states=True, save_states=False)
    # It should still be zeros since we didn't save it last time.
    prob2, value2 = a3c.predict(
        env.state, use_saved_states=True, save_states=True)
    # It should be different now.
    prob3, value3 = a3c.predict(
        env.state, use_saved_states=True, save_states=False)
    # This should be the same as the previous one.
    prob4, value4 = a3c.predict(
        env.state, use_saved_states=True, save_states=False)
    # Now we reset it, so we should get the same result as initially.
    prob5, value5 = a3c.predict(
        env.state, use_saved_states=False, save_states=True)
    assert np.array_equal(prob1, prob2)
    assert np.array_equal(prob1, prob5)
    assert np.array_equal(prob3, prob4)
    assert not np.array_equal(prob2, prob3)
