import os
import sys
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

# Append the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrate_shapes_robots import (
    HumanShapeGenerator, Custom3DRobotEnv, train_agent, save_model
)
from walking_agents.walking_agent import DQNAgent


def test_human_shape_generator():
    generator = HumanShapeGenerator()
    shape = generator.generate()
    assert shape.shape == (10, 5, 2, 3)  # 3D shape with color channels
    assert np.any(shape > 0)  # Ensure the shape is not empty


@patch('integrate_shapes_robots.Custom3DRobotEnv')
def test_custom_3d_robot_env(mock_env):
    mock_env.return_value.reset.return_value = np.zeros((64, 64, 64, 3))
    mock_env.return_value.step.return_value = (
        np.zeros((64, 64, 64, 3)), 1.0, False, {}
    )
    mock_env.return_value.action_space.sample.return_value = np.zeros(6)

    env = mock_env()
    initial_state = env.reset()
    assert initial_state.shape == (64, 64, 64, 3)  # 3D observation space
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    assert next_state.shape == (64, 64, 64, 3)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_dqn_agent():
    state_size = (64, 64, 64, 3)
    action_size = 6
    agent = DQNAgent(state_size, action_size)

    # Test act
    state = np.random.rand(*state_size)
    action = agent.act(state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (6,)
    assert np.all((action >= -1) & (action <= 1)), "Action values should be between -1 and 1"

    # Test remember and replay
    next_state = np.random.rand(*state_size)
    agent.remember(state, action, 1.0, next_state, False)
    agent.replay(32)  # Assuming batch size of 32

    # Test epsilon decay
    initial_epsilon = agent.epsilon
    agent.replay(32)
    assert agent.epsilon < initial_epsilon, "Epsilon should decay after replay"


@patch('integrate_shapes_robots.Custom3DRobotEnv')
@patch('integrate_shapes_robots.DQNAgent')
@patch('integrate_shapes_robots.save_model')
def test_training_loop(mock_save_model, mock_agent_class, mock_env_class):
    mock_env = Mock()
    mock_env.reset.return_value = np.random.rand(64, 64, 64, 3)
    mock_env.step.return_value = (np.random.rand(64, 64, 64, 3), 1.0, False, {})
    mock_env.action_space = Mock()
    mock_env.action_space.shape = (6,)
    mock_env_class.return_value = mock_env

    mock_agent = Mock()
    mock_agent.act.return_value = np.random.rand(6)
    mock_agent.epsilon = 0.5
    mock_agent.memory = []
    mock_agent.batch_size = 32
    mock_agent_class.return_value = mock_agent

    # Run a short training loop
    num_episodes = 2
    train_agent(num_episodes)

    assert mock_env.reset.call_count == num_episodes
    assert mock_agent.act.call_count > 0
    assert mock_agent.remember.call_count > 0
    assert mock_agent.replay.call_count > 0
    mock_save_model.assert_called_once()


def test_agent():
    mock_env = Mock(spec=Custom3DRobotEnv)
    mock_env.reset.return_value = Mock(is_last=Mock(return_value=False))
    mock_env.step.return_value = Mock(is_last=Mock(return_value=True), reward=1.0)

    mock_agent = Mock()
    mock_agent.act.return_value = np.zeros(6)  # 6D action space

    time_step = mock_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action = mock_agent.act(time_step)
        time_step = mock_env.step(action)
        episode_return += time_step.reward

    assert isinstance(episode_return, float)
    assert episode_return == 1.0

    # Test visualization
    with patch('matplotlib.pyplot.show') as mock_show:
        mock_fig = Mock()
        def render_side_effect():
            plt.show()
            return mock_fig
        mock_env.render.side_effect = render_side_effect
        result = mock_env.render()
        assert mock_show.called, "plt.show() was not called during rendering"
        assert result == mock_fig, "render method should return the figure object"


@patch('integrate_shapes_robots.lazy_import')
@patch('integrate_shapes_robots.os.path.join')
@patch('integrate_shapes_robots.tf.keras.models.save_model')
def test_save_model(mock_lazy_import, mock_path_join, mock_save_model):
    mock_lazy_import.return_value = Mock()
    mock_model = Mock()
    mock_path_join.return_value = '/fake/path/model.h5'

    save_model(mock_model, 'test_model')

    mock_path_join.assert_called_once_with('models', 'test_model.h5')
    mock_save_model.assert_called_once_with(mock_model, '/fake/path/model.h5')


if __name__ == "__main__":
    pytest.main([__file__])
