import pytest
import random
from unittest.mock import Mock, patch
import numpy as np
import matplotlib.pyplot as plt

# Append the project root directory to the Python path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrate_shapes_robots import HumanShapeGenerator, Custom3DRobotEnv, train_agent, visualize_training, save_model
from walking_agents.walking_agent import DQNAgent

def test_human_shape_generator():
    generator = HumanShapeGenerator()
    shape = generator.generate()
    assert shape.shape == (10, 5, 2, 3)  # 3D shape with color channels
    assert np.any(shape > 0)  # Ensure the shape is not empty

@patch('integrate_shapes_robots.Custom3DRobotEnv')
def test_custom_3d_robot_env(mock_env):
    mock_env.return_value.reset.return_value = np.zeros((64, 64, 64, 3))
    mock_env.return_value.step.return_value = (np.zeros((64, 64, 64, 3)), 1.0, False, {})
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

    # Test remember and replay
    next_state = np.random.rand(*state_size)
    agent.remember(state, action, 1.0, next_state, False)
    agent.replay(32)  # Assuming batch size of 32

@patch('integrate_shapes_robots.Custom3DRobotEnv')
@patch('integrate_shapes_robots.DQNAgent')
@patch('integrate_shapes_robots.visualize_training')
@patch('integrate_shapes_robots.save_model')
def test_training_loop(mock_save_model, mock_visualize, mock_agent_class, mock_env_class):
    mock_env = Mock()
    mock_env.reset.return_value = np.random.rand(64, 64, 64, 3)
    mock_env.step.return_value = (np.random.rand(64, 64, 64, 3), 1.0, False, {})
    mock_env_class.return_value = mock_env

    mock_agent = Mock()
    mock_agent.act.return_value = np.random.rand(6)
    mock_agent.epsilon = 0.5
    mock_agent_class.return_value = mock_agent

    # Run a short training loop
    num_episodes = 2
    train_agent(num_episodes)

    assert mock_env.reset.call_count == num_episodes
    assert mock_agent.act.call_count > 0
    assert mock_agent.remember.call_count > 0
    assert mock_agent.replay.call_count > 0
    mock_visualize.assert_called_once()
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

@patch('integrate_shapes_robots.plt')
def test_visualize_training(mock_plt):
    episodes = list(range(10))
    scores = [random.random() for _ in range(10)]
    epsilons = [random.random() for _ in range(10)]

    visualize_training(episodes, scores, epsilons)

    assert mock_plt.figure.call_count == 1
    assert mock_plt.plot.call_count == 2
    assert mock_plt.xlabel.call_count == 1
    assert mock_plt.ylabel.call_count == 2
    assert mock_plt.legend.call_count == 1
    assert mock_plt.show.call_count == 1

@patch('integrate_shapes_robots.os.path.join')
@patch('integrate_shapes_robots.tf.keras.models.save_model')
def test_save_model(mock_save_model, mock_path_join):
    mock_model = Mock()
    mock_path_join.return_value = '/fake/path/model.h5'

    save_model(mock_model, 'test_model')

    mock_path_join.assert_called_once_with('models', 'test_model.h5')
    mock_save_model.assert_called_once_with(mock_model, '/fake/path/model.h5')

if __name__ == "__main__":
    pytest.main([__file__])
