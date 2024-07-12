import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
import numpy as np
import os
import sys

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.environment import BirdRobotEnvironment
from config.config import POLICY_DIR

# Print the current working directory and Python path for debugging
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Create the environment
eval_py_env = BirdRobotEnvironment()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Load the trained policy
policy_dir = POLICY_DIR
if not os.path.exists(policy_dir):
    raise FileNotFoundError(f"Policy directory '{policy_dir}' does not exist. Please ensure the model is trained and saved correctly.")

try:
    saved_policy = tf.compat.v2.saved_model.load(policy_dir)
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(saved_policy, time_step_spec=eval_env.time_step_spec())
    # Check if the policy object has the 'action' method
    if not hasattr(policy, 'action'):
        raise AttributeError("Loaded policy object does not have an 'action' method.")
except Exception as e:
    raise RuntimeError(f"Error loading policy from '{policy_dir}': {e}")

# Run a few episodes and print the results
num_episodes = 10
for _ in range(num_episodes):
    time_step = eval_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward

    print('Episode return: {}'.format(episode_return))
