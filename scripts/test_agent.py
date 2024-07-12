import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
import numpy as np
import os
import sys

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import BirdRobotEnvironment
from config.config import POLICY_DIR

# Print the current working directory and Python path for debugging
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Print the TensorFlow Agents library version for debugging
import tf_agents

print("TF-Agents version:", tf_agents.__version__)

# Create the environment
eval_py_env = BirdRobotEnvironment()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Load the trained policy
policy_dir = POLICY_DIR
if not os.path.exists(policy_dir):
    raise FileNotFoundError(
        f"Policy directory '{policy_dir}' does not exist. Please ensure the model is trained and saved correctly."
    )

try:
    # Load the saved policy using tf.saved_model.load
    saved_policy = tf.saved_model.load(policy_dir)
    # Debugging: Print the contents of the policy directory and inspect the loaded policy object
    print(f"Contents of policy directory '{policy_dir}': {os.listdir(policy_dir)}")
    print(f"Loaded policy object: {saved_policy}")
    print(f"Attributes of loaded policy object: {dir(saved_policy)}")
except Exception as e:
    raise RuntimeError(f"Error loading policy from '{policy_dir}': {e}")

# Print TensorFlow version for debugging
print("TensorFlow version:", tf.__version__)

# Run a few episodes and print the results
num_episodes = 10
for _ in range(num_episodes):
    time_step = eval_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        try:
            action_step = saved_policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            episode_return += time_step.reward
        except AttributeError as e:
            print(f"AttributeError during policy execution: {e}")
            print(f"Attributes of policy object: {dir(saved_policy)}")
            raise

    print("Episode return: {}".format(episode_return))
