import os
import sys
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from src.environment import BirdRobotEnvironment
from config.config import POLICY_DIR

# Append the src directory to the Python path
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'src')
)  # noqa: E402

# Create the environment
eval_py_env = BirdRobotEnvironment()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Load the trained policy
policy_dir = POLICY_DIR
if not os.path.exists(policy_dir):
    raise FileNotFoundError(
        f"Policy directory '{policy_dir}' does not exist. "
        "Please ensure the model is trained and saved correctly."
    )

# Debugging: List contents of policy_dir
print(f"Contents of policy_dir ({policy_dir}): {os.listdir(policy_dir)}")

# Debugging: Print the POLICY_DIR value
print(f"POLICY_DIR: {POLICY_DIR}")

try:
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_dir, time_step_spec=eval_env.time_step_spec()
    )
    # Debugging: Print the loaded policy object
    print(f"Loaded policy: {policy}")
    # Debugging: Check if the 'action' method exists
    if hasattr(policy, 'action'):
        print("The loaded policy has an 'action' method.")
    else:
        print("The loaded policy does NOT have an 'action' method.")
except Exception as e:
    # Debugging: Print the exception details
    print(f"Exception details: {e}")
    raise RuntimeError(
        f"Error loading policy from '{policy_dir}': {e}"
    )

# Run a few episodes and print the results
num_episodes = 10
for _ in range(num_episodes):
    time_step = eval_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward

    print(f'Episode return: {episode_return}')
