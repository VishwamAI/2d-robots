import os
import sys

# Append the src directory to the Python path
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'src')
)  # noqa: E402

from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from src.environment import BirdRobotEnvironment
from config.config import POLICY_DIR

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

# Print the contents of the policy directory for debugging
print(f"Contents of policy directory '{policy_dir}': {os.listdir(policy_dir)}")

try:
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_dir,
        time_step_spec=eval_env.time_step_spec(),
        action_spec=eval_env.action_spec()
    )
    if hasattr(policy, 'action'):
        print("The loaded policy has an 'action' method.")
    else:
        print("The loaded policy does NOT have an 'action' method.")
        print(f"Available attributes of the policy object: {dir(policy)}")
        # Print the details of the policy object for debugging
        print(f"Policy object details: {policy}")
        # Print the signatures of the loaded policy
        print(f"Policy signatures: {policy.signatures}")
        # Attempt to retrieve the concrete function for 'action'
        if 'action' in policy.signatures:
            print("The 'action' method is present in the policy signatures.")
        else:
            print("The 'action' method is NOT present in the policy signatures.")
except Exception as e:
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

# New debugging statement to confirm changes
print("Debugging statement: End of script reached.")
