import sys
sys.path.append('..')  # Adds the project root to the Python path

from tf_agents.policies import random_tf_policy
from tf_agents.policies import PolicySaver
import tensorflow as tf
from src.environment import BirdRobotEnvironment
from tf_agents.environments import tf_py_environment

# Initialize the environment
train_py_env = BirdRobotEnvironment()
env = tf_py_environment.TFPyEnvironment(train_py_env)

# Assuming 'env' is already defined and initialized
policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

# Initialize the PolicySaver with the 'action' method included in the signatures
policy_saver = PolicySaver(policy, batch_size=None)

# Save the policy
policy_dir = '/tmp/policy_saver_test'
policy_saver.save(policy_dir)

# Load the saved policy
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# Check if the 'action' method is present in the signatures
if 'action' in saved_policy.signatures:
    print(f"'action' method signature: {saved_policy.signatures['action']}")
else:
    print("The 'action' method is not present in the saved policy signatures.")
