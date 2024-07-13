import sys

sys.path.append("..")  # Adds the project root to the Python path

from tf_agents.policies import random_tf_policy
from tf_agents.policies import PolicySaver
import tensorflow as tf
from src.environment import BirdRobotEnvironment
from tf_agents.environments import tf_py_environment
from tensorflow.python.saved_model import nested_structure_coder

# Initialize the environment
train_py_env = BirdRobotEnvironment()
env = tf_py_environment.TFPyEnvironment(train_py_env)

# Assuming 'env' is already defined and initialized
policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

# Register the 'action' method as a concrete function
time_step_spec = policy.time_step_spec
time_step_placeholder = tf.nest.map_structure(
    lambda spec: tf.TensorSpec(shape=[None] + list(spec.shape), dtype=spec.dtype),
    time_step_spec,
)

# Ensure the 'action' method is a tf.function
@tf.function(input_signature=[time_step_placeholder])
def action_fn(time_step):
    return policy.action(time_step)

# Initialize the PolicySaver without the 'signatures' argument
policy_saver = PolicySaver(policy, batch_size=None)

# Save the policy
policy_dir = "/tmp/policy_saver_test"
policy_saver.save(policy_dir)

# Load the saved policy
saved_policy = tf.compat.v2.saved_model.load(policy_dir)

# Check if the 'action' method is present in the signatures
if "action" in saved_policy.signatures:
    print(f"'action' method signature: {saved_policy.signatures['action']}")
else:
    print("The 'action' method is not present in the saved policy signatures.")
