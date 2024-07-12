import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from environment import BirdRobotEnvironment
from config import POLICY_DIR

# Set up the environment
train_py_env = BirdRobotEnvironment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

# Set up the DQN agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100,)
)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)
agent.initialize()

# Register the 'action' method as a concrete function
agent.policy.action = tf.function(agent.policy.action)
print(
    f"Registered 'action' method as a concrete function: "
    f"{agent.policy.action}"
)

# Save the 'action' method explicitly
concrete_action_function = agent.policy.action.get_concrete_function(
    tf.TensorSpec(
        shape=[None, *train_env.observation_spec().shape],
        dtype=train_env.observation_spec().dtype
    )
)
tf.saved_model.save(
    agent.policy, POLICY_DIR,
    signatures={'action': concrete_action_function}
)

# Initialize the PolicySaver
tf_policy_saver = policy_saver.PolicySaver(
    agent.policy,
    batch_size=1,  # Specify a batch size of 1
    use_nest_path_signatures=True,
    train_step=train_step_counter  # Include the train step counter
)

# Save the policy
# Commenting out the PolicySaver save operation to preserve the explicitly saved 'action' method
# tf_policy_saver.save(POLICY_DIR)
