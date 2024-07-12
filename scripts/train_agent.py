import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import tf_agents

print("TF-Agents version:", tf_agents.__version__)
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.policies import policy_saver

from src.environment import BirdRobotEnvironment
from config.config import (
    CONTROL_FREQUENCY,
    REWARD_COLLISION,
    REWARD_GOAL,
    REWARD_STEP,
    NUM_ITERATIONS,
    COLLECT_STEPS_PER_ITERATION,
    LOG_INTERVAL,
    EVAL_INTERVAL,
    POLICY_DIR,
)
import os

print(f"POLICY_DIR is set to: {POLICY_DIR}")

# Set up the environment
train_py_env = BirdRobotEnvironment()
eval_py_env = BirdRobotEnvironment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Set up the Q-Network
fc_layer_params = (100, 50)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params,
)

# Set up the DQN agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)
agent.initialize()

# Set up the replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000,
)

# Set up the random policy
random_policy = random_tf_policy.RandomTFPolicy(
    train_env.time_step_spec(), train_env.action_spec()
)

# Set up the metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

# Set up the driver
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    random_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=1,
)

# Collect initial data
initial_collect_steps = 1000
collect_driver.run = common.function(collect_driver.run)
for _ in range(initial_collect_steps):
    collect_driver.run()

# Set up the dataset
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=64, num_steps=2
).prefetch(3)

# Set up the iterator
iterator = iter(dataset)

# Set up the training loop
num_iterations = NUM_ITERATIONS
collect_steps_per_iteration = COLLECT_STEPS_PER_ITERATION
log_interval = LOG_INTERVAL
eval_interval = EVAL_INTERVAL

# Create a placeholder TimeStep object using the spec provided by time_step_spec
time_step_spec = agent.policy.time_step_spec
time_step_placeholder = tf.nest.map_structure(
    lambda spec: tf.TensorSpec(shape=[None] + list(spec.shape), dtype=spec.dtype),
    time_step_spec,
)

# Initialize the PolicySaver with the 'signatures' keyword argument to include the 'action' method
concrete_function = agent.policy.action.get_concrete_function(
    time_step=time_step_placeholder
)
print(f"Concrete function for 'action' method: {concrete_function}")
policy_saver = policy_saver.PolicySaver(
    agent.policy, batch_size=None, signatures={"action": concrete_function}
)
print(f"PolicySaver initialized with 'action' method included in signatures: {policy_saver.signatures}")
print(f"Signatures before saving: {policy_saver.signatures}")

# Ensure the 'action' method is a callable TensorFlow graph
assert callable(
    agent.policy.action
), "The 'action' method of the policy is not callable."
print(
    f"The 'action' method of the policy is a callable TensorFlow graph: {agent.policy.action}"
)

# Log the structure of the 'action' method
concrete_function = tf.function(agent.policy.action).get_concrete_function(
    time_step=time_step_placeholder
)
print(f"Concrete function for 'action' method: {concrete_function}")
# Removed lines that attempt to access 'input_signature' and 'output_shapes' attributes

# Training loop
try:
    for _ in range(num_iterations):
        # Collect a few steps and save to the replay buffer
        collect_driver.run()

        # Sample a batch of data from the replay buffer and update the agent's network
        experience, _ = next(iterator)
        observations = experience.observation
        print(f"Shape of observations: {tf.shape(observations)}")
        # Ensure observations have a batch dimension
        batched_observations = tf.nest.map_structure(
            lambda x: tf.expand_dims(x, axis=0) if len(x.shape) == 1 else x,
            observations,
        )
        # Ensure observations have the correct shape expected by the QNetwork
        batched_observations = tf.nest.map_structure(
            lambda x: tf.ensure_shape(
                x, [None] + list(q_net.input_tensor_spec.shape[1:])
            ),
            batched_observations,
        )
        print(
            f"Observations shape: {observations.shape}\nBatched observations shape: {batched_observations.shape}"
        )
        print(f"Shape of batched observations: {tf.shape(batched_observations)}")
        print(f"QNetwork input spec: {q_net.input_tensor_spec}")
        with tf.GradientTape() as tape:
            # Ensure the input to the QNetwork has the correct shape
            q_values, _ = q_net(batched_observations, training=True)
            # Use batched observations for loss calculation
            loss = agent._loss(experience)
            print(f"Shape of input to QNetwork: {tf.shape(batched_observations)}")
        gradients = tape.gradient(loss, agent.trainable_variables)
        agent._optimizer.apply_gradients(zip(gradients, agent.trainable_variables))
        train_loss = loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print("step = {0}: loss = {1}".format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = metric_utils.compute_summaries(
                metrics=train_metrics,
                environment=eval_env,
                policy=agent.policy,
                num_episodes=10,
                tf_summaries=False,
                log=True,
            )
            print("step = {0}: Average Return = {1}".format(step, avg_return))

        # Periodic save of the policy
        if step % (eval_interval // 10) == 0:
            policy_dir = POLICY_DIR
            print(f"Attempting to save policy at step {step} in directory {policy_dir}")
            if not os.path.exists(policy_dir):
                try:
                    os.makedirs(policy_dir)
                    print(f"Directory {policy_dir} created successfully.")
                except Exception as e:
                    print(f"Error creating directory {policy_dir}: {e}")
            try:
                # Debugging: Print the policy object before saving
                print(f"Policy object before saving: {agent.policy}")
                print(
                    f"Concrete function for 'action' method before saving: {concrete_function}"
                )
                policy_saver.save(policy_dir)
                print(f"Policy saved successfully in {policy_dir} at step {step}")
                print(
                    f"Contents of policy directory '{policy_dir}' after saving: {os.listdir(policy_dir)}"
                )
                saved_policy = tf.compat.v2.saved_model.load(policy_dir)
                # Debugging: Print the saved model object after loading
                print(f"Saved model object after loading: {saved_policy}")
                print(f"Attributes of saved policy object: {dir(saved_policy)}")
                print(f"Signatures of the loaded policy: {saved_policy.signatures}")
                if "action" in saved_policy.signatures:
                    print(
                        f"'action' method signature: {saved_policy.signatures['action']}"
                    )
                else:
                    print(
                        "The 'action' method is not present in the saved policy signatures."
                    )
                    # Additional debugging: Print the available methods in the saved policy
                    print(f"Available methods in saved policy: {dir(saved_policy)}")
                    # Additional debugging: Print the concrete function for 'action' method
                    concrete_function = saved_policy.signatures.get("action")
                    print(f"Concrete function for 'action': {concrete_function}")
            except Exception as e:
                print(f"Error saving policy at step {step}: {e}")

    # Final save of the trained policy
    print(f"Attempting final save of policy in directory {policy_dir}")
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)

    try:
        # Use the previously initialized PolicySaver for the final save
        policy_saver.save(policy_dir)
        print(f"Policy saved successfully in {policy_dir}")
        # Print the signatures of the saved model for debugging
        saved_model = tf.saved_model.load(policy_dir)
        print(f"Signatures of the saved model: {saved_model.signatures}")
        # Debugging: Print the attributes of the saved model
        print(f"Attributes of the saved model: {dir(saved_model)}")
        # Debugging: Print the 'action' method of the saved model
        if "action" in saved_model.signatures:
            print(f"'action' method signature: {saved_model.signatures['action']}")
        else:
            print("The 'action' method is not present in the saved model signatures.")
            # Additional debugging: Print the available methods in the saved model
            print(f"Available methods in saved model: {dir(saved_model)}")
    except Exception as e:
        print(f"Error saving policy: {e}")
except Exception as e:
    print(f"An unexpected error occurred during training: {e}")

# Debugging: Confirm 'action' method in policy signatures
if "action" in agent.policy.signatures:
    print(
        f"'action' method is in policy signatures: {agent.policy.signatures['action']}"
    )
else:
    print("'action' method is NOT in policy signatures")

# Additional Debugging
print('Final policy signatures:', policy_saver.signatures)
print("Debugging - Signatures before saving:", policy_saver.signatures)
print("Debugging - Signatures after saving:", saved_policy.signatures)
