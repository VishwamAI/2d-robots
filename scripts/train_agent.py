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
from config.config import CONTROL_FREQUENCY, REWARD_COLLISION, REWARD_GOAL, REWARD_STEP, NUM_ITERATIONS, COLLECT_STEPS_PER_ITERATION, LOG_INTERVAL, EVAL_INTERVAL, POLICY_DIR
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
    fc_layer_params=fc_layer_params)

# Set up the DQN agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()

# Set up the replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000)

# Set up the random policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

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
    num_steps=1)

# Collect initial data
initial_collect_steps = 1000
collect_driver.run = common.function(collect_driver.run)
for _ in range(initial_collect_steps):
    collect_driver.run()

# Set up the dataset
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2).prefetch(3)

# Set up the iterator
iterator = iter(dataset)

# Set up the training loop
num_iterations = NUM_ITERATIONS
collect_steps_per_iteration = COLLECT_STEPS_PER_ITERATION
log_interval = LOG_INTERVAL
eval_interval = EVAL_INTERVAL

# Initialize the PolicySaver
policy_saver = policy_saver.PolicySaver(agent.policy, batch_size=None, signatures={'action': tf.function(agent.policy.action).get_concrete_function()})

# Ensure the 'action' method is a callable TensorFlow graph
assert callable(agent.policy.action), "The 'action' method of the policy is not callable."
print(f"The 'action' method of the policy is a callable TensorFlow graph: {agent.policy.action}")

# Training loop
try:
    for _ in range(num_iterations):
        # Collect a few steps and save to the replay buffer
        collect_driver.run()

        # Sample a batch of data from the replay buffer and update the agent's network
        experience, _ = next(iterator)
        with tf.GradientTape() as tape:
            loss = agent._loss(experience)
        gradients = tape.gradient(loss, agent.trainable_variables)
        agent._optimizer.apply_gradients(zip(gradients, agent.trainable_variables))
        train_loss = loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = metric_utils.compute_summaries(
                metrics=train_metrics,
                environment=eval_env,
                policy=agent.policy,
                num_episodes=10,
                tf_summaries=False,
                log=True)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

        # Attempt to save the policy more frequently for debugging purposes
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
                policy_saver.save(policy_dir, signatures={'action': tf.function(agent.policy.action).get_concrete_function()})
                print(f"Policy saved successfully in {policy_dir} at step {step}")
            except Exception as e:
                print(f"Error saving policy at step {step}: {e}")

    # Final save of the trained policy
    print(f"Attempting final save of policy in directory {policy_dir}")
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)

    try:
        policy_saver.save(policy_dir, signatures={'action': tf.function(agent.policy.action).get_concrete_function()})
        print(f"Policy saved successfully in {policy_dir}")
    except Exception as e:
        print(f"Error saving policy: {e}")
except Exception as e:
    print(f"An unexpected error occurred during training: {e}")
