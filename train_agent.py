import tensorflow as tf
import tf_agents
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

from environment import BirdRobotEnvironment
from config import CONTROL_FREQUENCY, REWARD_COLLISION, REWARD_GOAL, REWARD_STEP

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
num_iterations = 20000
collect_steps_per_iteration = 1
log_interval = 200
eval_interval = 1000

# Training loop
for _ in range(num_iterations):
    # Collect a few steps and save to the replay buffer
    collect_driver.run()

    # Sample a batch of data from the replay buffer and update the agent's network
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = metric_utils.compute_summaries(
            metrics=train_metrics,
            environment=eval_env,
            policy=agent.policy,
            num_episodes=10)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
