import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver, SavedModelPyTFEagerPolicy
from tf_agents.trajectories import time_step as ts
from src.environment import BirdRobotEnvironment
from config.config import POLICY_DIR

def load_policy(policy_dir, time_step_spec, action_spec):
    """
    Load the trained policy from the specified directory.

    Args:
        policy_dir (str): The directory where the policy is saved.
        time_step_spec: The time step specification.
        action_spec: The action specification.

    Returns:
        tf_policy: The loaded policy.
    """
    try:
        policy = SavedModelPyTFEagerPolicy(policy_dir, time_step_spec=time_step_spec, action_spec=action_spec)
        print(f"Policy loaded successfully from {policy_dir}")
        print(f"Loaded policy object: {policy}")
        print(f"Policy methods: {dir(policy)}")
        return policy
    except Exception as e:
        print(f"Error loading policy: {e}")
        print(f"Policy directory: {policy_dir}")
        print(f"Time step spec: {time_step_spec}")
        print(f"Action spec: {action_spec}")
        return None

def evaluate_policy(policy, environment, num_episodes=10):
    """
    Evaluate the policy by running it through a series of episodes in the environment.

    Args:
        policy: The trained policy to be evaluated.
        environment: The environment in which to evaluate the policy.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        List[float]: A list of total rewards for each episode.
    """
    total_rewards = []
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_reward = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward
        total_rewards.append(episode_reward.numpy())
    return total_rewards

def main():
    # Create the environment
    env = BirdRobotEnvironment()
    tf_env = tf_py_environment.TFPyEnvironment(env)

    # Load the trained policy
    policy = load_policy(POLICY_DIR, tf_env.time_step_spec(), tf_env.action_spec())
    if policy is None:
        print("Failed to load policy. Exiting.")
        return

    # Evaluate the policy
    rewards = evaluate_policy(policy, tf_env)
    print(f"Total rewards for each episode: {rewards}")
    print(f"Average reward: {np.mean(rewards)}")

if __name__ == "__main__":
    main()
