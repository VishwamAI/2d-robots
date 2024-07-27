import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import h5py
from dmlab2d import Lab2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

# Load the 3D shapes dataset
def load_3d_shapes(file_path: str = '/home/ubuntu/3d-shapes/3dshapes.h5') -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
    return images, labels

def visualize_training(episodes, scores, epsilons):
    # Implementation will be added later
    pass

class HumanShapeGenerator:
    def __init__(self):
        self.shape_size = (10, 5, 2)  # Height, width, depth

    def generate(self):
        human_shape = np.zeros((*self.shape_size, 3), dtype=np.uint8)
        # Head
        human_shape[8:10, 2:4, :] = [255, 192, 203]  # Pink
        # Body
        human_shape[4:8, 1:4, :] = [0, 0, 255]  # Blue
        # Arms
        human_shape[5:7, 0:1, :] = [255, 255, 0]  # Yellow
        human_shape[5:7, 4:5, :] = [255, 255, 0]  # Yellow
        # Legs
        human_shape[0:4, 1:2, :] = [0, 255, 0]  # Green
        human_shape[0:4, 3:4, :] = [0, 255, 0]  # Green
        return human_shape

# Create a custom environment using dmlab2d and the 3D shapes
class Custom3DRobotEnv(gym.Env):
    def __init__(self):
        super(Custom3DRobotEnv, self).__init__()
        self.lab2d = Lab2d()  # Updated to use Lab2d class directly
        self.current_shape_index = 0
        self.human_generator = HumanShapeGenerator()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # 3D movement + rotation
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 64, 3), dtype=np.uint8)  # 3D observation space
        self.robot_position = np.zeros(3)
        self.robot_rotation = np.zeros(3)
        self.human_position = np.random.randint(0, 54, size=3)  # Random initial position for human
        self.images, _ = load_3d_shapes()  # Load the 3D shapes dataset

    def reset(self):
        self.state = self.images[np.random.randint(0, len(self.images))]
        self.robot_position = np.zeros(3)
        self.robot_rotation = np.zeros(3)
        self.human_position = np.random.randint(0, 54, size=3)  # Random initial position for human
        return self._get_observation()

    def step(self, action):
        # Implement deep walking mechanism
        self.robot_position += action[:3] * 0.1  # Scale movement
        self.robot_rotation += action[3:] * 0.1  # Scale rotation

        # Check for falling
        if self.robot_position[2] < 0:
            done = True
            reward = -10
        else:
            done = False
            reward = 1

        # Check for failing (e.g., robot tipped over)
        if np.abs(self.robot_rotation).max() > np.pi/2:
            done = True
            reward = -5

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # Get a random 3D shape from the loaded dataset
        obs = self.images[np.random.randint(0, len(self.images))]

        # Add robot
        robot_shape = np.zeros((5, 5, 5, 3), dtype=np.uint8)
        robot_shape[2, 2, 2] = [255, 0, 0]  # Red cube representing the robot
        x, y, z = (self.robot_position + 32).astype(int)  # Center the robot
        x = np.clip(x, 2, obs.shape[0]-3)
        y = np.clip(y, 2, obs.shape[1]-3)
        z = np.clip(z, 2, obs.shape[2]-3)
        obs[x-2:x+3, y-2:y+3, z-2:z+3] = robot_shape

        # Add human
        human_shape = self.human_generator.generate()
        hx, hy, hz = self.human_position
        hx = np.clip(hx, 0, obs.shape[0]-10)
        hy = np.clip(hy, 0, obs.shape[1]-5)
        hz = np.clip(hz, 0, obs.shape[2]-2)
        obs[hx:hx+10, hy:hy+5, hz:hz+2] = human_shape

        return obs

    def render(self, mode='human'):
        if mode == 'human':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            obs = self._get_observation()
            x, y, z = np.where(obs[:,:,:,0] == 255)
            ax.scatter(x, y, z, c='r', marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(f'Robot Position: {self.robot_position}, Rotation: {self.robot_rotation}')
            plt.show()
        return fig  # Return the figure object for testing purposes

# Define the neural network model for the DQN agent
def create_model(input_shape, action_space):
    model = tf.keras.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_space, activation='tanh')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def visualize_environment(env, episode_rewards, episode_lengths):
    # Visualize the 3D environment
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get the current observation
    obs = env._get_observation()

    # Plot the 3D shapes
    x, y, z = np.where(np.any(obs > 0, axis=3))
    colors = obs[x, y, z] / 255.0
    ax.scatter(x, y, z, c=colors, s=50)

    # Plot the robot
    robot_pos = env.robot_position + 32  # Center the robot
    ax.scatter(*robot_pos, c='red', s=200, marker='o', label='Robot')

    # Plot the human
    human_pos = env.human_position
    ax.scatter(*human_pos, c='green', s=200, marker='^', label='Human')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title('3D Environment Visualization')
    plt.savefig('environment_visualization.png')
    plt.close()

    # Visualize the training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(122)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def train_agent(num_episodes):
    env = Custom3DRobotEnv()
    state_size = env.observation_space.shape
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if len(agent.memory) > agent.batch_size:
                agent.replay(agent.batch_size)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")

    visualize_environment(env, episode_rewards, episode_lengths)
    save_model(agent.model, f"dqn_agent_episodes_{num_episodes}")

def save_model(model, name):
    model.save(f"models/{name}.h5")

if __name__ == "__main__":
    from walking_agents.walking_agent import DQNAgent
    num_episodes = 1000  # Or any other desired number of episodes
    env = Custom3DRobotEnv()
    state_size = env.observation_space.shape
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)
    train_agent(num_episodes)
