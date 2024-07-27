import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from google_deepmind_3d_shapes.shape_generator import ShapeGenerator
from google_deepmind_lab2d import Lab2D
from vishwamai_2d_robots.walking_agents import DQNAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize the 3D shape generator
shape_generator = ShapeGenerator()

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

# Create a custom environment using Lab2D and the 3D shapes
class Custom3DRobotEnv(gym.Env):
    def __init__(self):
        super(Custom3DRobotEnv, self).__init__()
        self.lab2d = Lab2D()
        self.shape_generator = shape_generator
        self.human_generator = HumanShapeGenerator()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # 3D movement + rotation
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 64, 3), dtype=np.uint8)  # 3D observation space
        self.robot_position = np.zeros(3)
        self.robot_rotation = np.zeros(3)
        self.human_position = np.random.randint(0, 54, size=3)  # Random initial position for human

    def reset(self):
        self.state = self.shape_generator.generate()
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
        obs = self.shape_generator.generate()

        # Add robot
        robot_shape = np.zeros((5, 5, 5, 3), dtype=np.uint8)
        robot_shape[2, 2, 2] = [255, 0, 0]  # Red cube representing the robot
        x, y, z = (self.robot_position + 32).astype(int)  # Center the robot
        obs[x-2:x+3, y-2:y+3, z-2:z+3] = robot_shape

        # Add human
        human_shape = self.human_generator.generate()
        hx, hy, hz = self.human_position
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

# Create the custom 3D robot environment
env = Custom3DRobotEnv()

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

# Initialize the DQN agent with the custom environment
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]
agent = DQNAgent(state_size, action_size, create_model)

# Training loop
episodes = 1000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, *state_size])
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Save the trained model
model_save_path = 'trained_3d_robot_model.h5'
agent.model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Visualize results
test_episodes = 5
for episode in range(test_episodes):
    state = env.reset()
    done = False
    while not done:
        state = np.reshape(state, [1, *state_size])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
    print(f"Test Episode {episode + 1} completed")
