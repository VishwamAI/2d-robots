import tensorflow as tf
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from config.config import MAX_SPEED, ACCELERATION, TURN_RATE, SENSOR_RANGE, SENSOR_ANGLE, CONTROL_FREQUENCY, SIMULATION_TIME_STEP, COLLISION_DISTANCE, BOUNDARY_MIN, BOUNDARY_MAX, REWARD_COLLISION, REWARD_GOAL, REWARD_STEP, BOUNDARY_OFFSET, INITIAL_ORIENTATION

class BirdRobotEnvironment(py_environment.PyEnvironment):
    """
    Custom environment for a 2D bird robot navigating through obstacles towards a goal.

    The environment supports discrete actions for controlling the bird robot's movement and orientation.
    The state includes the robot's position, orientation, velocity, goal position, and information about obstacles.

    Attributes:
        _action_spec (array_spec.BoundedArraySpec): Specification of the action space.
        _obstacles (List[np.ndarray]): List of obstacle positions.
        _observation_spec (array_spec.BoundedArraySpec): Specification of the observation space.
        _state (np.ndarray): Current state of the environment.
        _episode_ended (bool): Flag indicating whether the episode has ended.
    """

    ACTION_ACCELERATE = 0
    ACTION_DECELERATE = 1
    ACTION_TURN_RIGHT = 2
    ACTION_TURN_LEFT = 3
    ACTION_MOVE_FORWARD = 4
    ACTION_MOVE_BACKWARD = 5

    def __init__(self) -> None:
        """
        Initializes the BirdRobotEnvironment with action and observation specifications,
        and sets up the initial state and obstacles.
        """
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=self.ACTION_ACCELERATE, maximum=self.ACTION_MOVE_BACKWARD, name='action')
        self._obstacles = [np.array([20, 20]), np.array([40, 40]), np.array([60, 60])]  # Example obstacles
        num_obstacles = len(self._obstacles)
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(6 + num_obstacles * 3,), dtype=np.float32, minimum=BOUNDARY_MIN, maximum=BOUNDARY_MAX, name='observation')
        # State array structure: [x, y, orientation, velocity, goal_x, goal_y, obstacle_x1, obstacle_y1, distance1, ...]
        self._state = np.zeros(6 + num_obstacles * 3, dtype=np.float32)  # Initialize state array
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """
        Resets the environment to its initial state at the start of a new episode.
        """
        num_obstacles = len(self._obstacles)
        self._state = np.zeros(6 + num_obstacles * 3, dtype=np.float32)  # Reset to initial position and goal
        self._state[:2] = [BOUNDARY_MIN + BOUNDARY_OFFSET, BOUNDARY_MIN + BOUNDARY_OFFSET]  # Set initial position (x, y)
        self._state[2] = INITIAL_ORIENTATION  # Set initial orientation
        self._state[3] = 0.0  # Set initial velocity
        self._state[4:6] = [BOUNDARY_MAX - BOUNDARY_OFFSET, BOUNDARY_MAX - BOUNDARY_OFFSET]  # Set goal position (goal_x, goal_y)
        for i, obstacle in enumerate(self._obstacles):
            self._state[6 + i * 3:8 + i * 3] = obstacle  # Set obstacle positions (obstacle_x, obstacle_y)
            self._state[8 + i * 3] = SENSOR_RANGE  # Initialize obstacle distances to SENSOR_RANGE
        self._episode_ended = False
        return ts.restart(self._get_observation())

    def _step(self, action):
        """
        Updates the environment state based on the action taken by the agent.

        Args:
            action (int): The action to be taken by the agent. Valid actions are defined by the ACTION_* constants.

        Returns:
            ts.TimeStep: A TimeStep object representing the new state of the environment. The TimeStep object contains:
                - observation: The new state of the environment.
                - reward: The reward received for taking the action.
                - step_type: The type of the time step (FIRST, MID, or LAST).
                - discount: The discount factor for future rewards.

        Termination Conditions:
            - The episode ends if the bird robot collides with an obstacle or goes out of bounds, resulting in a reward of REWARD_COLLISION.
            - The episode ends if the bird robot reaches the goal, resulting in a reward of REWARD_GOAL.
            - If the episode has ended, a termination TimeStep is returned with a reward of 0.0.
            - Otherwise, a transition TimeStep is returned with a reward of REWARD_STEP and a discount factor of 0.9.
        """
        if self._episode_ended:
            return self.reset()

        # Update the state based on the action
        if action == self.ACTION_ACCELERATE:
            self._state[3] += ACCELERATION  # Increase velocity
        elif action == self.ACTION_DECELERATE:
            self._state[3] -= ACCELERATION  # Decrease velocity
        elif action == self.ACTION_TURN_RIGHT:
            self._state[2] = (self._state[2] + TURN_RATE) % 360  # Turn right
        elif action == self.ACTION_TURN_LEFT:
            self._state[2] = (self._state[2] - TURN_RATE) % 360  # Turn left
        elif action == self.ACTION_MOVE_FORWARD:
            self._state[0] += self._state[3] * np.cos(np.deg2rad(self._state[2])) * SIMULATION_TIME_STEP  # Move forward
            self._state[1] += self._state[3] * np.sin(np.deg2rad(self._state[2])) * SIMULATION_TIME_STEP  # Move forward
        elif action == self.ACTION_MOVE_BACKWARD:
            self._state[0] -= self._state[3] * np.cos(np.deg2rad(self._state[2])) * SIMULATION_TIME_STEP  # Move backward
            self._state[1] -= self._state[3] * np.sin(np.deg2rad(self._state[2])) * SIMULATION_TIME_STEP  # Move backward

        # Ensure the orientation stays within 0 to 360 degrees
        self._state[2] = self._state[2] % 360

        # Ensure the velocity does not exceed MAX_SPEED
        self._state[3] = np.clip(self._state[3], -MAX_SPEED, MAX_SPEED)

        # Update obstacle information in the state
        for i, obstacle in enumerate(self._obstacles):
            self._state[6 + i * 3] = obstacle[0]
            self._state[7 + i * 3] = obstacle[1]
            distance = np.linalg.norm(self._state[:2] - obstacle)
            angle = np.arctan2(obstacle[1] - self._state[1], obstacle[0] - self._state[0]) - np.deg2rad(self._state[2])
            if distance <= SENSOR_RANGE and np.abs(np.rad2deg(angle)) <= SENSOR_ANGLE / 2:
                self._state[8 + i * 3] = distance
            else:
                self._state[8 + i * 3] = SENSOR_RANGE

        # Check if the episode has ended
        if np.any(self._state[:2] < BOUNDARY_MIN + BOUNDARY_OFFSET) or np.any(self._state[:2] > BOUNDARY_MAX - BOUNDARY_OFFSET):
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=REWARD_COLLISION)

        # Check for collisions with obstacles
        for obstacle in self._obstacles:
            if np.linalg.norm(self._state[:2] - obstacle) < COLLISION_DISTANCE:
                self._episode_ended = True
                return ts.termination(self._get_observation(), reward=REWARD_COLLISION)

        # Check if the goal is reached
        if np.linalg.norm(self._state[:2] - self._state[4:6]) < COLLISION_DISTANCE:
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward=REWARD_GOAL)

        if self._episode_ended:
            return ts.termination(self._get_observation(), reward=0.0)
        else:
            return ts.transition(self._get_observation(), reward=REWARD_STEP, discount=0.9)

    def _get_observation(self):
        """
        Returns the current state of the environment.
        """
        return self._state

# Create the environment
train_py_env = BirdRobotEnvironment()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
