# Configuration parameters for the 2D Bird Robots Project

# Movement parameters
MAX_SPEED = 10.0  # Maximum speed of the bird robots
ACCELERATION = 2.0  # Acceleration rate of the bird robots
TURN_RATE = 45.0  # Turn rate of the bird robots in degrees per second

# Sensor parameters
SENSOR_RANGE = 100.0  # Range of the sensors in units
SENSOR_ANGLE = 120.0  # Field of view of the sensors in degrees

# Control parameters
CONTROL_FREQUENCY = 10  # Frequency of the control loop in Hz

# Other parameters
SIMULATION_TIME_STEP = 0.1  # Time step for the simulation in seconds

# Collision parameters
COLLISION_DISTANCE = 1.0  # Distance threshold for collision detection

# Boundary parameters
BOUNDARY_MIN = 0  # Minimum boundary value for the environment
BOUNDARY_MAX = 200  # Maximum boundary value for the environment
BOUNDARY_OFFSET = 10  # Offset from boundaries for initial and goal positions

# Initial state parameters
INITIAL_ORIENTATION = 0.0  # Initial orientation of the bird robots

# Reward parameters
REWARD_COLLISION = -10.0  # Reward for collision
REWARD_GOAL = 10.0  # Reward for reaching the goal
REWARD_STEP = 1.0  # Reward for each step taken

# Training parameters
NUM_ITERATIONS = 5000  # Total number of training iterations
COLLECT_STEPS_PER_ITERATION = 1  # Number of steps to collect per iteration
LOG_INTERVAL = 200  # Interval for logging training progress
EVAL_INTERVAL = 1000  # Interval for evaluating the agent's performance

# Policy directory
import os

POLICY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "policy"
)  # Directory to save the trained policy
