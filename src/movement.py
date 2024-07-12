import numpy as np
from config import MAX_SPEED, ACCELERATION, TURN_RATE, SIMULATION_TIME_STEP

class BirdRobotMovement:
    """
    Movement system for the 2D bird robot.

    This class provides methods to update the bird robot's position and orientation based on control commands.
    """

    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.velocity = 0.0

    def update_position(self, control_command):
        """
        Updates the bird robot's position and orientation based on the control command.

        Args:
            control_command (dict): A dictionary containing control commands with keys
            'accelerate', 'decelerate', 'turn_right', 'turn_left', 'move_forward',
            'move_backward'.

        Returns:
            np.ndarray: The new position of the bird robot [x, y].
        """
        if control_command.get('accelerate'):
            self.velocity = np.clip(self.velocity + ACCELERATION, -MAX_SPEED, MAX_SPEED)
        if control_command.get('decelerate'):
            self.velocity = np.clip(self.velocity - ACCELERATION, -MAX_SPEED, MAX_SPEED)
        if control_command.get('turn_right'):
            self.orientation = (self.orientation + TURN_RATE) % 360
        if control_command.get('turn_left'):
            self.orientation = (self.orientation - TURN_RATE) % 360
        if control_command.get('move_forward'):
            self.position[0] += (
                self.velocity * np.cos(np.deg2rad(self.orientation)) * SIMULATION_TIME_STEP
            )
            self.position[1] += (
                self.velocity * np.sin(np.deg2rad(self.orientation)) * SIMULATION_TIME_STEP
            )
        if control_command.get('move_backward'):
            self.position[0] -= (
                self.velocity * np.cos(np.deg2rad(self.orientation)) * SIMULATION_TIME_STEP
            )
            self.position[1] -= (
                self.velocity * np.sin(np.deg2rad(self.orientation)) * SIMULATION_TIME_STEP
            )

        return self.position
