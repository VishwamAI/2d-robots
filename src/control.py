import numpy as np
from config import (
    MAX_SPEED,
    ACCELERATION,
    TURN_RATE,
    SIMULATION_TIME_STEP,
)


class BirdRobotControl:
    """
    Control system for the 2D bird robot.

    This class provides methods to control the bird robot's movement and
    orientation.
    """

    def __init__(self):
        self.velocity = 0.0
        self.orientation = 0.0

    def accelerate(self):
        """
        Increases the velocity of the bird robot.
        """
        self.velocity = np.clip(
            self.velocity + ACCELERATION, -MAX_SPEED, MAX_SPEED
        )

    def decelerate(self):
        """
        Decreases the velocity of the bird robot.
        """
        self.velocity = np.clip(
            self.velocity - ACCELERATION, -MAX_SPEED, MAX_SPEED
        )

    def turn_right(self):
        """
        Turns the bird robot to the right.
        """
        self.orientation = (self.orientation + TURN_RATE) % 360

    def turn_left(self):
        """
        Turns the bird robot to the left.
        """
        self.orientation = (self.orientation - TURN_RATE) % 360

    def move_forward(self, position):
        """
        Moves the bird robot forward based on its current velocity and
        orientation.

        Args:
            position (np.ndarray): The current position of the bird robot
            [x, y].

        Returns:
            np.ndarray: The new position of the bird robot [x, y].
        """
        position[0] += (
            self.velocity * np.cos(np.deg2rad(self.orientation))
            * SIMULATION_TIME_STEP
        )
        position[1] += (
            self.velocity * np.sin(np.deg2rad(self.orientation))
            * SIMULATION_TIME_STEP
        )
        return position

    def move_backward(self, position):
        """
        Moves the bird robot backward based on its current velocity and
        orientation.

        Args:
            position (np.ndarray): The current position of the bird robot
            [x, y].

        Returns:
            np.ndarray: The new position of the bird robot [x, y].
        """
        position[0] -= (
            self.velocity * np.cos(np.deg2rad(self.orientation))
            * SIMULATION_TIME_STEP
        )
        position[1] -= (
            self.velocity * np.sin(np.deg2rad(self.orientation))
            * SIMULATION_TIME_STEP
        )
        return position
