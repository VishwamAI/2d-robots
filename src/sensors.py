import numpy as np
from config import SENSOR_RANGE, SENSOR_ANGLE


class BirdRobotSensors:
    """
    Sensor system for the 2D bird robot.

    This class provides methods to simulate sensor input, such as detecting
    obstacles and the bird robot's current state relative to the environment.
    """

    def __init__(self, obstacles):
        self.obstacles = obstacles

    def detect_obstacles(self, position, orientation):
        """
        Detects obstacles within the sensor range and angle.

        Args:
            position (np.ndarray): The current position of the bird robot
            [x, y].
            orientation (float): The current orientation of the bird robot in
            degrees.

        Returns:
            List[float]: A list of distances to detected obstacles. If an
            obstacle is not detected, the distance is set to SENSOR_RANGE.
        """
        distances = []
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle)
            angle_to_obstacle = np.arctan2(
                obstacle[1] - position[1], obstacle[0] - position[0]
            )
            angle = angle_to_obstacle - np.deg2rad(orientation)
            if (
                distance <= SENSOR_RANGE
                and np.abs(np.rad2deg(angle)) <= SENSOR_ANGLE / 2
            ):
                distances.append(distance)
            else:
                distances.append(SENSOR_RANGE)
        return distances

    def get_state(self, position, orientation):
        """
        Returns the current state of the bird robot, including its position,
        orientation, and detected obstacles.

        Args:
            position (np.ndarray): The current position of the bird robot
            [x, y].
            orientation (float): The current orientation of the bird robot in
            degrees.

        Returns:
            np.ndarray: The current state of the bird robot [x, y, orientation,
            distance1, distance2, ...].
        """
        distances = self.detect_obstacles(position, orientation)
        state = np.concatenate(
            ([position[0], position[1], orientation], distances)
        )
        return state
