# 2D Bird Robots Project

## Introduction
This project aims to develop a 2D simulation of bird robots with reinforcement learning capabilities. The simulation includes various functionalities such as movement, control, sensor integration, and a digital room environment for practice. The architecture is designed to be modular and extensible, allowing for easy integration of new features and functionalities.

## Project Structure
- `src/`
  - `environment.py`: Defines the reinforcement learning environment for the bird robots.
  - `control.py`: Implements the control system to manage the bird robot's movement and orientation.
  - `movement.py`: Develops the movement system to update the bird robot's position based on control commands.
  - `sensors.py`: Creates the sensor system to detect obstacles and provide the robot's state.
- `train_agent.py`: Script for training the reinforcement learning agent.
- `test_agent.py`: Script for testing the trained reinforcement learning agent.
- `config/`
  - `config.py`: Defines configuration parameters for the project.
- `requirements.txt`: Lists the project dependencies.
- `ARCHITECTURE.md`: Provides an overview of the project architecture.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAI/2d-robots.git
   cd 2d-robots
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the reinforcement learning agent:
   ```bash
   python3 train_agent.py
   ```
   **Note:** The training process can take a significant amount of time. Please ensure that the process completes before proceeding to the next step. During training, progress will be reported with loss values at regular intervals. The training is complete when the final model is saved in the `policy` directory.

5. Test the trained reinforcement learning agent:
   ```bash
   python3 test_agent.py
   ```
   **Note:** The `test_agent.py` script will run multiple episodes to evaluate the performance of the trained agent. The output will include the results of these episodes, providing insights into the agent's behavior and performance.

## Error Handling
### Policy Directory Not Found
If you encounter a `FileNotFoundError` indicating that the policy directory does not exist, ensure that the model has been trained and saved correctly. The `train_agent.py` script should create the `policy` directory and save the trained model. If the directory is missing, rerun the `train_agent.py` script to train and save the model.

### TensorFlow Errors
If you encounter TensorFlow-specific errors, ensure that TensorFlow and TF-Agents are installed correctly in your virtual environment. Refer to the `requirements.txt` file for the required versions and install them using `pip install -r requirements.txt`.

### AttributeError in `test_agent.py`
If you encounter an `AttributeError` indicating that the 'action' method is not found in the loaded policy object, ensure that the policy is saved correctly with the 'serving_default' signature. The `train_agent.py` script should include the 'action' method in the 'serving_default' signature when saving the policy. If the error persists, retrain the policy and save it again, ensuring that the 'action' method is included in the 'serving_default' signature.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests. When submitting a pull request, provide a clear description of the changes and the motivation behind them.

## Contact
For support or to report issues, please open an issue on the GitHub repository or contact the project maintainers.

## License
This project is licensed under the Apache License, Version 2.0. See the LICENSE file for more details.

## Acknowledgments
We would like to thank the contributors and the open-source community for their valuable resources and support in making this project possible. Special thanks to the Vishwam Airobotics team for their dedication and hard work.

## Devin Run
This project was developed with the assistance of [Devin](https://preview.devin.ai/devin/55c5ca45dd624ecca086fe995ce1368b).
