# 2d-walking

2d-walking with reinforcement learning and development

## Description

This project aims to create a 2D walking model for robotics using reinforcement learning. The goal is to explore different types of walking and integrate them into this model. The project includes the implementation of a Soft Actor-Critic (SAC) agent to achieve digital walking.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAirobotics/2d-robots.git
   cd 2d-robots
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the agent:
   ```bash
   python agents/train_agent.py
   ```

2. Evaluate the agent:
   ```bash
   python agents/evaluate_agent.py
   ```

## Running Tests

To run the tests locally, use the following command:
```bash
pytest
```

## Continuous Integration

This project uses GitHub Actions for Continuous Integration (CI). The CI workflow is defined in the `.github/workflows/ci.yml` file. It includes the following steps:
- Set up Python environment
- Install dependencies
- Lint the code with flake8
- Run tests with pytest

## Project Structure

The project is organized into the following directories:
- `agents/`: Contains the reinforcement learning agents, including `train_agent.py` and `evaluate_agent.py`.
- `behaviors/`: Contains the implementation of different behaviors and exercises for the walking model.
- `environments/`: Contains the environment definitions for the 2D walking model.
- `models/`: Contains the trained models and related files.
- `tests/`: Contains the test files for validating the functionality of the project.
- `utils/`: Contains utility scripts and helper functions.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
