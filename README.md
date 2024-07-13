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

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

<!-- Triggering CI workflow -->
<!-- Timestamp: 2024-07-11 02:06:04 -->
