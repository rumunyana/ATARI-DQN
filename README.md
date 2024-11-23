# DQN Agent for Atari Breakout

A Deep Q-Network (DQN) implementation for playing Atari's Breakout game using Stable-Baselines3 and Gymnasium.

## Project Structure
- `train.py`: Script for training the DQN agent
- `play.py`: Script for running the trained agent
- `requirements.txt`: List of required packages

## Training Results
The agent was trained for 50,000 timesteps with the following results:
- Episodes completed: 164
- Mean episode length: 299
- Mean reward: 2.89
- Final exploration rate: 0.05

## Installation

1. Create a virtual environment:
```bash
python -m venv gymvenv
```

2. Activate the virtual environment:
```bash
# Windows
gymvenv\Scripts\activate

# Linux/Mac
source gymvenv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

**Training the agent:**
```bash
python train.py
```

**Playing with trained agent:**
```bash
python play.py
```

## Requirements
- gymnasium
- stable-baselines3[extra]
- ale-py
- numpy

## Training Configuration
- Learning rate: 0.00025
- Buffer size: 10000
- Batch size: 32
- Gamma: 0.99
- Exploration fraction: 0.1
- Training steps: 50000

The trained model is saved as 'policy' and can be loaded to watch the agent play Breakout.