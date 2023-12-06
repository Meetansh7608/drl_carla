# Autonomous Driving Project for CSCE 642 Deep Reinforcement Learning

## Deployment Guide

Welcome to the Autonomous Driving Project deployment guide, crafted by Jaydeep and Meetansh. This guide offers comprehensive instructions for deploying both agents: DQN and DDPG.

### Prerequisites

Before diving into the deployment, ensure your system has the following dependencies:

- Carla
- Pygame
- PyTorch
- OpenCV
- NumPy

### Local System Setup

Begin by setting up your local system with the necessary dependencies.

#### Carla

Install the latest version of Carla using the provided repository URL and instructions at [Carla Documentation](https://carla.readthedocs.io/en/latest/build_windows/).

#### Clone this Repository

To get started, clone this repository to your GitHub account:

```bash
git clone [repository-url]
```

### Training Your Model

This section covers the training process for the model, which typically takes 8-10 hours before visible signs of learning on your setup. Adjust the training process by modifying variables in `config.py`.

- `target_speed`: Desired speed for the car in km/h
- `max_iter`: Maximum number of steps before starting a new episode
- `start_buffer`: Number of episodes to run before initiating training
- `train_freq`: Training frequency (set to 1 to train every step, 2 to train every 2 steps, etc.)
- `save_freq`: Model saving frequency
- `start_ep`: Episode to start training from (update if the program crashes during training)
- `max_dist_from_waypoint`: Maximum distance from waypoint/road before terminating the episode

### Evaluating the Model

To evaluate your agent's performance, run the following code:

```python
env = SimEnv(visuals=False)
```

This call initializes the simulation environment. Set `visuals` to `False` to disable the Pygame window or `True` to enable it along with the simulator.

Load a trained or pre-trained model for evaluation. The number 200 indicates that this model was trained for 200 episodes. If you've trained your own model for 200 episodes, you'll find the following files in the weights folder:

- `model_ep_200_optimizer`
- `model_ep_200_Q`

Load the model with:

```python
model.load('weights/model_ep_200')
```

Reach out to Jaydeep --> jdr@tamu.edu and Meetansh --> guptameetansh@tamu.edu
