# RL_aerohopper

Reinforcement learning environment for a single-legged hopping robot with quadrotor propellers.

## Tech Stack

- **Isaac Gym**: Physics simulation engine
- **PyTorch**: Deep learning framework  
- **PPO-CSE**: Policy gradient algorithm
- **Go1 Gym**: Base environment framework from Walk These Ways

## Architecture

Hybrid control system:
- **Leg**: 3 DOF (ball_x, ball_y, spring) controlled by policy
- **Propellers**: 4 rotors with automatic PD control for attitude stabilization

Policy outputs 3 actions for leg control. Rotors are managed automatically via PD controller to maintain orientation.

## Demo

![Hybrid Hopper Demo](hybrid_hopper_play.gif)

## Files

- `hybrid_hopper_env.py`: Environment implementation
- `hybrid_hopper_config.py`: Configuration parameters
- `play_hybrid_record.py`: Script for recording demonstration videos
