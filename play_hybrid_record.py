#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/home/abc/walk_these_ways_learning/walk-these-ways')

import isaacgym
import torch
import numpy as np
import subprocess
import shutil
from pathlib import Path

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.hybrid_hopper.hybrid_hopper_config import config_hybrid_hopper
from go1_gym.envs.hybrid_hopper.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse.actor_critic import ActorCritic
from isaacgym import gymapi

def play_hybrid_hopper_with_recording():
    config_hybrid_hopper(Cfg)
    
    Cfg.env.num_observations = 36
    Cfg.env.num_scalar_observations = 36
    Cfg.env.num_privileged_obs = 2
    Cfg.env.num_observation_history = 30
    Cfg.env.observe_two_prev_actions = True
    Cfg.env.observe_command = True
    Cfg.env.observe_vel = True
    
    Cfg.env.num_envs = min(Cfg.env.num_envs, 1)
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 3
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.curriculum = False
    
    Cfg.noise.add_noise = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_base_com = False
    Cfg.domain_rand.push_robots = False
    
    Cfg.commands.lin_vel_x = [0.3, 0.3]
    Cfg.commands.lin_vel_y = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0]
    Cfg.env.episode_length_s = 20
    
    Cfg.env.priv_observe_friction = True
    Cfg.env.priv_observe_restitution = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.env.priv_observe_com_displacement = False
    Cfg.env.priv_observe_motor_strength = False
    Cfg.env.priv_observe_motor_offset = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_gravity = False
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    
    EnvBase = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    Env = HistoryWrapper(EnvBase)
    Env.env.cfg.env.max_episode_length = 10000
    Obs = Env.reset()
    
    ObsHistoryBufferSize = Cfg.env.num_observations * Cfg.env.num_observation_history
    
    ActorCritic = ActorCritic(Cfg.env.num_observations, Cfg.env.num_privileged_obs, ObsHistoryBufferSize, Cfg.env.num_actions).to('cuda:0')
    
    RunsRoot = Path("/home/abc/walk_these_ways_learning/walk-these-ways/runs/gait-conditioned-agility")
    TrainDirs = sorted(RunsRoot.glob("*/train/*/"), key=os.path.getmtime, reverse=True)
    CheckpointPath = TrainDirs[0] / "checkpoints" / "ac_weights_last.pt"
    Checkpoint = torch.load(CheckpointPath)
    ActorCritic.load_state_dict(Checkpoint)
    ActorCritic.eval()
    
    OutputDir = Path("/home/abc/walk_these_ways_learning/videos/play_frames")
    OutputDir.mkdir(parents=True, exist_ok=True)
    
    for f in OutputDir.glob("*.png"):
        f.unlink()
    
    RobotIndex = 0
    Gym = EnvBase.gym
    Viewer = EnvBase.viewer
    
    CamOffset = np.array([-2.0, 1.5, 0.6])
    CameraTarget = np.array([0.0, 0.0, 0.6])
    Smoothing = 0.15
    
    def update_camera():
        nonlocal CameraTarget
        BaseState = Env.env.root_states[RobotIndex].cpu().numpy()
        DesiredTarget = np.array([BaseState[0], BaseState[1], 0.6])
        CameraTarget = (1 - Smoothing) * CameraTarget + Smoothing * DesiredTarget
        CamPosNp = CameraTarget + CamOffset
        CamPosVec = gymapi.Vec3(*CamPosNp.tolist())
        CamTargetVec = gymapi.Vec3(*CameraTarget.tolist())
        Gym.viewer_camera_look_at(Viewer, None, CamPosVec, CamTargetVec)
    
    update_camera()
    
    Fps = 30
    Duration = 15
    TotalFrames = Fps * Duration
    SimHz = 50
    StepsPerFrame = SimHz / Fps
    
    FrameCount = 0
    StepCount = 0
    NextFrameStep = 0
    
    for _ in range(20):
        with torch.inference_mode():
            Actions = ActorCritic.act_inference(Obs)
        Obs, Rews, Dones, Infos = Env.step(Actions.detach())
    
    while FrameCount < TotalFrames:
        with torch.inference_mode():
            Actions = ActorCritic.act_inference(Obs)
        Obs, Rews, Dones, Infos = Env.step(Actions.detach())
        
        update_camera()
        
        if StepCount >= NextFrameStep:
            FramePath = OutputDir / f"frame_{FrameCount:04d}.png"
            Gym.write_viewer_image_to_file(Viewer, str(FramePath))
            FrameCount += 1
            NextFrameStep += StepsPerFrame
        
        StepCount += 1
    
    OutputVideo = "/home/abc/walk_these_ways_learning/videos/hybrid_hopper_play.mp4"
    
    FfmpegCmd = ["ffmpeg", "-y", "-framerate", str(Fps), "-i", str(OutputDir / "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", OutputVideo]
    
    try:
        subprocess.run(FfmpegCmd, check=True, capture_output=True)
        shutil.rmtree(OutputDir)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except FileNotFoundError:
        print("FFmpeg not found")

if __name__ == '__main__':
    play_hybrid_hopper_with_recording()
