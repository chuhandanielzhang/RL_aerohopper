#!/usr/bin/env python3
"""
Play Hybrid Hopper - Headless模式测试
参考Hopper_rl_t-master/hopper_gym/scripts/play.py
"""

import sys
import os

# 必须先import isaacgym
sys.path.append('/home/abc/walk_these_ways_learning/isaacgym/python')
sys.path.append('/home/abc/walk_these_ways_learning/walk-these-ways')

from isaacgym import gymutil, gymapi

# 现在可以import其他模块
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.hybrid_hopper.hybrid_hopper_config import config_hybrid_hopper
from go1_gym.envs.hybrid_hopper.velocity_tracking import VelocityTrackingEasyEnv

import torch
import numpy as np

def play_hybrid_hopper(headless=True, num_steps=1000):
    """
    Play Hybrid Hopper环境
    
    Args:
        headless: 是否headless模式（不显示GUI）
        num_steps: 运行步数
    """
    print("\n" + "="*70)
    print("🎮 Play Hybrid Hopper (Headless Mode)")
    print("="*70)
    
    # 配置环境
    config_hybrid_hopper(Cfg)
    
    # 简化配置用于play
    Cfg.env.num_envs = 4  # 少量环境
    Cfg.terrain.mesh_type = 'plane'  # 平面地形
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.push_robots = False
    Cfg.commands.resampling_time = 100.0
    
    # 设置初始命令（让机器人尝试前进）
    Cfg.commands.lin_vel_x = [0.3, 0.3]  # 0.3 m/s前进
    Cfg.commands.lin_vel_y = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0]
    
    print(f"\n📊 配置信息:")
    print(f"  Headless: {headless}")
    print(f"  环境数量: {Cfg.env.num_envs}")
    print(f"  DOF: {Cfg.env.num_actions}")
    print(f"  观察维度: {Cfg.env.num_observations}")
    print(f"  运行步数: {num_steps}")
    
    # 创建sim_params
    sim_params = gymapi.SimParams()
    gymutil.parse_sim_config(vars(Cfg.sim), sim_params)
    
    print(f"\n⚙️ Sim参数:")
    print(f"  use_gpu_pipeline: {sim_params.use_gpu_pipeline}")
    print(f"  dt: {sim_params.dt}")
    print(f"  substeps: {sim_params.substeps}")
    
    # 创建环境
    print(f"\n🤖 创建环境（{'headless' if headless else 'GUI'}模式）...")
    try:
        env = VelocityTrackingEasyEnv(
            sim_device='cuda:0',
            headless=headless,
            cfg=Cfg
        )
        print("✅ 环境创建成功！")
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 环境信息
    print(f"\n📋 环境信息:")
    print(f"  自由度数量: {env.num_dof}")
    print(f"  刚体数量: {env.num_bodies}")
    print(f"  Foot indices: {env.feet_indices}")
    
    # DOF names
    if hasattr(env, 'dof_names'):
        print(f"\n📋 DOF列表:")
        for i, name in enumerate(env.dof_names):
            print(f"  [{i}] {name}")
    
    # 重置环境
    print(f"\n🔄 重置环境...")
    obs = env.reset()
    print(f"✅ 环境已重置")
    print(f"   Observation shape: {obs.shape}")
    
    # 准备统计数据
    rewards_history = []
    heights_history = []
    velocities_history = []
    
    print("\n" + "="*70)
    print("⚡ 开始运行仿真...")
    print("="*70)
    
    # 主循环
    for step in range(num_steps):
        # 生成随机动作（或使用策略）
        # actions shape: (num_envs, num_actions)
        if step < 100:
            # 前100步：保持静止
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            # 之后：随机小动作测试
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
            
            # 旋翼始终给一些推力（帮助悬停）
            if env.num_actions >= 7:  # 有旋翼
                actions[:, 3:7] = 0.2  # 小推力
        
        # 执行动作
        obs, _, rewards, dones, infos = env.step(actions)
        
        # 收集统计数据
        rewards_history.append(rewards.mean().item())
        
        # 获取高度和速度
        heights = env.root_states[:, 2].mean().item()
        velocities = env.root_states[:, 7:10].norm(dim=1).mean().item()
        heights_history.append(heights)
        velocities_history.append(velocities)
        
        # 定期打印
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{num_steps}:")
            print(f"  Mean reward: {rewards.mean():.4f}")
            print(f"  Mean height: {heights:.3f} m")
            print(f"  Mean velocity: {velocities:.3f} m/s")
            print(f"  Dones: {dones.sum().item()}/{env.num_envs}")
    
    print("\n" + "="*70)
    print("📊 仿真统计:")
    print("="*70)
    
    # 计算统计
    rewards_np = np.array(rewards_history)
    heights_np = np.array(heights_history)
    velocities_np = np.array(velocities_history)
    
    print(f"Rewards:")
    print(f"  Mean: {rewards_np.mean():.4f}")
    print(f"  Std:  {rewards_np.std():.4f}")
    print(f"  Min:  {rewards_np.min():.4f}")
    print(f"  Max:  {rewards_np.max():.4f}")
    
    print(f"\nHeights (m):")
    print(f"  Mean: {heights_np.mean():.3f}")
    print(f"  Std:  {heights_np.std():.3f}")
    print(f"  Min:  {heights_np.min():.3f}")
    print(f"  Max:  {heights_np.max():.3f}")
    
    print(f"\nVelocities (m/s):")
    print(f"  Mean: {velocities_np.mean():.3f}")
    print(f"  Std:  {velocities_np.std():.3f}")
    print(f"  Min:  {velocities_np.min():.3f}")
    print(f"  Max:  {velocities_np.max():.3f}")
    
    print("\n✅ 仿真完成！")
    
    # 保存数据（可选）
    save_data = False
    if save_data:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        axes[0].plot(rewards_history)
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward History')
        axes[0].grid(True)
        
        axes[1].plot(heights_history)
        axes[1].set_ylabel('Height (m)')
        axes[1].set_title('Body Height')
        axes[1].grid(True)
        axes[1].axhline(y=0.5, color='r', linestyle='--', label='Target')
        axes[1].legend()
        
        axes[2].plot(velocities_history)
        axes[2].set_ylabel('Velocity (m/s)')
        axes[2].set_xlabel('Step')
        axes[2].set_title('Body Velocity')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/abc/walk_these_ways_learning/hybrid_hopper_play.png')
        print(f"\n📊 统计图已保存: hybrid_hopper_play.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to run')
    args = parser.parse_args()
    
    play_hybrid_hopper(headless=args.headless, num_steps=args.steps)
