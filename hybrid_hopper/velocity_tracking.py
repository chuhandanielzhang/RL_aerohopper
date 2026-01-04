"""
Velocity Tracking Environment for Hybrid Hopper
"""

from go1_gym.envs.hybrid_hopper.hybrid_hopper_env import HybridHopperEnv
from isaacgym import gymapi, gymutil

class VelocityTrackingEasyEnv(HybridHopperEnv):
    """
    Velocity tracking task for Hybrid Hopper
    Same structure as Go1's VelocityTrackingEasyEnv
    """
    
    def __init__(self, sim_device, headless, cfg=None, eval_cfg=None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):
        """Initialize velocity tracking environment"""
        
        # Create sim_params using gymutil (same as Go1)
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        
        super().__init__(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=sim_device,
            headless=headless,
            eval_cfg=eval_cfg,
            initial_dynamics_dict=initial_dynamics_dict
        )
    
    def step(self, actions):
        """
        Step the environment
        Returns 5 values like base LeggedRobot class
        """
        # Call parent step (returns obs, privileged_obs, rew, done, info)
        return super().step(actions)

