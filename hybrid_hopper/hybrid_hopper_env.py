import torch
from go1_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
from isaacgym import gymapi

class HybridHopperEnv(LeggedRobot):
    
    def __init__(self, Cfg, SimParams, PhysicsEngine, SimDevice, Headless, EvalCfg=None, InitialDynamicsDict=None):
        super().__init__(Cfg, SimParams, PhysicsEngine, SimDevice, Headless, EvalCfg, InitialDynamicsDict)
        
        self.ActionScaleTensor = torch.tensor(self.cfg.control.action_scale, dtype=torch.float, device=self.device) if isinstance(self.cfg.control.action_scale, (list, tuple)) else self.cfg.control.action_scale
        
        self.RotorCurriculum = {0: 1.0}
        self.MaxRotorThrust = 5.0
        self.CurrentRotorLimit = 1.0
        self.Iteration = 0
        
        self._get_rotor_rigid_body_indices()
        self.RotorThrusts = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
    
    def _get_rotor_rigid_body_indices(self):
        RotorNames = ["rotor_FL", "rotor_FR", "rotor_RL", "rotor_RR"]
        self.RotorRbIndices = torch.zeros(4, dtype=torch.long, device=self.device)
        for i, RotorName in enumerate(RotorNames):
            self.RotorRbIndices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], RotorName)
    
    def refresh_actor_rigid_shape_props(self, EnvIds, Cfg):
        try:
            super().refresh_actor_rigid_shape_props(EnvIds, Cfg)
        except IndexError:
            pass
    
    def _get_env_origins(self, EnvIds, Cfg):
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if Cfg.env.num_envs > 1:
            NumCols = int(Cfg.env.num_envs ** 0.5)
            NumRows = int(torch.ceil(torch.tensor(Cfg.env.num_envs / NumCols)).item())
            Xx, Yy = torch.meshgrid(torch.arange(NumRows, device=self.device), torch.arange(NumCols, device=self.device))
            OriginsX = Cfg.env.env_spacing * Xx.flatten()
            OriginsY = Cfg.env.env_spacing * Yy.flatten()
            self.env_origins[EnvIds, 0] = OriginsX[:len(EnvIds)]
            self.env_origins[EnvIds, 1] = OriginsY[:len(EnvIds)]
            self.env_origins[EnvIds, 2] = 0.
    
    def _compute_torques(self, Actions):
        ActionsScaled = Actions * self.ActionScaleTensor
        ControlType = self.cfg.control.control_type
        
        if ControlType == "P":
            TorquesBall = self.p_gains[:2] * (ActionsScaled[:, :2] + self.default_dof_pos[:, :2] - self.dof_pos[:, :2]) - self.d_gains[:2] * self.dof_vel[:, :2]
            SpringStiffness = 200.0
            SpringDamping = 2.0
            SpringPassiveForce = -SpringStiffness * self.dof_pos[:, 2:3] - SpringDamping * self.dof_vel[:, 2:3]
            SpringActiveForce = ActionsScaled[:, 2:3] * 100.0
            SpringForce = SpringPassiveForce + SpringActiveForce
            Torques = torch.cat([TorquesBall, SpringForce], dim=1)
        else:
            raise NameError(f"Unknown controller type: {ControlType}")
        
        return torch.clip(Torques, -self.torque_limits[:3], self.torque_limits[:3])
    
    def step(self, Actions):
        self.actions = torch.clip(Actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions).to(self.device)
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()
        
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape[0], 3)
            FullTorques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device)
            FullTorques[:, :3] = self.torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(FullTorques))
            self._apply_rotor_forces()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        self.post_physics_step()
        
        self.obs_buf = torch.clip(self.obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _apply_rotor_forces(self):
        Quat = self.base_quat
        Roll = torch.atan2(2 * (Quat[:, 0] * Quat[:, 1] + Quat[:, 2] * Quat[:, 3]), 1 - 2 * (Quat[:, 1]**2 + Quat[:, 2]**2))
        Pitch = torch.asin(torch.clamp(2 * (Quat[:, 0] * Quat[:, 2] - Quat[:, 3] * Quat[:, 1]), -1.0, 1.0))
        
        KPitchP = 10.0
        KRollP = 10.0
        KPitchD = 2.5
        KRollD = 2.5
        
        BaseLiftPerRotor = 6.15 * self.CurrentRotorLimit
        BaseThrust = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device) * BaseLiftPerRotor
        AngVel = self.base_ang_vel
        RollVel = AngVel[:, 0]
        PitchVel = AngVel[:, 1]
        
        PitchCorrection = (KPitchP * Pitch + KPitchD * PitchVel).unsqueeze(1)
        RollCorrection = (KRollP * Roll + KRollD * RollVel).unsqueeze(1)
        
        AdjustedThrust = torch.zeros_like(BaseThrust)
        AdjustedThrust[:, 0] = BaseThrust[:, 0] - PitchCorrection[:, 0] - RollCorrection[:, 0]
        AdjustedThrust[:, 1] = BaseThrust[:, 1] - PitchCorrection[:, 0] + RollCorrection[:, 0]
        AdjustedThrust[:, 2] = BaseThrust[:, 2] + PitchCorrection[:, 0] - RollCorrection[:, 0]
        AdjustedThrust[:, 3] = BaseThrust[:, 3] + PitchCorrection[:, 0] + RollCorrection[:, 0]
        
        AdjustedThrust = torch.clamp(AdjustedThrust, min=0.0, max=self.MaxRotorThrust * self.CurrentRotorLimit)
        self.RotorThrusts = AdjustedThrust
        
        NumRigidBodies = self.gym.get_env_rigid_body_count(self.envs[0])
        Forces = torch.zeros(self.num_envs, NumRigidBodies, 3, dtype=torch.float, device=self.device)
        for i in range(4):
            Forces[:, self.RotorRbIndices[i], 2] = AdjustedThrust[:, i]
        
        ForcesFlat = Forces.view(-1, 3)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(ForcesFlat), None, gymapi.ENV_SPACE)
    
    def _post_physics_step_callback(self):
        self.foot_contacts = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1), dim=1)
        
        FixedFrequency = 3.0
        Frequencies = torch.full((self.num_envs,), FixedFrequency, device=self.device)
        self.gait_indices = torch.remainder(self.episode_length_buf * self.dt * Frequencies, 1.0)
        Phase = self.gait_indices
        
        if not hasattr(self, 'desired_contact_states') or self.desired_contact_states.shape[1] != max(1, self.feet_indices.shape[0]):
            self.desired_contact_states = torch.zeros(self.num_envs, max(1, self.feet_indices.shape[0]), dtype=torch.float, device=self.device)
        
        if self.feet_indices.shape[0] > 0:
            self.desired_contact_states[:, 0] = (Phase < 0.5).float()
        
        for IterThreshold, Limit in sorted(self.RotorCurriculum.items(), reverse=True):
            if self.Iteration >= IterThreshold:
                self.CurrentRotorLimit = Limit
                break
        
        if "env_bins" not in self.extras:
            self.extras["env_bins"] = torch.zeros(self.num_envs, device=self.device)
    
    def update_iteration(self, Iteration):
        self.Iteration = Iteration
    
    def _resample_commands(self, EnvIds):
        from isaacgym.torch_utils import torch_rand_float
        
        self.commands[EnvIds, 0] = torch_rand_float(self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_x[1], (len(EnvIds), 1), device=self.device).squeeze(1)
        self.commands[EnvIds, 1] = torch_rand_float(self.cfg.commands.lin_vel_y[0], self.cfg.commands.lin_vel_y[1], (len(EnvIds), 1), device=self.device).squeeze(1)
        self.commands[EnvIds, 2] = torch_rand_float(self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.ang_vel_yaw[1], (len(EnvIds), 1), device=self.device).squeeze(1)
        
        if self.cfg.commands.num_commands >= 10:
            self.commands[EnvIds, 5] = torch_rand_float(self.cfg.commands.period[0], self.cfg.commands.period[1], (len(EnvIds), 1), device=self.device).squeeze(1)
            self.commands[EnvIds, 6] = torch_rand_float(self.cfg.commands.swing_height[0], self.cfg.commands.swing_height[1], (len(EnvIds), 1), device=self.device).squeeze(1)
            self.commands[EnvIds, 7] = torch_rand_float(self.cfg.commands.walking_height[0], self.cfg.commands.walking_height[1], (len(EnvIds), 1), device=self.device).squeeze(1)
            self.commands[EnvIds, 8] = torch_rand_float(self.cfg.commands.roll[0], self.cfg.commands.roll[1], (len(EnvIds), 1), device=self.device).squeeze(1)
            self.commands[EnvIds, 9] = torch_rand_float(self.cfg.commands.pitch[0], self.cfg.commands.pitch[1], (len(EnvIds), 1), device=self.device).squeeze(1)
        
        self.commands[EnvIds, :2] *= (torch.norm(self.commands[EnvIds, :2], dim=1) > 0.05).unsqueeze(1)
        self.commands[EnvIds, 2] *= torch.norm(self.commands[EnvIds, 2]) > 0.05
    
    def compute_reward(self):
        super().compute_reward()
    
    def compute_observations(self):
        NumDofForObs = 3
        self.obs_buf = torch.cat((self.projected_gravity, (self.dof_pos[:, :NumDofForObs] - self.default_dof_pos[:, :NumDofForObs]) * self.obs_scales.dof_pos, self.dof_vel[:, :NumDofForObs] * self.obs_scales.dof_vel, self.actions), dim=-1)
        
        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity, self.commands * self.commands_scale, (self.dof_pos[:, :NumDofForObs] - self.default_dof_pos[:, :NumDofForObs]) * self.obs_scales.dof_pos, self.dof_vel[:, :NumDofForObs] * self.obs_scales.dof_vel, self.actions), dim=-1)
        
        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf, self.last_actions), dim=-1)
        
        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)
        
        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf, self.clock_inputs), dim=-1)
        
        if self.cfg.env.observe_vel:
            self.obs_buf = torch.cat((self.obs_buf, self.base_lin_vel * self.obs_scales.lin_vel, self.base_ang_vel * self.obs_scales.ang_vel), dim=-1)
        
        if self.cfg.env.num_privileged_obs is not None:
            if hasattr(self, 'privileged_obs_buf') and self.privileged_obs_buf is not None:
                pass
            else:
                if hasattr(self, 'friction_coeffs') and hasattr(self, 'restitution_coeffs'):
                    self.privileged_obs_buf = torch.cat((self.friction_coeffs.unsqueeze(1), self.restitution_coeffs.unsqueeze(1)), dim=-1)
                else:
                    self.privileged_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs, dtype=torch.float, device=self.device)
