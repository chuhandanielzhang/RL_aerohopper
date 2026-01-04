from typing import Union
from params_proto import Meta
from go1_gym.envs.base.legged_robot_config import Cfg

def config_hybrid_hopper(Cnfg: Union[Cfg, Meta]):
    _ = Cnfg.init_state
    _.pos = [0.0, 0.0, 1.0]
    _.default_joint_angles = {
        'ball_joint_x': 0.0,
        'ball_joint_y': 0.0,
        'spring_joint': 0.0,
        'rotor_FL_joint': 0.0,
        'rotor_FR_joint': 0.0,
        'rotor_RL_joint': 0.0,
        'rotor_RR_joint': 0.0,
    }
    
    _ = Cnfg.control
    _.control_type = 'P'
    _.stiffness = {
        'ball_joint_x': 20.0,
        'ball_joint_y': 20.0,
        'spring_joint': 200.0,
        'rotor_FL_joint': 0.0,
        'rotor_FR_joint': 0.0,
        'rotor_RL_joint': 0.0,
        'rotor_RR_joint': 0.0,
    }
    _.damping = {
        'ball_joint_x': 0.5,
        'ball_joint_y': 0.5,
        'spring_joint': 2.0,
        'rotor_FL_joint': 0.0,
        'rotor_FR_joint': 0.0,
        'rotor_RL_joint': 0.0,
        'rotor_RR_joint': 0.0,
    }
    _.action_scale = [0.5, 0.5, 10.0]
    _.decimation = 4
    
    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/hybrid_hopper/urdf/hybrid_hopper.urdf'
    _.foot_name = "lower_leg"
    _.penalize_contacts_on = ["upper_leg"]
    _.terminate_after_contacts_on = ["base"]
    _.self_collisions = 0
    _.flip_visual_attachments = False
    _.fix_base_link = False
    
    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9
    _.base_height_target = 0.85
    _.tracking_sigma = 0.2
    _.tracking_sigma_yaw = 0.2
    _.kappa_gait_probs = 0.07
    _.gait_force_sigma = 100.
    _.gait_vel_sigma = 10.
    _.use_terminal_body_height = True
    _.terminal_body_height = 0.25
    _.use_terminal_roll_pitch = True
    _.terminal_body_ori = 1.0
    
    _ = Cnfg.reward_scales
    _.torques = -0.00001
    _.action_rate = -0.01
    _.dof_pos_limits = -10.0
    _.orientation = -2.0
    _.base_height = 2.0
    _.tracking_lin_vel = 0.0
    _.tracking_ang_vel = 0.0
    _.tracking_contacts_shaped_force = 0.0
    _.tracking_contacts_shaped_vel = 0.0
    _.feet_air_time = 0.0
    _.rotor_energy = 0.0
    
    _ = Cnfg.terrain
    _.mesh_type = 'plane'
    _.measure_heights = False
    _.terrain_noise_magnitude = 0.0
    _.teleport_robots = True
    _.border_size = 50
    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    _.curriculum = False
    
    _ = Cnfg.env
    _.num_envs = 1024
    _.num_observations = 36
    _.num_scalar_observations = 36
    _.num_privileged_obs = 2
    _.num_actions = 3
    _.num_observation_history = 30
    _.env_spacing = 3.0
    _.episode_length_s = 20
    _.observe_vel = True
    
    _ = Cnfg.commands
    _.heading_command = False
    _.resampling_time = 5.0
    _.curriculum = False
    _.command_curriculum = False
    _.num_lin_vel_bins = 30
    _.num_ang_vel_bins = 30
    _.lin_vel_x = [-0.5, 0.5]
    _.lin_vel_y = [-0.5, 0.5]
    _.ang_vel_yaw = [-0.5, 0.5]
    _.swing_height = [0.08, 0.20]
    _.walking_height = [0.85, 0.85]
    _.period = [0.3, 0.5]
    _.roll = [-0.0, 0.0]
    _.pitch = [-0.0, 0.0]
    _.num_commands = 15
    
    _ = Cnfg.domain_rand
    _.randomize_base_mass = True
    _.added_mass_range = [-0.5, 0.5]
    _.randomize_base_com = True
    _.com_displacement_range = [-0.12, 0.12]
    _.randomize_all_link_mass = True
    _.link_mass_ratio = [0.8, 1.2]
    _.randomize_all_link_inertia = True
    _.link_inertia_ratio_range = [0.8, 1.2]
    _.randomize_pdgains = True
    _.pd_ratio = [0.9, 1.1]
    _.randomize_Kp_factor = True
    _.Kp_factor_range = [0.9, 1.1]
    _.randomize_Kd_factor = True
    _.Kd_factor_range = [0.9, 1.1]
    _.randomize_motor_strength = True
    _.motor_strength_range = [0.8, 1.2]
    _.add_motor_offset = True
    _.motor_offset_range = [-0.05, 0.05]
    _.randomize_motor_offset = True
    _.randomize_motor_friction = False
    _.motor_friction_range = [0.0, 0.1]
    _.randomize_motor_damping = False
    _.motor_damping_range = [0.0, 0.07]
    _.add_delay = True
    _.delay_prob = 0.5
    _.delay_bound = [0.0, 0.02]
    _.push_robots = False
    _.max_push_vel_xy = 0.3
    _.randomize_friction = True
    _.friction_range = [0.5, 1.25]
    _.randomize_restitution = True
    _.restitution_range = [0.0, 0.8]
    _.restitution = 0.0
    _.rand_interval_s = 6

class HybridHopperCfg(Cfg):
    pass
