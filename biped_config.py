
def get_cfgs():
    """
    Returns the configuration dictionaries for the environment, observations, rewards, and commands.
    This configuration remains unchanged as it is specific to the environment's internal logic.
    """
    env_cfg = {
        "num_actions": 9,  # 9 DOF for biped: 4 per leg + 1 torso
        # joint/link names - based on your URDF with neutral standing pose
        "default_joint_angles": {  # [rad] - neutral standing pose with ground contact
            "right_hip1": 0.0,     # hip abduction/adduction
            "right_hip2": -0.652,   # hip flexion/extension
            "right_knee": 1.30,    # knee flexion
            "right_ankle": -0.634,  # ankle flexion
            "left_hip1": 0.0,      # hip abduction/adduction
            "left_hip2": 0.652,    # hip flexion/extension
            "left_knee": -1.30,    # knee flexion (negative for left leg)
            "left_ankle": 0.634,   # ankle flexion
            "revolute_torso": 0.0,          # torso rotation
        },
        "joint_names": [
            # Right leg first (as per your configuration)
            "right_hip1",
            "right_hip2",
            "right_knee",
            "right_ankle",
            # Left leg
            "left_hip1",
            "left_hip2",
            "left_knee",
            "left_ankle",
            # Torso
            "revolute_torso",
        ],
        # PD control parameters - reduced for more stable operation
        "kp": 15.0,  # Reduced from 30.0 - high gains cause excessive torques
        "kd": 0.8,   # Slightly reduced damping
        # termination conditions - tighter for biped
        "termination_if_roll_greater_than": 30,  # degree - bipeds can lean more
        "termination_if_pitch_greater_than": 30, # degree
        "termination_if_height_below": 0.30,  # meters - terminate if base link height drops below this

        # Fall penalty thresholds (in degrees)
        "fall_roll_threshold": 25.0,   # Roll threshold for fall penalty (slightly less than termination)
        "fall_pitch_threshold": 25.0,  # Pitch threshold for fall penalty (slightly less than termination)
        # base pose - height adjusted for neutral configuration ground contact
        "base_init_pos": [0.0, 0.0, 0.50],  # Lower spawn height for ground contact with neutral pose
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 90.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # Conservative scaling
        "simulate_action_latency": True,
        "clip_actions": 100.0,

        # Domain Randomization Configuration
        "domain_rand": {
            "randomize_friction":True,
            "friction_range": [0.4, 1.25],

            "randomize_mass": False,
            "added_mass_range": [0.0, 0.4],

            "randomize_motor_strength": True,
            "motor_strength_range": [0.6, 1.2],

            "push_robot": False,
            "push_interval_s": 7,
            "max_push_vel_xy": 1.0,

            "add_motor_backlash": True ,
            "backlash_range": [0.01, 0.07],

            "add_observation_noise": True ,
            "noise_scales": {
                "dof_pos": 0.02,
                "dof_vel": 0.2,
                "lin_vel": 0.1,
                "ang_vel": 0.15,
                "base_pos": 0.01,
                "base_euler": 0.03,
                "foot_contact": 0.1,
            },

            "randomize_foot_contacts": False,
            "foot_contact_params": {
                "contact_threshold_range": [0.01, 0.15],
                "contact_noise_range": [0.0, 0.2],
                "false_positive_rate": 0.05,
                "false_negative_rate": 0.05,
                "contact_delay_range": [0, 2],
            }
        }
    }

    obs_cfg = {
        "num_obs": 38,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "base_euler": 1.0,
            "base_height": 1.0,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.25,
        "feet_height_target": 0.1,

        "forward_velocity_target": 0.5,
        "stability_factor": 1.0,
        "height_target": 0.25,
        "movement_threshold": 2.0,
        "movement_scale": 0.1,

        "reward_enables": {
            "tracking_lin_vel_x": True,
            "tracking_lin_vel_y": True,
            "lin_vel_z": True,
            "action_rate": True,
            "similar_to_default": True,
            "alive_bonus": True,
            "fall_penalty": True,
            "torso_stability": True,
            "height_maintenance": True,
            "joint_movement": True,
            "height_penalty": True,
        },

        "reward_scales": {
            "tracking_lin_vel_x": 10.0,
            "tracking_lin_vel_y": 6.0,
            "lin_vel_z": -2.0,
            "action_rate": -0.02,
            "similar_to_default": -0.1,
            "alive_bonus": 0.5,
            "fall_penalty": -100.0,
            "torso_stability": 5.0,
            "height_maintenance": -2.0,
            "joint_movement": 1.0,
            "height_penalty": -50.0,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [0.0, 0.0],
    }

    adaptive_lr_cfg = {
        # Adaptive Learning Rate Configuration
        # These parameters control the KL-divergence based learning rate adaptation
        "target_kl": 0.015,           # Target KL divergence threshold
        "lr_factor": 0.8,            # Factor to multiply LR (< 1.0 for reduction, > 1.0 for increase)
        "patience": 5,               # Number of consecutive violations before adapting
        "smoothing_window": 5,      # Window size for smoothing KL values to reduce noise
        "min_lr": 1e-6,             # Minimum learning rate (prevents lr from going too low)
        "adaptation_threshold": 0.8, # Fraction of target_kl to trigger LR increase (0.8 * target_kl)
        "verbose": 0                 # Verbosity level (0=silent, 1=basic logging, 2=detailed)
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, adaptive_lr_cfg
