import torch
import numpy as np
import genesis as gs

class RewardManager:
    """
    Manages the calculation of all reward components for the biped environment.
    """
    def __init__(self, env, reward_cfg, num_envs, dt):
        self.env = env
        self.reward_cfg = reward_cfg
        self.reward_scales = reward_cfg["reward_scales"]
        self.num_envs = num_envs
        self.device = gs.device
        self.dt = dt

        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            if self.reward_cfg.get("reward_enables", {}).get(name, True):
                self.reward_scales[name] *= self.dt
                self.reward_functions[name] = getattr(self, "_reward_" + name)
                self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def compute_rewards(self):
        """
        Computes the total reward for the current step by summing all enabled reward components.
        """
        rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            rew_buf += rew
            self.episode_sums[name] += rew
        return rew_buf

    def get_episode_sums(self):
        """
        Returns the cumulative reward sums for the current episode.
        """
        return self.episode_sums

    def reset_episode_sums(self, envs_idx):
        """
        Resets the cumulative reward sums for specified environments.
        """
        for key in self.episode_sums.keys():
            self.episode_sums[key][envs_idx] = 0.0

    # --- Reward Functions ---
    # (All _reward_* functions from the original biped_env.py are moved here)

    def _reward_lin_vel_z(self):
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        return torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _reward_forward_velocity(self):
        v_target = self.reward_cfg.get("forward_velocity_target", 0.5)
        vel_error = torch.square(self.env.base_lin_vel[:, 0] - v_target)
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-vel_error / sigma)

    def _reward_tracking_lin_vel_x(self):
        lin_vel_error = torch.square(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        lin_vel_error = torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive_bonus(self):
        return torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _reward_fall_penalty(self):
        fall_condition = (
            (torch.abs(self.env.base_euler[:, 0]) > self.env.env_cfg.get("fall_roll_threshold", 30.0)) |
            (torch.abs(self.env.base_euler[:, 1]) > self.env.env_cfg.get("fall_pitch_threshold", 30.0))
        )
        return torch.where(
            fall_condition,
            torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float),
            torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        )

    def _reward_torso_stability(self):
        orientation_error = torch.sum(torch.square(self.env.base_euler[:, :2]), dim=1)
        k_stability = self.reward_cfg.get("stability_factor", 1.0)
        return torch.exp(-k_stability * orientation_error)

    def _reward_height_maintenance(self):
        z_target = self.reward_cfg.get("height_target", 0.35)
        height_error = torch.square(z_target - self.env.base_pos[:, 2])
        return -height_error

    def _reward_joint_movement(self):
        joint_vel_magnitude = torch.sum(torch.abs(self.env.dof_vel), dim=1)
        movement_threshold = self.reward_cfg.get("movement_threshold", 0.1)
        movement_scale = self.reward_cfg.get("movement_scale", 1.0)
        return torch.clamp(joint_vel_magnitude * movement_scale, 0.0, movement_threshold)

    def _reward_sinusoidal_gait(self):
        amplitude = self.reward_cfg.get("gait_amplitude", 0.5)
        frequency = self.reward_cfg.get("gait_frequency", 0.5)
        phase_offsets = torch.tensor([0, 0, 0, 0, np.pi, 0, 0, 0], device=self.device, dtype=gs.tc_float)
        time = self.env.episode_length_buf * self.dt
        time = time.unsqueeze(1)
        leg_joints_default = self.env.default_dof_pos[:-1]
        target_leg_pos = leg_joints_default + amplitude * torch.sin(2 * np.pi * frequency * time + phase_offsets)
        leg_joints_current = self.env.dof_pos[:, :-1]
        error = torch.sum(torch.square(leg_joints_current - target_leg_pos), dim=1)
        sigma = self.reward_cfg.get("gait_sigma", 0.25)
        return torch.exp(-error / sigma)

    def _reward_torso_sinusoidal(self):
        torso_amplitude = self.reward_cfg.get("torso_amplitude", 0.2)
        torso_frequency = self.reward_cfg.get("torso_frequency", 0.3)
        torso_phase = self.reward_cfg.get("torso_phase", 0.0)
        time = self.env.episode_length_buf * self.dt
        time = time.unsqueeze(1)
        torso_default = self.env.default_dof_pos[8]
        target_torso_pos = torso_default + torso_amplitude * torch.sin(2 * np.pi * torso_frequency * time + torso_phase)
        torso_current = self.env.dof_pos[:, 8]
        error = torch.square(torso_current - target_torso_pos.squeeze())
        sigma = self.reward_cfg.get("torso_sigma", 0.25)
        return torch.exp(-error / sigma)

    def _reward_actuator_constraint(self):
        constraint_limit = self.reward_cfg.get("actuator_constraint_limit", 6.16)
        torque_coeff = self.reward_cfg.get("actuator_torque_coeff", 3.5)
        tolerance = self.reward_cfg.get("actuator_tolerance", 0.5)
        constraint_values = torch.abs(self.env.dof_vel) + torque_coeff * torch.abs(self.env.joint_torques)
        target_with_tolerance = constraint_limit + tolerance
        violations = torch.relu(constraint_values - target_with_tolerance)
        total_violation_per_env = torch.sum(violations, dim=1)
        self.env.actuator_constraint_violations = total_violation_per_env
        return -total_violation_per_env