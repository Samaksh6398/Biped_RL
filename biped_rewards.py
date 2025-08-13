import torch
import numpy as np

class RewardManager:
    def __init__(self, env):
        self.env = env

    def lin_vel_z(self):
        return torch.square(self.env.base_lin_vel[:, 2])

    def action_rate(self):
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def similar_to_default(self):
        return torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def foot_clearance(self):
        return torch.zeros((self.env.num_envs,), device=self.env.device, dtype=torch.float32)

    def forward_velocity(self):
        v_target = self.env.reward_cfg.get("forward_velocity_target", 0.5)
        vel_error = torch.square(self.env.base_lin_vel[:, 0] - v_target)
        sigma = self.env.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-vel_error / sigma)

    def tracking_lin_vel_x(self):
        lin_vel_error = torch.square(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.env.reward_cfg["tracking_sigma"])

    def tracking_lin_vel_y(self):
        lin_vel_error = torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.env.reward_cfg["tracking_sigma"])

    def alive_bonus(self):
        return torch.ones((self.env.num_envs,), device=self.env.device, dtype=torch.float32)

    def fall_penalty(self):
        fall_condition = (
            (torch.abs(self.env.base_euler[:, 0]) > self.env.env_cfg.get("fall_roll_threshold", 30.0)) |
            (torch.abs(self.env.base_euler[:, 1]) > self.env.env_cfg.get("fall_pitch_threshold", 30.0))
        )
        return torch.where(fall_condition, torch.ones_like(fall_condition, dtype=torch.float32), torch.zeros_like(fall_condition, dtype=torch.float32))

    def torso_stability(self):
        orientation_error = torch.sum(torch.square(self.env.base_euler[:, :2]), dim=1)
        k_stability = self.env.reward_cfg.get("stability_factor", 1.0)
        return torch.exp(-k_stability * orientation_error)

    def height_maintenance(self):
        z_target = self.env.reward_cfg.get("height_target", 0.35)
        height_error = torch.square(z_target - self.env.base_pos[:, 2])
        return -height_error

    def joint_movement(self):
        joint_vel_magnitude = torch.sum(torch.abs(self.env.dof_vel), dim=1)
        movement_threshold = self.env.reward_cfg.get("movement_threshold", 0.1)
        movement_scale = self.env.reward_cfg.get("movement_scale", 1.0)
        return torch.clamp(joint_vel_magnitude * movement_scale, 0.0, movement_threshold)
    
    def sinusoidal_gait(self):
        amplitude = self.env.reward_cfg.get("gait_amplitude", 0.5)
        frequency = self.env.reward_cfg.get("gait_frequency", 0.5)
        phase_offsets = torch.tensor([0, 0, 0, 0, np.pi, 0, 0, 0], device=self.env.device, dtype=torch.float32)
        time = (self.env.episode_length_buf * self.env.dt).unsqueeze(1)
        leg_joints_default = self.env.default_dof_pos[:-1]
        target_leg_pos = leg_joints_default + amplitude * torch.sin(2 * np.pi * frequency * time + phase_offsets)
        error = torch.sum(torch.square(self.env.dof_pos[:, :-1] - target_leg_pos), dim=1)
        sigma = self.env.reward_cfg.get("gait_sigma", 0.25)
        return torch.exp(-error / sigma)

    def torso_sinusoidal(self):
        torso_amplitude = self.env.reward_cfg.get("torso_amplitude", 0.2)
        torso_frequency = self.env.reward_cfg.get("torso_frequency", 0.3)
        torso_phase = self.env.reward_cfg.get("torso_phase", 0.0)
        time = (self.env.episode_length_buf * self.env.dt).unsqueeze(1)
        torso_default = self.env.default_dof_pos[8]
        target_torso_pos = torso_default + torso_amplitude * torch.sin(2 * np.pi * torso_frequency * time + torso_phase)
        error = torch.square(self.env.dof_pos[:, 8] - target_torso_pos.squeeze())
        sigma = self.env.reward_cfg.get("torso_sigma", 0.25)
        return torch.exp(-error / sigma)

    def actuator_constraint(self):
        constraint_limit = self.env.reward_cfg.get("actuator_constraint_limit", 6.16)
        torque_coeff = self.env.reward_cfg.get("actuator_torque_coeff", 3.5)
        tolerance = self.env.reward_cfg.get("actuator_tolerance", 0.5)
        constraint_values = torch.abs(self.env.dof_vel) + torque_coeff * torch.abs(self.env.joint_torques)
        violations = torch.relu(constraint_values - (constraint_limit + tolerance))
        total_violation_per_env = torch.sum(violations, dim=1)
        self.env.actuator_constraint_violations = total_violation_per_env
        return -total_violation_per_env