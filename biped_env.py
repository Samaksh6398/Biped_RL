import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.sensors import RigidContactForceGridSensor
import numpy as np
import time

from biped_rewards import RewardHandler
from biped_domain_rand import DomainRandomizationHandler, gs_rand_float


class BipedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/biped_v4.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # foot contact sensors
        self.left_foot_contact_sensor = None
        self.right_foot_contact_sensor = None
        for link in self.robot.links:
            if link.name == "revolute_leftfoot":
                self.left_foot_contact_sensor = RigidContactForceGridSensor(entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2))
            elif link.name == "revolute_rightfoot":
                self.right_foot_contact_sensor = RigidContactForceGridSensor(entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2))

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Domain randomization setup
        self.orig_kp = torch.tensor([self.env_cfg["kp"]] * self.num_actions, device=gs.device)
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        
        # Pre-allocate noise buffers
        self.noise_buffers = {
            'dof_pos': torch.zeros((self.num_envs, self.num_actions), device=gs.device),
            'dof_vel': torch.zeros((self.num_envs, self.num_actions), device=gs.device),
            'lin_vel': torch.zeros((self.num_envs, 3), device=gs.device),
            'ang_vel': torch.zeros((self.num_envs, 3), device=gs.device),
            'base_pos': torch.zeros((self.num_envs, 3), device=gs.device),
            'base_euler': torch.zeros((self.num_envs, 3), device=gs.device),
            'foot_contact': torch.zeros((self.num_envs, 2), device=gs.device),
        }
        
        self.randomization_intervals = { 'motor_strength': 50, 'friction': 100, 'mass': 200, 'observation_noise': 1, 'foot_contacts': 1, 'motor_backlash': 20 }
        self.randomization_counters = {k: 0 for k in self.randomization_intervals}
        self.randomization_step_counter = 0

        # Initialize handlers
        self.reward_handler = RewardHandler(self)
        self.dr_handler = DomainRandomizationHandler(self)

        # prepare episode sums for logging
        self.episode_sums = dict()
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.episode_sums["fps"] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=gs.device, dtype=gs.tc_float)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]], device=gs.device, dtype=gs.tc_float)
        
        self.motor_backlash = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.motor_backlash_direction = torch.ones((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_motor_positions = torch.zeros_like(self.actions)
        
        self.joint_torques = torch.zeros_like(self.actions)
        self.actuator_constraint_violations = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        self.foot_contacts = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.foot_contacts_raw = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        
        self.contact_thresholds = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_noise_scale = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_false_positive_prob = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_false_negative_prob = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)
        self.contact_delay_steps = torch.zeros((self.num_envs, 2), device=gs.device, dtype=torch.long)
        self.contact_delay_buffer = torch.zeros((self.num_envs, 2, 5), device=gs.device, dtype=gs.tc_float)
        self.contact_delay_idx = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)
        
        self.step_count = 0
        self.fps_timer = time.time()
        self.fps_update_interval = 100
        self.current_fps = 0.0
        
        self.extras = dict()
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = 0.0

    def step(self, actions):
        self.step_count += 1
        self.randomization_step_counter += 1
        
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        
        exec_actions = self.dr_handler.apply_on_step(exec_actions)
        
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        pos_error = target_dof_pos - self.dof_pos
        vel_error = -self.dof_vel
        self.joint_torques = self.env_cfg["kp"] * pos_error + self.env_cfg["kd"] * vel_error
        
        if self.left_foot_contact_sensor is not None:
            left_contact_data = self.left_foot_contact_sensor.read()
            left_contact_tensor = torch.as_tensor(left_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            self.foot_contacts_raw[:, 0] = torch.max(torch.norm(left_contact_tensor, dim=-1), dim=-1)[0]

        if self.right_foot_contact_sensor is not None:
            right_contact_data = self.right_foot_contact_sensor.read()
            right_contact_tensor = torch.as_tensor(right_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            self.foot_contacts_raw[:, 1] = torch.max(torch.norm(right_contact_tensor, dim=-1), dim=-1)[0]
        
        if self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            self.foot_contacts = self.dr_handler._apply_foot_contact_randomization_optimized(self.foot_contacts_raw)
        else:
            self.foot_contacts = self.foot_contacts_raw.clone()
        
        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)).nonzero(as_tuple=False).reshape((-1,))
        self._resample_commands(envs_idx)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        
        if self.env_cfg.get("terminate_on_actuator_violation", False):
            constraint_values = torch.abs(self.dof_vel) + self.reward_cfg.get("actuator_torque_coeff", 3.5) * torch.abs(self.joint_torques)
            constraint_limit = self.reward_cfg.get("actuator_constraint_limit", 6.16)
            termination_threshold = self.env_cfg.get("actuator_violation_termination_threshold", 2.0)
            violations = constraint_values - constraint_limit
            max_violation_per_env = torch.max(violations, dim=1)[0]
            self.reset_buf |= max_violation_per_env > termination_threshold

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        self.reward_handler.compute_rewards()
        self.episode_sums["fps"][:] = self.current_fps

        hip_angles = torch.cat([(self.dof_pos[:, [0, 1]] - self.default_dof_pos[[0, 1]]) * self.obs_scales["dof_pos"], (self.dof_pos[:, [4, 5]] - self.default_dof_pos[[4, 5]]) * self.obs_scales["dof_pos"]], dim=1)
        hip_velocities = torch.cat([self.dof_vel[:, [0, 1]] * self.obs_scales["dof_vel"], self.dof_vel[:, [4, 5]] * self.obs_scales["dof_vel"]], dim=1)
        knee_angles = torch.cat([(self.dof_pos[:, [2]] - self.default_dof_pos[[2]]) * self.obs_scales["dof_pos"], (self.dof_pos[:, [6]] - self.default_dof_pos[[6]]) * self.obs_scales["dof_pos"]], dim=1)
        knee_velocities = torch.cat([self.dof_vel[:, [2]] * self.obs_scales["dof_vel"], self.dof_vel[:, [6]] * self.obs_scales["dof_vel"]], dim=1)
        ankle_angles = torch.cat([(self.dof_pos[:, [3]] - self.default_dof_pos[[3]]) * self.obs_scales["dof_pos"], (self.dof_pos[:, [7]] - self.default_dof_pos[[7]]) * self.obs_scales["dof_pos"]], dim=1)
        ankle_velocities = torch.cat([self.dof_vel[:, [3]] * self.obs_scales["dof_vel"], self.dof_vel[:, [7]] * self.obs_scales["dof_vel"]], dim=1)
        foot_contacts_normalized = torch.clamp(self.foot_contacts, 0, 1)
        
        if self.env_cfg["domain_rand"]["add_observation_noise"] and self.dr_handler._should_update_randomization('observation_noise'):
            self.dr_handler._generate_noise_batch()
            base_euler_noisy = self.base_euler[:, :2] + self.noise_buffers['base_euler'][:, :2]
            base_ang_vel_xy_noisy = self.base_ang_vel[:, :2] + self.noise_buffers['ang_vel'][:, :2]
            base_ang_vel_z_noisy = self.base_ang_vel[:, [2]] + self.noise_buffers['ang_vel'][:, [2]]
            base_lin_vel_noisy = self.base_lin_vel[:, :2] + self.noise_buffers['lin_vel'][:, :2]
            base_pos_z_noisy = self.base_pos[:, [2]] + self.noise_buffers['base_pos'][:, [2]]
            hip_angles_noisy = hip_angles + self.noise_buffers['dof_pos'][:, :4]
            hip_velocities_noisy = hip_velocities + self.noise_buffers['dof_vel'][:, :4]
            knee_angles_noisy = knee_angles + self.noise_buffers['dof_pos'][:, 4:6]
            knee_velocities_noisy = knee_velocities + self.noise_buffers['dof_vel'][:, 4:6]
            ankle_angles_noisy = ankle_angles + self.noise_buffers['dof_pos'][:, 6:8]
            ankle_velocities_noisy = ankle_velocities + self.noise_buffers['dof_vel'][:, 6:8]
            foot_contacts_noisy = torch.clamp(foot_contacts_normalized + self.noise_buffers['foot_contact'], 0, 1)
        else:
            base_euler_noisy, base_ang_vel_xy_noisy, base_ang_vel_z_noisy, base_lin_vel_noisy, base_pos_z_noisy = self.base_euler[:, :2], self.base_ang_vel[:, :2], self.base_ang_vel[:, [2]], self.base_lin_vel[:, :2], self.base_pos[:, [2]]
            hip_angles_noisy, hip_velocities_noisy, knee_angles_noisy, knee_velocities_noisy, ankle_angles_noisy, ankle_velocities_noisy = hip_angles, hip_velocities, knee_angles, knee_velocities, ankle_angles, ankle_velocities
            foot_contacts_noisy = foot_contacts_normalized
        
        obs_components = [
            base_euler_noisy * self.obs_scales.get("base_euler", 1.0), base_ang_vel_xy_noisy * self.obs_scales["ang_vel"], base_ang_vel_z_noisy * self.obs_scales["ang_vel"],
            base_lin_vel_noisy * self.obs_scales["lin_vel"], base_pos_z_noisy * self.obs_scales.get("base_height", 1.0), self.commands[:, :3] * self.commands_scale,
            hip_angles_noisy, hip_velocities_noisy, knee_angles_noisy, knee_velocities_noisy, ankle_angles_noisy, ankle_velocities_noisy, foot_contacts_noisy, self.last_actions,
        ]
        torch.cat(obs_components, dim=-1, out=self.obs_buf)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.step_count % self.fps_update_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.fps_timer
            if elapsed_time > 0:
                self.current_fps = (self.fps_update_interval * self.num_envs) / elapsed_time
            self.fps_timer = current_time
        
        self.extras["observations"]["critic"] = self.obs_buf
        self.extras["fps"] = self.current_fps

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        self.extras["fps"] = self.current_fps
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.dr_handler.apply_on_reset(envs_idx)

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(position=self.dof_pos[envs_idx], dofs_idx_local=self.motors_dof_idx, zero_velocity=True, envs_idx=envs_idx)

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.foot_contacts[envs_idx] = 0.0
        self.randomized_kp[envs_idx] = self.orig_kp
        self.joint_torques[envs_idx] = 0.0
        self.actuator_constraint_violations[envs_idx] = 0.0
        
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if key == "fps":
                self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][envs_idx]).item()
            else:
                self.extras["episode"]["rew_" + key] = (torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"])
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self.extras["fps"] = self.current_fps
        return self.obs_buf, self.extras