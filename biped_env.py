import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np
import time

# Import the new modular components
from rewards import RewardManager
from observations import compute_observations
from domain_randomization import DomainRandomizationManager
from sensors import SensorManager
from commands import CommandManager

class BipedEnv:
    """
    The main biped environment class, orchestrating all modular components.
    """
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]

        # --- Scene Setup ---
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(max_FPS=int(0.5 / self.dt), camera_pos=(2.0, 0.0, 2.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=40),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(dt=self.dt, constraint_solver=gs.constraint_solver.Newton, enable_collision=True, enable_joint_limit=True),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(gs.morphs.URDF(file="urdf/biped_v4.urdf", pos=self.base_init_pos.cpu().numpy(), quat=self.base_init_quat.cpu().numpy()))
        self.scene.build(n_envs=num_envs)

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        
        # --- Instantiate Managers ---
        self.sensor_manager = SensorManager(self.robot, self.num_envs, self.device)
        self.dr_manager = DomainRandomizationManager(self)
        self.reward_manager = RewardManager(self, self.reward_cfg, self.num_envs, self.dt)
        self.command_manager = CommandManager(self.num_envs, self.command_cfg, self.dt, self.device)
        self.commands = self.command_manager.commands  # Get a reference to the commands tensor

        # --- PD Control ---
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # --- Initialize Buffers ---
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]], device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=self.device, dtype=gs.tc_float)
        
        self.joint_torques = torch.zeros_like(self.actions)
        self.actuator_constraint_violations = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.foot_contacts = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)
        self.foot_contacts_raw = torch.zeros((self.num_envs, 2), device=self.device, dtype=gs.tc_float)

        self.extras = {"observations": {}}
        self.step_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

    def step(self, actions):
        self.step_count += 1
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Apply step-based domain randomization (e.g., backlash)
        exec_actions = self.dr_manager.apply_on_step(self.last_actions if self.simulate_action_latency else self.actions)
        
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # Update state buffers from simulation
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
        
        # Read sensors and apply DR
        self.foot_contacts_raw = self.sensor_manager.read_contacts(self.foot_contacts_raw)
        self.dr_manager.apply_on_step(self.actions)
        
        # Resample commands
        self.command_manager.check_and_resample(self.episode_length_buf)

        # Check termination conditions
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # Compute rewards and observations
        self.rew_buf = self.reward_manager.compute_rewards()
        self.obs_buf = compute_observations(self)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Apply reset-based domain randomization
        self.dr_manager.apply_on_reset(envs_idx)

        # Reset robot state
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(self.dof_pos[envs_idx], self.motors_dof_idx, zero_velocity=True, envs_idx=envs_idx)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.foot_contacts[envs_idx] = 0.0
        self.joint_torques[envs_idx] = 0.0
        self.actuator_constraint_violations[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Fill extras from reward manager
        self.extras["episode"] = {}
        episode_sums = self.reward_manager.get_episode_sums()
        for key in episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (torch.mean(episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"])
        self.reward_manager.reset_episode_sums(envs_idx)

        # Resample commands for new episodes
        self.command_manager.resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None