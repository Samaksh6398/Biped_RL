import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.sensors import RigidContactForceGridSensor
import numpy as np
import time


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BipedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
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

        # add plain
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
        
        # Find foot links and create contact sensors
        for link in self.robot.links:
            if link.name == "revolute_leftfoot":
                self.left_foot_contact_sensor = RigidContactForceGridSensor(
                    entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )
            elif link.name == "revolute_rightfoot":
                self.right_foot_contact_sensor = RigidContactForceGridSensor(
                    entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Domain randomization setup
        self.orig_kp = torch.tensor([self.env_cfg["kp"]] * self.num_actions, device=gs.device)
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        # External force application disabled - removed push_interval initialization
        
        # Pre-allocate noise buffers for optimization
        self.noise_buffers = {
            'dof_pos': torch.zeros((self.num_envs, self.num_actions), device=gs.device),
            'dof_vel': torch.zeros((self.num_envs, self.num_actions), device=gs.device),
            'lin_vel': torch.zeros((self.num_envs, 3), device=gs.device),
            'ang_vel': torch.zeros((self.num_envs, 3), device=gs.device),
            'base_pos': torch.zeros((self.num_envs, 3), device=gs.device),
            'base_euler': torch.zeros((self.num_envs, 3), device=gs.device),
            'foot_contact': torch.zeros((self.num_envs, 2), device=gs.device),
        }
        
        # Different update intervals for different randomizations - optimization
        self.randomization_intervals = {
            'motor_strength': 50,  # Every 50 steps
            'friction': 100,       # Every 100 steps  
            'mass': 200,           # Every 200 steps
            'observation_noise': 1, # Every step
            'foot_contacts': 1,    # Every step
            'motor_backlash': 20,  # Every 20 steps
        }
        self.randomization_counters = {k: 0 for k in self.randomization_intervals}
        self.randomization_step_counter = 0

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        
        # Add FPS tracking as a pseudo-reward for logging (zero weight)
        self.episode_sums["fps"] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        
        # Motor Backlash Buffers
        self.motor_backlash = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.motor_backlash_direction = torch.ones((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)  # Direction of last movement
        self.last_motor_positions = torch.zeros_like(self.actions)
        
        # Actuator constraint tracking buffers
        self.joint_torques = torch.zeros_like(self.actions)  # Estimated joint torques
        self.actuator_constraint_violations = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # Track violations
        
        # Additional buffers for new observations
        self.foot_contacts = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # L/R foot contact
        self.foot_contacts_raw = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # Raw contact readings
        
        # Foot Contact Domain Randomization Buffers
        self.contact_thresholds = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # Per-foot contact thresholds
        self.contact_noise_scale = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # Per-foot noise scales
        self.contact_false_positive_prob = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # False positive rates
        self.contact_false_negative_prob = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # False negative rates
        self.contact_delay_steps = torch.zeros((self.num_envs, 2), device=gs.device, dtype=torch.long)  # Delay in timesteps
        self.contact_delay_buffer = torch.zeros((self.num_envs, 2, 5), device=gs.device, dtype=gs.tc_float)  # Circular buffer for delays
        self.contact_delay_idx = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.long)  # Current delay buffer index
        
        # FPS tracking
        self.step_count = 0
        self.fps_timer = time.time()
        self.fps_update_interval = 100  # Update FPS every 100 steps
        self.current_fps = 0.0
        
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = 0.0  # Set angular velocity to zero (no turning)

    def step(self, actions):
        # FPS tracking
        self.step_count += 1
        self.randomization_step_counter += 1
        
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        
        # Apply motor backlash if enabled - with optimization check
        if self.env_cfg["domain_rand"]["add_motor_backlash"] and self._should_update_randomization('motor_backlash'):
            exec_actions = self._apply_motor_backlash(exec_actions)
        
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Estimate joint torques using PD control law: τ = kp*(pos_target - pos_current) + kd*(vel_target - vel_current)
        # For position control, vel_target = 0
        pos_error = target_dof_pos - self.dof_pos
        vel_error = -self.dof_vel  # Target velocity is 0 for position control
        self.joint_torques = self.env_cfg["kp"] * pos_error + self.env_cfg["kd"] * vel_error
        
        # Update foot contact data
        if self.left_foot_contact_sensor is not None:
            left_contact_data = self.left_foot_contact_sensor.read()
            # Convert to a tensor, move to the correct device, then reshape
            left_contact_tensor = torch.as_tensor(left_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            # Now perform torch operations
            self.foot_contacts_raw[:, 0] = torch.max(torch.norm(left_contact_tensor, dim=-1), dim=-1)[0]

        if self.right_foot_contact_sensor is not None:
            right_contact_data = self.right_foot_contact_sensor.read()
            right_contact_tensor = torch.as_tensor(right_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            self.foot_contacts_raw[:, 1] = torch.max(torch.norm(right_contact_tensor, dim=-1), dim=-1)[0]
        
        # Apply foot contact domain randomization - optimized version
        if self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            self.foot_contacts = self._apply_foot_contact_randomization_optimized(self.foot_contacts_raw)
        else:
            self.foot_contacts = self.foot_contacts_raw.clone()
        
        # Domain randomization: Apply external perturbations (robot pushing)
        # Note: External force application is disabled in domain randomization config
        dr_cfg = self.env_cfg["domain_rand"]
        if dr_cfg["push_robot"]:
            # External force application has been disabled
            # This code is kept for potential future use but will not execute
            pass
        
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        
        # Check actuator constraint violations for termination
        if self.env_cfg.get("terminate_on_actuator_violation", False):
            # Calculate constraint values: speed + 3.5*|torque|
            constraint_values = torch.abs(self.dof_vel) + self.reward_cfg.get("actuator_torque_coeff", 3.5) * torch.abs(self.joint_torques)
            constraint_limit = self.reward_cfg.get("actuator_constraint_limit", 6.16)
            termination_threshold = self.env_cfg.get("actuator_violation_termination_threshold", 2.0)
            
            # Check for severe violations (beyond termination threshold)
            # Fix: The constraint violation should be compared to constraint_limit first
            violations = constraint_values - constraint_limit
            max_violation_per_env = torch.max(violations, dim=1)[0]
            self.reset_buf |= max_violation_per_env > termination_threshold

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            # Check if the reward is enabled in the config
            if self.reward_cfg.get("reward_enables", {}).get(name, True):
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew
        
        # Update FPS tracking in episode sums for logging
        # For FPS, we want the current FPS value, not a sum
        self.episode_sums["fps"][:] = self.current_fps

        # compute observations
        # Extract specific joint angles and velocities based on joint order: 
        # [left_hip1, left_hip2, left_knee, left_ankle, right_hip1, right_hip2, right_knee, right_ankle, torso]
        
        # Hip joints (left_hip1, left_hip2, right_hip1, right_hip2)
        hip_angles = torch.cat([
            (self.dof_pos[:, [0, 1]] - self.default_dof_pos[[0, 1]]) * self.obs_scales["dof_pos"],  # Left hip
            (self.dof_pos[:, [4, 5]] - self.default_dof_pos[[4, 5]]) * self.obs_scales["dof_pos"],  # Right hip
        ], dim=1)  # 4 values
        
        hip_velocities = torch.cat([
            self.dof_vel[:, [0, 1]] * self.obs_scales["dof_vel"],  # Left hip
            self.dof_vel[:, [4, 5]] * self.obs_scales["dof_vel"],  # Right hip
        ], dim=1)  # 4 values
        
        # Knee joints (left_knee, right_knee)
        knee_angles = torch.cat([
            (self.dof_pos[:, [2]] - self.default_dof_pos[[2]]) * self.obs_scales["dof_pos"],  # Left knee
            (self.dof_pos[:, [6]] - self.default_dof_pos[[6]]) * self.obs_scales["dof_pos"],  # Right knee
        ], dim=1)  # 2 values
        
        knee_velocities = torch.cat([
            self.dof_vel[:, [2]] * self.obs_scales["dof_vel"],  # Left knee
            self.dof_vel[:, [6]] * self.obs_scales["dof_vel"],  # Right knee
        ], dim=1)  # 2 values
        
        # Ankle joints (left_ankle, right_ankle)
        ankle_angles = torch.cat([
            (self.dof_pos[:, [3]] - self.default_dof_pos[[3]]) * self.obs_scales["dof_pos"],  # Left ankle
            (self.dof_pos[:, [7]] - self.default_dof_pos[[7]]) * self.obs_scales["dof_pos"],  # Right ankle
        ], dim=1)  # 2 values
        
        ankle_velocities = torch.cat([
            self.dof_vel[:, [3]] * self.obs_scales["dof_vel"],  # Left ankle
            self.dof_vel[:, [7]] * self.obs_scales["dof_vel"],  # Right ankle
        ], dim=1)  # 2 values
        
        # Normalize foot contacts (binary or normalized force)
        foot_contacts_normalized = torch.clamp(self.foot_contacts, 0, 1)  # 2 values
        
        # Apply observation noise if enabled - optimized version
        if self.env_cfg["domain_rand"]["add_observation_noise"] and self._should_update_randomization('observation_noise'):
            # Generate all noise types in a single batch operation
            self._generate_noise_batch()
            
            # Add noise to sensor readings using pre-allocated buffers
            base_euler_noisy = self.base_euler[:, :2] + self.noise_buffers['base_euler'][:, :2]
            base_ang_vel_xy_noisy = self.base_ang_vel[:, :2] + self.noise_buffers['ang_vel'][:, :2]
            base_ang_vel_z_noisy = self.base_ang_vel[:, [2]] + self.noise_buffers['ang_vel'][:, [2]]
            base_lin_vel_noisy = self.base_lin_vel[:, :2] + self.noise_buffers['lin_vel'][:, :2]
            base_pos_z_noisy = self.base_pos[:, [2]] + self.noise_buffers['base_pos'][:, [2]]
            
            # Add noise to joint measurements
            hip_angles_noisy = hip_angles + self.noise_buffers['dof_pos'][:, :4]
            hip_velocities_noisy = hip_velocities + self.noise_buffers['dof_vel'][:, :4]
            knee_angles_noisy = knee_angles + self.noise_buffers['dof_pos'][:, 4:6]
            knee_velocities_noisy = knee_velocities + self.noise_buffers['dof_vel'][:, 4:6]
            ankle_angles_noisy = ankle_angles + self.noise_buffers['dof_pos'][:, 6:8]
            ankle_velocities_noisy = ankle_velocities + self.noise_buffers['dof_vel'][:, 6:8]
            
            # Add noise to foot contact readings
            foot_contacts_noisy = torch.clamp(foot_contacts_normalized + self.noise_buffers['foot_contact'], 0, 1)
        else:
            # Use clean observations
            base_euler_noisy = self.base_euler[:, :2]
            base_ang_vel_xy_noisy = self.base_ang_vel[:, :2]
            base_ang_vel_z_noisy = self.base_ang_vel[:, [2]]
            base_lin_vel_noisy = self.base_lin_vel[:, :2]
            base_pos_z_noisy = self.base_pos[:, [2]]
            hip_angles_noisy = hip_angles
            hip_velocities_noisy = hip_velocities
            knee_angles_noisy = knee_angles
            knee_velocities_noisy = knee_velocities
            ankle_angles_noisy = ankle_angles
            ankle_velocities_noisy = ankle_velocities
            foot_contacts_noisy = foot_contacts_normalized
        
        # Use in-place operation for observation buffer creation - optimization
        obs_components = [
            base_euler_noisy * self.obs_scales.get("base_euler", 1.0),  # Torso pitch/roll angle (2)
            base_ang_vel_xy_noisy * self.obs_scales["ang_vel"],  # Torso pitch/roll velocity (2)
            base_ang_vel_z_noisy * self.obs_scales["ang_vel"],  # Torso yaw velocity (1)
            base_lin_vel_noisy * self.obs_scales["lin_vel"],  # Torso linear velocity X,Y (2)
            base_pos_z_noisy * self.obs_scales.get("base_height", 1.0),  # Torso height (1)
            self.commands[:, :3] * self.commands_scale,  # Velocity commands [lin_vel_x, lin_vel_y, ang_vel] (3)
            hip_angles_noisy,  # Hip joint angles L/R (4)
            hip_velocities_noisy,  # Hip joint velocities L/R (4)
            knee_angles_noisy,  # Knee joint angles L/R (2)
            knee_velocities_noisy,  # Knee joint velocities L/R (2)
            ankle_angles_noisy,  # Ankle joint angles L/R (2)
            ankle_velocities_noisy,  # Ankle joint velocities L/R (2)
            foot_contacts_noisy,  # Foot contact L/R (2) - with noise applied
            self.last_actions,  # Previous actions (9)
        ]
        
        # In-place concatenation - optimization
        torch.cat(obs_components, dim=-1, out=self.obs_buf)  # Total: 2+2+1+2+1+3+4+4+2+2+2+2+2+9 = 38 observations

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Calculate and update FPS
        if self.step_count % self.fps_update_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - self.fps_timer
            if elapsed_time > 0:
                self.current_fps = (self.fps_update_interval * self.num_envs) / elapsed_time
            self.fps_timer = current_time
        
        # Add FPS to extras for logging
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

        # --- Optimized Domain Randomization ---
        dr_cfg = self.env_cfg["domain_rand"]

        # Vectorized motor strength randomization - optimization  
        if dr_cfg["randomize_motor_strength"] and self._should_update_randomization('motor_strength'):
            # Single vectorized operation for random generation - major optimization
            strength_scale = gs_rand_float(
                dr_cfg["motor_strength_range"][0],
                dr_cfg["motor_strength_range"][1], 
                (len(envs_idx), self.num_actions),
                gs.device
            )
            self.randomized_kp[envs_idx] = self.orig_kp * strength_scale
            
            # Apply kp values - individual calls due to Genesis API constraints
            # The optimization comes from vectorized random generation above
            for i, env_idx in enumerate(envs_idx):
                self.robot.set_dofs_kp(
                    self.randomized_kp[env_idx],  # 1D tensor for single environment
                    self.motors_dof_idx, 
                    envs_idx=[env_idx]
                )

        # Optimize friction randomization with batch operations
        if dr_cfg["randomize_friction"] and self._should_update_randomization('friction'):
            try:
                friction = gs_rand_float(
                    dr_cfg["friction_range"][0],
                    dr_cfg["friction_range"][1],
                    (len(envs_idx),),
                    gs.device
                )
                # Vectorized friction setting if API supports it
                if hasattr(self.robot, 'set_friction_batch'):
                    self.robot.set_friction_batch(friction, envs_idx=envs_idx)
                else:
                    # Fallback to individual setting
                    for i, env_idx in enumerate(envs_idx):
                        if hasattr(self.robot, 'set_friction'):
                            self.robot.set_friction(friction[i].item())
            except (AttributeError, TypeError) as e:
                # If the method doesn't exist or has different signature, skip friction randomization
                if not hasattr(self, '_friction_warning_shown'):
                    print(f"Warning: Friction randomization not supported: {e}")
                    self._friction_warning_shown = True

        # Optimize mass randomization with reduced frequency
        if dr_cfg["randomize_mass"] and self._should_update_randomization('mass'):
            try:
                # Find torso link - try different possible names
                torso_link = None
                for link in self.robot.links:
                    if link.name in ["revolute_torso", "base_link", "torso_link", "base", "servo1"]:
                        torso_link = link
                        break
                
                if torso_link is not None:
                    # Vectorized mass randomization
                    base_mass = 1.0  # Default base mass in kg
                    added_mass = gs_rand_float(
                        dr_cfg["added_mass_range"][0],
                        dr_cfg["added_mass_range"][1],
                        (len(envs_idx),),
                        gs.device
                    )
                    # Try vectorized mass setting if API supports it
                    if hasattr(self.robot, 'set_link_mass_batch'):
                        new_masses = base_mass + added_mass
                        self.robot.set_link_mass_batch(torso_link.idx, new_masses, envs_idx=envs_idx)
                    else:
                        # Fallback to individual setting
                        for i, env_idx in enumerate(envs_idx):
                            new_mass = base_mass + added_mass[i].item()
                            try:
                                if hasattr(self.robot, 'set_link_mass'):
                                    self.robot.set_link_mass(torso_link.idx, new_mass)
                                elif hasattr(torso_link, 'set_mass'):
                                    torso_link.set_mass(new_mass)
                                elif hasattr(torso_link, 'mass'):
                                    torso_link.mass = new_mass
                                else:
                                    if not hasattr(self, '_mass_api_warning_shown'):
                                        print("Warning: Mass randomization not supported - no mass modification API found")
                                        self._mass_api_warning_shown = True
                                    break
                            except Exception as e:
                                if not hasattr(self, '_mass_api_warning_shown'):
                                    print(f"Warning: Mass randomization not supported: {e}")
                                    self._mass_api_warning_shown = True
                                break
                else:
                    # If torso link not found, print warning only once
                    if not hasattr(self, '_mass_warning_shown'):
                        print("Warning: Torso link not found for mass randomization")
                        self._mass_warning_shown = True
            except (AttributeError, TypeError) as e:
                # If mass randomization methods don't exist, skip and warn once
                if not hasattr(self, '_mass_api_warning_shown'):
                    print(f"Warning: Mass randomization not supported: {e}")
                    self._mass_api_warning_shown = True

        # --- End of Optimized Domain Randomization ---

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.foot_contacts[envs_idx] = 0.0
        self.randomized_kp[envs_idx] = self.orig_kp  # Reset to original kp values
        self.joint_torques[envs_idx] = 0.0  # Reset torque estimates
        self.actuator_constraint_violations[envs_idx] = 0.0  # Reset violation tracking
        
        # Reset motor backlash buffers
        if self.env_cfg["domain_rand"]["add_motor_backlash"]:
            # Randomize backlash for each joint in each environment
            backlash_values = gs_rand_float(
                self.env_cfg["domain_rand"]["backlash_range"][0],
                self.env_cfg["domain_rand"]["backlash_range"][1],
                (len(envs_idx), self.num_actions),
                gs.device
            )
            self.motor_backlash[envs_idx] = backlash_values
            self.motor_backlash_direction[envs_idx] = 1.0  # Reset direction
            self.last_motor_positions[envs_idx] = 0.0
        
        # Reset foot contact domain randomization parameters
        if self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            fc_params = self.env_cfg["domain_rand"]["foot_contact_params"]
            
            # Randomize contact thresholds
            self.contact_thresholds[envs_idx] = gs_rand_float(
                fc_params["contact_threshold_range"][0],
                fc_params["contact_threshold_range"][1],
                (len(envs_idx), 2),
                gs.device
            )
            
            # Randomize noise scales
            self.contact_noise_scale[envs_idx] = gs_rand_float(
                fc_params["contact_noise_range"][0],
                fc_params["contact_noise_range"][1],
                (len(envs_idx), 2),
                gs.device
            )
            
            # Randomize false positive/negative rates
            self.contact_false_positive_prob[envs_idx] = fc_params["false_positive_rate"]
            self.contact_false_negative_prob[envs_idx] = fc_params["false_negative_rate"]
            
            # Randomize delay steps
            self.contact_delay_steps[envs_idx] = torch.randint(
                fc_params["contact_delay_range"][0],
                fc_params["contact_delay_range"][1] + 1,
                (len(envs_idx), 2),
                device=gs.device
            )
            
            # Reset delay buffers
            self.contact_delay_buffer[envs_idx] = 0.0
            self.contact_delay_idx[envs_idx] = 0
        
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if key == "fps":
                # For FPS, show the current FPS value
                self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][envs_idx]).item()
            else:
                # For actual rewards, show the sum divided by episode length
                self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
                )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        
        # Initialize FPS in extras
        self.extras["fps"] = self.current_fps
        
        return self.obs_buf, self.extras

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
       
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
       
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        
        return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_forward_velocity(self):
  
        v_target = self.reward_cfg.get("forward_velocity_target", 0.5)
        vel_error = torch.square(self.base_lin_vel[:, 0] - v_target)
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-vel_error / sigma)

    def _reward_tracking_lin_vel_x(self):
      
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
       
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive_bonus(self):
 
        return torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_fall_penalty(self):
        fall_condition = (
            (torch.abs(self.base_euler[:, 0]) > self.env_cfg.get("fall_roll_threshold", 30.0)) |  # Roll > 30 degrees
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg.get("fall_pitch_threshold", 30.0))   # Pitch > 30 degrees
        )
        return torch.where(
            fall_condition,
            torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float),  # Apply penalty
            torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # No penalty
        )

    def _reward_torso_stability(self):
        orientation_error = torch.sum(torch.square(self.base_euler[:, :2]), dim=1)  # φ² + θ² (roll² + pitch²)
        k_stability = self.reward_cfg.get("stability_factor", 1.0)
        return torch.exp(-k_stability * orientation_error)

    def _reward_height_maintenance(self):
        z_target = self.reward_cfg.get("height_target", 0.35)
        height_error = torch.square(z_target - self.base_pos[:, 2])
        return -height_error  # Return negative error (will be scaled by negative weight in config)

    def _reward_joint_movement(self):
        joint_vel_magnitude = torch.sum(torch.abs(self.dof_vel), dim=1)
        movement_threshold = self.reward_cfg.get("movement_threshold", 0.1)
        movement_scale = self.reward_cfg.get("movement_scale", 1.0)

        return torch.clamp(joint_vel_magnitude * movement_scale, 0.0, movement_threshold)
    
    def _reward_sinusoidal_gait(self):


        amplitude = self.reward_cfg.get("gait_amplitude", 0.5)  # rad
        frequency = self.reward_cfg.get("gait_frequency", 0.5)  # Hz
        

        phase_offsets = torch.tensor(
            [0, 0, 0, 0, np.pi, 0, 0, 0], 
            device=self.device, dtype=gs.tc_float
        )

        time = self.episode_length_buf * self.dt
        time = time.unsqueeze(1) # Reshape for broadcasting

        leg_joints_default = self.default_dof_pos[:-1]  # All joints except the last one (torso)
        target_leg_pos = leg_joints_default + amplitude * torch.sin(
            2 * np.pi * frequency * time + phase_offsets
        )

        leg_joints_current = self.dof_pos[:, :-1]  # All joints except the last one (torso)
        error = torch.sum(torch.square(leg_joints_current - target_leg_pos), dim=1)

        sigma = self.reward_cfg.get("gait_sigma", 0.25)
        return torch.exp(-error / sigma)

    def _reward_torso_sinusoidal(self):
        # Get torso sine wave parameters from config, with defaults
        torso_amplitude = self.reward_cfg.get("torso_amplitude", 0.2)  # rad (smaller amplitude for torso)
        torso_frequency = self.reward_cfg.get("torso_frequency", 0.3)  # Hz (different frequency from legs)
        torso_phase = self.reward_cfg.get("torso_phase", 0.0)  # Phase offset for torso

        # Calculate the current time in the episode
        time = self.episode_length_buf * self.dt
        time = time.unsqueeze(1) # Reshape for broadcasting

        # Calculate the target angle for torso joint (index 8)
        torso_default = self.default_dof_pos[8]  # Torso joint default position
        target_torso_pos = torso_default + torso_amplitude * torch.sin(
            2 * np.pi * torso_frequency * time + torso_phase
        )

        # Calculate the error between current and target torso position
        torso_current = self.dof_pos[:, 8]  # Current torso joint position
        error = torch.square(torso_current - target_torso_pos.squeeze())

        # Use an exponential function to convert the error to a reward
        sigma = self.reward_cfg.get("torso_sigma", 0.25)
        return torch.exp(-error / sigma)

    def _reward_actuator_constraint(self):
        """
        Reward function that enforces actuator constraints: speed + 3.5*|torque| <= 6.16
        
        This prevents motor overheating and ensures realistic operation within hardware limits.
        The constraint is based on typical servo motor specifications where high speed and 
        high torque cannot be sustained simultaneously.
        
        Returns:
            Negative reward (penalty) for constraint violations with tolerance
        """
        # Get constraint parameters from config
        constraint_limit = self.reward_cfg.get("actuator_constraint_limit", 6.16)
        torque_coeff = self.reward_cfg.get("actuator_torque_coeff", 3.5)
        tolerance = self.reward_cfg.get("actuator_tolerance", 0.5)
        
        # Calculate constraint values for all joints: speed + 3.5*|torque|
        # Using absolute values since the constraint applies in both directions
        constraint_values = torch.abs(self.dof_vel) + torque_coeff * torch.abs(self.joint_torques)
        
        # Calculate violations with tolerance
        # Only penalize when constraint exceeds (limit + tolerance)
        target_with_tolerance = constraint_limit + tolerance
        violations = torch.relu(constraint_values - target_with_tolerance)
        
        # Sum violations across all joints for each environment
        total_violation_per_env = torch.sum(violations, dim=1)
        
        # Store violations for monitoring/debugging
        self.actuator_constraint_violations = total_violation_per_env
        
        # Debug logging (print only occasionally to avoid spam)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 1000 == 0 and total_violation_per_env.max() > 0:
            max_constraint = constraint_values.max().item()
            max_violation = total_violation_per_env.max().item()
            max_torque = torch.abs(self.joint_torques).max().item()
            max_vel = torch.abs(self.dof_vel).max().item()
            print(f"Actuator Debug - Step {self._debug_counter}: max_constraint={max_constraint:.2f}, "
                  f"max_violation={max_violation:.2f}, max_torque={max_torque:.2f}, max_vel={max_vel:.2f}")
        
        # Return negative sum of violations (penalty increases with violation magnitude)
        return -total_violation_per_env
    
    #############helper functions for domain randomization#############
    def _should_update_randomization(self, randomization_type):
        
        interval = self.randomization_intervals.get(randomization_type, 1)
        return self.randomization_step_counter % interval == 0
    
    def _generate_noise_batch(self):
       
        noise_scales = self.env_cfg["domain_rand"]["noise_scales"]
        
        for noise_type, scale in noise_scales.items():
            if noise_type in self.noise_buffers:
                # In-place random generation - optimization
                torch.randn(
                    self.noise_buffers[noise_type].shape, 
                    out=self.noise_buffers[noise_type], 
                    device=gs.device
                )
                # In-place scaling - optimization
                self.noise_buffers[noise_type] *= scale
    
    def _apply_foot_contact_randomization_optimized(self, raw_contacts):
        
        # All operations in parallel across environments - major optimization
        contact_detected = raw_contacts > self.contact_thresholds
        
        # Vectorized noise addition - optimization
        if self.env_cfg["domain_rand"]["add_observation_noise"]:
            # Reuse noise buffer if available
            if 'foot_contact' in self.noise_buffers:
                contact_noise = self.noise_buffers['foot_contact']
            else:
                contact_noise = torch.randn_like(raw_contacts) * self.contact_noise_scale
            randomized_contacts = raw_contacts + contact_noise
        else:
            randomized_contacts = raw_contacts.clone()
        
        # Vectorized false positive/negative application - optimization
        false_pos_rand = torch.rand_like(raw_contacts)
        false_neg_rand = torch.rand_like(raw_contacts)
        
        false_pos_mask = (false_pos_rand < self.contact_false_positive_prob) & ~contact_detected
        false_neg_mask = (false_neg_rand < self.contact_false_negative_prob) & contact_detected
        
        randomized_contacts[false_pos_mask] = 1.0
        randomized_contacts[false_neg_mask] = 0.0
        if hasattr(self, 'contact_delay_steps'):
            self._apply_contact_delays_vectorized(randomized_contacts)
        
        return torch.clamp(randomized_contacts, 0.0, 1.0)
    
    def _apply_contact_delays_vectorized(self, contacts):
        if self.contact_delay_buffer.numel() == 0:
            return contacts
        
        # Update delay buffer with current readings - vectorized
        current_idx = self.contact_delay_idx[0] % self.contact_delay_buffer.shape[2]
        
        # Vectorized buffer update
        self.contact_delay_buffer[:, :, current_idx] = contacts
        
        # Apply delays vectorized where delay is uniform
        uniform_delays = torch.mode(self.contact_delay_steps.flatten())[0]
        if uniform_delays > 0 and uniform_delays < self.contact_delay_buffer.shape[2]:
            delay_idx = (current_idx - uniform_delays) % self.contact_delay_buffer.shape[2]
            # Apply uniform delay to all environments/feet that match
            delay_mask = (self.contact_delay_steps == uniform_delays)
            contacts[delay_mask] = self.contact_delay_buffer[:, :, delay_idx][delay_mask]
        
        # Handle non-uniform delays (fallback to original method for mixed delays)
        non_uniform_mask = (self.contact_delay_steps != uniform_delays)
        if non_uniform_mask.any():
            for env_idx in range(self.num_envs):
                for foot_idx in range(2):
                    if non_uniform_mask[env_idx, foot_idx]:
                        delay = self.contact_delay_steps[env_idx, foot_idx]
                        if delay > 0 and delay < self.contact_delay_buffer.shape[2]:
                            delay_idx = (current_idx - delay) % self.contact_delay_buffer.shape[2]
                            contacts[env_idx, foot_idx] = self.contact_delay_buffer[env_idx, foot_idx, delay_idx]
        
        # Update delay buffer index for all environments
        self.contact_delay_idx += 1
        
        return contacts
    
    def _apply_motor_backlash(self, actions):
        
        # Calculate the direction of movement
        position_diff = actions - self.last_motor_positions
        
        # Create mask for direction changes
        direction_change = (position_diff * self.motor_backlash_direction) < 0
        
        # Where direction changes, apply backlash offset
        backlash_offset = self.motor_backlash * self.motor_backlash_direction
        actions_with_backlash = actions.clone()
        actions_with_backlash[direction_change] += backlash_offset[direction_change]
        
        # Update direction tracking
        self.motor_backlash_direction = torch.sign(position_diff)
        self.motor_backlash_direction[torch.abs(position_diff) < 1e-6] = self.motor_backlash_direction[torch.abs(position_diff) < 1e-6]  # Keep previous direction for tiny movements
        
        # Update last positions
        self.last_motor_positions = actions.clone()
        
        return actions_with_backlash
    
    def _get_noise(self, noise_type, shape):
        
        noise_scale = self.env_cfg["domain_rand"]["noise_scales"][noise_type]
        return torch.randn(shape, device=self.device, dtype=gs.tc_float) * noise_scale
    
    def _apply_foot_contact_randomization(self, raw_contacts):
        
        randomized_contacts = raw_contacts.clone()
        
        # Apply contact thresholds
        contact_detected = raw_contacts > self.contact_thresholds
        
        # Add sensor noise
        if self.env_cfg["domain_rand"]["add_observation_noise"]:
            noise = torch.randn_like(raw_contacts) * self.contact_noise_scale
            randomized_contacts = raw_contacts + noise
        
        # Apply false positives (random contact detection when no contact)
        false_positive_mask = torch.rand_like(raw_contacts) < self.contact_false_positive_prob
        no_contact_mask = ~contact_detected
        false_positive_final = false_positive_mask & no_contact_mask
        randomized_contacts[false_positive_final] = 1.0
        
        # Apply false negatives (missing actual contact)
        false_negative_mask = torch.rand_like(raw_contacts) < self.contact_false_negative_prob
        contact_mask = contact_detected
        false_negative_final = false_negative_mask & contact_mask
        randomized_contacts[false_negative_final] = 0.0
        
        # Apply contact detection delays
        # Update delay buffer with current readings
        current_idx = self.contact_delay_idx[0] % self.contact_delay_buffer.shape[2]  # Use first env as reference
        
        # Store current readings in delay buffer for all environments
        for env_idx in range(self.num_envs):
            self.contact_delay_buffer[env_idx, :, current_idx] = randomized_contacts[env_idx, :]
        
        # Get delayed readings for each environment/foot
        for env_idx in range(self.num_envs):
            for foot_idx in range(2):
                delay = self.contact_delay_steps[env_idx, foot_idx]
                if delay > 0 and delay < self.contact_delay_buffer.shape[2]:
                    delay_idx = (current_idx - delay) % self.contact_delay_buffer.shape[2]
                    randomized_contacts[env_idx, foot_idx] = self.contact_delay_buffer[env_idx, foot_idx, delay_idx]
        
        # Update delay buffer index for all environments
        self.contact_delay_idx += 1
        
        # Ensure contacts are in valid range [0, 1]
        randomized_contacts = torch.clamp(randomized_contacts, 0.0, 1.0)
        
        return randomized_contacts
