import torch
import genesis as gs

def gs_rand_float(lower, upper, shape, device):
    """
    Generates random floats in a given range.
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class CommandManager:
    """
    Manages the generation and resampling of velocity commands for the agent.
    """
    def __init__(self, num_envs, command_cfg, dt, device):
        self.num_envs = num_envs
        self.command_cfg = command_cfg
        self.dt = dt
        self.device = device
        self.commands = torch.zeros((self.num_envs, self.command_cfg["num_commands"]), device=self.device, dtype=gs.tc_float)
        self.resampling_time_s = 4.0  # From env_cfg

    def resample_commands(self, envs_idx):
        """
        Resamples velocity commands for the specified environments.
        """
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = 0.0

    def check_and_resample(self, episode_length_buf):
        """
        Checks if it's time to resample commands based on the episode length.
        """
        resample_interval = int(self.resampling_time_s / self.dt)
        envs_idx = (episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).reshape((-1,))
        self.resample_commands(envs_idx)