import torch
import genesis as gs
from genesis.sensors import RigidContactForceGridSensor

class SensorManager:
    """
    Manages the initialization and reading of sensors for the robot.
    """
    def __init__(self, robot, num_envs, device):
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
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

    def read_contacts(self, foot_contacts_raw):
        """
        Reads data from the foot contact sensors and updates the raw contacts tensor.
        """
        if self.left_foot_contact_sensor is not None:
            left_contact_data = self.left_foot_contact_sensor.read()
            left_contact_tensor = torch.as_tensor(left_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            foot_contacts_raw[:, 0] = torch.max(torch.norm(left_contact_tensor, dim=-1), dim=-1)[0]

        if self.right_foot_contact_sensor is not None:
            right_contact_data = self.right_foot_contact_sensor.read()
            right_contact_tensor = torch.as_tensor(right_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            foot_contacts_raw[:, 1] = torch.max(torch.norm(right_contact_tensor, dim=-1), dim=-1)[0]
        
        return foot_contacts_raw