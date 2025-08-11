import genesis as gs
from genesis.sensors import RigidContactForceGridSensor
import pathlib
import numpy as np
import time

class GenesisSimulator:
    def __init__(self):
        print("Initializing Genesis Simulator...")

        # --- Initialize Genesis Simulator ---
        script_dir = pathlib.Path(__file__).parent.resolve()
        biped_urdf_path = script_dir / 'urdf/biped_v4.urdf'

        gs.init(backend=gs.cuda)
        self.scene = gs.Scene(show_viewer=False)  # Run without viewer to avoid graphics issues
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Define the two link names separately
        self.right_foot_link_name = "revolute_rightfoot"
        self.left_foot_link_name = "revolute_leftfoot"

        # --- Load Robot ---
        links_to_keep_list = [self.right_foot_link_name, self.left_foot_link_name]

        self.biped_robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(biped_urdf_path),
                fixed=False,
                links_to_keep=links_to_keep_list
            )
        )

        # --- Create Two Separate Sensors ---
        self.right_foot_sensor = None
        self.left_foot_sensor = None

        for link in self.biped_robot.links:
            if link.name == self.right_foot_link_name:
                self.right_foot_sensor = RigidContactForceGridSensor(
                    entity=self.biped_robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )
                print(f"Attached grid contact sensor to '{self.right_foot_link_name}'")
            elif link.name == self.left_foot_link_name:
                self.left_foot_sensor = RigidContactForceGridSensor(
                    entity=self.biped_robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )
                print(f"Attached grid contact sensor to '{self.left_foot_link_name}'")

        self.scene.build()
        print('Genesis simulation with grid sensors initialized.')

        # --- Get Joint Names ---
        num_dofs = len(self.biped_robot.get_dofs_position().cpu().numpy())
        self.joint_names = [f'joint_{i}' for i in range(num_dofs)]

    def run_simulation(self):
        """Run the simulation and print values"""
        step_count = 0
        max_steps = 100  # Run for 100 steps for testing
        
        while step_count < max_steps:
            step_count += 1
            self.step_simulation()
            
            # Print values every 10 steps to see more output
            if step_count % 10 == 0:
                print(f"\n--- Step {step_count} ---")
            
            time.sleep(0.02)  # 50 Hz

    def step_simulation(self):
        try:
            self.scene.step()
            
            # --- Get robot data with proper error handling ---
            # Get quaternion - might need different indexing
            quat_data = self.biped_robot.get_quat().cpu().numpy()
            
            # Handle different possible shapes
            if quat_data.ndim == 1 and len(quat_data) == 4:
                orientation_quat = quat_data
            elif quat_data.ndim == 2 and quat_data.shape[0] > 0:
                orientation_quat = quat_data[0]
            else:
                print(f"Unexpected quaternion shape: {quat_data.shape}")
                orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Default identity quaternion
            
            # Get velocities
            vel_data = self.biped_robot.get_vel().cpu().numpy()
            if vel_data.ndim == 2 and vel_data.shape[0] > 0:
                linear_velocity = vel_data[0]
            elif vel_data.ndim == 1:
                linear_velocity = vel_data
            else:
                linear_velocity = np.array([0.0, 0.0, 0.0])
                
            ang_data = self.biped_robot.get_ang().cpu().numpy()
            if ang_data.ndim == 2 and ang_data.shape[0] > 0:
                angular_velocity = ang_data[0]
            elif ang_data.ndim == 1:
                angular_velocity = ang_data
            else:
                angular_velocity = np.array([0.0, 0.0, 0.0])

            # Get joint data
            joint_positions = self.biped_robot.get_dofs_position().cpu().numpy()
            joint_velocities = self.biped_robot.get_dofs_velocity().cpu().numpy()

            # Print robot state data
            print(f"Orientation (quat): [{orientation_quat[0]:.3f}, {orientation_quat[1]:.3f}, {orientation_quat[2]:.3f}, {orientation_quat[3]:.3f}]")
            print(f"Linear velocity: [{linear_velocity[0]:.3f}, {linear_velocity[1]:.3f}, {linear_velocity[2]:.3f}]")
            print(f"Angular velocity: [{angular_velocity[0]:.3f}, {angular_velocity[1]:.3f}, {angular_velocity[2]:.3f}]")
            print(f"Joint positions: {joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            
            # --- Read from each sensor and print contact forces ---
            if self.right_foot_sensor:
                try:
                    grid_forces = self.right_foot_sensor.read()
                    max_force = np.max(np.linalg.norm(grid_forces, axis=-1))
                    print(f"Right foot contact force: {max_force:.3f}")
                except Exception as e:
                    print(f"Error reading right foot sensor: {e}")

            if self.left_foot_sensor:
                try:
                    grid_forces = self.left_foot_sensor.read()
                    max_force = np.max(np.linalg.norm(grid_forces, axis=-1))
                    print(f"Left foot contact force: {max_force:.3f}")
                except Exception as e:
                    print(f"Error reading left foot sensor: {e}")
                    
        except Exception as e:
            print(f"Error in simulation step: {e}")

def main(args=None):
    simulator = GenesisSimulator()
    print("Genesis simulation running. Starting simulation loop...")
    simulator.run_simulation()
    print("Simulation completed.")

if __name__ == '__main__':
    main()