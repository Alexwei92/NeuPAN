from turtle import distance
from neupan import neupan
import irsim
import numpy as np
import argparse
import torch

from neupan import configuration
from neupan.configuration import tensor_to_np
from neupan.util import time_it
import time
from neupan.blocks import MPPIHandler


def main(
    env_file,
    planner_file,
    save_animation=False,
    ani_name="mppi_animation",
):

    env = irsim.make(env_file, save_ani=save_animation)
    env.step(np.array([0, 0]))
    
    neupan_planner = neupan.init_from_yaml(planner_file)
    pan = neupan_planner.pan
    
    mppi = MPPIHandler(
        robot=neupan_planner.robot,
        ref_speed=neupan_planner.ref_speed,
        device=configuration.device
    )
    mppi.update_goal(env.robot.goal)
    mppi.set_pan(pan)
        
    for i in range(2000):
        if env.status == "Arrived":
            print("Arrived at the goal")
            break
        
        robot_state = env.get_robot_state()
        robot_vel = env.get_robot_velocity()
        lidar_scan = env.get_lidar_scan() 
        
        points = neupan_planner.scan_to_point(robot_state, lidar_scan)
        
        state = torch.from_numpy(np.vstack([robot_state[:3], robot_vel[:2]])) # extended state
        action = mppi.command(state, points)
        
        ref_trajectory = mppi.get_trajectory()
        all_traj_rollouts = mppi.get_all_trajectory_rollouts()
              
        env.draw_points(tensor_to_np(mppi.obs_points), s=20, c="r", refresh=True) 
        for i in range(all_traj_rollouts.shape[0]):
            env.draw_trajectory(all_traj_rollouts[i].T, "c", linewidth=0.5, alpha=0.15, refresh=True)
        env.draw_trajectory(ref_trajectory.T, "g", linewidth=2.0, alpha=1.0, refresh=True)  
        
        env.step(action)
        env.render()
    
    env.end(3, ani_name=ani_name)
    # input("Press Enter to close...")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    env_path_file = "env.yaml"
    planner_path_file = "planner.yaml"

    parser.add_argument("-a", "--save_animation", action="store_true", help="save animation")
    args = parser.parse_args()
    main(env_path_file, planner_path_file, args.save_animation)
