from neupan import neupan
import irsim
import numpy as np
import argparse
import torch

from neupan.configuration import np_to_tensor, tensor_to_np
from neupan.util import downsample_decimation, time_it
import time

# import warnings

# warnings.filterwarnings(
#     "ignore",
#     category=UserWarning,
#     message=r"Converting [GA] to a CSC matrix; may take a while\.",
#     module=r"^ecos(\.ecos)?$",
# )


class Timer:
    @time_it("Obstacle Processing")
    def process_obstacles(self, obs_points, pan, nom_s, point_velocities):
        if obs_points is not None and not pan.no_obs:
            point_flow_list, R_list, obs_points_list = pan.generate_point_flow(
                nom_s, obs_points, point_velocities
            )

            if pan.is_multipolygon:
                mu_list, lam_list, sort_point_list = [], [], []
                for i, dune_layer in enumerate(pan.dune_layer_list):
                    mu_list_i, lam_list_i, sort_point_list_i, distance_list_i = dune_layer(
                        point_flow_list, R_list, obs_points_list
                    )
                    mu_list.append(mu_list_i)
                    lam_list.append(lam_list_i)
                    sort_point_list.append(sort_point_list_i)
            else:  
                mu_list, lam_list, sort_point_list, distance_list = pan.dune_layer(
                    point_flow_list, R_list, obs_points_list
                )
        else:
            mu_list, lam_list, sort_point_list = [], [], []
        return mu_list, lam_list, sort_point_list

def main(
    env_file,
    planner_file,
):
    
    env = irsim.make(env_file)
    env.step(np.array([0, 0]))
    neupan_planner = neupan.init_from_yaml(planner_file, device='cuda', time_print=True)

    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    points = neupan_planner.scan_to_point(robot_state, lidar_scan)
    point_velocities = None

    pan = neupan_planner.pan

    nom_s = torch.stack([np_to_tensor(robot_state[:3]) for _ in range(pan.T+1)], dim=1)
    obs_points = np_to_tensor(points) if points is not None else None
    # print(f"Number of obs points: {obs_points.shape[1]}")
    point_velocities = None

    # Use Timer class with @time_it decorator
    timer = Timer()
    mu_list, lam_list, sort_point_list = timer.process_obstacles(obs_points, pan, nom_s, point_velocities)

    env.draw_points(tensor_to_np(obs_points), s=20, c="r", refresh=False)

    id_to_display = min(0, len(sort_point_list) - 1)
    env.draw_points(
        tensor_to_np(sort_point_list[id_to_display][0][:, :pan.nrmp_max_num]),
        s=15, c='g', refresh=False
    )

    env.render()
        
    input("Press Enter to close...")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    env_path_file = "env.yaml"
    planner_path_file = "planner.yaml"

    main(env_path_file, planner_path_file)
