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
    
    @time_it("Obstacle Processing")
    def process_obstacles_scale(self, obs_points, pan, nom_s, point_velocities):
        point_flow_list, R_list, obs_points_list = pan.generate_point_flow(
            nom_s, obs_points, point_velocities
        )

        mu_scaled, lam_scaled, sort_point_scaled, distance_scaled, = [], [], [], []
        mu_scaled_from_unit, lam_scaled_from_unit, sort_point_scaled_from_unit, distance_scaled_from_unit = [], [], [], []
        unit_layer = pan.dune_layer_list[0] 
        scaled_layer = pan.dune_layer_list[1] 

        # calculate the latent features for scaled polygon
        mu_scaled, lam_scaled, sort_point_scaled, distance_scaled = scaled_layer(
                point_flow_list, R_list, obs_points_list
            )
        # check the shape of point_flow_list and mu_scaled
        # print("point_flow_list shapes: ", [pf.shape for pf in point_flow_list])
        # print(f"mu_scaled shapes:{[m.shape for m in mu_scaled]}, length of mu_scaled: {len(mu_scaled)}")
        # calculate the latent features for the scaled polygon with unit layer
        # 1. scale the obs points and translation
        scaled_nom_s = nom_s.clone()
        # print(f"scaled_nom_s before scaling: {scaled_nom_s[:,0,0]}")
        scaled_nom_s[:2, :] *= 2
        # print(f"scaled_nom_s after scaling: {scaled_nom_s[:,0,0]}")

        scaled_obs = obs_points.clone()
        print(f"obs_points before scaling: {scaled_obs[:,0]}")
        scaled_obs[:2, :] *= 2
        print(f"obs_points after scaling: {scaled_obs[:,0]}")

        point_flow_scaled, R_list, obs_points_scaled = pan.generate_point_flow(
            scaled_nom_s, scaled_obs, point_velocities
        )
        # 2. scale it to the scaled case
        mu_scaled_from_unit, lam_scaled_from_unit, sort_point_scaled_from_unit, distance_scaled_from_unit = unit_layer(
                point_flow_scaled, R_list, obs_points_scaled
            )
        # scale back the outputs from the unit layer to match the scaled case
        distance_scaled_from_unit = [d / 2. for d in distance_scaled_from_unit]
        sort_point_scaled_from_unit = [p / 2. for p in sort_point_scaled_from_unit]
        # mu_scaled_from_unit = [m * 2. for m in mu_scaled_from_unit]
        return (mu_scaled, lam_scaled, sort_point_scaled, distance_scaled, 
                mu_scaled_from_unit, lam_scaled_from_unit, sort_point_scaled_from_unit, distance_scaled_from_unit)

def main(
    env_file,
    planner_file,
):
    
    env = irsim.make(env_file)
    env.step()
    neupan_planner = neupan.init_from_yaml(planner_file, device='cuda', time_print=True)

    robot_state = env.get_robot_state()
    lidar_scan = env.get_lidar_scan()

    points = neupan_planner.scan_to_point(robot_state, lidar_scan)
    # print(f"Number of obs points: {points.shape[1] if points is not None else 0}")
    point_velocities = None

    pan = neupan_planner.pan
    print(neupan_planner.robot.G_list)
    print(neupan_planner.robot.h_list)
    nom_s = torch.stack([np_to_tensor(robot_state[:3]) for _ in range(pan.T+1)], dim=1)
    obs_points = np_to_tensor(points) if points is not None else None
    print(f"Number of obs points: {obs_points.shape[1]}, type of the obs points: {type(obs_points[:,0])}")
    print(f"Number of obs points: {nom_s.shape[1]}, type of the nom_s: {type(nom_s)}")
    # I just want to choose one point form the obs_points
    # obs_points = obs_points[:,0].unsqueeze(1)
    # print(f"Number of obs points: {obs_points.shape[1]}, type of the obs points: {type(obs_points[:,0])}")
    point_velocities = None

    # Use Timer class with @time_it decorator
    timer = Timer()
    (mu_scaled, lam_scaled, sort_point_scaled, distance_scaled, 
    mu_scaled_from_unit, lam_scaled_from_unit, sort_point_scaled_from_unit, distance_scaled_from_unit) = timer.process_obstacles_scale(obs_points, pan, nom_s, point_velocities)

    env.draw_points(tensor_to_np(obs_points), s=20, c="r", refresh=False)


    mu_scaled_sorted = [m[:, m.sum(dim=0).argsort()] for m in mu_scaled]
    mu_scaled_from_unit_sorted = [m[:, m.sum(dim=0).argsort()] for m in mu_scaled_from_unit]

    lam_scaled_sorted = [l[:, l.sum(dim=0).argsort()] for l in lam_scaled]
    lam_scaled_from_unit_sorted = [l[:, l.sum(dim=0).argsort()] for l in lam_scaled_from_unit]

    sort_point_scaled_sorted = [p[:, p.sum(dim=0).argsort()] for p in sort_point_scaled]
    sort_point_scaled_from_unit_sorted = [p[:, p.sum(dim=0).argsort()] for p in sort_point_scaled_from_unit]

    distance_scaled_sorted = [d[d.argsort()] for d in distance_scaled]
    distance_scaled_from_unit_sorted = [d[d.argsort()] for d in distance_scaled_from_unit]

    diff_mu = torch.abs(mu_scaled[0] - mu_scaled_from_unit[0])
    print(f"mu_scaled_sorted:{mu_scaled[0][:,0]}")
    print(f"mu_scaled_from_unit_sorted:{mu_scaled_from_unit[0][:,0]}")
    print(f"Max difference in mu: {diff_mu.max().item()}")
    print(f"Mean difference in mu: {diff_mu.mean().item()}")

    diff_lam = torch.abs(lam_scaled_sorted[0] - lam_scaled_from_unit_sorted[0])
    print(f"lam_scaled_sorted:{lam_scaled_sorted[0][:,0]}")
    print(f"lam_scaled_from_unit_sorted:{lam_scaled_from_unit_sorted[0][:,0]}")
    print(f"Max difference in lam: {diff_lam.max().item()}")
    print(f"Mean difference in lam: {diff_lam.mean().item()}") 

    diff_point = torch.abs(sort_point_scaled_sorted[0] - sort_point_scaled_from_unit_sorted[0])
    print(f"Max difference in point: {diff_point.max().item()}")
    print(f"Mean difference in point: {diff_point.mean().item()}")

    diff_distance = torch.abs(distance_scaled_sorted[0] - distance_scaled_from_unit_sorted[0])
    print(f"Max difference in distance: {diff_distance.max().item()}")
    print(f"Mean difference in distance: {diff_distance.mean().item()}")
    # id_to_display = min(0, len(sort_point_scaled) - 1)
    env.draw_points(
        tensor_to_np(sort_point_scaled[0][:, :pan.nrmp_max_num]),
        s=100, c='g', refresh=False, alpha=0.2
    )
    env.draw_points(
        tensor_to_np(sort_point_scaled_from_unit[0][:, :pan.nrmp_max_num]),
        s=50, c='b', refresh=False, alpha=0.5
    )

    env.render()
        
    input("Press Enter to close...")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    env_path_file = "env.yaml"
    planner_path_file = "planner.yaml"

    main(env_path_file, planner_path_file)
