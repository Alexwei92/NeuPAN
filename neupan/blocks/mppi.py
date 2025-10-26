from typing import Optional, Union
from dataclasses import dataclass

import torch
import numpy as np

from neupan.pytorch_mppi import MPPI
from neupan.blocks import PAN
from neupan.util import downsample_decimation, time_it

import time


@dataclass
class RobotParams:
    kinematics: str
    nx: int
    dt: float

    L: Optional[float]
    max_u: torch.Tensor
    max_du: torch.Tensor

    is_multipolygon: bool
    num_of_polygons: int


def normalize_with_minmax(x):
    min_val, max_val = torch.aminmax(x)
    return (x - min_val) / (max_val - min_val + 1e-8)


def normalize_with_median(x, eps=1e-8):
    med = x.median(dim=-1, keepdim=True).values
    mad = (x - med).abs().median(dim=-1, keepdim=True).values
    return (x - med) / (mad + eps)


class MPPIHandler:
    def __init__(
        self,
        robot,
        receding: int = 30,
        step_time: float = 0.1,
        ref_speed: float = 4.0,
        num_samples: int = 500,
        noise_sigma: list = [0.5, 0.5],
        lambda_: float = 1.0,
        max_obs_num: int = 10,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:

        self.device = device
        self.dtype = dtype

        # MPPI parameters
        self.receding = receding
        self.dt = step_time
        self.ref_speed = torch.tensor(ref_speed, dtype=self.dtype, device=self.device)
        self.num_samples = num_samples
        self.noise_sigma = torch.diag(
            torch.tensor(noise_sigma, dtype=self.dtype, device=self.device)
        )
        self.lambda_ = lambda_

        # Cost
        self.goal = torch.tensor([0, 0, 0], dtype=self.dtype, device=self.device)
        self.max_obs_num = max_obs_num

        # Robot parameters
        self.robot_params = RobotParams(
            kinematics=robot.kinematics,
            nx=5,
            dt=step_time,
            L=robot.L,
            max_u=torch.tensor(
                robot.max_speed.squeeze(1), dtype=self.dtype, device=self.device
            ),
            max_du=torch.tensor(
                robot.max_acce.squeeze(1), dtype=self.dtype, device=self.device
            ),
            is_multipolygon=robot.is_multipolygon,
            num_of_polygons=robot.num_of_polygons,
        )

        # PAN
        self.pan = None
        self.obs_points = None
        self.mppi_obs_points = None

        # MPPI controller
        self.state = None
        self.build_controller()
        
        self.hyperparameters = {
            "gamma": 0.98,
            "r_gate": 10.0,
            "pos_tol": 0.2,
            "yaw_tol": 0.1,
            "R_goal": 200.0,
            "d_safe": 2.0,
            "beta": 1.0,
        }

    def set_pan(self, pan: PAN):
        assert self.robot_params.is_multipolygon == pan.is_multipolygon

        self.pan = pan

    def update_goal(self, goal: Union[torch.Tensor, list, np.ndarray]):
        if isinstance(goal, list):
            self.goal = torch.tensor(goal, dtype=self.dtype, device=self.device)
        elif isinstance(goal, np.ndarray):
            self.goal = torch.from_numpy(goal).type(self.dtype).to(self.device)
        elif isinstance(goal, torch.Tensor):
            self.goal = goal.type(self.dtype).to(self.device)
        else:
            raise ValueError(f"Goal must be a list, numpy array, or torch tensor")

        if self.goal.ndim == 1:
            self.goal = self.goal.unsqueeze(0)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, t: int = None):
        if self.robot_params.kinematics == "acker":
            return self.ackermann_model(state, action, self.robot_params)
        elif self.params.kinematics == "diff":
            return self.diff_model(state, action, self.robot_params)
        else:
            raise ValueError(f"Kinematics {self.robot_params.kinematics} not supported")

    def ackermann_model(
        self, state: torch.Tensor, action: torch.Tensor, params: RobotParams
    ):
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        dt = params.dt
        x, y, yaw, v, delta = state.unbind(-1)

        action_feas_min = torch.maximum(
            -params.max_du * torch.ones_like(action),
            (-params.max_u - state[:, 3:5]) / dt,
        )
        action_feas_max = torch.minimum(
            params.max_du * torch.ones_like(action), (params.max_u - state[:, 3:5]) / dt
        )
        action_feas = torch.clamp(action, action_feas_min, action_feas_max)

        a, d_delta = action_feas.unbind(-1)
        v_next = v + a * dt
        delta_next = delta + d_delta * dt

        new_state = torch.stack(
            [
                x + v_next * torch.cos(yaw) * dt,
                y + v_next * torch.sin(yaw) * dt,
                yaw + v_next * torch.tan(delta_next) / params.L * dt,
                v_next,
                delta_next,
            ],
            dim=-1,
        )

        return new_state

    def diff_model(
        self, state: torch.Tensor, action: torch.Tensor, params: RobotParams
    ):
        if state.ndim == 1:
            state = state.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        dt = params.dt
        x, y, yaw, v, w = state.unbind(-1)

        action_feas_min = torch.maximum(
            -params.max_du * torch.ones_like(action),
            (-params.max_u - state[:, 3:5]) / dt,
        )
        action_feas_max = torch.minimum(
            params.max_du * torch.ones_like(action), (params.max_u - state[:, 3:5]) / dt
        )
        action_feas = torch.clamp(action, action_feas_min, action_feas_max)

        a, d_w = action_feas.unbind(-1)
        v_next = v + a * dt
        w_next = w + d_w * dt

        new_state = torch.stack(
            [
                x + v_next * torch.cos(yaw) * dt,
                y + v_next * torch.sin(yaw) * dt,
                yaw + w_next * dt,
                v_next,
                w_next,
            ],
            dim=-1,
        )

        return new_state

    def terminal_state_cost(self, state: torch.Tensor, action: torch.Tensor):
        pass

    def running_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ):
        """
        state: (M, K, T, nx)
        action: (M, K, T, 2)
        t: (T,)
        """

        # x, y, yaw, v, delta = state.unbind(-1)
        # a_cmd, d_delta_cmd = action.unbind(-1)

        M, K, T, nx = state.shape
        eps = 1e-8
        
        
        gamma = self.hyperparameters.get("gamma", 0.98)
        r_gate = self.hyperparameters.get("r_gate", 10.0)

        # Time discount
        td = (gamma ** torch.arange(T, device=self.device)).view(1, 1, T)  # (1, 1, T) # TODO: can be cached

        # --------- control cost ---------
        control_cost = (action**2).sum(dim=-1)  # (M, K, T)

        # --------- progress cost ---------
        goal_vec = self.goal[0:2, 0].view(1, 1, 1, 2) - state[..., 0:2]  # (M, K, T, 2)
        r2 = (goal_vec**2).sum(dim=-1)  # (M, K, T)
        dr2 = r2.diff(dim=-1, prepend=r2[..., :1])  # (M, K, T)
        progress_cost = torch.nn.functional.relu(dr2)  # (M, K, T)

        # --------- yaw cost (gated) ---------
        gate = torch.exp(-r2 / (2 * (r_gate**2)))  # (M, K, T)
        yaw_err = (state[..., 2] - self.goal[2, 0] + torch.pi) % (2 * torch.pi) - torch.pi  # (M, K, T)
        goal_yaw_cost = gate * (yaw_err**2)  # (M, K, T)

        # --------- speed toward goal cost ---------
        inv_r = torch.rsqrt(r2 + eps)  # (M, K, T)
        unit_dir = goal_vec * inv_r.unsqueeze(-1)  # (M, K, T, 2)
        fwd = torch.stack(
            [torch.cos(state[..., 2]), torch.sin(state[..., 2])], dim=-1
        )  # (M, K, T, 2)
        v_toward = state[..., 3] * (fwd * unit_dir).sum(dim=-1)  # (M, K, T)
        speed_toward_goal_cost = -v_toward  # (M, K, T)

        # --------- reference speed cost (gated) ---------
        ref_speed_cost = (1.0 - gate) * (
            state[..., 3] - self.ref_speed
        ) ** 2  # (M, K, T)

        # --------- collision cost ---------
        collision_cost = torch.zeros_like(control_cost)
        if self.obs_points is not None:
            distance_list = self.pan_forward(
                state.view(M * K * T, nx)[:, :3], self.obs_points
            )
            
            if self.robot_params.is_multipolygon:
                all_distances = []
                for poly_id in range(self.robot_params.num_of_polygons):
                    all_distances = torch.cat(distance_list[poly_id], dim=0).view(
                        M, K, T, -1
                    )
                    topk_distance = all_distances[
                        ..., : min(self.max_obs_num, all_distances.shape[-1])
                    ]  # (M, K, T, topk)
            else:
                all_distances = torch.cat(distance_list, dim=0).view(M, K, T, -1)
                topk_distance = all_distances[
                    ..., : min(self.max_obs_num, all_distances.shape[-1])
                ]  # (M, K, T, topk)

            d_safe = 2.0 # TODO: hyperparameter
            beta = 1.0 # TODO: hyperparameter

            collision_penalty = torch.nn.functional.softplus(
                d_safe - topk_distance
            )  # (M, K, T, topk)
            collision_cost = beta * collision_penalty.mean(dim=-1)  # (M, K, T)
            
        # --------- early termination reward ---------
        pos_tol = 0.2 # TODO: hyperparameter
        yaw_tol = 0.1 # TODO: hyperparameter
        R_goal = 200.0 # TODO: hyperparameter

        at_goal = (r2 < (pos_tol**2)) & (yaw_err.abs() < yaw_tol)  # (M, K, T)
        reached_cumsum = at_goal.cumsum(dim=-1).clamp_max_(1)  # (M, K)
        first_mask = at_goal & (reached_cumsum == 1)  # (M, K, T)
        term_bonus = -R_goal * first_mask.float()



        # w_control = 0.01
        # w_goal_abs = 0.2
        # w_progress = 1.0
        # w_heading = 0.5
        # w_speed_toward = 0.3
        # w_ref_speed = 0.05
        # w_collision = 1.0

        w_control = 0.1
        w_goal_abs = 2.0
        w_progress = 1.0
        w_heading = 5.0
        w_speed_toward = 2.0
        w_ref_speed = 1.0
        w_collision = 5.0

        per_step = (
            w_control * control_cost
            + w_goal_abs * r2 / (10.0 ** 2)
            + w_progress * progress_cost / (10 ** 2)
            + w_heading * goal_yaw_cost / (torch.pi ** 2)
            + w_speed_toward * speed_toward_goal_cost / 2.0
            + w_ref_speed * ref_speed_cost / (2.0 ** 2)
            + w_collision * collision_cost
        )  # (M, K, T)

        # per_step = (
        #     w_control * normalize_with_median(control_cost)
        #     + w_goal_abs * normalize_with_minmax(r2)
        #     + w_progress * normalize_with_median(progress_cost)
        #     # + w_heading * normalize_with_median(goal_yaw_cost)
        #     # + w_speed_toward * normalize_with_median(speed_toward_goal_cost)
        #     + w_ref_speed * normalize_with_median(ref_speed_cost)
        #     # + w_collision * normalize_with_median(collision_cost)
        # )  # (M, K, T)

        cost = (per_step + term_bonus) * td  # (M, K, T)

        ###

        # M, K, T, nx = state.shape
        
        # # Control cost
        # control_cost = (action**2).sum(dim=-1)  # (M, K, T)

        # # Goal cost
        # goal_xy_cost = torch.sum(
        #     (state[..., 0:2] - self.goal[0:2, 0]) ** 2, dim=-1
        # )  # (M, K, T)
        # goal_yaw_cost = (
        #     (state[..., 2] - self.goal[2, 0] + torch.pi) % (2 * torch.pi) - torch.pi
        # ) ** 2  # (M, K, T)

        # # Ref speed cost (M, K, T)
        # ref_speed_cost = (state[..., 3] - self.ref_speed) ** 2

        # # Collision cost (M, K, T)
        # collision_cost = torch.zeros_like(control_cost)

        # if self.obs_points is not None:
        #     distance_list = self.pan_forward(
        #         state.view(M * K * T, nx)[:, :3], self.obs_points
        #     )

        #     if self.robot_params.is_multipolygon:
        #         for poly_id in range(self.robot_params.num_of_polygons):
        #             all_distances = torch.cat(distance_list[poly_id], dim=0).view(
        #                 M, K, T, -1
        #             )
        #     else:
        #         all_distances = torch.cat(distance_list, dim=0).view(M, K, T, -1)

        #     topk_distance = all_distances[
        #         ..., : min(self.max_obs_num, all_distances.shape[-1])
        #     ]

        #     min_distance = topk_distance[..., 0]  # (M, K, T)
        #     # collision_cost = -min_distance
        #     collision_cost = torch.where(
        #         min_distance > 3.0, torch.zeros_like(min_distance), -min_distance
        #     )

        # # print(f"control cost: {control_cost.sum(dim=-1)}")
        # # print(f"ref speed cost: {ref_speed_cost.sum(dim=-1)}")
        # # print(f"goal xy cost: {goal_xy_cost.sum(dim=-1)}")
        # # print(f"goal yaw cost: {goal_yaw_cost.sum(dim=-1)}")
        # # print(f"collision cost: {collision_cost.sum(dim=-1)}")

        # cost = (
        #     control_cost
        #     + ref_speed_cost
        #     + goal_xy_cost * 10
        #     + goal_yaw_cost * 20
        #     + collision_cost * 400
        # )

        return cost

    def build_controller(self):
        self.controller = MPPI(
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            nx=self.robot_params.nx,
            noise_sigma=self.noise_sigma,
            num_samples=self.num_samples,
            horizon=self.receding,
            device=self.device,
            terminal_state_cost=None,
            lambda_=self.lambda_,
            u_min=-self.robot_params.max_du,
            u_max=self.robot_params.max_du,
            step_dependent_dynamics=False,
        )

    def dune_batch_forward(self, nom_s, obs_points):
        if obs_points is not None:
            point_flow_list, R_list, obs_points_list = self.generate_point_flow(
                nom_s, obs_points
            )

            if self.robot_params.is_multipolygon:
                distance_list = []
                for i, dune_layer in enumerate(self.pan.dune_layer_list):
                    _, _, _, distance_list_i = dune_layer.batch_forward(
                        point_flow_list, R_list, obs_points_list
                    )

                    distance_list.append(distance_list_i)
            else:
                _, _, _, distance_list = self.pan.dune_layer.batch_forward(
                    point_flow_list, R_list, obs_points_list
                )
        else:
            distance_list = []

        return distance_list

    def pan_forward(self, nom_s, obs_points):
        if not torch.is_tensor(nom_s):
            nom_s = self.np_to_tensor(nom_s)

        if (obs_points is not None) and (not torch.is_tensor(obs_points)):
            obs_points = self.np_to_tensor(obs_points)

        distance_list = self.dune_batch_forward(nom_s, obs_points)

        return distance_list

    def point_state_transform(self, state: torch.Tensor, obs_points: torch.Tensor):
        state = state.reshape((3, 1))
        trans = state[0:2]
        theta = state[2, 0]
        R = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        ).to(self.device)

        p0 = R.T @ (obs_points - trans)

        return p0, R

    def batch_state_transform(self, states: torch.Tensor, obs_points: torch.Tensor):
        N = states.shape[0]

        trans = states[:, :2]
        theta = states[:, 2]

        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            (torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2
        )  # (N, 2, 2)

        obs_points_b = obs_points.unsqueeze(0).expand(N, -1, -1)  # (N, 2, M)
        trans_b = trans.unsqueeze(-1)  # (N, 2, 1)

        p0_b = R.transpose(1, 2) @ (obs_points_b - trans_b)  # (N, 2, M)

        point_flow_list = [p0_b[n] for n in range(N)]
        R_list = [R[n] for n in range(N)]

        return point_flow_list, R_list

    def generate_point_flow(self, nom_s: torch.Tensor, obs_points: torch.Tensor):
        max_num = self.pan.dune_max_num
        if obs_points.shape[1] > max_num:
            obs_points = downsample_decimation(obs_points, max_num)

        # # by point
        # obs_points_list = []
        # point_flow_list = []
        # R_list = []

        # for i in range(nom_s.shape[0]):
        #     obs_points_list.append(obs_points)
        #     p0, R = self.point_state_transform(nom_s[i, :], obs_points)
        #     point_flow_list.append(p0)
        #     R_list.append(R)

        # by batch
        point_flow_list, R_list = self.batch_state_transform(nom_s, obs_points)
        obs_points_list = [obs_points] * nom_s.shape[0]

        return point_flow_list, R_list, obs_points_list

    @time_it("- mppi command")
    def command(
        self,
        state: Union[torch.Tensor, list, np.ndarray],
        obs_points: Optional[np.ndarray] = None,
    ):
        if isinstance(state, list):
            state = torch.tensor(state, dtype=self.dtype, device=self.device)
        elif isinstance(state, np.ndarray):
            state = self.np_to_tensor(state)
        elif isinstance(state, torch.Tensor):
            state = state.type(self.dtype).to(self.device)
        else:
            raise ValueError(f"State must be a list, numpy array, or torch tensor")

        self.state = state
        self.obs_points = (
            self.np_to_tensor(obs_points) if obs_points is not None else None
        )

        with torch.no_grad():
            action = self.controller.command(state)
            next_state = self.dynamics(state.T, action)
            u_0 = next_state[0, 3:5].detach().cpu().numpy()  # v, delta (or w)
            return u_0

    def get_action_sequence(self):
        return self.controller.get_action_sequence()

    def get_trajectory(self):
        if self.state is None:
            return None

        trajectory = []
        action_sequence = self.get_action_sequence()
        state = self.state.T

        for i in range(action_sequence.shape[0]):
            next_state = self.dynamics(state, action_sequence[i])
            state = next_state
            trajectory.append(state[:, :3])

        trajectory = torch.cat(trajectory, dim=0).detach().cpu().numpy()  # (T, 3)
        return trajectory

    def get_all_trajectory_rollouts(self):
        return (
            self.controller.states[..., 0:3].mean(dim=0).detach().cpu().numpy()
        )  # (K, T, 3)

    def np_to_tensor(self, array):
        if np.isscalar(array):
            return torch.tensor(array).type(torch.float32).to(self.device)

        return torch.from_numpy(array).type(torch.float32).to(self.device)

    def tensor_to_np(self, tensor):
        if tensor is None:
            return None

        tensor = tensor.cpu()
        return tensor.detach().numpy()
