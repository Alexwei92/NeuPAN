from typing import Optional, Union
from dataclasses import dataclass

import torch
import numpy as np

from neupan.pytorch_mppi import MPPI
from neupan.blocks import PAN
from neupan.util import downsample_decimation, time_it


@dataclass
class RobotParams:
    kinematics: str = "acker"
    nx: int = 5
    dt: float = 0.1

    L: Optional[float] = 1.9
    max_u: torch.Tensor = torch.tensor([1.0, 1.0])
    max_du: torch.Tensor = torch.tensor([1.0, 1.0])

    is_multipolygon: bool = False
    num_of_polygons: int = 1

    is_mosaic: bool = False
    mosaic_ratios: Optional[torch.Tensor] = None
    mosaic_translations: Optional[torch.Tensor] = None


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
        num_samples: int = 400,
        noise_sigma: list = [0.5, 0.5],
        lambda_: float = 0.1,
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
            max_u=torch.tensor(robot.max_speed.squeeze(1)),
            max_du=torch.tensor(robot.max_acce.squeeze(1)),
            is_multipolygon=robot.is_multipolygon,
            num_of_polygons=robot.num_of_polygons,
            is_mosaic=robot.is_mosaic,
        )

        self.robot_params.max_u = self.robot_params.max_u.to(
            dtype=self.dtype, device=self.device
        )
        self.robot_params.max_du = self.robot_params.max_du.to(
            dtype=self.dtype, device=self.device
        )

        if robot.is_mosaic:
            self.robot_params.mosaic_ratios = robot.mosaic_ratios.to(
                dtype=self.dtype, device=self.device
            )
            self.robot_params.mosaic_translations = robot.mosaic_translations.to(
                dtype=self.dtype, device=self.device
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

        # Hyperparameters
        gamma = self.hyperparameters.get("gamma", 0.98)  # time discount factor
        map_scale = self.hyperparameters.get("map_scale", 30.0)  # map scale [m]
        r_gate = self.hyperparameters.get("r_gate", 10.0)  # goal gate radius [m]
        d_safe = self.hyperparameters.get("d_safe", 2.0)  # safety distance [m]
        pos_tol = self.hyperparameters.get(
            "pos_tol", 0.2
        )  # position tolerance to goal [m]
        yaw_tol = self.hyperparameters.get(
            "yaw_tol", 0.1
        )  # yaw tolerance to goal [rad]
        early_R = self.hyperparameters.get("early_R", 500.0)  # early arrival reward

        # Normalization
        inv_U2 = 1.0 / (self.robot_params.max_du**2)
        inv_scale2 = 1.0 / (map_scale**2)
        inv_pi2 = 1.0 / (torch.pi**2)
        inv_v = 1.0 / self.ref_speed
        inv_v2 = inv_v**2

        # Time discount
        td = (gamma ** torch.arange(T, device=self.device)).view(
            1, 1, T
        )  # (1, 1, T) # TODO: can be cached

        # --------- control cost ---------
        # control_cost = (action**2).sum(dim=-1)  # (M, K, T)
        control_cost = (action**2 * inv_U2).sum(dim=-1)  # (M, K, T)

        # --------- progress cost ---------
        goal_vec = self.goal[0:2, 0].view(1, 1, 1, 2) - state[..., 0:2]  # (M, K, T, 2)
        r2 = (goal_vec**2).sum(dim=-1)  # (M, K, T)
        dr2 = r2.diff(dim=-1, prepend=r2[..., :1])  # (M, K, T)
        progress_cost = torch.nn.functional.relu(dr2)  # (M, K, T)
        progress_cost *= inv_scale2  # normalization

        # --------- yaw cost (gated) ---------
        gate = torch.exp(-r2 / (2 * (r_gate**2)))  # (M, K, T)
        yaw_err = (state[..., 2] - self.goal[2, 0] + torch.pi) % (
            2 * torch.pi
        ) - torch.pi  # (M, K, T)
        goal_yaw_cost = gate * (yaw_err**2) * inv_pi2  # (M, K, T)
        goal_yaw_cost *= inv_pi2  # normalization

        # --------- speed toward goal cost ---------
        inv_r = torch.rsqrt(r2 + eps)  # (M, K, T)
        unit_dir = goal_vec * inv_r.unsqueeze(-1)  # (M, K, T, 2)
        fwd = torch.stack(
            [torch.cos(state[..., 2]), torch.sin(state[..., 2])], dim=-1
        )  # (M, K, T, 2)
        v_toward = state[..., 3] * (fwd * unit_dir).sum(dim=-1)  # (M, K, T)
        speed_toward_goal_cost = -v_toward  # (M, K, T)
        speed_toward_goal_cost *= inv_v  # normalization

        # --------- reference speed cost (gated) ---------
        ref_speed_cost = (1.0 - gate) * (
            state[..., 3] - self.ref_speed
        ) ** 2  # (M, K, T)
        ref_speed_cost *= inv_v2  # normalization

        # --------- collision cost ---------
        collision_cost = torch.zeros_like(control_cost)
        if self.obs_points is not None:
            distance_b = self.pan_forward(
                state.view(M * K * T, nx)[:, :3].T, self.obs_points
            )

            B = self.robot_params.num_of_polygons
            P = self.obs_points.shape[1]
            all_distances = distance_b.view(M, K, T, B, P)
            all_distances = all_distances.reshape(M, K, T, B*P)

            if B > 1:
                topk_distance, _ = torch.topk(
                    all_distances,
                    min(self.max_obs_num, all_distances.shape[-1]),
                    dim=-1,
                    largest=False,
                )  # (M, K, T, topk)
            else:
                topk_distance = all_distances[
                    ..., : min(self.max_obs_num, all_distances.shape[-1])
                ]  # (M, K, T, topk)

            collision_penalty = torch.nn.functional.softplus(
                d_safe - topk_distance
            )  # (M, K, T, topk)
            collision_cost = collision_penalty.mean(dim=-1)  # (M, K, T)

        # --------- near goal helper cost ---------
        # near = (torch.sqrt(r2 + eps) < r_gate).float()

        # v_min_near = 0.07 * self.ref_speed
        # keep_move_cost = near * torch.nn.functional.relu(
        #     v_min_near - state[..., 3]
        # )  # (M, K, T)
        # yaw_rate = state[..., 3] * torch.tan(state[..., 4]) / self.robot_params.L
        # align_turn_cost = -(yaw_rate**2) * (1.0 + 4.0 * near)

        # --------- early termination reward & cost ---------
        at_goal = (r2 < (pos_tol**2)) & (yaw_err.abs() < yaw_tol)  # (M, K, T)
        reached_cumsum = at_goal.cumsum(dim=-1).clamp_max_(1)  # (M, K)
        first_mask = at_goal & (reached_cumsum == 1)  # (M, K, T)
        term_bonus = -early_R * first_mask.float()
        is_last = torch.zeros_like(r2)
        is_last[..., -1] = 1.0
        term_pose_cost = is_last * (3.0 * r2 * inv_scale2 + 2.0 * yaw_err**2 * inv_pi2)

        # w_control = 0.1
        # w_goal_abs = 2.0
        # w_progress = 1.0
        # w_heading = 5.0
        # w_speed_toward = 2.0
        # w_ref_speed = 1.0
        # w_collision = 5.0

        w_control = 0.01
        w_goal_abs = 0.25
        w_progress = 1.0
        w_yaw = 1.0
        w_speed_toward = 0.35
        w_ref_speed = 0.1
        w_collision = 1.0
        w_term_funnel = 1.0
        w_move_near = 0.3
        w_align_turn = 0.3

        # w_yaw = w_yaw * (1.0 + 4.0 * near)
        # w_control = w_control * (1.0 - 0.5 * near)
        # w_progress = w_progress * (
        #     1.0
        #     + 2.0
        #     * ((r2[..., 0] - r2[..., -1] < 0.05 * (map_scale**2)).float().unsqueeze(-1))
        # )  # anti-stuck

        per_step = (
            w_control * control_cost
            + w_goal_abs * r2 / (map_scale**2)
            + w_progress * progress_cost
            + w_yaw * goal_yaw_cost
            + w_speed_toward * speed_toward_goal_cost
            + w_ref_speed * ref_speed_cost
            + w_collision * collision_cost
            + w_term_funnel * term_pose_cost
            # + w_move_near * keep_move_cost
            # + w_align_turn * align_turn_cost
        )  # (M, K, T)

        cost = (per_step + term_bonus) * td  # (M, K, T)

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

    def pan_forward(
        self, nom_s: torch.Tensor, obs_points: Optional[torch.Tensor] = None
    ):
        """
        Args:
            nom_s: tensor of shape (3, N)
            obs_points: tensor of shape (2, P) or None

        Returns:
            distance_b: tensor of shape (N, P) or (N, B, P) or None
            (B - number of polygons)
        """
        if obs_points is None:
            return None

        distance_b = self.dune_batch_forward(nom_s, obs_points)
        return distance_b

    def dune_batch_forward(
        self, nom_s: torch.Tensor, obs_points: Optional[torch.Tensor] = None
    ):
        """
        Args:
            nom_s: tensor of shape (3, N)
            obs_points: tensor of shape (2, P)

        Returns:
            distance_b: tensor of shape (N, B, P)
            (B - number of polygons)
        """
        if obs_points is None:
            return None

        point_flow_b, R_b, obs_points_b = self.generate_point_flow(
            nom_s, obs_points
        )

        if self.robot_params.is_multipolygon and not self.robot_params.is_mosaic:
            distance_b_list = []
            for i, dune_layer in enumerate(self.pan.dune_layer_list):
                distance_b_i = dune_layer.batch_forward_fast(
                    point_flow_b, R_b, obs_points_b
                ) # (N, P)
                distance_b_list.append(distance_b_i)
            distance_b = torch.stack(distance_b_list, dim=1) # (N, B, P)
        else:
            distance_b = self.pan.dune_layer.batch_forward_fast(
                point_flow_b, R_b, obs_points_b
            ) # (N*?, P)

            if self.robot_params.is_mosaic:
                N = nom_s.shape[1]
                B = self.robot_params.num_of_polygons
                ratios = self.robot_params.mosaic_ratios # (B, 1)
                distance_b = distance_b.reshape(N, B, -1) # (N, B, P)
                distance_b = distance_b * ratios.view(1, B, 1) # (N, B, P)
            else:
                distance_b = distance_b.unsqueeze(1) # (N, 1, P)

        return distance_b

    def generate_point_flow(self, nom_s: torch.Tensor, obs_points: torch.Tensor):
        """
        Args:
            nom_s: (3, N)
            obs_points: (2, P)

        Returns:
            point_flow_b: (N, 2, P) or (N*B, 2, P)
            R_b: (N, 2, 2) or (N*B, 2, 2)
            obs_points_b: (N, 2, P) or (N*B, 2, P)

        """
        if obs_points.shape[1] > self.pan.dune_max_num:
            obs_points = downsample_decimation(obs_points, self.pan.dune_max_num)

        if self.robot_params.is_mosaic:
            point_flow_b, R_b, obs_points_b = self.generate_point_flow_mosaic(
                nom_s, obs_points
            )
        else:
            point_flow_b, R_b = self.batch_state_transform(nom_s, obs_points) # (N, 2, P), (N, 2, 2)
            obs_points_b = obs_points.unsqueeze(0).expand(nom_s.shape[1], -1, -1) # (N, 2, P)

        return point_flow_b, R_b, obs_points_b

    def batch_state_transform(self, states: torch.Tensor, obs_points: torch.Tensor):
        """
        Args:
            states: (3, N)
            obs_points: (2, P)

        Returns:
            point_flow_b: (N, 2, P)
            R_b: (N, 2, 2)
        """
        N = states.shape[1]

        trans = states[:2, :].T  # (N, 2)
        theta = states[2, :]  # (N,)

        c, s = torch.cos(theta), torch.sin(theta)
        R = torch.stack(
            (torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2
        )  # (N, 2, 2)

        obs_points_b = obs_points.unsqueeze(0).expand(N, -1, -1)  # (N, 2, P)
        trans_b = trans.unsqueeze(-1)  # (N, 2, 1)

        p0_b = R.transpose(1, 2) @ (obs_points_b - trans_b)  # (N, 2, P)

        point_flow_b = p0_b # (N, 2, P)
        R_b = R # (N, 2, 2)

        return point_flow_b, R_b

    def generate_point_flow_mosaic(self, nom_s: torch.Tensor, obs_points: torch.Tensor):
        """
        Args:
            nom_s: (3, N)
            obs_points: (2, P)

        Returns:
            point_flow_b: (N*B, 2, P)
            R_b: (N*B, 2, 2)
            obs_points_b: (N*B, 2, P)
            (B - number of polygons)
        """
        ratios = self.robot_params.mosaic_ratios # (B, 1)
        translations = self.robot_params.mosaic_translations # (B, 2)

        B = self.robot_params.num_of_polygons
        N = nom_s.shape[1]
        P = obs_points.shape[1]

        # translate then scale nom_s per poly
        nom_s_b = nom_s.unsqueeze(0).expand(B, -1, -1).clone()  # (B, 3, N)
        nom_s_b[:, :2, :] += translations.view(B, 2, 1)
        scaled_nom_s_b = nom_s_b.clone()  # (B, 3, N)
        scaled_nom_s_b[:, :2, :] /= ratios.view(B, 1, 1)

        # scale obstacles + velocities per poly (stay in global frame, only scale xy)
        scaled_obs_b = obs_points.unsqueeze(0).expand(B, -1, -1).clone()  # (B, 2, N)
        scaled_obs_b[:, :2, :] /= ratios.view(B, 1, 1)

        # Batch generate point flows for all polys at once
        point_flow_b, _, obs_points_b = self.generate_point_flow_batched(
            scaled_nom_s_b, scaled_obs_b
        ) # (N, B, 2, P)

        point_flow_b = point_flow_b.reshape(N * B, 2, P)
        obs_points_b = obs_points_b.reshape(N * B, 2, P)

        # Compute R_b directly from nom_s to avoid duplicates
        R_base_b = self.compute_R_b(nom_s) # (N, 2, 2)
        R_b = R_base_b.unsqueeze(1).expand(N, B, -1, -1).reshape(N * B, 2, 2)

        return point_flow_b, R_b, obs_points_b

    def generate_point_flow_batched(
        self, nom_s: torch.Tensor, obs_points: torch.Tensor
    ):
        """
        Args:
            nom_s: (B, 2, N)
            obs_points: (B, 2, P)

        Returns:
            point_flow_b: (N, B, 2, P)
            R_b: (N, B, 2, 2)
            obs_points_b: (N, B, 2, P)
        """
        N = nom_s.shape[-1]

        nom_perm = nom_s.permute(2, 0, 1).contiguous()  # (N, B, 3)
        trans = nom_perm[..., :2].unsqueeze(-1)  # (N, B, 2, 1)
        theta = nom_perm[..., 2]  # (N, B)

        c, s = torch.cos(theta), torch.sin(theta) # (N, B)
        R_b = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2
        )  # (N, B, 2, 2)

        obs_points_b = obs_points.unsqueeze(0).expand(N, -1, -1, -1)  # (N, B, 2, P)
        point_flow_b = R_b.transpose(-1, -2) @ (
            obs_points_b - trans
        )  # (N, B, 2, P)

        return point_flow_b, R_b, obs_points_b

    def compute_R_b(self, nom_s: torch.Tensor):
        """
        Compute rotation matrices R_b from nom_s

        Args:
            nom_s: (B, 3, N) or (3, N)

        Returns:
            R_b: (N, B, 2, 2) or (N, 2, 2)
        """
        if nom_s.dim() == 2:
            nom_s_b = nom_s.unsqueeze(0)
        elif nom_s.dim() == 3:
            nom_s_b = nom_s
        else:
            raise ValueError("nom_s must have ndim 2 or 3")

        nom_perm = nom_s_b.permute(2, 0, 1) # (N, B, 3)
        theta = nom_perm[..., 2]  # (N, B)

        c, s = torch.cos(theta), torch.sin(theta)
        R_b = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2
        )  # (N, B, 2, 2)

        if nom_s.dim() == 2:
            R_b = R_b.squeeze(1) # (N, 2, 2)

        return R_b

    @time_it("- mppi command")
    def command(
        self,
        state: Union[torch.Tensor, list, np.ndarray],
        obs_points: Optional[Union[torch.Tensor, list, np.ndarray]] = None,
    ):
        if isinstance(state, list):
            state = torch.tensor(state, dtype=self.dtype, device=self.device)
        elif isinstance(state, np.ndarray):
            state = self.np_to_tensor(state)
        elif isinstance(state, torch.Tensor):
            state = state.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"State must be a list, numpy array, or torch tensor")

        if obs_points is not None:
            if isinstance(obs_points, list):
                obs_points = torch.tensor(
                    obs_points, dtype=self.dtype, device=self.device
                )
            elif isinstance(obs_points, np.ndarray):
                obs_points = self.np_to_tensor(obs_points)
            elif isinstance(obs_points, torch.Tensor):
                obs_points = obs_points.to(dtype=self.dtype, device=self.device)
            else:
                raise ValueError(
                    f"obs_points must be a list, numpy array, or torch tensor"
                )

        self.state = state
        self.obs_points = obs_points

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
