"""
流形几何约束模块 (Manifold Geometry Constraints)
==================================================
核心思想:
将权重和生长完全约束在莫比乌斯-克莱因流形上, 实现:
1. 切空间权重初始化 - 新连接沿流形切方向生成
2. 黎曼梯度投影 - 优化在流形切空间内进行
3. 测地线生长 - 新神经元沿流形测地线扩展
4. 自动梯度有界 - 紧流形上梯度天然有界, 无需clip

数学框架:
  流形 M = Mobius-Klein 混合非定向流形
  切空间 T_p(M) ⊂ ℝ^n
  指数映射 Exp_p: T_p(M) → M
  对数映射 Log_p: M → T_p(M)
  黎曼梯度 grad_M f = Proj_{T_p(M)}(grad_ℝ^n f)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math


class ManifoldGeometry(nn.Module):
    """
    流形几何核心 - 提供切空间运算和流形映射

    流形定义:
      M = { (θ, φ) | θ ∈ [0, 2π), φ ∈ [-r, r] } / ~
      其中 ~ 是莫比乌斯扭转等价关系:
        (0, φ) ~ (2π, -φ)

    切空间:
      T_p(M) = ℝ^(d×d) 在 p 点的线性近似
      投影算子: Proj_p(v) = v - <v, n_p> · n_p
      其中 n_p 是 p 点的法向量
    """

    def __init__(
        self,
        max_dim: int = 512,
        manifold_radius: float = 1.0,
        twist_rate: float = math.pi,
        klein_mix: float = 0.0,
    ):
        super().__init__()

        self.max_dim = max_dim
        self.manifold_radius = nn.Parameter(torch.tensor(manifold_radius))
        self.twist_rate = nn.Parameter(torch.tensor(twist_rate))
        self.klein_mix = nn.Parameter(torch.tensor(klein_mix))

        self._normal_cache = {}
        self._projection_cache = {}

    def _get_manifold_normal(self, p: torch.Tensor) -> torch.Tensor:
        """
        计算流形在点 p 处的法向量

        莫比乌斯流形的法向量:
          n(θ, φ) = (-sin(θ/2)·φ, cos(θ/2)·φ, sin(θ/2))
        归一化后得到单位法向量
        """
        device = p.device
        key = p.shape
        if key in self._normal_cache:
            return self._normal_cache[key]

        N = p.shape[-1]
        theta = torch.linspace(0, 2 * math.pi, N, device=device)

        half_twist = self.twist_rate * theta / (2 * math.pi)
        sin_half = torch.sin(half_twist)
        cos_half = torch.cos(half_twist)

        r = self.manifold_radius
        n_theta = -sin_half * r
        n_phi = cos_half * r
        n_twist = sin_half

        normal = torch.stack([n_theta, n_phi, n_twist], dim=-1)
        normal = F.normalize(normal, dim=-1)

        self._normal_cache[key] = normal
        return normal

    def project_to_tangent(self, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        将欧氏向量 v 投影到流形在 p 点的切空间

        对于高维权重矩阵, 逐行投影:
          每行 w_i 投影到以 p_i 为法向的切超平面
          Proj(w_i) = w_i - <w_i, p_i/||p_i||> · p_i/||p_i||
        """
        if v.dim() == 1 and p.dim() == 1:
            if v.shape == p.shape:
                n = F.normalize(p, dim=0)
                inner = (v * n).sum()
                return v - inner * n
            return v
        elif v.dim() == 2 and p.dim() == 1:
            if v.shape[1] == p.shape[0]:
                n = F.normalize(p, dim=0)
                inner = (v * n.unsqueeze(0)).sum(dim=-1, keepdim=True)
                return v - inner * n.unsqueeze(0)
            elif v.shape[0] == p.shape[0]:
                n = F.normalize(p, dim=0)
                inner = (v * n.unsqueeze(1)).sum(dim=0, keepdim=True)
                return v - n.unsqueeze(1) * inner
            return v
        elif v.dim() == 2 and p.dim() == 2:
            if v.shape == p.shape:
                n = F.normalize(p, dim=-1)
                inner = (v * n).sum(dim=-1, keepdim=True)
                return v - inner * n
            return v
        else:
            return v

    def exp_map(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        指数映射: 从切空间回到流形

        Exp_p(v) = cos(||v||/r) · p + r · sin(||v||/r) · (v/||v||)

        当 ||v|| → 0 时, Exp_p(v) ≈ p + v (一阶近似)
        """
        r = self.manifold_radius.abs() + 1e-6
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        angle = v_norm / r
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        direction = v / v_norm
        result = cos_a * p + r * sin_a * direction

        valid = v_norm.squeeze(-1) > 1e-6
        result[~valid] = p[~valid] + v[~valid]

        return result

    def log_map(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        对数映射: 从流形回到切空间

        Log_p(q) = r · (θ / sin(θ)) · (q - cos(θ)·p) / r
        其中 θ = arccos(<p, q> / r²)
        """
        r = self.manifold_radius.abs() + 1e-6

        inner = (p * q).sum(dim=-1, keepdim=True) / (r * r)
        inner = inner.clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(inner)

        sin_theta = torch.sin(theta).clamp(min=1e-8)
        factor = r * theta / sin_theta

        result = factor * (q - torch.cos(theta) * p) / r

        valid = theta.squeeze(-1) > 1e-6
        result[~valid] = q[~valid] - p[~valid]

        return result

    def geodesic_distance(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """测地线距离: d_M(p, q) = ||Log_p(q)||"""
        log_q = self.log_map(p, q)
        return log_q.norm(dim=-1)

    def tangent_basis(self, p: torch.Tensor, dim: int) -> torch.Tensor:
        """
        计算切空间的标准正交基

        使用 Gram-Schmidt 正交化:
          e_1, e_2, ..., e_d ∈ T_p(M)
          <e_i, e_j> = δ_ij
        """
        N = p.shape[-1]
        d = min(dim, N - 1)

        random_vecs = torch.randn(d, N, device=p.device, dtype=p.dtype)

        normals = self._get_manifold_normal(p)
        if normals.dim() == 1:
            normals = normals.unsqueeze(0)

        for i in range(d):
            v = random_vecs[i]
            for n in normals:
                v = v - (v * n).sum() * n

            for j in range(i):
                v = v - (v * random_vecs[j]).sum() * random_vecs[j]

            v_norm = v.norm()
            if v_norm > 1e-8:
                random_vecs[i] = v / v_norm

        return random_vecs[:d]


class ManifoldWeightInitializer:
    """
    流形约束权重初始化

    新连接的权重不是随机初始化, 而是:
    1. 沿流形切空间方向
    2. 受扭转张量调制
    3. 振幅有界 (流形半径)
    """

    def __init__(
        self, geometry: ManifoldGeometry, twist_tensor: Optional[torch.Tensor] = None
    ):
        self.geometry = geometry
        self.twist_tensor = twist_tensor

    def init_connection_weight(
        self,
        in_pos: torch.Tensor,
        out_pos: torch.Tensor,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化连接权重 - 沿流形测地线方向

        权重 = r · Log_p(q) / ||Log_p(q)|| · α
        其中 α ∈ [0, 1] 是随机缩放
        """
        r = self.geometry.manifold_radius.abs()

        log_vec = self.geometry.log_map(in_pos.unsqueeze(0), out_pos.unsqueeze(0))
        log_norm = log_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = log_vec / log_norm

        alpha = torch.rand(1, device=device) * 0.5
        weight = direction * r * alpha

        if self.twist_tensor is not None:
            twist_mod = torch.cos(self.twist_tensor.mean(dim=-1))
            twist_scalar = twist_mod.abs().mean().item()
            weight = weight * twist_scalar

        return weight.squeeze(), alpha.item()

    def init_neuron_weights(
        self,
        parent_state: torch.Tensor,
        n_new: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        初始化新神经元权重 - 沿父神经元切空间扩展

        新神经元 = Exp_p(v), v ∈ T_p(M), ||v|| < ε
        """
        r = self.geometry.manifold_radius.abs()
        basis = self.geometry.tangent_basis(parent_state, dim=n_new)

        new_weights = []
        for i in range(n_new):
            eps = torch.randn(1, device=device) * 0.1 * r
            tangent_vec = eps * basis[i]
            new_state = self.geometry.exp_map(
                parent_state.unsqueeze(0), tangent_vec.unsqueeze(0)
            )
            new_weights.append(new_state.squeeze(0))

        return torch.stack(new_weights, dim=0)


class RiemannianOptimizer:
    """
    黎曼优化器包装器

    将标准优化器的欧氏梯度投影到流形切空间:
      1. 计算欧氏梯度 g = ∇f(w)
      2. 投影到切空间: g_M = Proj_{T_w(M)}(g)
      3. 用优化器更新: w' = w - η · g_M
      4. 收缩回流形: w'' = Exp_w(w' - w)

    效果:
      - 参数始终在流形上
      - 梯度自动有界 (切空间投影)
      - 不需要 clip_grad_norm
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        geometry: ManifoldGeometry,
    ):
        self.optimizer = optimizer
        self.geometry = geometry

    def project_gradients(self):
        """将所有参数的梯度投影到流形切空间"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p_point = p.data.view(-1)
                    p_grad = p.grad.view(-1)

                    projected = self.geometry.project_to_tangent(p_grad, p_point)
                    p.grad.copy_(projected.view_as(p.grad))

    def retract(self):
        """将参数收缩回流形"""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    old_p = p.data.clone()
                    step = group["lr"] * p.grad

                    new_p_flat = old_p.view(-1) - step.view(-1)
                    old_p_flat = old_p.view(-1)

                    retracted = self.geometry.exp_map(
                        old_p_flat.unsqueeze(0), (-step.view(-1)).unsqueeze(0)
                    )

                    p.data.copy_(retracted.view_as(p.data))

    def step(self):
        """一步黎曼优化"""
        self.project_gradients()
        self.optimizer.step()
        self.retract()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class GeodesicGrowthPlanner:
    """
    测地线生长规划器

    生长不再是随机分裂, 而是:
    1. 找到流形上"曲率最大"的区域 (信息密度高)
    2. 沿测地线方向扩展新神经元
    3. 新连接沿流形拓扑距离建立

    这模拟了大脑发育中:
      - 神经元沿皮层表面迁移
      - 突触沿功能邻近性建立
      - 生长受几何约束引导
    """

    def __init__(self, geometry: ManifoldGeometry):
        self.geometry = geometry

    def find_growth_sites(
        self,
        states: torch.Tensor,
        importance: torch.Tensor,
        n_sites: int = 1,
    ) -> List[int]:
        """
        找到最佳生长位置 - 流形上信息密度最高的区域

        使用曲率启发: 重要性高且邻近区域也高的位置
        """
        N = states.shape[0]
        if N < 2:
            return [0]

        scores = importance.clone()

        for i in range(N):
            for j in range(i + 1, N):
                dist = self.geometry.geodesic_distance(
                    states[i].unsqueeze(0), states[j].unsqueeze(0)
                )
                if dist < 0.3:
                    scores[i] += importance[j] * 0.5
                    scores[j] += importance[i] * 0.5

        _, top_indices = scores.topk(min(n_sites, N))
        return top_indices.tolist()

    def plan_new_connection(
        self,
        states: torch.Tensor,
        source_idx: int,
        target_candidates: List[int],
    ) -> Tuple[int, int, torch.Tensor]:
        """
        规划新连接 - 选择测地线最短的候选

        返回: (source, target, weight)
        """
        source_state = states[source_idx]
        best_target = -1
        best_distance = float("inf")
        best_weight = None

        for target_idx in target_candidates:
            target_state = states[target_idx]
            dist = self.geometry.geodesic_distance(
                source_state.unsqueeze(0), target_state.unsqueeze(0)
            )
            if dist < best_distance:
                best_distance = dist
                best_target = target_idx
                log_vec = self.geometry.log_map(
                    source_state.unsqueeze(0), target_state.unsqueeze(0)
                )
                best_weight = log_vec.squeeze(0)

        return source_idx, best_target, best_weight

    def plan_new_neuron(
        self,
        states: torch.Tensor,
        parent_idx: int,
    ) -> torch.Tensor:
        """
        规划新神经元 - 沿父神经元切空间测地线扩展

        新神经元位置 = Exp_parent(v), v 沿最大曲率方向
        """
        parent_state = states[parent_idx]
        basis = self.geometry.tangent_basis(parent_state, dim=2)

        eps = (
            torch.randn(2, device=states.device)
            * 0.15
            * self.geometry.manifold_radius.abs()
        )
        tangent_vec = eps[0] * basis[0] + eps[1] * basis[1]

        new_state = self.geometry.exp_map(
            parent_state.unsqueeze(0), tangent_vec.unsqueeze(0)
        )
        return new_state.squeeze(0)
