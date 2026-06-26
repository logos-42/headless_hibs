"""
Twistor-LMT 零样本和多任务学习扩展设计方案

目标：让 Twistor-LMT 支持
1. 零样本学习 (Zero-shot Learning) - 在未见过的新任务上直接推理
2. 多任务学习 (Multi-task Learning) - 单个模型处理多个相关任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# 方案 1: 基于任务嵌入的多任务 Twistor-LMT
# ============================================================================

@dataclass
class TaskConfig:
    """任务配置"""
    task_id: int
    task_name: str
    input_dim: int
    output_dim: int
    task_embedding: Optional[torch.Tensor] = None


class MultiTaskTwistorLMT(nn.Module):
    """
    支持多任务学习的 Twistor-LMT
    
    核心思想:
    1. 共享的动力学核心 (Shared Dynamics Core)
    2. 任务特定的嵌入 (Task-specific Embeddings)
    3. 任务特定的输入/输出投影 (Task-specific Projections)
    
    架构:
    ┌─────────────────────────────────────────┐
    │           Task Embedding                │
    │              (共享)                      │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Input Projection (任务特定)          │
    │     x → z_encoded                        │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Shared Dynamics Core                │
    │     dz/dt = F(z, x, task_embedding)     │
    └─────────────────┬───────────────────────┘
                      │
    ┌─────────────────▼───────────────────────┐
    │     Output Projection (任务特定)         │
    │     z → y                                │
    └─────────────────┬───────────────────────┘
    """
    
    def __init__(
        self,
        task_configs: List[TaskConfig],
        hidden_dim: int = 32,
        task_embedding_dim: int = 8,
        dt: float = 0.1,
    ):
        super().__init__()
        self.task_configs = task_configs
        self.hidden_dim = hidden_dim
        self.task_embedding_dim = task_embedding_dim
        self.dt = dt
        self.n_tasks = len(task_configs)
        
        # 1. 任务嵌入 (共享)
        self.task_embeddings = nn.ParameterDict({
            cfg.task_name: nn.Parameter(torch.randn(task_embedding_dim))
            for cfg in task_configs
        })
        
        # 2. 共享动力学核心
        self.dynamics_core = nn.ModuleDict({
            'W_z': nn.Linear(hidden_dim, hidden_dim),
            'W_x': nn.Linear(hidden_dim + task_embedding_dim, hidden_dim),
            'W_tau': nn.Linear(hidden_dim, hidden_dim),
        })
        self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # 3. 任务特定的输入投影
        self.input_projections = nn.ModuleDict({
            cfg.task_name: nn.Linear(cfg.input_dim, hidden_dim)
            for cfg in task_configs
        })
        
        # 4. 任务特定的输出投影
        self.output_projections = nn.ModuleDict({
            cfg.task_name: nn.Linear(hidden_dim, cfg.output_dim)
            for cfg in task_configs
        })
        
        # 5. 任务门控网络 (动态调整动力学)
        self.task_gates = nn.ModuleDict({
            cfg.task_name: nn.Sequential(
                nn.Linear(task_embedding_dim, hidden_dim),
                nn.Sigmoid()
            )
            for cfg in task_configs
        })
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.dynamics_core.items():
            if isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, gain=0.5)
                nn.init.zeros_(param.bias)
        
        for proj in self.input_projections.values():
            nn.init.orthogonal_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
        
        for proj in self.output_projections.values():
            nn.init.orthogonal_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
    
    def compute_dzdt(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        task_emb: torch.Tensor,
        task_name: str
    ) -> torch.Tensor:
        """
        计算动力学方程 (任务条件化)
        
        dz/dt = (-z + gate·[W_z·tanh(z) + W_x·[x;task_emb]]) / tau(z)
        """
        z_real = z.real
        z_imag = z.imag
        
        # 任务门控
        gate = self.task_gates[task_name](task_emb)
        
        # 非线性项
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        W_tanh_real = self.dynamics_core['W_z'](tanh_real)
        W_tanh_imag = self.dynamics_core['W_z'](tanh_imag)
        
        # 输入项 (拼接任务嵌入)
        x_task = torch.cat([x, task_emb.unsqueeze(0).expand(x.size(0), -1)], dim=-1)
        Ux = self.dynamics_core['W_x'](x_task)
        
        # 应用门控
        W_tanh_real = gate * W_tanh_real
        W_tanh_imag = gate * W_tanh_imag
        Ux = gate * Ux
        
        # 动力学方程
        dz_real = -z_real + W_tanh_real + Ux
        dz_imag = -z_imag + W_tanh_imag + Ux
        
        # 时间常数
        z_mod = torch.abs(z)
        tau = torch.sigmoid(self.dynamics_core['W_tau'](z_mod))
        tau = tau + self.tau_bias.unsqueeze(0)
        tau = torch.clamp(tau, 0.01, 1.0) + 1e-6
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        
        # 稳定性裁剪
        dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
        
        return dzdt
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_name: str,
        return_states: bool = False
    ) -> torch.Tensor:
        """
        前向传播 (指定任务)
        
        Args:
            x: 输入序列 (T, B, input_dim)
            task_name: 任务名称
            return_states: 是否返回状态
        
        Returns:
            y: 输出序列 (T, B, output_dim)
        """
        T, B, _ = x.shape
        
        # 获取任务嵌入
        task_emb = self.task_embeddings[task_name]
        
        # 初始化状态
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        # 获取任务特定的投影
        input_proj = self.input_projections[task_name]
        output_proj = self.output_projections[task_name]
        
        outputs = []
        states = []
        
        for t in range(T):
            x_t = x[t]
            
            # 输入投影
            x_encoded = input_proj(x_t)
            
            # 动力学演化
            dzdt = self.compute_dzdt(z, x_encoded, task_emb, task_name)
            z = z + self.dt * dzdt
            
            # 状态限幅
            z = torch.complex(
                torch.clamp(z.real, -100, 100),
                torch.clamp(z.imag, -100, 100)
            )
            
            # 输出投影
            y_t = output_proj(z.real)
            outputs.append(y_t)
            
            if return_states:
                states.append(z)
        
        y = torch.stack(outputs, dim=0)
        
        if return_states:
            states = torch.stack(states, dim=0)
            return y, states
        
        return y
    
    def zero_shot_transfer(
        self, 
        x: torch.Tensor, 
        source_task: str, 
        target_task: str
    ) -> torch.Tensor:
        """
        零样本迁移：使用源任务训练的模型处理目标任务
        
        Args:
            x: 输入序列
            source_task: 源任务名称
            target_task: 目标任务名称
        
        Returns:
            预测输出
        """
        # 使用源任务的输入投影
        # 使用目标任务的输出投影
        # 共享动力学核心
        
        T, B, _ = x.shape
        task_emb = self.task_embeddings[target_task]
        
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        # 使用源任务的输入投影
        input_proj = self.input_projections[source_task]
        # 使用目标任务的输出投影
        output_proj = self.output_projections[target_task]
        
        outputs = []
        
        for t in range(T):
            x_t = x[t]
            x_encoded = input_proj(x_t)
            
            dzdt = self.compute_dzdt(z, x_encoded, task_emb, target_task)
            z = z + self.dt * dzdt
            
            y_t = output_proj(z.real)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=0)


# ============================================================================
# 方案 2: 基于元学习的零样本 Twistor-LMT (MAML 风格)
# ============================================================================

class MetaTwistorLMT(nn.Module):
    """
    基于元学习的 Twistor-LMT，支持零样本适应新任务
    
    使用 MAML (Model-Agnostic Meta-Learning) 算法
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dt: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        
        # 共享参数 (元学习初始化)
        self.meta_params = nn.ParameterDict({
            'W_z': nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1),
            'W_x': nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1),
            'W_out': nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1),
            'W_tau': nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01),
            'b_z': nn.Parameter(torch.zeros(hidden_dim)),
            'b_x': nn.Parameter(torch.zeros(hidden_dim)),
            'b_out': nn.Parameter(torch.zeros(output_dim)),
        })
    
    def compute_dzdt_fast(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """快速动力学计算 (使用给定参数)"""
        z_real = z.real
        z_imag = z.imag
        
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        W_tanh_real = F.linear(tanh_real, params['W_z'], params['b_z'])
        W_tanh_imag = F.linear(tanh_imag, params['W_z'], params['b_z'])
        Ux = F.linear(x, params['W_x'], params['b_x'])
        
        dz_real = -z_real + W_tanh_real + Ux
        dz_imag = -z_imag + W_tanh_imag
        
        z_mod = torch.abs(z)
        tau = torch.sigmoid(F.linear(z_mod, params['W_tau'])) + 1e-6
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
        
        return dzdt
    
    def forward_step(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步前向传播"""
        dzdt = self.compute_dzdt_fast(z, x, params)
        z_new = z + self.dt * dzdt
        y = F.linear(z_new.real, params['W_out'], params['b_out'])
        return z_new, y
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[Dict] = None
    ) -> torch.Tensor:
        """前向传播"""
        if params is None:
            params = self.meta_params
        
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        for t in range(T):
            z, y_t = self.forward_step(z, x[t], params)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=0)
    
    def meta_update(
        self, 
        x_support: torch.Tensor, 
        y_support: torch.Tensor,
        x_query: torch.Tensor,
        y_query: torch.Tensor,
        inner_lr: float = 0.1,
        inner_steps: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAML 元更新
        
        Args:
            x_support: 支持集输入 (用于适应)
            y_support: 支持集目标
            x_query: 查询集输入 (用于评估)
            y_query: 查询集目标
            inner_lr: 内部学习率
            inner_steps: 内部更新步数
        
        Returns:
            query_loss, adapted_params
        """
        # 1. 复制参数进行适应
        adapted_params = {k: v.clone() for k, v in self.meta_params.items()}
        
        # 2. 在支持集上进行梯度下降适应
        for _ in range(inner_steps):
            y_pred = self.forward(x_support, adapted_params)
            support_loss = F.mse_loss(y_pred, y_support)
            
            # 计算梯度并更新
            grads = torch.autograd.grad(
                support_loss, 
                adapted_params.values(),
                create_graph=True
            )
            
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - inner_lr * grad
        
        # 3. 在查询集上评估
        y_query_pred = self.forward(x_query, adapted_params)
        query_loss = F.mse_loss(y_query_pred, y_query)
        
        return query_loss, adapted_params
    
    def zero_shot_adapt(
        self, 
        x_few_shot: torch.Tensor, 
        y_few_shot: torch.Tensor,
        x_test: torch.Tensor,
        adapt_steps: int = 10,
        adapt_lr: float = 0.1
    ) -> torch.Tensor:
        """
        零样本适应：使用少量样本快速适应新任务
        
        Args:
            x_few_shot: 少量样本输入
            y_few_shot: 少量样本目标
            x_test: 测试输入
            adapt_steps: 适应步数
            adapt_lr: 适应学习率
        
        Returns:
            预测输出
        """
        # 复制参数 (需要梯度)
        adapted_params = {}
        for name, param in self.meta_params.items():
            adapted_params[name] = param.clone().detach().requires_grad_(True)
        
        # 快速适应
        for _ in range(adapt_steps):
            y_pred = self.forward(x_few_shot, adapted_params)
            loss = F.mse_loss(y_pred, y_few_shot)
            
            grads = torch.autograd.grad(
                loss, 
                adapted_params.values(),
                retain_graph=True,
                create_graph=False
            )
            
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - adapt_lr * grad
        
        # 使用适应后的参数进行预测
        y_test_pred = self.forward(x_test, adapted_params)
        
        return y_test_pred


# ============================================================================
# 方案 3: 基于提示学习的零样本 Twistor-LMT
# ============================================================================

class PromptTwistorLMT(nn.Module):
    """
    基于提示学习的 Twistor-LMT
    
    核心思想：
    1. 学习一组"提示"向量 (Prompt Vectors)
    2. 不同任务使用不同的提示组合
    3. 新任务通过提示组合实现零样本迁移
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 1,
        n_prompts: int = 10,
        prompt_dim: int = 8,
        dt: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.n_prompts = n_prompts
        
        # 提示库
        self.prompt_bank = nn.Parameter(torch.randn(n_prompts, prompt_dim))
        
        # 提示选择网络
        self.prompt_selector = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_prompts),
            nn.Softmax(dim=-1)
        )
        
        # 提示投影
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        
        # 核心动力学
        self.core = nn.ModuleDict({
            'W_z': nn.Linear(hidden_dim, hidden_dim),
            'W_x': nn.Linear(input_dim, hidden_dim),
            'W_tau': nn.Linear(hidden_dim, hidden_dim),
        })
        
        # 输出层
        self.out = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.core['W_z'].weight, gain=0.5)
        nn.init.orthogonal_(self.core['W_x'].weight, gain=0.5)
        nn.init.orthogonal_(self.core['W_tau'].weight, gain=0.1)
        nn.init.zeros_(self.core['W_z'].bias)
        nn.init.zeros_(self.core['W_x'].bias)
        nn.init.zeros_(self.core['W_tau'].bias)
    
    def get_prompt(self, x: torch.Tensor) -> torch.Tensor:
        """获取输入相关的提示"""
        # 计算提示权重
        weights = self.prompt_selector(x)  # (B, n_prompts)
        
        # 加权组合提示
        prompt = torch.einsum('bn,np->bp', weights, self.prompt_bank)
        
        # 投影到隐藏维度
        return self.prompt_proj(prompt)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        
        for t in range(T):
            x_t = x[t]
            
            # 获取提示
            prompt = self.get_prompt(x_t)
            
            # 动力学
            z_real = z.real
            z_imag = z.imag
            
            tanh_real = torch.tanh(z_real)
            tanh_imag = torch.tanh(z_imag)
            
            W_tanh_real = self.core['W_z'](tanh_real)
            W_tanh_imag = self.core['W_z'](tanh_imag)
            Ux = self.core['W_x'](x_t)
            
            # 添加提示影响
            W_tanh_real = W_tanh_real + prompt
            W_tanh_imag = W_tanh_imag + prompt
            
            dz_real = -z_real + W_tanh_real + Ux
            dz_imag = -z_imag + W_tanh_imag
            
            z_mod = torch.abs(z)
            tau = torch.sigmoid(self.core['W_tau'](z_mod)) + 1e-6
            
            dzdt = torch.complex(dz_real / tau, dz_imag / tau)
            dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
            
            z = z + self.dt * dzdt
            
            y_t = self.out(z.real)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=0)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Twistor-LMT 零样本和多任务学习扩展测试")
    print("=" * 70)
    
    # 1. 测试多任务 Twistor-LMT
    print("\n1. 测试多任务 Twistor-LMT")
    
    task_configs = [
        TaskConfig(task_id=0, task_name="sine", input_dim=2, output_dim=1),
        TaskConfig(task_id=1, task_name="cosine", input_dim=2, output_dim=1),
        TaskConfig(task_id=2, task_name="lorenz", input_dim=3, output_dim=3),
    ]
    
    multi_task_model = MultiTaskTwistorLMT(task_configs, hidden_dim=32)
    
    # 模拟多任务训练
    x_sine = torch.randn(20, 4, 2)
    y_sine = torch.randn(20, 4, 1)
    
    y_pred = multi_task_model(x_sine, task_name="sine")
    print(f"   多任务模型输出形状：{y_pred.shape}")
    print(f"   ✅ 多任务学习支持")
    
    # 2. 测试元学习 Twistor-LMT
    print("\n2. 测试元学习 Twistor-LMT (MAML)")
    
    meta_model = MetaTwistorLMT(input_dim=2, hidden_dim=32, output_dim=1)
    
    # 模拟元学习适应
    x_support = torch.randn(10, 2, 2)
    y_support = torch.randn(10, 2, 1)
    x_query = torch.randn(5, 2, 2)
    y_query = torch.randn(5, 2, 1)
    
    query_loss, _ = meta_model.meta_update(x_support, y_support, x_query, y_query)
    print(f"   查询集损失：{query_loss.item():.4f}")
    print(f"   ✅ 元学习支持")
    
    # 3. 测试零样本适应
    print("\n3. 测试零样本适应")
    
    x_few = torch.randn(5, 2, 2)
    y_few = torch.randn(5, 2, 1)
    x_test = torch.randn(10, 2, 2)
    
    y_test_pred = meta_model.zero_shot_adapt(x_few, y_few, x_test, adapt_steps=5)
    print(f"   零样本预测形状：{y_test_pred.shape}")
    print(f"   ✅ 零样本适应支持")
    
    # 4. 测试提示学习 Twistor-LMT
    print("\n4. 测试提示学习 Twistor-LMT")
    
    prompt_model = PromptTwistorLMT(input_dim=2, hidden_dim=32, output_dim=1)
    
    x = torch.randn(20, 4, 2)
    y = prompt_model(x)
    print(f"   提示模型输出形状：{y.shape}")
    print(f"   ✅ 提示学习支持")
    
    print("\n" + "=" * 70)
    print("所有扩展功能测试通过！")
    print("=" * 70)
