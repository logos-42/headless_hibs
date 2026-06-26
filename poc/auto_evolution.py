"""
Twistor-LMT 自动化进化系统
============================
目标：让模型能够自动测试、自动评估、自动改进

核心组件:
1. 测试基准 (Benchmarks)
2. 性能评估 (Evaluation)
3. 架构搜索 (Architecture Search)
4. 进化算法 (Evolution)
5. 自动迭代 (Auto-Iteration)

架构:
┌─────────────┐
│  初始模型   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  性能测试   │ ← Benchmarks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  性能评估   │ ← Metrics
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  架构变异   │ ← Mutation
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  选择最优   │ ← Selection
└──────┬──────┘
       │
       └──────→ (循环迭代)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import random
from copy import deepcopy


# ============================================================================
# 1. 测试基准 (Benchmarks)
# ============================================================================

@dataclass
class BenchmarkResult:
    """测试结果"""
    task_name: str
    metric_name: str
    metric_value: float
    higher_is_better: bool = True
    
    def is_better_than(self, other: 'BenchmarkResult') -> bool:
        """判断是否比其他结果好"""
        if other is None:
            return True
        if self.higher_is_better:
            return self.metric_value > other.metric_value
        else:
            return self.metric_value < other.metric_value


class BenchmarkSuite:
    """
    测试基准套件
    
    包含多个测试任务，用于评估模型性能
    """
    
    def __init__(self):
        self.tasks = {}
    
    def register_task(self, name: str, task_fn):
        """注册测试任务"""
        self.tasks[name] = task_fn
    
    def run(self, model, device='cpu') -> Dict[str, BenchmarkResult]:
        """运行所有测试"""
        results = {}
        
        for name, task_fn in self.tasks.items():
            result = task_fn(model, device)
            results[name] = result
        
        return results


# ============================================================================
# 2. 测试任务定义
# ============================================================================

def create_sine_forecast_task(seq_len=50, n_samples=100):
    """创建正弦波预测任务"""
    def task(model, device):
        model.eval()
        
        # 生成测试数据
        X, y = [], []
        for _ in range(n_samples):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            t = np.linspace(0, 4*np.pi, seq_len+1)
            signal = np.sin(freq * t + phase) + np.random.randn(len(t)) * 0.1
            
            X.append(signal[:-1].reshape(-1, 1))
            y.append(signal[1:].reshape(-1, 1))
        
        X = torch.FloatTensor(np.stack(X)).to(device)
        y = torch.FloatTensor(np.stack(y)).to(device)
        
        # 测试
        with torch.no_grad():
            x_input = X.transpose(0, 1)
            y_pred = model(x_input)
            
            # 计算 MSE
            mse = nn.functional.mse_loss(y_pred.transpose(0, 1), y).item()
        
        return BenchmarkResult(
            task_name='sine_forecast',
            metric_name='MSE',
            metric_value=mse,
            higher_is_better=False,
        )
    
    return task


def create_lorenz_task(seq_len=50, n_samples=50):
    """创建 Lorenz 吸引子预测任务"""
    def task(model, device):
        model.eval()
        
        # 生成 Lorenz 数据
        X, y = [], []
        for _ in range(n_samples):
            x0 = np.random.uniform(-10, 10, 3)
            trajectory = [x0]
            
            sigma, rho, beta = 10.0, 28.0, 8.0/3.0
            dt = 0.01
            
            for _ in range(seq_len):
                x, y_, z = trajectory[-1]
                dx = sigma * (y_ - x) * dt
                dy = (x * (rho - z) - y_) * dt
                dz = (x * y_ - beta * z) * dt
                trajectory.append([x + dx, y_ + dy, z + dz])
            
            trajectory = np.array(trajectory) + np.random.randn(seq_len + 1, 3) * 0.01
            X.append(trajectory[:-1])
            y.append(trajectory[1:])
        
        X = torch.FloatTensor(np.stack(X)).to(device)
        y = torch.FloatTensor(np.stack(y)).to(device)
        
        # 测试 (需要调整模型输入维度)
        with torch.no_grad():
            x_input = X.transpose(0, 1)
            try:
                y_pred = model(x_input)
                mse = nn.functional.mse_loss(y_pred.transpose(0, 1), y).item()
            except:
                mse = float('inf')
        
        return BenchmarkResult(
            task_name='lorenz_forecast',
            metric_name='MSE',
            metric_value=mse,
            higher_is_better=False,
        )
    
    return task


def create_stability_task(seq_len=100):
    """创建稳定性测试任务"""
    def task(model, device):
        model.eval()
        
        # 测试长序列稳定性
        x = torch.randn(seq_len, 1, 1).to(device)
        
        with torch.no_grad():
            try:
                y = model(x)
                
                # 检查是否有 NaN/Inf
                has_nan = torch.isnan(y).any().item()
                has_inf = torch.isinf(y).any().item()
                
                # 稳定性分数 (0-1)
                stability_score = 1.0 if not (has_nan or has_inf) else 0.0
                
            except Exception as e:
                stability_score = 0.0
        
        return BenchmarkResult(
            task_name='stability',
            metric_name='stability_score',
            metric_value=stability_score,
            higher_is_better=True,
        )
    
    return task


def create_speed_task(seq_len=50, n_runs=10):
    """创建推理速度测试任务"""
    def task(model, device):
        import time
        
        model.eval()
        x = torch.randn(seq_len, 1, 1).to(device)
        
        # 预热
        with torch.no_grad():
            model(x)
        
        # 测试
        times = []
        for _ in range(n_runs):
            start = time.time()
            with torch.no_grad():
                model(x)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        return BenchmarkResult(
            task_name='inference_speed',
            metric_name='avg_time_sec',
            metric_value=avg_time,
            higher_is_better=False,
        )
    
    return task


# ============================================================================
# 3. 架构搜索空间
# ============================================================================

@dataclass
class ArchitectureConfig:
    """架构配置"""
    hidden_dim: int = 256
    n_layers: int = 4
    dt: float = 0.1
    tau_min: float = 0.01
    tau_max: float = 1.0
    sparsity: float = 0.3
    multi_scale_tau: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'dt': self.dt,
            'tau_min': self.tau_min,
            'tau_max': self.tau_max,
            'sparsity': self.sparsity,
            'multi_scale_tau': self.multi_scale_tau,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ArchitectureConfig':
        return cls(**d)


class ArchitectureSearchSpace:
    """
    架构搜索空间
    
    定义可以变异的参数范围
    """
    
    def __init__(self):
        # 参数范围
        self.ranges = {
            'hidden_dim': [64, 128, 256, 512, 1024, 2048],
            'n_layers': [2, 4, 8, 12, 16, 24],
            'dt': [0.01, 0.05, 0.1, 0.2, 0.5],
            'tau_min': [0.001, 0.01, 0.1],
            'tau_max': [0.5, 1.0, 2.0, 5.0],
            'sparsity': [0.0, 0.3, 0.5, 0.7],
            'multi_scale_tau': [True, False],
        }
    
    def sample(self) -> ArchitectureConfig:
        """随机采样一个配置"""
        return ArchitectureConfig(
            hidden_dim=random.choice(self.ranges['hidden_dim']),
            n_layers=random.choice(self.ranges['n_layers']),
            dt=random.choice(self.ranges['dt']),
            tau_min=random.choice(self.ranges['tau_min']),
            tau_max=random.choice(self.ranges['tau_max']),
            sparsity=random.choice(self.ranges['sparsity']),
            multi_scale_tau=random.choice(self.ranges['multi_scale_tau']),
        )
    
    def mutate(self, config: ArchitectureConfig, mutation_rate: float = 0.3) -> ArchitectureConfig:
        """变异配置"""
        new_config = deepcopy(config)
        
        for param_name, values in self.ranges.items():
            if random.random() < mutation_rate:
                current_value = getattr(new_config, param_name)
                
                # 找到当前值在列表中的位置
                if current_value in values:
                    idx = values.index(current_value)
                    
                    # 随机选择相邻值或随机值
                    if random.random() < 0.5 and 0 < idx < len(values) - 1:
                        # 相邻值
                        new_idx = idx + random.choice([-1, 1])
                    else:
                        # 随机值
                        new_idx = random.randint(0, len(values) - 1)
                    
                    setattr(new_config, param_name, values[new_idx])
        
        return new_config


# ============================================================================
# 4. 进化算法
# ============================================================================

@dataclass
class Individual:
    """进化算法中的个体"""
    config: ArchitectureConfig
    fitness: float = 0.0
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)
    generation: int = 0


class EvolutionaryOptimizer:
    """
    进化优化器
    
    使用遗传算法搜索最优架构
    """
    
    def __init__(
        self,
        search_space: ArchitectureSearchSpace,
        population_size: int = 20,
        elite_size: int = 4,
        mutation_rate: float = 0.3,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history = []
    
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            config = self.search_space.sample()
            individual = Individual(
                config=config,
                generation=0,
            )
            self.population.append(individual)
    
    def evaluate_population(self, benchmark_suite: BenchmarkSuite, model_class, device='cpu'):
        """评估种群中所有个体"""
        for individual in self.population:
            # 创建模型
            model = model_class(individual.config)
            model = model.to(device)
            
            # 运行测试
            results = benchmark_suite.run(model, device)
            individual.results = results
            
            # 计算适应度
            fitness = self._compute_fitness(results)
            individual.fitness = fitness
            
            # 更新最优个体
            if self.best_individual is None or fitness > self.best_individual.fitness:
                self.best_individual = individual
        
        self.generation += 1
    
    def _compute_fitness(self, results: Dict[str, BenchmarkResult]) -> float:
        """
        计算适应度分数
        
        综合多个测试任务的结果
        """
        fitness = 0.0
        
        # 定义各任务权重
        weights = {
            'sine_forecast': 1.0,
            'lorenz_forecast': 1.0,
            'stability': 2.0,  # 稳定性更重要
            'inference_speed': 0.5,
        }
        
        for task_name, result in results.items():
            weight = weights.get(task_name, 1.0)
            
            # 归一化分数 (0-1)
            if task_name == 'stability':
                score = result.metric_value
            elif task_name == 'inference_speed':
                # 速度越快分数越高 (假设 0.01s 是满分)
                score = max(0, 1.0 - result.metric_value / 0.1)
            else:
                # MSE 越低分数越高 (假设 0.01 是满分)
                score = max(0, 1.0 - result.metric_value / 1.0)
            
            fitness += weight * score
        
        return fitness
    
    def selection(self) -> List[Individual]:
        """选择精英"""
        # 按适应度排序
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # 选择精英
        elites = sorted_pop[:self.elite_size]
        
        return elites
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """交叉"""
        # 简单交叉：随机选择一个父本的配置
        if random.random() < 0.5:
            child_config = deepcopy(parent1.config)
        else:
            child_config = deepcopy(parent2.config)
        
        return Individual(
            config=child_config,
            generation=self.generation,
        )
    
    def evolve(self, benchmark_suite: BenchmarkSuite, model_class, n_generations: int = 10, device='cpu'):
        """进化循环"""
        print(f"开始进化优化...")
        print(f"  种群大小：{self.population_size}")
        print(f"  进化代数：{n_generations}")
        print()
        
        # 初始化
        self.initialize_population()
        
        for gen in range(n_generations):
            print(f"Generation {gen + 1}/{n_generations}")
            
            # 评估
            self.evaluate_population(benchmark_suite, model_class, device)
            
            # 记录历史
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            best_fitness = self.best_individual.fitness
            self.history.append({
                'generation': gen,
                'avg_fitness': avg_fitness,
                'best_fitness': best_fitness,
                'best_config': self.best_individual.config.to_dict(),
            })
            
            print(f"  平均适应度：{avg_fitness:.4f}")
            print(f"  最佳适应度：{best_fitness:.4f}")
            print(f"  最佳配置：hidden_dim={self.best_individual.config.hidden_dim}, "
                  f"n_layers={self.best_individual.config.n_layers}")
            
            # 选择
            elites = self.selection()
            
            # 生成新一代
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # 选择父本 (锦标赛选择)
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                
                # 交叉
                child = self.crossover(parent1, parent2)
                
                # 变异
                child.config = self.search_space.mutate(child.config, self.mutation_rate)
                
                new_population.append(child)
            
            self.population = new_population
        
        print()
        print(f"进化完成！最佳适应度：{self.best_individual.fitness:.4f}")
        
        return self.best_individual


# ============================================================================
# 5. 自动化进化系统
# ============================================================================

class AutoEvolutionSystem:
    """
    自动化进化系统
    
    整合所有组件，实现完整的自动化进化流程
    """
    
    def __init__(self, model_class):
        self.model_class = model_class
        
        # 创建测试套件
        self.benchmark_suite = BenchmarkSuite()
        self._register_benchmarks()
        
        # 创建搜索空间
        self.search_space = ArchitectureSearchSpace()
        
        # 创建进化优化器
        self.optimizer = EvolutionaryOptimizer(
            self.search_space,
            population_size=10,
            elite_size=3,
            mutation_rate=0.3,
        )
        
        # 进化历史
        self.evolution_history = []
    
    def _register_benchmarks(self):
        """注册测试任务"""
        self.benchmark_suite.register_task('sine_forecast', create_sine_forecast_task())
        self.benchmark_suite.register_task('stability', create_stability_task())
        self.benchmark_suite.register_task('inference_speed', create_speed_task())
    
    def run_evolution(self, n_generations: int = 5, device='cpu') -> Individual:
        """运行进化"""
        best_individual = self.optimizer.evolve(
            self.benchmark_suite,
            self.model_class,
            n_generations=n_generations,
            device=device,
        )
        
        self.evolution_history.append({
            'best_config': best_individual.config.to_dict(),
            'best_fitness': best_individual.fitness,
            'results': {k: v.metric_value for k, v in best_individual.results.items()},
        })
        
        return best_individual
    
    def save_history(self, path: str):
        """保存进化历史"""
        with open(path, 'w') as f:
            json.dump({
                'evolution_history': self.evolution_history,
                'optimizer_history': self.optimizer.history,
            }, f, indent=2)
        
        print(f"进化历史已保存到：{path}")
    
    def get_best_config(self) -> ArchitectureConfig:
        """获取最佳配置"""
        if self.optimizer.best_individual:
            return self.optimizer.best_individual.config
        return None


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Twistor-LMT 自动化进化系统测试")
    print("=" * 70)
    
    # 导入 TwistorLMT
    from poc.twistor_100m_config import StackedTwistorLMT, TwistorLMTConfig
    
    # 包装模型类以适配进化系统
    class ModelWrapper(nn.Module):
        def __init__(self, config: ArchitectureConfig):
            super().__init__()
            from poc.twistor_100m_config import StackedTwistorLMT, TwistorLMTConfig
            
            twistor_config = TwistorLMTConfig(
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                dt=config.dt,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                sparsity=config.sparsity,
                multi_scale_tau=config.multi_scale_tau,
            )
            
            self.model = StackedTwistorLMT(twistor_config)
        
        def forward(self, x):
            return self.model(x)
    
    # 创建进化系统
    system = AutoEvolutionSystem(ModelWrapper)
    
    # 运行进化 (小规模测试)
    print("\n运行进化优化 (测试模式)...")
    best = system.run_evolution(n_generations=2, device='cpu')
    
    # 输出最佳配置
    print("\n" + "=" * 70)
    print("最佳架构配置:")
    print(f"  hidden_dim: {best.config.hidden_dim}")
    print(f"  n_layers: {best.config.n_layers}")
    print(f"  dt: {best.config.dt}")
    print(f"  sparsity: {best.config.sparsity}")
    print(f"  适应度分数：{best.fitness:.4f}")
    print("=" * 70)
    
    # 保存历史
    system.save_history('evolution_history.json')
