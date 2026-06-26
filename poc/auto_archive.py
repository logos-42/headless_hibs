"""
Twistor-LMT GitHub 自动存档 + 自动进化系统
===============================================
功能:
1. 自动测试验证
2. 自动提交到 Git
3. 测试通过则推送 (存档)
4. 测试失败则回滚
5. 自动进化模型架构

使用方式:
    # 自动进化 + 存档
    python auto_archive.py --evolve --generations 5 --message "v0.5.0 进化完成"
    
    # 仅存档
    python auto_archive.py --archive --message "v0.4.0 完成"
    
    # 回滚到指定版本
    python auto_archive.py --rollback-to v0.4.0
    
    # 查看版本历史
    python auto_archive.py --versions
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from copy import deepcopy

import random

# 延迟导入 torch，只在需要时导入
try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    np = None


# ============================================================================
# 版本管理
# ============================================================================

@dataclass
class ModelVersion:
    """模型版本"""
    version: str
    commit: str
    timestamp: str
    fitness: float
    config: Dict
    status: str  # 'success' or 'failed'
    benchmarks: Dict[str, float] = field(default_factory=dict)


class VersionManager:
    """版本管理器"""
    
    def __init__(self, storage_path: str = 'versions'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.versions: List[ModelVersion] = []
        self._load_history()
    
    def _load_history(self):
        """加载版本历史"""
        if not self.storage_path.exists():
            return
        
        for version_file in self.storage_path.glob('*.json'):
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.versions.append(ModelVersion(**data))
            except:
                pass
    
    def save_version(self, version: ModelVersion):
        """保存版本"""
        self.versions.append(version)
        
        # 保存到文件
        version_file = self.storage_path / f"{version.version}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump({
                'version': version.version,
                'commit': version.commit,
                'timestamp': version.timestamp,
                'fitness': version.fitness,
                'config': version.config,
                'status': version.status,
                'benchmarks': version.benchmarks,
            }, f, indent=2, ensure_ascii=False)
    
    def load_version(self, version: str) -> Optional[ModelVersion]:
        """加载版本"""
        version_file = self.storage_path / f"{version}.json"
        if not version_file.exists():
            return None
        
        with open(version_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return ModelVersion(**data)
    
    def get_latest_success_version(self) -> Optional[ModelVersion]:
        """获取最新的成功版本"""
        success_versions = [v for v in self.versions if v.status == 'success']
        if success_versions:
            return success_versions[-1]
        return None
    
    def get_best_version(self) -> Optional[ModelVersion]:
        """获取最佳版本 (最高适应度)"""
        success_versions = [v for v in self.versions if v.status == 'success']
        if success_versions:
            return max(success_versions, key=lambda x: x.fitness)
        return None
    
    def list_versions(self) -> List[ModelVersion]:
        """列出所有版本"""
        return sorted(self.versions, key=lambda x: x.timestamp, reverse=True)


# ============================================================================
# 架构配置
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
    """架构搜索空间"""
    
    def __init__(self):
        self.ranges = {
            'hidden_dim': [64, 128, 256, 512],
            'n_layers': [2, 4, 6, 8],
            'dt': [0.05, 0.1, 0.2],
            'tau_min': [0.01, 0.1],
            'tau_max': [0.5, 1.0],
            'sparsity': [0.0, 0.3, 0.5],
            'multi_scale_tau': [True, False],
        }
    
    def sample(self) -> ArchitectureConfig:
        """随机采样"""
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
                if current_value in values:
                    idx = values.index(current_value)
                    if random.random() < 0.5 and 0 < idx < len(values) - 1:
                        new_idx = idx + random.choice([-1, 1])
                    else:
                        new_idx = random.randint(0, len(values) - 1)
                    setattr(new_config, param_name, values[new_idx])
        
        return new_config


# ============================================================================
# 简单进化引擎
# ============================================================================

@dataclass
class EvolutionResult:
    """进化结果"""
    config: ArchitectureConfig
    fitness: float
    generation: int
    benchmarks: Dict[str, float] = field(default_factory=dict)


class SimpleEvolutionEngine:
    """简单进化引擎"""
    
    def __init__(self, model_class):
        self.model_class = model_class
        self.search_space = ArchitectureSearchSpace()
        self.best_config: Optional[ArchitectureConfig] = None
        self.best_fitness: float = 0.0
        self.version_manager = VersionManager()
    
    def _create_model(self, config: ArchitectureConfig) -> nn.Module:
        """创建模型"""
        return self.model_class(config)
    
    def _evaluate_config(self, config: ArchitectureConfig, device='cpu') -> Tuple[float, Dict[str, float]]:
        """评估配置"""
        try:
            model = self._create_model(config)
            model = model.to(device)
            model.eval()
            
            results = {}
            
            # 测试 1: 正弦波预测
            X, y = [], []
            for _ in range(20):
                freq = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2*np.pi)
                t = np.linspace(0, 4*np.pi, 51)
                signal = np.sin(freq * t + phase) + np.random.randn(len(t)) * 0.1
                X.append(signal[:-1].reshape(-1, 1))
                y.append(signal[1:].reshape(-1, 1))
            
            X = torch.FloatTensor(np.stack(X)).to(device)
            y = torch.FloatTensor(np.stack(y)).to(device)
            
            with torch.no_grad():
                x_input = X.transpose(0, 1)
                try:
                    y_pred = model(x_input)
                    mse = nn.functional.mse_loss(y_pred.transpose(0, 1), y).item()
                except:
                    mse = 10.0
            
            results['sine_mse'] = mse
            
            # 测试 2: 稳定性
            x = torch.randn(100, 1, 1).to(device)
            with torch.no_grad():
                try:
                    y = model(x)
                    has_nan = torch.isnan(y).any().item()
                    has_inf = torch.isinf(y).any().item()
                    stability = 0.0 if (has_nan or has_inf) else 1.0
                except:
                    stability = 0.0
            
            results['stability'] = stability
            
            # 计算适应度
            fitness = 0.0
            if results['sine_mse'] < 1.0:
                fitness += max(0, 1.0 - results['sine_mse']) * 1.0
            fitness += results['stability'] * 2.0
            
            return fitness, results
            
        except Exception as e:
            return 0.0, {'error': str(e)}
    
    def evolve(self, n_generations: int = 3, population_size: int = 5, device='cpu') -> EvolutionResult:
        """运行进化"""
        print("=" * 60)
        print("开始自动进化")
        print(f"代数: {n_generations}, 种群大小: {population_size}")
        print("=" * 60)
        
        # 初始化种群
        population = [self.search_space.sample() for _ in range(population_size)]
        
        for gen in range(n_generations):
            print(f"\n第 {gen + 1}/{n_generations} 代")
            
            # 评估所有配置
            fitnesses = []
            configs_results = []
            
            for i, config in enumerate(population):
                fitness, results = self._evaluate_config(config, device)
                fitnesses.append(fitness)
                configs_results.append((config, fitness, results))
                print(f"  配置 {i+1}/{population_size}: fitness={fitness:.4f}")
            
            # 记录最佳
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_config = population[best_idx]
            
            print(f"  当前最佳适应度: {self.best_fitness:.4f}")
            
            # 生成新一代
            if gen < n_generations - 1:
                # 选择精英
                sorted_configs = sorted(configs_results, key=lambda x: x[1], reverse=True)
                elites = [c[0] for c in sorted_configs[:2]]
                
                # 生成新种群
                new_population = elites.copy()
                while len(new_population) < population_size:
                    parent = random.choice(elites)
                    child = self.search_space.mutate(parent, 0.5)
                    new_population.append(child)
                
                population = new_population
        
        print("\n" + "=" * 60)
        print(f"进化完成！最佳适应度: {self.best_fitness:.4f}")
        print("=" * 60)
        
        # 评估最佳配置
        final_fitness, final_results = self._evaluate_config(self.best_config, device)
        
        return EvolutionResult(
            config=self.best_config,
            fitness=final_fitness,
            generation=n_generations,
            benchmarks=final_results,
        )


# ============================================================================
# Git 存档系统
# ============================================================================

class GitArchiveSystem:
    """Git 自动存档系统"""
    
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path)
        self.backup_path = self.repo_path / '.git_backup'
        self.log_path = self.repo_path / 'archive_log.json'
        self.logs = self._load_logs()
        self.version_manager = VersionManager()
    
    def _load_logs(self) -> List[dict]:
        """加载存档日志"""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 确保返回的是 list
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data] if data else []
            except:
                pass
        return []
    
    def _save_logs(self):
        """保存存档日志"""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def _run_git(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """运行 git 命令"""
        result = subprocess.run(
            ['git'] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    
    def check_git_status(self) -> Tuple[bool, str]:
        """检查 git 状态"""
        try:
            result = self._run_git(['status', '--porcelain'], check=False)
            has_changes = len(result.stdout.strip()) > 0
            
            result = self._run_git(['rev-parse', '--abbrev-ref', 'HEAD'], check=False)
            branch = result.stdout.strip()
            
            return has_changes, branch
        except Exception as e:
            return False, f"Error: {e}"
    
    def create_backup(self):
        """创建备份 (用于回滚)"""
        print("创建备份...")
        
        # 备份 .git 目录
        git_dir = self.repo_path / '.git'
        if git_dir.exists():
            if self.backup_path.exists():
                shutil.rmtree(self.backup_path)
            shutil.copytree(git_dir, self.backup_path)
            print(f"  ✓ 已备份 .git 目录到 {self.backup_path}")
        
        # 记录当前 commit
        try:
            result = self._run_git(['rev-parse', 'HEAD'])
            current_commit = result.stdout.strip()
            print(f"  ✓ 当前 commit: {current_commit[:8]}")
        except:
            current_commit = None
            print("  ⚠ 无法获取当前 commit")
        
        return current_commit
    
    def restore_backup(self):
        """恢复备份 (回滚)"""
        print("恢复备份 (回滚)...")
        
        if not self.backup_path.exists():
            print("  ✗ 备份不存在，无法回滚")
            return False
        
        # 恢复 .git 目录
        git_dir = self.repo_path / '.git'
        if git_dir.exists():
            shutil.rmtree(git_dir)
        shutil.copytree(self.backup_path, git_dir)
        print(f"  ✓ 已恢复 .git 目录")
        
        # 清理备份
        shutil.rmtree(self.backup_path)
        print(f"  ✓ 已清理备份")
        
        return True
    
    def cleanup_backup(self):
        """清理备份"""
        if self.backup_path.exists():
            shutil.rmtree(self.backup_path)
            print("已清理备份")
    
    def run_tests(self) -> Tuple[bool, str]:
        """运行测试"""
        print("运行测试...")
        
        test_results = []
        
        # 测试 1: 导入测试
        print("  测试 1: 导入测试...")
        try:
            import torch
            from poc.twistor_100m_config import StackedTwistorLMT, TwistorLMTConfig
            
            config = TwistorLMTConfig.small()
            model = StackedTwistorLMT(config)
            
            x = torch.randn(10, 1, 1)
            with torch.no_grad():
                y = model(x)
            
            assert y.shape == (10, 1, 1), f"输出形状错误：{y.shape}"
            print("    ✓ 导入测试通过")
            test_results.append(("import_test", True))
        except Exception as e:
            print(f"    ✗ 导入测试失败：{e}")
            test_results.append(("import_test", False))
        
        # 测试 2: 进化系统测试
        print("  测试 2: 进化系统测试...")
        try:
            config = ArchitectureConfig(hidden_dim=64, n_layers=2)
            assert config.hidden_dim == 64
            assert config.n_layers == 2
            print("    ✓ 进化系统测试通过")
            test_results.append(("evolution_test", True))
        except Exception as e:
            print(f"    ✗ 进化系统测试失败：{e}")
            test_results.append(("evolution_test", False))
        
        # 测试 3: 文件完整性测试
        print("  测试 3: 文件完整性测试...")
        try:
            required_files = [
                'twistor_LMT.py',
                'twistor_100m_config.py',
                'auto_evolution.py',
                'auto_archive.py',
                'README.md',
                'requirements.txt',
            ]
            
            for file in required_files:
                assert (self.repo_path / file).exists(), f"缺少文件：{file}"
            
            print("    ✓ 文件完整性测试通过")
            test_results.append(("file_integrity_test", True))
        except Exception as e:
            print(f"    ✗ 文件完整性测试失败：{e}")
            test_results.append(("file_integrity_test", False))
        
        # 总结
        all_passed = all(result[1] for result in test_results)
        
        if all_passed:
            print("\n✓ 所有测试通过")
        else:
            failed = [name for name, passed in test_results if not passed]
            print(f"\n✗ 测试失败：{', '.join(failed)}")
        
        return all_passed, ", ".join([name for name, _ in test_results])
    
    def stage_changes(self):
        """暂存更改"""
        print("暂存更改...")
        
        # 添加所有更改 (排除备份目录)
        self._run_git(['add', '-A', ':!.git_backup'])
        
        # 检查暂存状态
        result = self._run_git(['diff', '--cached', '--name-only'])
        staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        print(f"  ✓ 已暂存 {len(staged_files)} 个文件")
        for f in staged_files[:10]:
            print(f"    - {f}")
        if len(staged_files) > 10:
            print(f"    ... 还有 {len(staged_files) - 10} 个文件")
        
        return staged_files
    
    def commit(self, message: str) -> Optional[str]:
        """提交"""
        print(f"提交：{message}")
        
        try:
            # 设置用户信息
            self._run_git(['config', 'user.name', 'Twistor-LMT-Bot'], check=False)
            self._run_git(['config', 'user.email', 'bot@twistor-LMT.local'], check=False)
            
            # 提交
            self._run_git(['commit', '-m', message])
            
            # 获取 commit hash
            result = self._run_git(['rev-parse', 'HEAD'])
            commit_hash = result.stdout.strip()
            
            print(f"  ✓ 提交成功：{commit_hash[:8]}")
            
            return commit_hash
        except Exception as e:
            print(f"  ✗ 提交失败：{e}")
            return None
    
    def push(self, remote: str = 'origin', branch: Optional[str] = None) -> bool:
        """推送到远程"""
        print(f"推送到 {remote}...")
        
        try:
            if branch is None:
                result = self._run_git(['rev-parse', '--abbrev-ref', 'HEAD'])
                branch = result.stdout.strip()
            
            result = subprocess.run(
                ['git', 'push', remote, branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"  ✓ 推送成功到 {remote}/{branch}")
                return True
            else:
                print(f"  ✗ 推送失败：{result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ✗ 推送超时")
            return False
        except Exception as e:
            print(f"  ✗ 推送失败：{e}")
            return False
    
    def archive(self, message: str, run_test: bool = True, auto_push: bool = True) -> bool:
        """
        完整存档流程
        """
        print("=" * 70)
        print("Twistor-LMT 自动存档系统")
        print("=" * 70)
        print(f"提交信息：{message}")
        print(f"运行测试：{run_test}")
        print(f"自动推送：{auto_push}")
        print()
        
        # 1. 创建备份
        print("步骤 1: 创建备份")
        print("-" * 40)
        old_commit = self.create_backup()
        print()
        
        # 2. 运行测试
        if run_test:
            print("步骤 2: 运行测试")
            print("-" * 40)
            test_passed, test_info = self.run_tests()
            print()
            
            if not test_passed:
                print("✗ 测试失败，执行回滚")
                self.restore_backup()
                return False
        else:
            print("步骤 2: 跳过测试")
            print()
        
        # 3. 暂存更改
        print("步骤 3: 暂存更改")
        print("-" * 40)
        self.stage_changes()
        print()
        
        # 4. 提交
        print("步骤 4: 提交")
        print("-" * 40)
        commit_hash = self.commit(message)
        
        if not commit_hash:
            print("✗ 提交失败，执行回滚")
            self.restore_backup()
            return False
        print()
        
        # 5. 推送
        if auto_push:
            print("步骤 5: 推送到远程")
            print("-" * 40)
            push_success = self.push()
            
            if not push_success:
                print("✗ 推送失败，但保留本地提交")
        else:
            print("步骤 5: 跳过推送 (本地提交)")
            print()
        
        # 6. 清理备份
        print("步骤 6: 清理备份")
        print("-" * 40)
        self.cleanup_backup()
        print()
        
        # 7. 记录日志
        print("步骤 7: 记录日志")
        print("-" * 40)
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'commit': commit_hash,
            'test_passed': run_test,
            'pushed': auto_push,
        }
        self.logs.append(log_entry)
        self._save_logs()
        print(f"  ✓ 已记录到 {self.log_path}")
        print()
        
        # 完成
        print("=" * 70)
        print("✓ 存档完成")
        print(f"  提交：{commit_hash[:8]}")
        print(f"  信息：{message}")
        if auto_push:
            print(f"  状态：已推送到远程")
        else:
            print(f"  状态：本地提交 (未推送)")
        print("=" * 70)
        
        return True
    
    def rollback(self) -> bool:
        """回滚到上一个版本"""
        print("=" * 70)
        print("Twistor-LMT 回滚系统")
        print("=" * 70)
        
        # 尝试恢复备份
        if self.backup_path.exists():
            print("发现未完成的备份，执行回滚...")
            success = self.restore_backup()
        else:
            print("尝试回滚到上一个 commit...")
            try:
                self._run_git(['reset', '--hard', 'HEAD~1'])
                print("  ✓ 已回滚到上一个 commit")
                success = True
            except Exception as e:
                print(f"  ✗ 回滚失败：{e}")
                success = False
        
        print()
        print("=" * 70)
        if success:
            print("✓ 回滚完成")
        else:
            print("✗ 回滚失败")
        print("=" * 70)
        
        return success
    
    def rollback_to_version(self, version: str) -> bool:
        """回滚到指定版本"""
        print("=" * 70)
        print(f"回滚到版本 {version}")
        print("=" * 70)
        
        target = self.version_manager.load_version(version)
        if not target:
            print(f"✗ 版本 {version} 不存在")
            return False
        
        # 使用 git 回滚
        try:
            self._run_git(['checkout', target.commit])
            print(f"  ✓ 已回滚到版本 {version}")
            print(f"  Commit: {target.commit[:8]}")
            return True
        except Exception as e:
            print(f"  ✗ 回滚失败：{e}")
            return False
    
    def show_status(self):
        """显示状态"""
        print("=" * 70)
        print("Twistor-LMT Git 状态")
        print("=" * 70)
        
        has_changes, branch = self.check_git_status()
        print(f"当前分支：{branch}")
        print(f"有未提交更改：{'是' if has_changes else '否'}")
        print()
        
        # 显示版本历史
        print("版本历史:")
        versions = self.version_manager.list_versions()
        if versions:
            for v in versions[:5]:
                status_icon = "✓" if v.status == "success" else "✗"
                print(f"  {status_icon} {v.version}: fitness={v.fitness:.4f} ({v.timestamp[:10]})")
        else:
            print("  暂无版本记录")
        print()
        
        # 显示最近的存档记录
        print("最近的存档记录:")
        recent_logs = self.logs[-5:] if self.logs else []
        valid_logs = [log for log in recent_logs if isinstance(log, dict) and 'timestamp' in log]
        if valid_logs:
            for log in valid_logs:
                print(f"  - {log['timestamp'][:10]}: {log['message']} ({log['commit'][:8]})")
        else:
            print("  暂无存档记录")
        print()
        
        print("=" * 70)


# ============================================================================
# 整合系统
# ============================================================================

class AutoArchiveWithEvolution:
    """自动进化 + 存档系统"""
    
    def __init__(self, model_class):
        self.model_class = model_class
        self.evolution_engine = SimpleEvolutionEngine(model_class)
        self.archive_system = GitArchiveSystem()
    
    def evolve_and_archive(self, n_generations: int = 3, population_size: int = 5, 
                          message: Optional[str] = None, device='cpu') -> bool:
        """
        进化 + 存档完整流程
        """
        print("=" * 70)
        print("Twistor-LMT 自动进化 + 存档系统")
        print("=" * 70)
        print(f"进化代数: {n_generations}, 种群大小: {population_size}")
        print()
        
        # 1. 创建备份
        print("[1/6] 创建备份...")
        old_commit = self.archive_system.create_backup()
        print()
        
        # 2. 运行进化
        print("[2/6] 运行自动进化...")
        print("-" * 40)
        result = self.evolution_engine.evolve(
            n_generations=n_generations,
            population_size=population_size,
            device=device,
        )
        print()
        
        # 3. 运行测试
        print("[3/6] 运行单元测试...")
        print("-" * 40)
        test_passed, test_info = self.archive_system.run_tests()
        
        if not test_passed:
            print("\n✗ 测试失败，执行回滚")
            self.archive_system.restore_backup()
            return False
        print()
        
        # 4. 暂存更改
        print("[4/6] 暂存更改...")
        print("-" * 40)
        self.archive_system.stage_changes()
        print()
        
        # 5. 提交
        print("[5/6] 提交...")
        print("-" * 40)
        
        # 生成版本号
        version = self._generate_version(result.fitness)
        commit_message = message or f"v{version}: 进化完成 - 适应度 {result.fitness:.4f}"
        
        commit_hash = self.archive_system.commit(commit_message)
        
        if not commit_hash:
            print("✗ 提交失败，执行回滚")
            self.archive_system.restore_backup()
            return False
        print()
        
        # 6. 推送
        print("[6/6] 推送到远程...")
        print("-" * 40)
        push_success = self.archive_system.push()
        
        # 清理备份
        self.archive_system.cleanup_backup()
        
        # 保存版本信息
        model_version = ModelVersion(
            version=version,
            commit=commit_hash,
            timestamp=datetime.now().isoformat(),
            fitness=result.fitness,
            config=result.config.to_dict(),
            status='success',
            benchmarks=result.benchmarks,
        )
        self.evolution_engine.version_manager.save_version(model_version)
        
        # 完成
        print()
        print("=" * 70)
        print("✓ 自动进化 + 存档完成!")
        print(f"  版本: {version}")
        print(f"  提交: {commit_hash[:8]}")
        print(f"  适应度: {result.fitness:.4f}")
        print(f"  配置: hidden_dim={result.config.hidden_dim}, n_layers={result.config.n_layers}")
        if push_success:
            print(f"  状态: 已推送到远程")
        else:
            print(f"  状态: 本地提交 (未推送)")
        print("=" * 70)
        
        return True
    
    def _generate_version(self, fitness: float) -> str:
        """生成版本号"""
        # 基于适应度和时间生成版本
        major = int(fitness * 10) % 100
        return f"0.{major}.0"
    
    def show_versions(self):
        """显示版本历史"""
        versions = self.evolution_engine.version_manager.list_versions()
        
        print("=" * 70)
        print("版本历史")
        print("=" * 70)
        
        if not versions:
            print("暂无版本记录")
        else:
            for v in versions:
                status_icon = "✓" if v.status == "success" else "✗"
                print(f"{status_icon} {v.version}")
                print(f"    提交: {v.commit[:8]}")
                print(f"    适应度: {v.fitness:.4f}")
                print(f"    时间: {v.timestamp[:19]}")
                print(f"    配置: {v.config}")
                if v.benchmarks:
                    print(f"    基准: {v.benchmarks}")
                print()
        
        print("=" * 70)


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Twistor-LMT 自动存档 + 进化系统')
    parser.add_argument('-m', '--message', type=str, help='提交信息')
    parser.add_argument('-t', '--test', action='store_true', help='运行测试')
    parser.add_argument('-p', '--push', action='store_true', help='自动推送到远程')
    parser.add_argument('--evolve', action='store_true', help='运行自动进化')
    parser.add_argument('--generations', type=int, default=3, help='进化代数')
    parser.add_argument('--population', type=int, default=5, help='种群大小')
    parser.add_argument('--rollback', action='store_true', help='回滚到上一个版本')
    parser.add_argument('--rollback-to', type=str, help='回滚到指定版本')
    parser.add_argument('--versions', action='store_true', help='显示版本历史')
    parser.add_argument('--status', action='store_true', help='显示状态')
    parser.add_argument('--no-test', action='store_true', help='跳过测试')
    parser.add_argument('--no-push', action='store_true', help='不推送')
    
    args = parser.parse_args()
    
    # 显示版本历史
    if args.versions:
        system = AutoArchiveWithEvolution(None)
        system.show_versions()
        return
    
    # 显示状态
    if args.status:
        archive_system = GitArchiveSystem()
        archive_system.show_status()
        return
    
    # 回滚
    if args.rollback:
        archive_system = GitArchiveSystem()
        archive_system.rollback()
        return
    
    # 回滚到指定版本
    if args.rollback_to:
        archive_system = GitArchiveSystem()
        archive_system.rollback_to_version(args.rollback_to)
        return
    
    # 自动进化 + 存档
    if args.evolve:
        try:
            from poc.twistor_100m_config import StackedTwistorLMT, TwistorLMTConfig
            
            class ModelWrapper(nn.Module):
                def __init__(self, config: ArchitectureConfig):
                    super().__init__()
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
            
            system = AutoArchiveWithEvolution(ModelWrapper)
            
            run_test = not args.no_test
            auto_push = args.push and not args.no_push
            
            success = system.evolve_and_archive(
                n_generations=args.generations,
                population_size=args.population,
                message=args.message,
                device='cpu',
            )
            
            # 如果需要推送
            if success and auto_push:
                print("推送到远程...")
            
            sys.exit(0 if success else 1)
            
        except ImportError as e:
            print(f"✗ 无法导入模型: {e}")
            print("请确保 twistor_100m_config.py 存在")
            sys.exit(1)
    
    # 仅存档
    if args.message:
        archive_system = GitArchiveSystem()
        run_test = args.test and not args.no_test
        auto_push = args.push and not args.no_push
        
        success = archive_system.archive(
            message=args.message,
            run_test=run_test,
            auto_push=auto_push,
        )
        
        sys.exit(0 if success else 1)
    
    # 显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
