"""
V15: 在线域适应 — 纤维丛的独特价值
====================================

核心思路:
  1. 预训练 backbone(201K 冻结) = 通用语言能力
  2. 纤维丛(2K可训练) = 轻量"舵" → 在线适应新领域
  3. 领域切换时: 纤维丛自动生长/旋转方向, RNN 做不到因为:
     - RNN 需要完全重训练或忘记旧领域
     - 纤维丛可以生长新维度 —— 不遗忘, 只增加容量
  4. InternalCritic 自我评估检测领域漂移 → 触发自适应

与 LoRA 的区别:
  - LoRA: 在 attention weights 上加低秩更新 (适配器)
  - 纤维丛: 在 ODE 状态的复数子空间上加旋转 + 振幅 → 连续动力学适应
  - 可在线实时生长新维度 (LoRA rank 固定)

架构:
  PretrainedFiberTwistor (V14 复用)
     + InternalCritic (自我评估)
     + DomainAdaptation (域适应控制器)
           ├─ train_on_domain(text, domain_name)
           ├─ detect_domain_shift(critic_score) → 触发生长/旋转
           └─ compare_steering() → 可视化不同域的纤维方向差异
"""

import os, json, time, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained_fiber import PretrainedFiberTwistor
from .self_feedback import InternalCritic
from .text_generator import CharTokenizer


@dataclass
class DomainAdaptConfig:
    """域适应配置"""
    growth_threshold: float = 0.3       # critic 低于此值 → 生长纤维
    rotate_threshold: float = 0.4       # critic 低于此值 → 旋转方向
    critic_eval_interval: int = 100     # 每 N 步自我评估一次
    cooldown_growth: int = 500          # 两次生长之间至少 N 步
    cooldown_rotate: int = 200          # 两次旋转之间至少 N 步
    lr_adapt: float = 0.01              # 方向自适应学习率
    n_internal_forward: int = 1         # ODE 步数
    max_grow_per_event: int = 8         # 单次最多生长 N 个纤维
    report_every: int = 1000            # 日志频率


class DomainAdaptation(nn.Module):
    """
    域适应控制器 — 纤维丛的在线自适应.
    """

    def __init__(
        self,
        backbone: nn.Module,
        fiber_bundle,
        tokenizer: CharTokenizer,
        config: Optional[DomainAdaptConfig] = None,
        n_internal: int = 1,
    ):
        super().__init__()
        self.config = config or DomainAdaptConfig()
        self.tokenizer = tokenizer
        self.V = backbone.out.weight.shape[0]

        # 核心模型: V14 骨干 + 纤维丛 (不冻结 backbone)
        self.pf = PretrainedFiberTwistor(
            backbone, fiber_bundle,
            n_internal=n_internal,
            freeze_backbone=False,
        )

        # 自我评估器 (评估生成质量)
        self.critic = InternalCritic(
            hidden_dim=backbone.hidden_dim,
            output_dim=self.V,
        )

        # 域追踪
        self.current_domain: Optional[str] = None
        self.domain_log: Dict[str, dict] = {}
        self.steering_snapshots: Dict[str, torch.Tensor] = {}

        # 自适应事件日志
        self.growth_events: List[dict] = []
        self.rotate_events: List[dict] = []
        self.critic_history: List[dict] = []

        # 自适应冷却
        self._last_growth_step = 0
        self._last_rotate_step = 0
        self._total_steps = 0

        # 参数统计
        self.backbone_params = self.pf.backbone_params
        self.fiber_params = self.pf.fiber_params

    @property
    def current_fiber_dim(self) -> int:
        return self.pf.fiber_bundle.current_fiber_dim

    # ============================================================
    # 域训练 (在线 streaming)
    # ============================================================
    def train_on_domain(
        self,
        text: str,
        domain_name: str,
        n_tokens: int = 5000,
        lr: float = 1e-3,
        opt: Optional[torch.optim.Optimizer] = None,
    ) -> Dict:
        """
        在特定域文本上在线训练纤维丛.
        仅训练纤维丛参数, backbone 冻结.

        Args:
            text: 训练文本
            domain_name: 域名称 (如 "shakespeare", "tech")
            n_tokens: 训练 token 数
            lr: 学习率
        Returns:
            stats: 训练统计
        """
        self.current_domain = domain_name
        ids = self.tokenizer.encode(text).to(next(self.parameters()).device)
        ids = ids[:n_tokens + 1]

        if opt is None:
            opt = torch.optim.Adam(self.pf.fiber_bundle.parameters(), lr=lr)

        self.pf.fiber_bundle.train()

        losses = []
        z = self.pf.reset_state(1, ids.device)
        n = len(ids)

        t0 = time.time()

        for i in range(n - 1):
            x = F.one_hot(ids[i:i+1], self.V).float().to(ids.device)
            target = ids[i+1:i+2]

            logits, z = self.pf(x, z=z.detach(), return_state=True)
            loss = F.cross_entropy(logits, target)

            opt.zero_grad()
            loss.backward()
            trainable = list(self.pf.fiber_bundle.parameters())
            if not self.pf.freeze_backbone:
                trainable += list(self.pf.backbone.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            losses.append(float(loss.item()))
            self._total_steps += 1

            # 自我评估 (间隔)
            if self._total_steps % self.config.critic_eval_interval == 0:
                self._self_evaluate(logits, target, z)

            # 自适应触发 (critic 驱动)
            if self._total_steps % self.config.critic_eval_interval == 0:
                recent = self.critic_history[-5:]
                if len(recent) >= 3:
                    avg_critic = np.mean([r["overall"] for r in recent])
                    if avg_critic < self.config.growth_threshold:
                        self._maybe_grow_fibers(avg_critic)
                    if avg_critic < self.config.rotate_threshold:
                        self._maybe_rotate_directions(avg_critic)

            # 日志
            if (i + 1) % self.config.report_every == 0:
                avg_loss = np.mean(losses[-self.config.report_every:]) if losses else 0
                print(f"  [{domain_name}] step {i+1:6d}/{n-1}  "
                      f"loss={avg_loss:.4f}  fiber_dim={self.current_fiber_dim}")

        elapsed = time.time() - t0

        # 保存域快照
        self._save_domain_snapshot(domain_name, losses, elapsed)

        return {
            "domain": domain_name,
            "n_tokens": n - 1,
            "final_loss": float(np.mean(losses[-200:])),
            "elapsed_s": elapsed,
            "tok_s": (n - 1) / elapsed,
            "fiber_dim": self.current_fiber_dim,
            "n_growth_events": len(self.growth_events),
        }

    # ============================================================
    # 自我评估 (InternalCritic)
    # ============================================================
    def _self_evaluate(self, logits, target, z):
        with torch.no_grad():
            scores = self.critic.evaluate(
                current_output=logits,
                current_state=z.real,
            )
            scores["step"] = self._total_steps
            scores["domain"] = self.current_domain
            self.critic_history.append(scores)

    def evaluate_generation(self, prompt: str, max_len: int = 50) -> Dict:
        """生成并自我评估"""
        gen_text = self.pf.generate(self.tokenizer, prompt, max_len=max_len)
        encoded = self.tokenizer.encode(gen_text).to(next(self.parameters()).device)
        losses = []
        with torch.no_grad():
            z = self.pf.reset_state(1, encoded.device)
            for i in range(len(encoded) - 1):
                x = F.one_hot(encoded[i:i+1], self.V).float().to(encoded.device)
                logits, z = self.pf(x, z=z, return_state=True)
                loss = F.cross_entropy(logits, encoded[i+1:i+2])
                losses.append(float(loss.item()))
        return {
            "text": gen_text,
            "per_token_loss": float(np.mean(losses)) if losses else 0,
            "critic": self.critic_history[-1] if self.critic_history else None,
        }

    # ============================================================
    # 自适应生长/旋转
    # ============================================================
    def _maybe_grow_fibers(self, critic_value: float):
        """critic 低 → 模型不自信 → 生长新纤维"""
        if self._total_steps - self._last_growth_step < self.config.cooldown_growth:
            return
        fb = self.pf.fiber_bundle
        if fb.current_fiber_dim >= fb.max_fiber_dim:
            return
        n_new = min(self.config.max_grow_per_event,
                     fb.max_fiber_dim - fb.current_fiber_dim)
        fb.grow_fiber(n_new)
        self._last_growth_step = self._total_steps
        self.growth_events.append({
            "step": self._total_steps,
            "domain": self.current_domain,
            "critic": critic_value,
            "n_new": n_new,
            "new_dim": fb.current_fiber_dim,
        })
        print(f"  [+] grew {n_new} fibers (critic={critic_value:.3f}) "
              f"→ dim={fb.current_fiber_dim}")

    def _maybe_rotate_directions(self, critic_value: float):
        """critic 极低 → 模型在挣扎 → 旋转纤维方向"""
        if self._total_steps - self._last_rotate_step < self.config.cooldown_rotate:
            return
        fb = self.pf.fiber_bundle
        with torch.no_grad():
            cur = fb.current_fiber_dim
            if cur > 0:
                noise = torch.randn_like(fb.fiber_dirs[:cur]) * 0.05
                fb.fiber_dirs[:cur] += noise
                self._last_rotate_step = self._total_steps
                self.rotate_events.append({
                    "step": self._total_steps,
                    "domain": self.current_domain,
                    "critic": critic_value,
                })
                print(f"  [~] rotated fiber directions (critic={critic_value:.3f})")

    # ============================================================
    # 域快照 & 分析
    # ============================================================
    def _save_domain_snapshot(self, domain: str, losses: List[float], elapsed: float):
        fb = self.pf.fiber_bundle
        self.domain_log[domain] = {
            "fiber_dim": fb.current_fiber_dim,
            "fiber_amps": fb.fiber_amps[:fb.current_fiber_dim].detach().cpu().tolist(),
            "final_loss": float(np.mean(losses[-200:])),
            "n_tokens": len(losses),
            "elapsed_s": elapsed,
        }
        # 保存纤维方向快照
        self.steering_snapshots[domain] = fb.fiber_dirs[:fb.current_fiber_dim].detach().cpu().clone()

    def compare_steering(self, domain_a: str, domain_b: str) -> Dict:
        """比较两个域之间的纤维方向差异"""
        if domain_a not in self.steering_snapshots or domain_b not in self.steering_snapshots:
            return {"error": "one or both domains not found"}
        dirs_a = self.steering_snapshots[domain_a]
        dirs_b = self.steering_snapshots[domain_b]
        min_dim = min(dirs_a.shape[0], dirs_b.shape[0])
        if min_dim == 0:
            return {"cosine_sim": [], "mean_sim": 0}
        dirs_a = dirs_a[:min_dim].float()
        dirs_b = dirs_b[:min_dim].float()
        # 归一化
        dirs_a = dirs_a / (dirs_a.norm(dim=-1, keepdim=True) + 1e-8)
        dirs_b = dirs_b / (dirs_b.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_sim = (dirs_a * dirs_b).sum(dim=-1).tolist()
        return {
            "cosine_sim": cosine_sim,
            "mean_sim": float(np.mean(cosine_sim)),
            "std_sim": float(np.std(cosine_sim)),
            "min_dim": min_dim,
        }

    def get_adaptation_report(self) -> Dict:
        """完整的自适应报告"""
        return {
            "backbone_params": self.backbone_params,
            "fiber_params": self.fiber_params,
            "fiber_dim": self.current_fiber_dim,
            "max_fiber_dim": self.pf.fiber_bundle.max_fiber_dim,
            "total_steps": self._total_steps,
            "n_domains": len(self.domain_log),
            "domains": list(self.domain_log.keys()),
            "n_growth_events": len(self.growth_events),
            "n_rotate_events": len(self.rotate_events),
            "growth_events": self.growth_events,
            "rotate_events": self.rotate_events,
            "domain_log": self.domain_log,
            "critic_trace": self.critic_history,
        }
