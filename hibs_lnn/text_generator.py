"""
V12: 文本生成器 + 在线参数自适应
=================================

核心: 在 TwistFiberBundle 模型上搭建:
  1. 字符级 tokenizer
  2. 自回归生成 (温度采样)
  3. 在线学习 (streaming next-token prediction)
  4. 自我评估 + 微量调参 (self-feedback 风格)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class CharTokenizer:
    """字符级 tokenizer (复用已有 checkpoint 的 vocab)"""

    def __init__(self, char2idx: Dict[str, int], idx2char: Dict[int, str]):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx.get(c, 0) for c in text])

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.idx2char.get(int(i), "?") for i in ids)

    def encode_batch(self, texts: List[str], pad_len: int = None) -> torch.Tensor:
        encoded = [self.encode(t) for t in texts]
        if pad_len is None:
            pad_len = max(len(e) for e in encoded)
        out = torch.zeros(len(texts), pad_len, dtype=torch.long)
        for i, e in enumerate(encoded):
            out[i, :len(e)] = e[:pad_len]
        return out


class OnlineLearner:
    """
    在线学习器: streaming next-token prediction.

    每读一个 token, 做一次 CE loss + 梯度更新.
    用 Adam 自动适应学习率.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        clip_norm: float = 1.0,
        window: int = 100,
    ):
        self.params = list(params)
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.loss_history = []
        self.window = window
        self.steps = 0

    def step(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        loss = F.cross_entropy(logits, target)
        self.opt.zero_grad()
        loss.backward()
        if self.clip_norm > 0:
            nn.utils.clip_grad_norm_(self.params, self.clip_norm)
        self.opt.step()
        self.loss_history.append(float(loss.item()))
        self.steps += 1
        return float(loss.item())

    def get_avg_loss(self, n: int = 100) -> float:
        if not self.loss_history:
            return float("inf")
        return float(np.mean(self.loss_history[-n:]))

    def get_lr(self) -> float:
        return self.opt.param_groups[0]["lr"]


class SelfEvaluator:
    """
    自我评估器: 评估输出质量, 触发微量调参.

    评估指标:
    - 预测置信度 (softmax max prob)
    - 预测熵 (entropy)
    - loss 趋势 (上升/下降)
    """

    def __init__(self, fiber_bundle, threshold_entropy: float = 1.5):
        self.fiber_bundle = fiber_bundle
        self.threshold_entropy = threshold_entropy
        self.adapt_events = 0

    def evaluate(
        self, logits: torch.Tensor, loss: float, step: int
    ) -> Dict[str, float]:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
        confidence = probs.max(-1).values.mean().item()
        return {
            "entropy": entropy,
            "confidence": confidence,
            "loss": loss,
        }

    def maybe_adapt(self, eval_result: Dict[str, float], step: int) -> bool:
        """如果 loss 不降 (停滞), 微量扰动纤维方向探索新区域"""
        if self.fiber_bundle.current_fiber_dim < 2:
            return False
        if step > 100 and step % 200 == 0:
            # 定期微扰: 在 base 方向上加小噪声
            with torch.no_grad():
                cur = self.fiber_bundle.current_fiber_dim
                noise = torch.randn_like(self.fiber_bundle.fiber_dirs[:cur]) * 0.01
                self.fiber_bundle.fiber_dirs[:cur] += noise
                self.adapt_events += 1
                return True
        return False


def generate_text(
    model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_len: int = 100,
    temperature: float = 0.8,
    device: str = "cpu",
    top_k: int = 20,
) -> str:
    """
    自回归文本生成 (单步 = 1 ODE 步).

    Args:
        model: TwistGrowMultimodalTwistorLMT
        tokenizer: CharTokenizer
        prompt: 提示文本
        max_len: 生成最大长度
        temperature: 采样温度
        top_k: 只从 top-k 个 token 中采样 (0 = 不限)
    Returns:
        generated: 生成的文本
    """
    model.eval()
    encoded = tokenizer.encode(prompt).to(device)
    generated = list(encoded.cpu().numpy())

    with torch.no_grad():
        for _ in range(max_len):
            x = F.one_hot(encoded[-1:], num_classes=tokenizer.vocab_size).float().to(device)
            out = model(x, modality="text", seq_len=1)
            logits = out["output"]

            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            generated.append(next_idx)
            new_input = torch.tensor([next_idx], device=device)
            encoded = torch.cat([encoded, new_input])

    return tokenizer.decode(torch.tensor(generated))


def online_learn_text(
    model,
    tokenizer: CharTokenizer,
    text: str,
    learner: OnlineLearner,
    evaluator: Optional[SelfEvaluator] = None,
    device: str = "cpu",
    report_every: int = 200,
) -> Dict:
    """
    在线学习: 逐 token 读文本, 预测下一个, 梯度更新.

    Args:
        model: 模型
        tokenizer: tokenizer
        text: 训练文本
        learner: 在线学习器
        evaluator: 自我评估器 (可选)
        device: 设备
        report_every: 每 N 步报告一次
    Returns:
        stats: 学习统计
    """
    model.train()
    ids = tokenizer.encode(text).to(device)
    n = len(ids)

    losses = []
    adapt_count = 0
    start = time.time()

    for i in range(n - 1):
        x = F.one_hot(ids[i:i+1], num_classes=tokenizer.vocab_size).float().to(device)
        target = ids[i+1:i+2]

        out = model(x, modality="text", seq_len=1)
        logits = out["output"]

        loss = learner.step(logits, target)
        losses.append(loss)

        if evaluator is not None:
            eval_res = evaluator.evaluate(logits, loss, i)
            if evaluator.maybe_adapt(eval_res, i):
                adapt_count += 1

        if (i + 1) % report_every == 0:
            avg_l = np.mean(losses[-report_every:])
            print(f"  step {i+1:6d}/{n-1}  loss={avg_l:.4f}  lr={learner.get_lr():.2e}  adapt={adapt_count}")

    elapsed = time.time() - start
    return {
        "n_tokens": n - 1,
        "final_loss": float(np.mean(losses[-100:])),
        "losses": losses,
        "adapt_events": adapt_count,
        "elapsed_s": elapsed,
    }


import time
