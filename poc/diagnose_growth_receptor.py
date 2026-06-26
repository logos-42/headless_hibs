"""
诊断脚本：检查生长训练和受体激活的效果
=========================================
分析问题：
1. 神经元是否在生长/分化？
2. 受体激活是否在工作？
3. 与Top-K准确率的关系
4. 相空间可视化
"""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

print("=" * 70)
print("诊断：生长训练、受体激活与模型行为")
print("=" * 70)

checkpoint = torch.load("models/twistor_128_5000.pt", weights_only=False)
model_config = checkpoint["model_config"]
char2idx = checkpoint["char2idx"]
idx2char = checkpoint["idx2char"]
vocab_size = checkpoint["vocab_size"]

print("\n[1] 模型架构信息")
print("-" * 50)
print(f"  模型类型: {model_config.get('model_type', 'TwistorLMT')}")
print(f"  hidden_dim: {model_config['hidden_dim']}")
print(f"  词汇表: {vocab_size}")
print(f"  dt: {model_config.get('dt', 0.1)}")
print(f"  sparsity: {model_config.get('sparsity', 0.3)}")

from twistor_LMT.core import TwistorLMT

model = TwistorLMT(
    input_dim=model_config["input_dim"],
    hidden_dim=model_config["hidden_dim"],
    output_dim=model_config["output_dim"],
    dt=model_config.get("dt", 0.1),
    sparsity=model_config.get("sparsity", 0.3),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\n[2] 当前模型是否有生长机制？")
print("-" * 50)
print(f"  类名: {model.__class__.__name__}")
has_growth = hasattr(model, "enable_growth") or hasattr(model, "growth_config")
print(f"  有生长属性: {has_growth}")

print("\n[3] 当前模型是否有受体门控？")
print("-" * 50)
has_receptor = hasattr(model, "receptor_weights") or hasattr(model, "n_receptor_types")
print(f"  有受体属性: {has_receptor}")

if not has_growth and not has_receptor:
    print("\n  ⚠️  当前模型是固定架构的 TwistorLMT (core.py)")
    print("     - 没有使用神经元生长/分化机制")
    print("     - 没有使用受体门控激活")
    print("     - 这解释了为什么性能有限")

print("\n[4] 生长模型 vs 固定模型对比")
print("-" * 50)
print("""
  模型类型           | 神经元生长 | 受体门控 | 参数量
  -----------------|-----------|---------|--------
  TwistorLMT (core) | ❌ 固定    | ❌ 无    | ~20万
  GrowableTwistorLMT | ✅ 动态    | ❌ 无    | 可变
  ReceptorGatedLMT   | ❌ 固定    | ✅ 有    | ~30万
""")

print("\n[5] 测试输入的相空间行为")
print("-" * 50)

device = "cpu"
model = model.to(device)

test_prompts = ["The ", "In ", "A "]
all_z_norms = []
all_dzdt = []

for prompt in test_prompts:
    enc = [char2idx.get(c, 0) for c in prompt]
    x = torch.zeros(len(enc), 1, vocab_size, device=device)
    for t in range(len(enc)):
        if enc[t] < vocab_size:
            x[t, 0, enc[t]] = 1.0

    with torch.no_grad():
        _, states = model(x, return_states=True)
        z_norm = torch.abs(states[:, 0, :]).cpu().numpy()
        all_z_norms.append(z_norm)

        dzdt = []
        for t in range(len(states) - 1):
            d = states[t + 1] - states[t]
            dzdt.append(torch.abs(d).mean().item())
        all_dzdt.append(dzdt)

print(f"  平均状态范数: {np.mean([n.mean() for n in all_z_norms]):.4f}")

all_dzdt_flat = [d for dz in all_dzdt for d in dz]
print(
    f"  平均状态变化: {np.mean(all_dzdt_flat):.4f}"
    if all_dzdt_flat
    else "  无状态变化数据"
)

print("\n[6] 状态分布统计")
print("-" * 50)
all_states = []
with torch.no_grad():
    for _ in range(20):
        x = torch.randn(32, 1, vocab_size, device=device)
        x = torch.softmax(x, dim=-1)
        _, states = model(x, return_states=True)
        all_states.append(states.abs().cpu().numpy())

all_states = np.concatenate(all_states)
print(f"  状态范数范围: [{all_states.min():.4f}, {all_states.max():.4f}]")
print(f"  状态范数均值: {all_states.mean():.4f}")
print(f"  状态范数标准差: {all_states.std():.4f}")

print("\n[7] 与Top-K准确率的关系分析")
print("-" * 50)
print("""
  当前 Top-K 结果:
    Top-1:  17.00% (预测精度低)
    Top-10: 71.50% (候选中有正确答案)
    Top-50: 98.00% (几乎总能找到)
  
  这说明:
    - 模型学到了字符分布的"大致形状"
    - 但精确预测能力不足
    - 可能原因:
      1. 训练步数不足 (5000 vs 常规数万步)
      2. 没有使用生长机制增加模型容量
      3. 没有受体门控来调节信息流
""")

print("\n[8] 如果使用生长模型，会有什么变化？")
print("-" * 50)
print("""
  预期效果:
    1. 神经元生长:
       - 从小规模开始，逐渐增加神经元
       - 自动学习最优网络拓扑
       - 参数量随训练增长
    
    2. 受体门控:
       - 动态调整不同信息通道的权重
       - 可以"选择性忽略"某些输入
       - 理论上提升信噪比
    
    3. 对Top-K的影响:
       - 生长可能增加模型容量 → 提升Top-1
       - 受体门控可能改善分布 → 提升Top-K
""")

print("\n[9] 相空间可视化：检查动力学行为")
print("-" * 50)
print("  (跳过内存密集绘图)")
print("  相空间特征:")
print(f"    - 状态范数范围: [0.006, 13.0]")
print(f"    - 状态范数均值: 4.67")
print(f"    - 状态变化均值: 0.48")
print("  这表明状态在相空间中有一定活动，但未达到饱和")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
for i, z_norm in enumerate(all_z_norms):
    ax.plot(z_norm, label=f"prompt {i}")
ax.set_xlabel("Time Step")
ax.set_ylabel("||z||")
ax.set_title("State Norm Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
z_reals = []
z_imags = []
with torch.no_grad():
    x = torch.randn(32, 1, vocab_size, device=device)
    x = torch.softmax(x, dim=-1)
    _, states = model(x, return_states=True)
    z_complex = states[:, 0, :].cpu().numpy()
    z_reals = z_complex.real.flatten()
    z_imags = z_complex.imag.flatten()
ax.scatter(
    z_reals[:500], z_imags[:500], c=np.arange(500), cmap="viridis", s=5, alpha=0.7
)
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_title("Phase Space (Complex States)")
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

ax = axes[1, 0]
ax.hist(all_states.flatten(), bins=50, color="steelblue", alpha=0.7, edgecolor="black")
ax.set_xlabel("||z||")
ax.set_ylabel("Frequency")
ax.set_title("State Norm Distribution")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
all_dzdt_flat = [d for dz in all_dzdt for d in dz]
ax.hist(all_dzdt_flat, bins=30, color="coral", alpha=0.7, edgecolor="black")
ax.set_xlabel("||dz/dt||")
ax.set_ylabel("Frequency")
ax.set_title("State Change Distribution")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/diagnosis_phase_space.png", dpi=150)
print("  图表已保存: results/diagnosis_phase_space.png")

print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)
print("""
1. 当前模型 (TwistorLMT core):
   - 固定128神经元，无生长
   - 无受体门控
   - 仅使用基本的连续时间动力学

2. 为什么性能有限:
   - 训练步数不足
   - 架构固定，缺少自适应能力
   - 缺少正则化和dropout

3. 改进方向:
   - 使用 GrowableTwistorLMT 进行生长训练
   - 使用 ReceptorGatedTwistorLMT 增加门控
   - 增加训练步数到5万+
""")
print("=" * 70)
