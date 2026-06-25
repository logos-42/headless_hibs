# Twistor-LMT: 扭量驱动的液态神经网络

基于扭量理论的液态神经网络实现，支持智能体和时间序列建模。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 正弦波拟合任务（简单）- 推荐先运行这个
python3 twistor_LMT_main.py

# 完整版（含 Lorenz 预测）
python3 twistor_LMT.py --task sine --epochs 200

# Lorenz 系统预测（中等难度）
python3 twistor_LMT.py --task lorenz --epochs 100
```

### 预期输出

```
============================================================
Twistor-inspired Liquid Neural Network
复数状态的液态神经网络 - 时间序列预测
============================================================

生成正弦波数据...
输入形状：torch.Size([100, 32, 1])
目标形状：torch.Size([100, 32, 1])

创建模型...
设备：cpu
参数量：1,073
Hidden 维度：16

开始训练...
开始训练...
------------------------------------------------------------
Epoch  20/200 | Loss: 0.xxxxxx | MSE: 0.xxxxxx | Stab: 0.xxxxxx
Epoch  40/200 | Loss: 0.xxxxxx | MSE: 0.xxxxxx | Stab: 0.xxxxxx
...
------------------------------------------------------------
训练完成！最终 Loss: 0.xxxxxx

结果已保存到：twistor_LMT_results.png
```

## 核心概念

### 1. 扭量状态 Z

```python
Z ∈ ℂⁿ  # 复数向量，核心状态
```

替代传统 LMT 的实数状态 h ∈ ℝⁿ

### 2. 动力学方程

```python
dZ/dt = (-Z + W·tanh(Z) + U·x) / τ(Z)
```

- τ(Z) = sigmoid(W_τ · Re(Z)) - 可学习的时间常数
- 控制状态变化速度，实现"液态"特性

### 3. 多空间解码

```python
vector = Re(Z)           # 矢量
tensor = vector ⊗ vector # 张量
scalar = ||Z||          # 标量
```

## 架构说明

```
Input x(t)
   ↓
TwistorLMTCell → dZ/dt
   ↓
Euler/RK4 积分 → Z(t)
   ↓
TwistorDecoder → Output
```

## 使用示例

### 时间序列预测

```python
from twistor_LMT import TwistorLMT

model = TwistorLMT(
    input_dim=10,
    hidden_dim=32,
    output_dim=1
)

# 输入序列 [seq_len, batch, input_dim]
x_seq = torch.randn(50, 1, 10)

# 前向传播
output = model(x_seq, dt=0.1)
```

### 智能体应用

```python
from twistor_LMT import TwistorAgent

agent = TwistorAgent(
    obs_dim=4,      # 观测维度
    action_dim=2,   # 动作维度
    hidden_dim=32
)

# 环境循环
obs = env.reset()
agent.reset()

while True:
    action = agent.act(obs)
    obs, reward, done, _ = env.step(action)
```

## 项目结构

```
液态扭量模型/
├── twistor_LMT.py          # 核心实现
├── requirements.txt         # 依赖
├── README.md               # 说明文档
└── docs/
    ├── Twistor-LMT-架构设计.md  # 详细设计文档
    ├── 架构实验.md
    ├── 公式.md
    └── ...
```

## 实验任务

### 简单（入门）
- [x] 正弦波拟合
- [ ] 一阶系统辨识

### 中等（推荐）
- [x] Lorenz 系统预测
- [ ] Mackey-Glass 预测

### 进阶（研究）
- [ ] 多智能体协调
- [ ] 具身智能任务

## 调参建议

| 参数 | 推荐范围 | 说明 |
|-----|---------|------|
| hidden_dim | 16-64 | 扭量维度 |
| dt | 0.05-0.2 | 时间步长 |
| lr | 1e-4 - 1e-3 | 学习率 |
| τ初始 | ~1.0 | 时间常数 |

## 调试技巧

1. **数值稳定性**
   - 检查 ||dZ/dt|| 是否爆炸
   - 减小 dt
   - 限制 τ 范围

2. **可视化**
   - 绘制相空间轨迹
   - 观察 ||Z|| 演化
   - 对比输出 vs 目标

3. **梯度问题**
   - 使用梯度裁剪
   - 检查学习率

## 参考论文

1. Liquid Time-constant Networks (Hasani et al., 2021)
2. Neural Ordinary Differential Equations (Chen et al., 2018)
3. Twistor Theory (Penrose, 1967)

## 下一步

1. 运行正弦波示例，理解基本行为
2. 修改 hidden_dim，观察对结果的影响
3. 尝试 Lorenz 预测任务
4. 添加自定义任务

## 常见问题

**Q: 为什么要用复数？**

A: 复数天然支持相位、旋转、振荡，这些在实数空间需要额外参数。

**Q: 扭量和普通复数有什么区别？**

A: 工程实现上先当作复数向量，理论上有 Penrose 扭量几何意义。

**Q: 时间常数τ怎么初始化？**

A: 初始化为接近 1，通过 sigmoid(W_τ·Re(Z)) + ε实现。

**Q: 如何扩展到更复杂的任务？**

A: 
- 增加 hidden_dim
- 使用 RK4 积分器
- 添加物理约束损失

## License

MIT
