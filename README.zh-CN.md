# headless_hibs

研究型代码库,围绕**液态时间常数网络 (Liquid Time-Constant Networks, LTC)** 展开,
核心骨架是纤维丛几何,并配套训练、导出、检视模型的小型生态。本项目最好当作
"什么有效、什么无效"的演进笔记:在带有记忆的虚拟世界中,一个约 5 万参数的
主动推理系统在线学习。

本仓库**仅发布源码**。测试、数据集、实验产物、内部日志以及 AI 工具配置目录
均被有意排除——详见 `.gitignore`。

## 仓库结构

```
headless_hibs/
├── liquid_net/              # 液态神经网络核心库
│   ├── models/              # LiquidNet、LTC cell、稀疏 LTC cell
│   ├── solvers/             # Euler / RK4 / 通用 ODE 求解器
│   ├── training/            # 训练循环与损失
│   ├── analysis/            # 动力学分析工具
│   └── configs/             # 默认模型配置
├── hibs_studio/             # 模型浏览 / 训练 UI 辅助
├── hibs_export/             # 导出脚本(移动端 / 端侧格式)
├── hibs_cli/                # HIBS 命令行接口
├── twistor_studio/          # 扭量模型的工作台工具
├── twistor_LMT/             # 基于扭量的液态时间常数包
├── headless/                # Headless 桥接与训练模块
├── models/                  # 预训练模型检查点 (.pt)
├── poc/                     # 概念验证实验
├── scripts/                 # 训练与冒烟测试入口
├── main.py                  # 顶层入口
├── README.md                # 英文 README(默认)
├── README.zh-CN.md          # 本文件
└── .gitignore
```

## 安装

目标 **Python 3.12+**,依赖 PyTorch / NumPy。公开发布版未附带 `requirements.txt`,
按需安装即可:

```bash
pip install torch numpy matplotlib pyyaml
```

## 快速开始

```bash
# 冒烟测试安装
python scripts/smoke_test.py

# 训练一个小型 LiquidNet
python scripts/train_hibs_0_16.py

# 顶层入口
python main.py
```

## 库用法

`liquid_net` 包直接暴露模型、求解器和训练工具:

```python
from liquid_net.models import LiquidNet
from liquid_net.solvers import rk4
from liquid_net.training import train

model = LiquidNet(input_dim=1, hidden_dim=16, output_dim=1)
```

## 研究轨迹 —— V18 → V20.3

近期工作把静态推理栈变成**主动、持续学习的推理器**。关键数字
(完整细节见 `docs/v16_5_to_v20_3_synthesis.md`):

| 版本 | 思路 | 结果 |
| :--: | :-- | :-- |
| V18.0 | 朴素在线 BPTT(EFE 选动作) | 发散——PPL → ∞,参数漂移 3.60 |
| V18.1 | 波驱动局部更新 | 稳定,8K token 内 NLL −4.5 nats |
| V18.5 | 真实认知价值 (IG via logvar) | 动作匹配 6.0%(随机 2.2%) |
| V18.7 | 仅用内部信号更新(彻底移除 backward) | 稳定,KL 7.04→1.52,但 NLL 持平 |
| V18.9 | 三级世界(文本 / 文件系统 / Shell) | 反馈密度 0%,无法学习 |
| V19 | 离散基元(K=5:READ/WRITE/SEARCH/EDIT/ERROR) | 反馈密度 100%,world_loss 4.84→3.32 |
| V20.0 | Notebook World + 均匀 EFE | 基元平衡 17–23%,world_loss 3.56 |
| V20.1 | 不可逆"墓碑"token | TOME 在 98% 处合谋——新的捷径 |
| V20.2 | 衰减 + 冷却 + 加权损失 | 加权路径有效(world_loss 1.80);硬冷却崩溃 |
| **V20.3** | **两阶段训练 + 概率冷却** | **PPL 1e52 → 388,world_loss 0.09,WRITE→READ 配对率 99.5%** |
| V20.5 | 两阶段 + 健康世界 (V20.4b-4) | PPL 544,内容 84.5%,墓碑 1.9%,首次双赢 |
| V21 | 因果世界与位置绑定 | 87% set_decl 但仅 17.4% pos_decl(假阳性) |
| V21.1 | 严格位置评估 | 揭露 V21 的 87% 为集合论假阳性 |
| **V22** | **WRITE 冻结 + 大模型 + 位置加权** | **pos_decl 17.2% → 40.0% (+22.8%),首次真正陈述性因果** |

### V16.5 → V20.3 七条经验

1. **反馈密度是持续学习的先决条件。** V18–V18.9 花了六轮迭代才确认:
   在动作拿到 100% 反馈之前,其它都无意义。
2. **EFE 信号真实但有偏。** 用预期自由能选动作比随机好 1.57×,
   但"学什么"和"怎么学"是两件事。
3. **物理正确 ≠ 工程高效。** 波驱动更新能收敛,但需要 ~10⁵–10⁶ token,
   不是 10⁴。
4. **动作空间必须小。** 从 109⁴ token 序列改为 K=5 离散基元才是突破口。
5. **加权损失与冷却机制相互掣肘。** 让墓碑变得可学,反而让冷却不再触发。
6. **程序性因果与陈述性因果可分。** V20.3 学到了"WRITE 之后要 READ"(99.5%),
   但没学到"READ 会返回什么"——类比婴儿知道按按钮,却不知道屏幕会出现什么。
7. **解耦语言与世界学习是结构性发现。** Stage-1 离线 LM 预训练
   (lr=3e-4,2 epoch) + Stage-2 冻结 LM 只训练 WorldHead(lr=1e-4,1000 步)
   把 PPL 降了 50 个数量级,世界学习提速 20×。

### V20.3 留下的组件

| 组件 | 引入版本 | 状态 |
| :-- | :--: | :--: |
| 纤维丛骨架(`TwistFiberBundle`) | V16.6 | ✅ 保留 |
| 变分 SSM 扫描(含 KL) | V17 | ✅ 保留 |
| 结构化先验(`log σ_p² − α·log(1+‖κ‖)`) | V17.1 | ✅ 保留 |
| 真实认知价值(IG via logvar) | V18.5 | ✅ 保留 |
| 离散 K=5 基元 | V19 | ✅ 保留 |
| 均匀 EFE pragmatic 项 | V20 | ✅ 保留 |
| 加权 WorldHead 损失(`w_tome=0.3`) | V20.2 | ✅ 保留 |
| 两阶段训练 | V20.3 | ✅ 保留 |

### 路线图

- **V22.1** —— 更好训练的模型(更长预训练,PPL < 50)
- **V22.2** —— 更健康的世界(更低墓碑率)
- **V22.3** —— 多位置学习(突破位置 0)
- **V23** —— 迁移到真实世界交互(真实文件 I/O)
- **V24** —— 多智能体协作(两个 V20.3+ 模型共享一个笔记本)

完整报告留在工作树的 `docs/` 下;公开发布版省略了完整文档集。

## 许可证

MIT —— 详见各模块源文件头。
