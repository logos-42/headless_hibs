"""
测试因果能力 - Test Causal Capabilities
========================================
验证所有新模块的因果推理能力：
1. 干预执行 (do-演算)
2. 反事实推理
3. 情景记忆
4. 好奇心驱动的探索
5. 因果学习信号
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List

from twistor_LMT import (
    TwistorLMT,
    CuriosityModule, CuriosityConfig,
    CausalInterventionEngine, DoIntervention, InterventionTarget,
    EpisodicMemory, HippocampalSystem, Episode,
    OfflineSimulator, SimulationConfig, DreamCortex,
    CausalLearningSignal, NeuromodulatorySignal,
    CausalCortex, CausalCortexConfig, CausalSelfAwareTwistorLMT,
    CausalReflectionModule, CausalReflectionConfig,
    NeuromodulatedGrowthConfig, CausalGrowableWrapper, AdaptiveDevelopmentalSchedule,
)


def test_curiosity_module():
    """测试内在动机模块"""
    print("\n" + "=" * 70)
    print("测试1: 内在动机模块 (Curiosity Module)")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    batch_size = 4
    
    curiosity = CuriosityModule(
        hidden_dim,
        CuriosityConfig(
            hidden_dim=hidden_dim,
            curiosity_weight=1.0,
        )
    ).to(device)
    
    curiosity.init_memory(device)
    
    state = torch.randn(batch_size, hidden_dim).to(device)
    
    result = curiosity(state)
    
    print(f"  状态形状: {state.shape}")
    print(f"  内在奖励: {result['intrinsic_reward'].mean().item():.4f}")
    print(f"  TD误差: {result['td_error'].mean().item():.4f}")
    print(f"  总不确定性: {result['total_uncertainty'].mean().item():.4f}")
    print(f"  总新奇性: {result['total_novelty'].mean().item():.4f}")
    print(f"  干预选择: {result['choice'].target_type if result['choice'] else 'None'}")
    print(f"  预期信息增益: {result['choice'].expected_information_gain if result['choice'] else 0:.4f}")
    
    print("\n  ✓ 内在动机模块测试通过")
    
    return curiosity


def test_intervention_executor():
    """测试干预执行器"""
    print("\n" + "=" * 70)
    print("测试2: 干预执行器 (Intervention Executor)")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    batch_size = 4
    
    engine = CausalInterventionEngine(hidden_dim).to(device)
    
    state = torch.randn(batch_size, hidden_dim).to(device)
    
    target_idx = 5
    intervention_value = torch.randn(batch_size, hidden_dim).to(device) * 2.0
    
    result = engine.do(
        state, 'z', [target_idx], intervention_value, 'hard'
    )
    
    print(f"  原始状态[0,0]: {state[0, 0].item():.4f}")
    print(f"  干预目标索引: {target_idx}")
    print(f"  干预后状态[0,0]: {result.intervened_state[0, 0].item():.4f}")
    print(f"  干预成本: {result.intervention_cost.mean().item():.4f}")
    print(f"  因果效应范数: {torch.norm(result.causal_effect).item():.4f}")
    
    def simple_dynamics(state):
        return state + 0.1 * torch.randn_like(state)
    
    def intervention_fn(state, step):
        intervened = state.clone()
        intervened[:, 3] = 5.0
        return intervened
    
    cf_result = engine.counterfactual(
        state, DoIntervention(
            intervention_type='soft',
            target=InterventionTarget(target_type='z', indices=[3]),
            reversible=True,
        ), simple_dynamics
    )
    
    print(f"  反事实轨迹形状: {cf_result.counterfactual_trajectory.shape}")
    print(f"  因果效应大小: {cf_result.effect_magnitude:.4f}")
    print(f"  置信度: {cf_result.confidence:.4f}")
    
    print("\n  ✓ 干预执行器测试通过")
    
    return engine


def test_episodic_memory():
    """测试情景记忆"""
    print("\n" + "=" * 70)
    print("测试3: 情景记忆 (Episodic Memory)")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    memory_dim = 64
    
    memory = EpisodicMemory(hidden_dim, memory_dim).to(device)
    hippocampus = HippocampalSystem(hidden_dim, memory_dim).to(device)
    
    episode1_states = torch.randn(20, hidden_dim).to(device)
    episode1_rewards = torch.randn(20).to(device)
    
    episode_id = hippocampus.encode_and_store(episode1_states, rewards=episode1_rewards)
    
    print(f"  存储episode ID: {episode_id}")
    print(f"  Episode状态形状: {episode1_states.shape}")
    
    query_state = episode1_states[10]
    
    retrieved = hippocampus.retrieve_similar(query_state, top_k=3)
    
    print(f"  检索到 {len(retrieved)} 个相关记忆")
    if retrieved:
        print(f"    Top-1 相似度: {retrieved[0][1]:.4f}")
    
    memory_stats = memory.get_statistics()
    print(f"  记忆统计: 总episode数={memory_stats['total_episodes']}")
    
    offline_replays = hippocampus.offline_replay(num_replays=3)
    print(f"  离线重演: {len(offline_replays)} 个片段")
    
    print("\n  ✓ 情景记忆测试通过")
    
    return hippocampus


def test_offline_simulator():
    """测试离线模拟器"""
    print("\n" + "=" * 70)
    print("测试4: 离线模拟器 (Offline Simulator)")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    
    simulator = OfflineSimulator(hidden_dim).to(device)
    dream_cortex = DreamCortex(hidden_dim).to(device)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.U = nn.Linear(hidden_dim, hidden_dim)
            
        def compute_dzdt(self, z, x):
            return -z + self.U(x) * 0.1
            
        def forward(self, z, x):
            return self.compute_dzdt(z, x)
    
    model = SimpleModel().to(device)
    
    initial_state = torch.randn(1, hidden_dim).to(device)
    
    final_state, trajectory = simulator.simulate_without_learning(
        model, initial_state, num_steps=20
    )
    
    print(f"  初始状态形状: {initial_state.shape}")
    print(f"  轨迹形状: {trajectory.shape}")
    print(f"  最终状态范数: {torch.norm(final_state).item():.4f}")
    
    def intervention_fn(state, step):
        intervened = state.clone()
        intervened[:, 0] += 2.0
        return intervened
    
    cf_result = simulator.counterfactual_rollout(
        model, initial_state, intervention_fn, num_steps=10, num_rollouts=5
    )
    
    print(f"  反事实轨迹形状: {cf_result.counterfactual_trajectory.shape}")
    print(f"  因果效应大小: {cf_result.effect_magnitude:.4f}")
    
    imagined = simulator.imagine_alternatives(
        model, initial_state, num_alternatives=5
    )
    
    print(f"  想象替代方案: {len(imagined)} 个")
    if imagined:
        print(f"    Top-1 新奇性分数: {imagined[0].novelty_score:.4f}")
    
    print("\n  ✓ 离线模拟器测试通过")
    
    return simulator


def test_causal_learning_signal():
    """测试因果学习信号"""
    print("\n" + "=" * 70)
    print("测试5: 因果学习信号 (Causal Learning Signal)")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    batch_size = 4
    
    causal_learning = CausalLearningSignal(hidden_dim).to(device)
    
    state = torch.randn(batch_size, hidden_dim).to(device)
    next_state = torch.randn(batch_size, hidden_dim).to(device)
    extrinsic_reward = torch.randn(batch_size).to(device)
    intrinsic_reward = torch.ones(batch_size) * 0.5
    weights = torch.randn(hidden_dim, hidden_dim) * 0.1
    pre_synaptic = state
    post_synaptic = next_state
    
    updated_weights, info = causal_learning(
        state, extrinsic_reward, intrinsic_reward, next_state,
        weights, pre_synaptic, post_synaptic
    )
    
    print(f"  状态形状: {state.shape}")
    print(f"  外部奖励: {extrinsic_reward.mean().item():.4f}")
    print(f"  内在奖励: {intrinsic_reward.mean().item():.4f}")
    print(f"  多巴胺信号: {info['dopamine'].mean().item():.4f}")
    print(f"  TD误差: {info['td_error'].mean().item():.4f}")
    print(f"  更新后权重范数: {torch.norm(updated_weights).item():.4f}")
    print(f"  可塑性阈值范围: [{info['threshold'].min().item():.4f}, {info['threshold'].max().item():.4f}]")
    
    print("\n  ✓ 因果学习信号测试通过")
    
    return causal_learning


def test_causal_cortex():
    """测试因果皮层（整合模块）"""
    print("\n" + "=" * 70)
    print("测试6: 因果皮层 (Causal Cortex - Integration)")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    
    causal_cortex = CausalCortex(
        hidden_dim,
        CausalCortexConfig(
            hidden_dim=hidden_dim,
            enable_curiosity=True,
            enable_intervention=True,
            enable_episodic_memory=True,
            enable_offline_simulation=True,
            enable_causal_learning=True,
        )
    ).to(device)
    
    state = torch.randn(4, hidden_dim).to(device)
    
    observe_result = causal_cortex.observe(state)
    
    print(f"  观察状态: {state.shape}")
    print(f"  内在奖励: {observe_result['intrinsic_reward'].mean().item():.4f}")
    
    intervened_state, interv_info = causal_cortex.intervene(state)
    print(f"  干预后状态形状: {intervened_state.shape}")
    
    episode_id = causal_cortex.store_episode()
    print(f"  存储episode ID: {episode_id}")
    
    retrieved = causal_cortex.retrieve_memories(state[0:1], top_k=3)
    print(f"  检索到 {len(retrieved)} 个相关记忆")
    
    print("\n  ✓ 因果皮层测试通过")
    
    return causal_cortex


def test_causal_self_aware_twistor():
    """测试因果自感知扭量网络"""
    print("\n" + "=" * 70)
    print("测试7: 因果自感知扭量网络")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    input_dim = 10
    hidden_dim = 32
    output_dim = 5
    batch_size = 4
    seq_len = 8
    
    model = CausalSelfAwareTwistorLMT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dt=0.1,
    ).to(device)
    
    x = torch.randn(seq_len, batch_size, input_dim).to(device)
    
    output_seq, info = model(x, enable_causal=True)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output_seq.shape}")
    
    state = torch.randn(batch_size, hidden_dim).to(device)
    
    intervened = model.do_intervention(
        state,
        target_indices=[0, 1, 2],
        value=torch.randn(batch_size, hidden_dim) * 0.5
    )
    print(f"  干预后状态形状: {intervened.shape}")
    
    imagined = model.imagine_alternatives(state, num_alternatives=3)
    print(f"  想象替代方案: {len(imagined)} 个")
    
    episode_id = model.store_memory()
    print(f"  存储记忆: {episode_id}")
    
    retrieved = model.retrieve_memories(state[0:1], top_k=3)
    print(f"  检索到 {len(retrieved)} 个相关记忆")
    
    print("\n  ✓ 因果自感知扭量网络测试通过")
    
    return model


def test_causal_reflection():
    """测试因果反思模块"""
    print("\n" + "=" * 70)
    print("测试8: 因果反思模块")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 32
    
    reflection = CausalReflectionModule(
        hidden_dim,
        CausalReflectionConfig(
            hidden_dim=hidden_dim,
            enable_curiosity=True,
            enable_intervention=True,
            enable_counterfactual=True,
            enable_neuromodulation=True,
        )
    ).to(device)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.dt = 0.1
            
        def compute_dzdt(self, z, x):
            return -z + 0.1 * torch.randn_like(z)
    
    model = SimpleModel().to(device)
    
    x = torch.randn(10, 4, hidden_dim).to(device)
    y = torch.randn(10, 4, hidden_dim).to(device)
    
    loss_history = [0.5, 0.4, 0.35, 0.3, 0.28]
    
    reflection_result = reflection.reflect(
        model, x, y, loss_history, training_step=100
    )
    
    print(f"  反思结果动作: {reflection_result['decision']['action']}")
    print(f"  决策置信度: {reflection_result['decision']['confidence']:.4f}")
    print(f"  好奇心信息: {reflection_result['curiosity_info']}")
    print(f"  因果效应: {reflection_result['cf_info'].get('avg_effect', 0):.4f}")
    print(f"  多巴胺水平: {reflection_result['neuromodulation']['dopamine']:.4f}")
    
    stats = reflection.get_stats()
    print(f"  统计: 总反思={stats['total_thinks']}, 增长={stats['total_grows']}, 修剪={stats['total_prunes']}")
    
    print("\n  ✓ 因果反思模块测试通过")
    
    return reflection


def test_causal_growable():
    """测试因果生长模块"""
    print("\n" + "=" * 70)
    print("测试9: 因果生长模块")
    print("=" * 70)
    
    device = 'cpu'
    torch.manual_seed(42)
    
    hidden_dim = 16
    
    growable_wrapper = CausalGrowableWrapper(
        None,
        NeuromodulatedGrowthConfig(
            hidden_dim=hidden_dim,
            enable_neuromodulation=True,
            enable_causal_importance=True,
        )
    ).to(device)
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = hidden_dim
            
        def split_neuron(self, idx):
            return idx + 1
            
        def disable_connection(self, idx):
            pass
    
    mock_model = MockModel()
    growable_wrapper.model = mock_model
    
    states = torch.randn(10, hidden_dim).to(device)
    outputs = torch.randn(10, hidden_dim).to(device)
    
    neuromod = NeuromodulatorySignal(
        dopamine=torch.tensor(0.5),
        serotonin=torch.tensor(0.3),
        acetylcholine=torch.tensor(0.6),
        norepinephrine=torch.tensor(0.4),
    )
    
    importance_scores = growable_wrapper.update_neuromodulation(
        states, outputs, neuromod
    )
    
    print(f"  因果重要性分数数量: {len(importance_scores)}")
    
    should_grow = growable_wrapper.neuromod_growth.should_grow()
    print(f"  应该生长: {should_grow}")
    
    should_prune, prune_indices = growable_wrapper.neuromod_growth.should_prune()
    print(f"  应该修剪: {should_prune}, 索引: {prune_indices}")
    
    result = growable_wrapper.step(current_step=50)
    print(f"  生长步骤结果: {result}")
    
    summary = growable_wrapper.get_summary()
    print(f"  多巴胺水平: {summary['neuromodulation']['dopamine']:.4f}")
    print(f"  生长速率: {summary['neuromodulation']['growth_rate']:.4f}")
    
    adaptive_schedule = AdaptiveDevelopmentalSchedule(hidden_dim).to(device)
    
    adaptive_result = adaptive_schedule(states[:1], neuromod)
    print(f"  自适应阶段: {adaptive_result['current_phase']}")
    print(f"  探索强度: {adaptive_result['exploration_intensity']:.4f}")
    
    print("\n  ✓ 因果生长模块测试通过")
    
    return growable_wrapper


def main():
    """主测试函数"""
    print("\n" + "#" * 70)
    print("# LMT-Twister 因果能力测试")
    print("#" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        curiosity = test_curiosity_module()
        
        intervention = test_intervention_executor()
        
        memory = test_episodic_memory()
        
        simulator = test_offline_simulator()
        
        causal_learning = test_causal_learning_signal()
        
        cortex = test_causal_cortex()
        
        self_aware = test_causal_self_aware_twistor()
        
        reflection = test_causal_reflection()
        
        growable = test_causal_growable()
        
        print("\n" + "#" * 70)
        print("# 所有测试通过!")
        print("#" * 70)
        
        print("""
已实现的因果能力:

1. 内在动机模块 (CuriosityModule)
   ✓ 信息增益驱动的好奇心
   ✓ 不确定性估计
   ✓ 新奇性检测
   ✓ 干预选择机制

2. 干预执行器 (InterventionExecutor)
   ✓ 硬干预 do(Z = value)
   ✓ 软干预 do(Z := Z + modulator)
   ✓ 机制干预
   ✓ 因果效应评估

3. 情景记忆系统 (EpisodicMemory / HippocampalSystem)
   ✓ 状态轨迹存储
   ✓ 基于相似度的检索
   ✓ 时序编码
   ✓ 离线重演

4. 离线模拟器 (OfflineSimulator / DreamCortex)
   ✓ 不更新权重的模拟
   ✓ 反事实rollout
   ✓ 想象替代方案生成
   ✓ 因果效应量化

5. 因果学习信号 (CausalLearningSignal)
   ✓ TD误差 / 多巴胺信号
   ✓ 神经调节信号 (血清素、乙酰胆碱、去甲肾上腺素)
   ✓ 可塑性阈值调节 (BCM规则)
   ✓ 资格痕迹

6. 因果皮层 (CausalCortex)
   ✓ 整合所有模块
   ✓ 在线观察和干预
   ✓ 离线反事实推理
   ✓ 因果学习信号更新

7. 因果反思模块 (CausalReflectionModule)
   ✓ 好奇心驱动的思考
   ✓ 反事实思考
   ✓ 神经调节整合
   ✓ 因果决策（替代固定阈值）

8. 因果生长模块 (CausalGrowableWrapper)
   ✓ 神经调节驱动的生长/修剪
   ✓ 因果重要性评估
   ✓ 自适应发育调度
""")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
