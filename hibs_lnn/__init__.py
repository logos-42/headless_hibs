"""
Twistor-LMT: 扭量驱动的液态神经网络
===================================

完整的 Twistor-inspired Liquid Neural Network 实现，包含：
- 复数值隐藏状态
- 状态依赖的时间常数
- 稀疏连接
- 多尺度动力学
- 多种积分方法
- 智能体接口
- 不动点分析
- 相空间可视化

快速开始:
    from twistor_LMT import TwistorLMT, TwistorAgent

    model = TwistorLMT(input_dim=4, hidden_dim=32, output_dim=2)
    agent = TwistorAgent(obs_dim=4, action_dim=2)
"""

from .core import TwistorLMT
from .coupled import CoupledTwistorLMT, StackedCoupledLMT
from .agent import TwistorAgent, TwistorAgentWithPolicy, MultiAgent
from .decoder import TwistorDecoder, TensorTwistorDecoder, create_decoder
from .integrators import (
    euler_step,
    RK4Integrator,
    ODESolver,
    AdjointODESolver,
    TORCHDIFFEQ_AVAILABLE,
    rk4_step,
    heun_step,
    dopri5_step,
    create_integrator,
)
from .ode_solver import TwistorODE, ODEDynamics, odeint_wrapper, create_ode_solver
from .analysis import (
    FixedPointFinder,
    StabilityAnalyzer,
    BifurcationAnalyzer,
    analyze_model,
)
from .visualization import (
    plot_phase_space_2d,
    plot_phase_space_3d,
    plot_vector_field,
    plot_tau_evolution,
    plot_complex_plane,
    plot_stability_analysis,
    plot_training_diagnostics,
)
from .datasets import (
    generate_lorenz_dataset,
    generate_mackey_glass_dataset,
    generate_van_der_pol_dataset,
    generate_sine_dataset,
    create_dataset,
)
from .training import (
    train_model,
    train_on_task,
    plot_training_results,
    plot_predictions,
)
from .mobius import (
    MobiusConstraint,
    AdaptiveMobiusConstraint,
    create_mobius_constraint,
)
from .resonance import (
    TwistorResonance,
    MultiHeadResonance,
    create_resonance,
)
from .growable import GrowableTwistorLMT, create_growable_twistor_LMT
from .self_feedback import SelfAwareTwistorLMT, SelfFeedbackConfig
from .reflection import ReflectionModule, ReflectionConfig, ModificationRecord
from .curiosity_module import CuriosityModule, CuriosityConfig, IntrinsicRewardCalculator
from .intervention_executor import InterventionExecutor, CausalInterventionEngine, DoIntervention
from .episodic_memory import EpisodicMemory, HippocampalSystem, Episode
from .offline_simulator import OfflineSimulator, DreamCortex, SimulationConfig
from .causal_learning_signal import CausalLearningSignal, NeuromodulatorySignal
from .causal_cortex import CausalCortex, CausalCortexConfig, CausalSelfAwareTwistorLMT
from .causal_reflection import CausalReflectionModule, CausalReflectionConfig, CausalDecision
from .causal_growable import (
    NeuromodulatedGrowthConfig,
    CausalImportanceScore,
    CausalImportanceEstimator,
    NeuromodulationDrivenGrowth,
    CausalGrowableWrapper,
    AdaptiveDevelopmentalSchedule,
)
from .emotional_system import (
    EmotionalState,
    EmotionReaction,
    EmotionalEncoder,
    EmotionalModulator,
    EmotionalMemory,
    EmotionalLearning,
    EmotionalSystem,
    EmotionalCausalLoop,
)
from .twistor_causal import (
    TwistorCausalConfig,
    PhaseCausalCalculator,
    AmplitudeCausalCalculator,
    TwistorInterventionExecutor,
    TwistorOfflineSimulator,
    TwistorCausalModule,
)
from .twistor_emotional import (
    TwistorEmotionalConfig,
    TwistorEmotionalState,
    TwistorEmotionExtractor,
    TwistorEmotionModulator,
    TwistorEmotionMemory,
    TwistorEmotionalModule,
    TwistorCausalEmotionalLoop,
)
# ========== Multimodal (Möbius latent unification) ==========
from .multimodal_normalizer import (
    ModalityNormalizer,
    MultimodalNormalizer,
    create_multimodal_normalizer,
    MODALITY_TEXT,
    MODALITY_VISION,
    MODALITY_AUDIO,
    SUPPORTED_MODALITIES,
    DEFAULT_TAU_SCALE,
)
from .multimodal_bridge import (
    ModalityInjector,
    MobiusIsometryLoss,
    MultimodalBridge,
    mobius_geodesic_distance,
    mobius_raw_distance,
    mobius_retrieval_r1,
)
from .multimodal_heads import (
    TextHead,
    ContinuousHead,
    MultimodalHeads,
)
from .multimodal_model import MultimodalTwistorLMT
# ========== Twist Grow (扭量旋开生长) ==========
from .twist_grow import (
    TwistGrowCell,
    TwistGrowMultimodalTwistorLMT,
)

__version__ = "2.0.0"

__all__ = [
    # ========== Core ==========
    "TwistorLMT",
    # ========== Coupled Models ==========
    "CoupledTwistorLMT",
    "StackedCoupledLMT",
    # ========== Agent ==========
    "TwistorAgent",
    "TwistorAgentWithPolicy",
    "MultiAgent",
    # ========== Decoder ==========
    "TwistorDecoder",
    "TensorTwistorDecoder",
    "create_decoder",
    # ========== Integrators ==========
    "euler_step",
    "rk4_step",
    "RK4Integrator",
    "heun_step",
    "dopri5_step",
    "ODESolver",
    "AdjointODESolver",
    "TORCHDIFFEQ_AVAILABLE",
    "create_integrator",
    # ========== ODE Solver (Twistor-specific) ==========
    "TwistorODE",
    "ODEDynamics",
    "odeint_wrapper",
    "create_ode_solver",
    # ========== Analysis ==========
    "FixedPointFinder",
    "StabilityAnalyzer",
    "BifurcationAnalyzer",
    "analyze_model",
    # ========== Visualization ==========
    "plot_phase_space_2d",
    "plot_phase_space_3d",
    "plot_vector_field",
    "plot_tau_evolution",
    "plot_complex_plane",
    "plot_stability_analysis",
    "plot_training_diagnostics",
    # ========== Datasets ==========
    "generate_lorenz_dataset",
    "generate_mackey_glass_dataset",
    "generate_van_der_pol_dataset",
    "generate_sine_dataset",
    "create_dataset",
    # ========== Training ==========
    "train_model",
    "train_on_task",
    "plot_training_results",
    "plot_predictions",
    # ========== Mobius Manifold Constraint ==========
    "MobiusConstraint",
    "AdaptiveMobiusConstraint",
    "create_mobius_constraint",
    # ========== Resonance Attention ==========
    "TwistorResonance",
    "MultiHeadResonance",
    "create_resonance",
    # ========== Growable ==========
    "GrowableTwistorLMT",
    "create_growable_twistor_LMT",
    # ========== Self-Feedback ==========
    "SelfAwareTwistorLMT",
    "SelfFeedbackConfig",
    # ========== Reflection ==========
    "ReflectionModule",
    "ReflectionConfig",
    "ModificationRecord",
    # ========== Curiosity (Intrinsic Motivation) ==========
    "CuriosityModule",
    "CuriosityConfig",
    "IntrinsicRewardCalculator",
    # ========== Intervention Executor ==========
    "InterventionExecutor",
    "CausalInterventionEngine",
    "DoIntervention",
    # ========== Episodic Memory ==========
    "EpisodicMemory",
    "HippocampalSystem",
    "Episode",
    # ========== Offline Simulator ==========
    "OfflineSimulator",
    "DreamCortex",
    "SimulationConfig",
    # ========== Causal Learning Signal ==========
    "CausalLearningSignal",
    "NeuromodulatorySignal",
    # ========== Causal Cortex (Integration) ==========
    "CausalCortex",
    "CausalCortexConfig",
    "CausalSelfAwareTwistorLMT",
    # ========== Causal Reflection ==========
    "CausalReflectionModule",
    "CausalReflectionConfig",
    "CausalDecision",
    # ========== Causal Growable ==========
    "NeuromodulatedGrowthConfig",
    "CausalImportanceScore",
    "CausalImportanceEstimator",
    "NeuromodulationDrivenGrowth",
    "CausalGrowableWrapper",
    "AdaptiveDevelopmentalSchedule",
    # ========== Emotional System ==========
    "EmotionalState",
    "EmotionReaction",
    "EmotionalEncoder",
    "EmotionalModulator",
    "EmotionalMemory",
    "EmotionalLearning",
    "EmotionalSystem",
    "EmotionalCausalLoop",
    # ========== Twistor Causal Module ==========
    "TwistorCausalConfig",
    "PhaseCausalCalculator",
    "AmplitudeCausalCalculator",
    "TwistorInterventionExecutor",
    "TwistorOfflineSimulator",
    "TwistorCausalModule",
    # ========== Twistor Emotional Module ==========
    "TwistorEmotionalConfig",
    "TwistorEmotionalState",
    "TwistorEmotionExtractor",
    "TwistorEmotionModulator",
    "TwistorEmotionMemory",
    "TwistorEmotionalModule",
    "TwistorCausalEmotionalLoop",
    # ========== Multimodal ==========
    "ModalityNormalizer",
    "MultimodalNormalizer",
    "create_multimodal_normalizer",
    "ModalityInjector",
    "MobiusIsometryLoss",
    "MultimodalBridge",
    "mobius_geodesic_distance",
    "mobius_raw_distance",
    "mobius_retrieval_r1",
    "TextHead",
    "ContinuousHead",
    "MultimodalHeads",
    "MultimodalTwistorLMT",
    # ========== Twist Grow ==========
    "TwistGrowCell",
    "TwistGrowMultimodalTwistorLMT",
    "MODALITY_TEXT",
    "MODALITY_VISION",
    "MODALITY_AUDIO",
    "SUPPORTED_MODALITIES",
    "DEFAULT_TAU_SCALE",
]
