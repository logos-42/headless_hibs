"""
Hibs-LMT hibs 0.16 参数版本 (基于 V16.6 验证架构)
============================

继承自 V16.6 验证有效的核心组件:
  - 复数对角 SSM (ZOH 解析解)
  - 复值 κ (amplitude + phase modulation)
  - 扭量纤维丛 (HibsFiberBundle)

50M 扩展要点:
  - d_model: 128 -> 768
  - n_layers: 2-3 -> 12
  - d_state: 8 -> 16
  - vocab: 109 (char) -> 8192 (BPE)
  - 总参数: 247K -> ~50M
"""

from .model import Hibs_0_16_50M, Hibs_0_16_50M_Config
from .layer import SSM_Layer_V16_6_Scaled
from .fiber import HibsFiberBundleV2_Scaled

__all__ = [
    "Hibs_0_16_50M",
    "Hibs_0_16_50M_Config",
    "SSM_Layer_V16_6_Scaled",
    "HibsFiberBundleV2_Scaled",
]