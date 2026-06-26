"""
Hibs-LMT CLI 工具
===================

提供简单对话/服务/导出/训练命令.

安装:
    pip install -e .
    # 或直接运行:
    python -m hibs_cli chat --ckpt checkpoints/hibs_0_16_best.pt

命令:
    hibs chat     交互式对话
    hibs serve    启动 HTTP API 服务
    hibs export   导出多端格式
    hibs train    训练模型
    hibs info     查看模型信息
"""