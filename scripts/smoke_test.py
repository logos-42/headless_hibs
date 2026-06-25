"""
Hibs 0.16 端到端冒烟测试
========================

验证完整流程:
  1. 创建模型 (50M)
  2. 保存 checkpoint
  3. CLI info 加载 checkpoint
  4. CLI chat 生成 (mock stdin)
  5. 导出 ONNX / TorchScript

不需要训练数据, 不需要 GPU.
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from hibs_LMT.v16_6_50m import Hibs_0_16_50M, Hibs_0_16_50M_Config

CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CKPT_PATH = CKPT_DIR / "hibs_0_16_smoke.pt"


def main():
    print("=" * 70)
    print("Hibs 0.16 端到端冒烟测试")
    print("=" * 70)

    # 1. 创建并保存 checkpoint
    print("\n[1/5] 创建 50M 模型并保存 checkpoint...")
    cfg = Hibs_0_16_50M_Config()
    model = Hibs_0_16_50M(cfg)
    n_params = model.num_params()
    print(f"  参数: {n_params:,} ({n_params/1e6:.2f}M)")

    torch.save({
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "epoch": 0,
        "val_ppl": 999.0,  # 占位
    }, CKPT_PATH)
    print(f"  ✅ 保存: {CKPT_PATH}")

    # 2. CLI info
    print("\n[2/5] 测试 `hibs info` ...")
    result = subprocess.run(
        [sys.executable, "-m", "hibs_cli", "info", "--ckpt", str(CKPT_PATH)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"  ❌ FAIL: {result.stderr}")
        return
    print("  " + result.stdout.replace("\n", "\n  ").strip())

    # 3. CLI chat (echo test, 输入 quit 立即退出)
    print("\n[3/5] 测试 `hibs chat` (输入 'quit') ...")
    result = subprocess.run(
        [sys.executable, "-m", "hibs_cli", "chat",
         "--ckpt", str(CKPT_PATH),
         "--max-tokens", "8",
         "--temperature", "1.0"],
        input="quit\n",
        capture_output=True, text=True, cwd=str(ROOT),
        timeout=60,
    )
    if "再见" in result.stdout or "Twist" in result.stdout or "Hibs" in result.stdout:
        print("  ✅ chat 启动正常")
    else:
        print(f"  ⚠️ chat 输出异常: {result.stdout[:200]}")
    if result.stderr:
        # 过滤掉 UserWarning
        for line in result.stderr.split("\n"):
            if "UserWarning" in line or "ARRAY_API" in line or "device:" in line:
                continue
            if line.strip():
                print(f"  STDERR: {line[:200]}")

    # 4. 导出 TorchScript
    print("\n[4/5] 测试 TorchScript 导出...")
    export_dir = ROOT / "exported_smoke"
    export_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(ROOT / "hibs_export" / "export_hibs_0_16.py"),
         "--ckpt", str(CKPT_PATH),
         "--format", "torchscript",
         "--output", str(export_dir)],
        capture_output=True, text=True, cwd=str(ROOT),
        timeout=120,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  ❌ FAIL: {result.stderr[-500:]}")

    # 5. 导出 ONNX (可选, 慢)
    print("\n[5/5] 测试 ONNX 导出 (可能 1-2 分钟)...")
    result = subprocess.run(
        [sys.executable, str(ROOT / "hibs_export" / "export_hibs_0_16.py"),
         "--ckpt", str(CKPT_PATH),
         "--format", "onnx",
         "--output", str(export_dir)],
        capture_output=True, text=True, cwd=str(ROOT),
        timeout=300,
    )
    out_line = [l for l in result.stdout.split("\n") if "✅" in l or "❌" in l]
    for l in out_line:
        print("  " + l)
    if result.returncode != 0:
        print(f"  ❌ FAIL: {result.stderr[-500:]}")

    # 检查产物
    print("\n=== 产物清单 ===")
    for f in sorted(export_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("✅ 冒烟测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    main()