"""
hibs 0.16 (基于 hibs 0.16) 多端导出工具
====================

支持导出格式:
  - torchscript:  TorchScript (PyTorch 全平台)
  - onnx:         ONNX (跨框架推理)
  - mobile:       PyTorch Mobile (iOS/Android)
  - coreml:       CoreML (iOS 原生)
  - quantized:    INT8 动态量化 (减小体积)

使用方法:
    python export/export_v16_6_50m.py --ckpt checkpoints/hibs_0_16_best.pt --format onnx
    python export/export_v16_6_50m.py --ckpt checkpoints/hibs_0_16_best.pt --format quantized
    python export/export_v16_6_50m.py --ckpt checkpoints/hibs_0_16_best.pt --format mobile
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hibs_LMT.v16_6_50m import Hibs_0_16_50M, Hibs_0_16_50M_Config


def load_model(ckpt_path: str) -> Hibs_0_16_50M:
    """加载 checkpoint"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    cfg = Hibs_0_16_50M_Config(**cfg_dict)
    model = Hibs_0_16_50M(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"加载模型: {ckpt_path}")
    print(f"  参数: {model.num_params():,}")
    print(f"  Epoch: {ckpt.get('epoch', '?')} | Val PPL: {ckpt.get('val_ppl', '?')}")
    return model, cfg


def export_torchscript(model, cfg, output_dir):
    """导出 TorchScript"""
    out = Path(output_dir) / "hibs_0_16.pt"
    traced = torch.jit.trace(model, torch.zeros(1, 32, dtype=torch.long))
    traced.save(str(out))
    print(f"  ✅ TorchScript: {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def export_onnx(model, cfg, output_dir):
    """导出 ONNX (实数化包装, 因为 ONNX 不支持 complex64)

    复数 SSM 的 logits 输出是实数 (forward 末尾 .real), 所以 ONNX 兼容.
    """
    try:
        out = Path(output_dir) / "hibs_0_16.onnx"
        dummy = torch.zeros(1, 32, dtype=torch.long)

        # 临时切换 complex64 -> float32 的 context
        # 由于 forward 内部用 complex64, ONNX 导出时 PyTorch 会报错
        # 解法: 用 forward pre-hook 临时禁用 complex, 或用纯实数 fallback
        torch.onnx.export(
            model,
            (dummy,),
            str(out),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  ✅ ONNX: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return out
    except Exception as e:
        err_msg = str(e)
        if "complex" in err_msg.lower():
            print(f"  ⚠️ ONNX 不支持 complex64 SSM (V16.6 核心特征)")
            print(f"     建议: 用 TorchScript 或 PyTorch Mobile 部署")
            print(f"     或: 实现纯实数 SSM 版本 (需重写 forward)")
        else:
            print(f"  ❌ ONNX 导出失败: {e}")
        return None


def export_mobile(model, cfg, output_dir):
    """导出 PyTorch Mobile (iOS/Android Lite Interpreter)"""
    try:
        out = Path(output_dir) / "hibs_0_16_mobile.ptl"
        scripted = torch.jit.script(model)
        # 优化移动端
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(scripted)
        optimized._save_for_lite_interpreter(str(out))
        print(f"  ✅ Mobile: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return out
    except Exception as e:
        print(f"  ❌ Mobile 导出失败: {e}")
        return None


def export_coreml(model, cfg, output_dir):
    """导出 CoreML (iOS 原生)"""
    try:
        import coremltools as ct
        out = Path(output_dir) / "hibs_0_16.mlpackage"

        dummy = torch.zeros(1, 32, dtype=torch.long)
        traced = torch.jit.trace(model, dummy)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=dummy.shape, dtype=int)],
            convert_to="mlprogram",
        )
        mlmodel.save(str(out))
        print(f"  ✅ CoreML: {out}")
        return out
    except ImportError:
        print(f"  ⚠️ coremltools 未安装, 跳过 CoreML 导出")
        print(f"     安装: pip install coremltools")
        return None
    except Exception as e:
        print(f"  ❌ CoreML 导出失败: {e}")
        return None


def quantize_int8_dynamic(model, cfg, output_dir):
    """INT8 动态量化"""
    try:
        out = Path(output_dir) / "hibs_0_16_int8.pt"

        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d},
            dtype=torch.qint8,
        )
        torch.save({
            "model": quantized.state_dict(),
            "config": cfg.__dict__,
            "quantization": "int8_dynamic",
        }, out)

        size_mb = out.stat().st_size / 1e6
        print(f"  ✅ INT8 量化: {out} ({size_mb:.1f} MB)")

        # 简单敏感性测试
        test_quantization_sensitivity(model, quantized, cfg)

        return out
    except Exception as e:
        print(f"  ❌ INT8 量化失败: {e}")
        return None


def test_quantization_sensitivity(fp32_model, int8_model, cfg):
    """
    量化敏感性测试: 同一输入, 比较输出差异.
    hibs 0.16 复值 κ 对精度敏感, 必须验证.
    """
    print("\n  量化敏感性测试:")
    fp32_model.eval()
    int8_model.eval()

    # 随机输入
    torch.manual_seed(42)
    test_ids = torch.randint(0, cfg.vocab_size, (2, 64))

    with torch.no_grad():
        fp32_out = fp32_model(test_ids)
        int8_out = int8_model(test_ids)

    # 计算差异
    diff = (fp32_out - int8_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (fp32_out.abs() + 1e-6)).mean().item()

    print(f"    max_abs_diff:  {max_diff:.4f}")
    print(f"    mean_abs_diff: {mean_diff:.4f}")
    print(f"    mean_rel_diff: {rel_diff*100:.2f}%")

    if rel_diff < 0.05:
        print(f"    ✅ 量化友好 (< 5% 相对误差)")
    elif rel_diff < 0.10:
        print(f"    ⚠️ 量化有损 (5-10% 相对误差), 建议 PTQ 校准")
    else:
        print(f"    ❌ 量化破坏严重 (> 10% 相对误差), 考虑 QAT 或保持 FP32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径")
    parser.add_argument(
        "--format",
        choices=["torchscript", "onnx", "mobile", "coreml", "quantized", "all"],
        default="all",
    )
    parser.add_argument("--output", default="exported", help="输出目录")
    args = parser.parse_args()

    output_dir = ROOT / args.output
    output_dir.mkdir(exist_ok=True)

    print(f"=" * 70)
    print(f"hibs 0.16 (基于 hibs 0.16) 多端导出")
    print(f"=" * 70)

    model, cfg = load_model(args.ckpt)
    print(f"\n导出目录: {output_dir}")

    formats = (
        ["torchscript", "onnx", "mobile", "coreml", "quantized"]
        if args.format == "all"
        else [args.format]
    )

    for fmt in formats:
        print(f"\n[{fmt}]")
        t0 = time.time()

        if fmt == "torchscript":
            export_torchscript(model, cfg, output_dir)
        elif fmt == "onnx":
            export_onnx(model, cfg, output_dir)
        elif fmt == "mobile":
            export_mobile(model, cfg, output_dir)
        elif fmt == "coreml":
            export_coreml(model, cfg, output_dir)
        elif fmt == "quantized":
            quantize_int8_dynamic(model, cfg, output_dir)

        print(f"  耗时: {time.time()-t0:.1f}s")

    print(f"\n✅ 导出完成! 文件位于: {output_dir}/")


if __name__ == "__main__":
    main()