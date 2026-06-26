"""
Hibs CLI 推理引擎 (单例, 全平台)
================================

支持加载:
  - PyTorch checkpoint (.pt)
  - TorchScript (.pt)
  - ONNX (.onnx)
  - PyTorch Mobile Lite (.ptl)
"""
import os
import sys
from pathlib import Path
from typing import Optional

import torch


ROOT = Path(__file__).parent.parent


class InferenceEngine:
    """
    统一推理引擎, 自动适配 checkpoint 格式.
    """

    def __init__(self, model_path: str, device: str = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.format = self._detect_format()
        self.model = None
        self.tokenizer = None
        self.config = None

        print(f"加载模型: {self.model_path}")
        print(f"  格式: {self.format}")
        print(f"  设备: {self.device}")

        self._load()

    def _detect_format(self) -> str:
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        elif suffix == ".ptl":
            return "mobile"
        else:
            return "torchscript"

    def _load(self):
        if self.format == "torchscript":
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                # 标准 checkpoint
                from hibs_LMT.v16_6_50m import Hibs_0_16_50M, Hibs_0_16_50M_Config
                cfg = Hibs_0_16_50M_Config(**ckpt["config"])
                self.model = Hibs_0_16_50M(cfg).to(self.device)
                self.model.load_state_dict(ckpt["model"])
                self.config = cfg
            else:
                # TorchScript
                self.model = ckpt.to(self.device)
            self.model.eval()
        elif self.format == "onnx":
            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                self.model = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)
            except ImportError:
                raise ImportError("ONNX 推理需要 onnxruntime: pip install onnxruntime")
        elif self.format == "mobile":
            self.model = torch.jit.load(str(self.model_path))
            self.model.eval()

        # Tokenizer (暂用字符级, 后续替换为 BPE)
        from hibs_cli.tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """生成文本"""
        ids = self.tokenizer.encode(prompt)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)

        if self.format in ("torchscript", "mobile"):
            for _ in range(max_new_tokens):
                ids_cond = ids_tensor if ids_tensor.size(1) <= 1024 else ids_tensor[:, -1024:]
                logits = self.model(ids_cond)[:, -1, :] / max(temperature, 1e-5)

                # Top-k 过滤
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-p 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    sorted_mask = cumprobs > top_p
                    sorted_mask[:, 0] = False
                    mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                    logits[mask] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids_tensor = torch.cat([ids_tensor, next_id], dim=1)

            return self.tokenizer.decode(ids_tensor[0].tolist())
        elif self.format == "onnx":
            # ONNX 生成循环
            for _ in range(max_new_tokens):
                ids_cond = ids_tensor.numpy() if hasattr(ids_tensor, 'numpy') else ids_tensor
                if ids_cond.shape[1] > 1024:
                    ids_cond = ids_cond[:, -1024:]
                logits = self.model.run(None, {"input_ids": ids_cond.astype("int64")})[0]
                logits = torch.tensor(logits[:, -1, :]) / max(temperature, 1e-5)
                # ... (同上采样逻辑)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids_tensor = torch.cat([ids_tensor, next_id], dim=1)
            return self.tokenizer.decode(ids_tensor[0].tolist())

    def info(self) -> dict:
        """返回模型信息"""
        info = {
            "path": str(self.model_path),
            "format": self.format,
            "device": self.device,
            "size_mb": self.model_path.stat().st_size / 1e6,
        }
        if self.config:
            info["vocab_size"] = self.config.vocab_size
            info["d_model"] = self.config.d_model
            info["n_layers"] = self.config.n_layers
        return info