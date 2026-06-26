"""
轻量级 tokenizer 包装
====================

优先级:
  1. tiktoken (OpenAI BPE, 推荐)
  2. sentencepiece (Google)
  3. 字符级 fallback

训练数据 jsonl 中应保存 tokenizer 配置到:
  data/tokenizer_config.json: {"type": "tiktoken", "name": "cl100k_base"}
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_TOKENIZER_CONFIG = ROOT / "data" / "tokenizer_config.json"


class CharTokenizer:
    """字符级 tokenizer (fallback)"""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def encode(self, text: str) -> list:
        ids = []
        for c in text:
            if c not in self.char_to_id:
                self.char_to_id[c] = len(self.char_to_id)
                self.id_to_char[len(self.id_to_char)] = c
            ids.append(self.char_to_id[c])
        return ids

    def decode(self, ids: list) -> str:
        return "".join(self.id_to_char.get(i, "?") for i in ids)


class TiktokenWrapper:
    """tiktoken 包装"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.max_token_value + 1
        self.name = encoding_name

    def encode(self, text: str) -> list:
        return self.enc.encode(text)

    def decode(self, ids: list) -> str:
        return self.enc.decode(ids)


class SentencePieceWrapper:
    """sentencepiece 包装"""

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str) -> list:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list) -> str:
        return self.sp.DecodeIds(ids)


def get_tokenizer(config_path: str = None):
    """自动选择 tokenizer"""
    config_path = Path(config_path) if config_path else DEFAULT_TOKENIZER_CONFIG

    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            kind = cfg.get("type")
            if kind == "tiktoken":
                return TiktokenWrapper(cfg.get("name", "cl100k_base"))
            elif kind == "sentencepiece":
                return SentencePieceWrapper(cfg["path"])
        except Exception as e:
            print(f"加载 tokenizer 配置失败: {e}, 使用字符级 fallback")

    # 字符级 fallback
    print("⚠️ 使用字符级 tokenizer (vocab=256), 推荐配置 tiktoken")
    return CharTokenizer()