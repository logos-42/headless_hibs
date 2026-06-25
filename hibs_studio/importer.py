"""
个人数据导入器
==============

支持格式:
  - .txt  - 纯文本
  - .md   - Markdown
  - .pdf  - PDF 文档
  - .docx - Word 文档
  - .csv  - CSV (列名: text / content / message)
  - .json - JSON 数组 (元素含 text/content/message 字段)

输出: 标准 JSONL 训练格式 {text: "..."}
"""
import csv
import json
import os
import re
from pathlib import Path
from typing import List, Callable, Optional


def read_txt(path: str) -> str:
    """读取纯文本"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_md(path: str) -> str:
    """读取 Markdown (去除元数据)"""
    text = read_txt(path)
    # 去除 frontmatter
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    # 去除代码块 (可选: 也可以保留)
    # text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return text


def read_pdf(path: str) -> str:
    """读取 PDF"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() for page in reader.pages)
    except ImportError:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            return "\n\n".join(page.extract_text() for page in reader.pages)
        except ImportError:
            raise ImportError("需要安装 pypdf 或 PyPDF2: pip install pypdf")


def read_docx(path: str) -> str:
    """读取 Word 文档"""
    try:
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise ImportError("需要安装 python-docx: pip install python-docx")


def read_csv(path: str) -> List[str]:
    """读取 CSV (text/content/message 列)"""
    texts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 尝试多个字段名
            text = (
                row.get("text")
                or row.get("content")
                or row.get("message")
                or row.get("内容")
                or row.get("文本")
                or ""
            )
            if text:
                texts.append(text)
    return texts


def read_json(path: str) -> List[str]:
    """读取 JSON (数组格式)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [
            item.get("text") or item.get("content") or item.get("message") or str(item)
            for item in data
        ]
    elif isinstance(data, dict):
        # 尝试嵌套
        if "messages" in data:
            return [m.get("content", "") for m in data["messages"]]
        return [str(data)]
    return []


def read_file(path: str) -> List[str]:
    """根据扩展名读取单个文件, 返回文本段列表"""
    ext = Path(path).suffix.lower()

    if ext == ".txt":
        return [read_txt(path)]
    elif ext == ".md":
        return [read_md(path)]
    elif ext == ".pdf":
        return [read_pdf(path)]
    elif ext == ".docx":
        return [read_docx(path)]
    elif ext == ".csv":
        return read_csv(path)
    elif ext == ".json":
        return read_json(path)
    else:
        # 尝试作为文本
        return [read_txt(path)]


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    把长文本切分成训练块.

    策略:
      - 优先按段落切分 (\n\n)
      - 段落过长时按句子切
      - 保持 chunk_size 字符左右
    """
    # 按段落切
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if len(current) + len(p) + 2 <= chunk_size:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            # 段落本身过长, 按句子切
            if len(p) > chunk_size:
                sentences = re.split(r"(?<=[.!?。!?])\s+", p)
                current = ""
                for s in sentences:
                    if len(current) + len(s) + 1 <= chunk_size:
                        current = (current + " " + s).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = s
            else:
                current = p

    if current:
        chunks.append(current)

    return chunks


def convert_to_training_format(
    file_paths: List[str],
    output_path: str,
    chunk_size: int = 500,
    min_chunk_size: int = 50,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    把多个文件转换为统一训练格式 (JSONL).

    Args:
        file_paths: 输入文件列表
        output_path: 输出 JSONL 路径
        chunk_size: 每块字符数
        min_chunk_size: 最小块字符数 (过滤太短)
        progress_callback: 进度回调 (current, total, message)

    Returns:
        输出文件路径
    """
    all_chunks = []
    total_files = len(file_paths)

    for i, fp in enumerate(file_paths):
        if progress_callback:
            progress_callback(i, total_files, f"读取 {Path(fp).name}...")

        try:
            segments = read_file(fp)
        except Exception as e:
            print(f"⚠️ 读取 {fp} 失败: {e}")
            continue

        for seg in segments:
            if len(seg) < min_chunk_size:
                continue
            chunks = split_into_chunks(seg, chunk_size=chunk_size)
            all_chunks.extend(chunks)

    # 去重
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        # 用 hash 简单去重
        h = hash(c.strip())
        if h not in seen:
            seen.add(h)
            unique_chunks.append(c.strip())

    # 写入 JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in unique_chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    if progress_callback:
        progress_callback(total_files, total_files, f"完成: {len(unique_chunks)} 段")

    return str(output_path)


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="个人数据转换工具")
    parser.add_argument("inputs", nargs="+", help="输入文件 (支持通配符)")
    parser.add_argument("-o", "--output", default="train.jsonl")
    parser.add_argument("-c", "--chunk-size", type=int, default=500)
    args = parser.parse_args()

    import glob
    files = []
    for inp in args.inputs:
        if "*" in inp:
            files.extend(glob.glob(inp))
        else:
            files.append(inp)

    print(f"输入: {len(files)} 个文件")
    out = convert_to_training_format(files, args.output, chunk_size=args.chunk_size)
    print(f"✅ 输出: {out}")