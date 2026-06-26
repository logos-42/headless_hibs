"""
完整流水线: 扫描用户笔记 → 训练 → Chat 演示
"""
import sys, os, json, math, re, time, glob
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAULT = Path(r"D:\Notesystem\忘心复盘笔记")
OUT_DIR = ROOT / "user_training"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 扫描并提取所有 .md 文件
# ============================================================
print("=" * 60)
print("  1. 扫描并提取笔记数据")
print("=" * 60)

md_files = sorted(VAULT.rglob("*.md"))
# Exclude files in .git, .obsidian, .kilo
md_files = [f for f in md_files if not any(
    p.startswith('.') for p in f.relative_to(VAULT).parts
)]
print(f"  找到 {len(md_files)} 个 .md 文件")

# 检查一下目录覆盖是否完整
dirs_found = set(f.parent.relative_to(VAULT).as_posix() for f in md_files)
print(f"  覆盖目录: {len(dirs_found)} 个")
for d in sorted(dirs_found):
    cnt = len([f for f in md_files if f.parent.relative_to(VAULT).as_posix() == d])
    print(f"    {d}/ : {cnt} 文件")

# ============================================================
# 2. 提取和清洗文本
# ============================================================
print("\n" + "=" * 60)
print("  2. 清洗文本数据")
print("=" * 60)

def clean_text(text):
    """最小清洗: 只移除图片引用和 frontmatter"""
    text = re.sub(r'^---[\s\S]*?---\n*', '', text)  # front matter
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)      # ![alt](path)
    text = re.sub(r'!\[\[.*?\]\]', '', text)          # ![[file]]
    text = re.sub(r'\n{3,}', '\n\n', text)            # 压缩空行
    return text.strip()

all_texts = []
empty_count = 0
for f in md_files:
    try:
        content = f.read_text(encoding='utf-8')
        cleaned = clean_text(content)
        if len(cleaned) > 20:  # 至少 20 个字符才有意义
            all_texts.append(cleaned)
        else:
            empty_count += 1
    except Exception as e:
        print(f"  ⚠️ 读取失败: {f.name}: {e}")

print(f"  清洗后有效文本: {len(all_texts)} 条")
print(f"  跳过空/过短: {empty_count} 条")
total_chars = sum(len(t) for t in all_texts)
print(f"  总字符数: {total_chars:,} ({total_chars/1024:.1f} KB)")

if total_chars < 1000:
    print("  ❌ 数据太少，无法训练")
    sys.exit(1)

# 显示样本
print("\n  样本预览:")
for i in range(min(2, len(all_texts))):
    preview = all_texts[i][:200]
    print(f"  --- 样本 {i+1} ({len(all_texts[i])} chars) ---")
    print(f"  {preview}")
    print()

# ============================================================
# 3. 保存为 JSONL
# ============================================================
print("=" * 60)
print("  3. 保存训练/验证数据")
print("=" * 60)

import random
random.seed(42)

# 按长度排序，取前 20% 作为验证集
all_texts.sort(key=len, reverse=True)
split_idx = max(1, len(all_texts) // 5)
val_texts = all_texts[:split_idx]
train_texts = all_texts[split_idx:]

random.shuffle(train_texts)

train_file = OUT_DIR / "train.jsonl"
val_file = OUT_DIR / "val.jsonl"

with open(train_file, "w", encoding="utf-8") as f:
    for t in train_texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

with open(val_file, "w", encoding="utf-8") as f:
    for t in val_texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

print(f"  训练集: {len(train_texts)} 条 -> {train_file}")
print(f"  验证集: {len(val_texts)} 条 -> {val_file}")

# ============================================================
# 4. 训练
# ============================================================
print("\n" + "=" * 60)
print("  4. 训练 Hibs 模型")
print("=" * 60)

import torch
from torch.utils.data import DataLoader, Dataset
from scripts.train_hibs_0_16 import BPETokenizerWrapper
from hibs_lnn.v16_6_50m import Hibs_0_16_50M, Hibs_0_16_50M_Config

# --- 先扫描建立固定词表 ---
all_chars = set()
for t in all_texts:
    all_chars.update(t)
vocab_size = min(len(all_chars) + 5, 5000)
print(f"  唯一字符: {len(all_chars)} -> vocab_size={vocab_size}")

class FixedCharTokenizer:
    """固定词表的字符级 tokenizer"""
    def __init__(self, chars_set, max_vocab=5000):
        self.char_to_id = {}
        self.id_to_char = {}
        self.char_to_id['\x00'] = 0  # padding
        self.char_to_id['\x01'] = 1  # unknown
        self.id_to_char[0] = '\x00'
        self.id_to_char[1] = '\x01'
        for i, c in enumerate(sorted(chars_set), start=2):
            if i >= max_vocab - 1:
                break
            self.char_to_id[c] = i
            self.id_to_char[i] = c
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text):
        return [self.char_to_id.get(c, 1) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

tokenizer = FixedCharTokenizer(all_chars, max_vocab=vocab_size)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx])
        ids = ids[:self.max_seq_len + 1]
        if len(ids) < self.max_seq_len + 1:
            ids = ids + [0] * (self.max_seq_len + 1 - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

# --- 中型配置 ---
model_cfg = Hibs_0_16_50M_Config(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_layers=2,
    d_state=8,
    max_seq_len=64,
    conv_kernel=4,
)
device = "cpu"
model = Hibs_0_16_50M(model_cfg).to(device)
print(f"  模型参数量: {model.num_params()/1e6:.2f}M")

# 数据集
train_ds = TextDataset(train_texts, tokenizer, model_cfg.max_seq_len)
val_ds = TextDataset(val_texts, tokenizer, model_cfg.max_seq_len)
print(f"  训练 batch: {len(train_ds)} 条, 验证: {len(val_ds)} 条")

batch_size = 4
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 训练
n_epochs = 3
print(f"\n  开始训练 {n_epochs} epochs...")
print("-" * 50)

t0 = time.time()
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    train_ppl = math.exp(avg_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    val_ppl = math.exp(avg_val_loss)
    
    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1:2d}/{n_epochs}: train loss={avg_loss:.4f} (ppl={train_ppl:.2f}) | val loss={avg_val_loss:.4f} (ppl={val_ppl:.2f}) | {elapsed:.0f}s")

elapsed_total = time.time() - t0
print(f"\n  训练完成! 耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f} 分钟)")

# 保存模型
ckpt_path = ROOT / "checkpoints" / "user_trained_notes.pt"
torch.save({
    "model": model.state_dict(),
    "config": model_cfg.__dict__,
    "epochs": n_epochs,
    "final_train_ppl": train_ppl,
    "final_val_ppl": val_ppl,
}, ckpt_path)
print(f"  模型保存: {ckpt_path}")

# ============================================================
# 5. Chat 演示
# ============================================================
print("\n" + "=" * 60)
print("  5. 对话演示")
print("=" * 60)

tok = tokenizer  # 使用训练时的固定词表 tokenizer

def chat(prompt, max_new_tokens=100, temperature=0.8, top_k=30, top_p=0.85):
    """用训练好的模型生成回复"""
    model.eval()
    ids = tok.encode(prompt)
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        ids_cond = ids_tensor if ids_tensor.size(1) <= model_cfg.max_seq_len else ids_tensor[:, -model_cfg.max_seq_len:]
        logits = model(ids_cond)[:, -1, :] / max(temperature, 1e-5)
        
        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_mask = cumprobs > top_p
            sorted_mask[:, 0] = False
            mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
            logits[mask] = float('-inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids_tensor = torch.cat([ids_tensor, next_id], dim=1)
    
    return tok.decode(ids_tensor[0].tolist())

test_prompts = [
    "我今天心情不太好",
    "今天想写点什么",
    "我对未来感到迷茫",
    "关于创业我有些想法",
    "学习AI的时候遇到了困难",
]

print()
for prompt in test_prompts:
    response = chat(prompt, max_new_tokens=60)
    print(f"  你: {prompt}")
    # Extract just the generated part (after prompt)
    if response.startswith(prompt):
        generated = response[len(prompt):]
    else:
        generated = response
    print(f"  江木: {generated[:150]}")
    print()
    
print("=" * 60)
print("  ✅ 训练完成，模型可以对话!")
print("=" * 60)
