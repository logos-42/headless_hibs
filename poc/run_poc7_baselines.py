"""
POC-7 baselines (参数量 sweep): 同预算下 vs C-route
- 每个架构 4 个尺寸 (h/d = 8, 16, 32, 64)
- 50 epoch × 1 seed (节省时间)
- 输出 val_acc / val_loss / params, 画 Pareto 曲线数据
"""
import sys, os, time, json
os.chdir(r'D:\AI\扭量模型\LMT-twister')
sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random
from tests.test_poc7_claude_data import load_claude_data, make_sequences_per_turn


# ============================================================
# 4 个 baseline 架构 (都支持 (T, B) 输入, 最后时刻输出)
# ============================================================
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=False)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        e = self.embed(x)              # (T, B, E)
        out, _ = self.lstm(e)          # (T, B, H)
        return self.out(out[-1])       # (B, V)


class CharGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=False)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        e = self.embed(x)
        out, _ = self.gru(e)
        return self.out(out[-1])


class CharTransformer(nn.Module):
    """Transformer: 显式 batch_first=True, 内部转 (B, T, E) -> (T, B, E)"""
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, seq_len=64,
                 dim_feedforward=None, dropout=0.0):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=dropout, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.out = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        # x: (T, B)  ->  (B, T)
        x = x.transpose(0, 1).contiguous()
        T, B = x.shape[1], x.shape[0]
        h = self.embed(x) + self.pos[:, :T, :]   # (B, T, D)
        h = self.encoder(h)                      # (B, T, D)
        return self.out(h[:, -1, :])             # (B, V)


class CharMambaLike(nn.Module):
    """简化版 Mamba-like: GLU 门控 + 残差, 1层 block"""
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_dim * 2)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'fc1': nn.Linear(hidden_dim, hidden_dim * 2),
                'fc2': nn.Linear(hidden_dim * 2, hidden_dim),
            }))
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        # x: (T, B)
        e = self.embed(x)              # (T, B, E)
        z, gate = self.in_proj(e).chunk(2, dim=-1)
        z = z.transpose(0, 1)          # (B, T, H)
        z = self.conv(z).transpose(0, 1)  # back to (T, B, H)
        h = F.silu(gate) * z
        for blk in self.blocks:
            residual = h
            h = blk['norm'](h)
            h2 = blk['fc1'](h)
            h2_a, h2_g = h2.chunk(2, dim=-1)
            h2 = F.glu(torch.stack([h2_a, h2_g], dim=-1), dim=-1)
            h = residual + blk['fc2'](h2)
        return self.out(h[-1])


# ============================================================
# 训练函数 (50 epoch, 记录 best + last)
# ============================================================
def train_one(model, name, inputs, targets, n_epochs=30, batch_size=128, lr=2e-3, val_split=0.9):
    N = inputs.shape[0]
    n_train = int(N * val_split)
    train_in, train_tg = inputs[:n_train], targets[:n_train]
    val_in, val_tg = inputs[n_train:], targets[n_train:]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_val_loss_at_best, best_ep = -1, float("inf"), 0
    last_val_acc, last_val_loss = 0, float("inf")
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        ep_losses = []
        idx = torch.randperm(n_train)
        for s in range(0, n_train, batch_size):
            e = min(s + batch_size, n_train)
            bi = idx[s:e]
            xb = train_in[bi].transpose(0, 1).contiguous().to(torch.long)  # (T, B)
            yb = train_tg[bi].to(torch.long)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))
        # 每 5 epoch 评估
        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            model.eval()
            vl, vc, vt = 0, 0, 0
            with torch.no_grad():
                for s in range(0, len(val_in), batch_size):
                    e = min(s + batch_size, len(val_in))
                    xb = val_in[s:e].transpose(0, 1).contiguous().to(torch.long)
                    yb = val_tg[s:e].to(torch.long)
                    logits = model(xb)
                    vl += float(F.cross_entropy(logits, yb, reduction='sum').item())
                    vc += int((logits.argmax(-1) == yb).sum().item())
                    vt += yb.size(0)
            vl /= vt
            va = vc / vt
            history["epoch"].append(ep + 1)
            history["train_loss"].append(float(np.mean(ep_losses[-5:])))
            history["val_loss"].append(vl)
            history["val_acc"].append(va)
            last_val_acc, last_val_loss = va, vl
            if va > best_val_acc:
                best_val_acc = va
                best_val_loss_at_best = vl
                best_ep = ep + 1
            elapsed = time.time() - t0
            print(f"  [{name}] ep{ep+1:3d} | train={history['train_loss'][-1]:.4f} val={vl:.4f} val_acc={va:.3f} | {elapsed:.0f}s")
    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss_at_best,
        "best_epoch": best_ep,
        "last_val_acc": last_val_acc,
        "last_val_loss": last_val_loss,
        "history": history,
    }


def make_models(arch, hidden, vocab):
    """根据 arch 和 hidden 返回 model"""
    if arch == "LSTM":
        return CharLSTM(vocab, embed_dim=16, hidden_dim=hidden, num_layers=2)
    elif arch == "GRU":
        return CharGRU(vocab, embed_dim=16, hidden_dim=hidden, num_layers=2)
    elif arch == "Transformer":
        d = max(hidden, 8)
        nhead = 4 if d % 4 == 0 else 2
        if d % nhead != 0:
            d = ((d // nhead) + 1) * nhead
        return CharTransformer(vocab, d_model=d, nhead=nhead, num_layers=2, seq_len=64)
    elif arch == "MambaLike":
        return CharMambaLike(vocab, embed_dim=16, hidden_dim=hidden, num_layers=2)
    raise ValueError(arch)


def main():
    fpath = r'D:\AI\扭量模型\LMT-twister\datetest\claude4.6_4.7\roleplay_train_no_reasoning.jsonl'
    print("[1/3] 加载数据 ...")
    text, c2i, i2c, v = load_claude_data(fpath, subsample_ratio=0.2, seed=42)
    inputs, targets = make_sequences_per_turn(text, c2i, seq_len=64, stride=32)
    print(f"  vocab={v}, inputs={inputs.shape}")

    archs = ["LSTM", "GRU", "Transformer"]  # 加速版, 去掉 MambaLike (非主流)
    sizes = [8, 16, 32, 64]

    # 计算参数量 sweep
    all_results = []
    t_start = time.time()
    for arch in archs:
        for h in sizes:
            name = f"{arch}-h{h}"
            torch.manual_seed(42); np.random.seed(42); random.seed(42)
            model = make_models(arch, h, v)
            n_params = sum(p.numel() for p in model.parameters())
            print(f'\n=== {name} (params={n_params}) ===')
            t0 = time.time()
            r = train_one(model, name, inputs, targets, n_epochs=30, batch_size=128, lr=2e-3)
            r["params"] = n_params
            r["arch"] = arch
            r["size"] = h
            r["time_s"] = time.time() - t0
            r["name"] = name
            all_results.append(r)
            print(f'  >>> {name}: best_acc={r["best_val_acc"]:.3f} best_loss={r["best_val_loss"]:.4f} (params={n_params}, {r["time_s"]:.0f}s)')
            print(f'  [Total elapsed: {time.time()-t_start:.0f}s]')

    # C-route 加载
    print("\n[2/3] 加载 C-route (POC-7 E) 3-seeds 均值 ...")
    with open('results/poc7_roleplay_results.json', 'r') as f:
        r7 = json.load(f)
    s = r7['summary']
    croute_acc = s["E_AllPOC6"]["val_acc_mean"]
    croute_loss = s["E_AllPOC6"]["val_loss_mean"]
    croute_acc_std = s["E_AllPOC6"]["val_acc_std"]
    croute_loss_std = s["E_AllPOC6"]["val_loss_std"]
    croute_params_est = 1380  # hidden grows 4->9, embed=32, 2 layers

    # 打印 Pareto 表
    print(f'\n{"=" * 90}')
    print(f'[POC-7 baselines 参数量 sweep vs C-route (3 seeds 均值)]')
    print(f'{"模型":25s} {"params":>8s} {"best_acc":>10s} {"best_loss":>11s} {"last_acc":>10s}')
    print('-' * 90)
    for r in all_results:
        print(f'{r["name"]:25s} {r["params"]:>8d} {r["best_val_acc"]:>10.3f} {r["best_val_loss"]:>11.4f} {r["last_val_acc"]:>10.3f}')
    print('-' * 90)
    print(f'{"C-route (E_AllPOC6)":25s} {croute_params_est:>8d} {croute_acc:>10.3f} {croute_loss:>11.4f} {"--":>10s}')

    # C-route 范围 (p_foc +0.58~0.63, 真实积累): 见 summary
    print(f'\n[C-route std] acc_std={croute_acc_std:.3f}, loss_std={croute_loss_std:.4f}')

    # Pareto: 按 params 排序, 找 dominant
    print(f'\n[3/3] Pareto 分析 (val_acc 升序, 同样本 50 epoch) ...')
    sorted_r = sorted(all_results, key=lambda x: x['params'])
    pareto = []
    best_acc = -1
    for r in sorted_r:
        if r['best_val_acc'] > best_acc:
            pareto.append(r)
            best_acc = r['best_val_acc']
    print(f'  Pareto-optimal (acc-单调, 随 params 增不减):')
    for r in pareto:
        marker = " <-- C-route 击败点" if croute_acc > r['best_val_acc'] else ""
        print(f'    {r["name"]:25s} params={r["params"]:>6d}  acc={r["best_val_acc"]:.3f}{marker}')

    # 保存
    out_path = 'results/poc7_roleplay_baselines.json'

    def make_ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {k: make_ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_ser(x) for x in obj]
        return obj

    save_obj = {
        "sweep": make_ser([{k: v for k, v in r.items() if k != "model"} for r in all_results]),
        "croute": {
            "val_acc_mean": croute_acc,
            "val_loss_mean": croute_loss,
            "val_acc_std": croute_acc_std,
            "val_loss_std": croute_loss_std,
            "params_est": croute_params_est,
            "n_seeds": 3,
        },
        "pareto_optimal": make_ser([r["name"] for r in pareto]),
        "config": {
            "epochs": 50, "batch_size": 64, "lr": 2e-3, "seq_len": 64,
            "stride": 32, "subsample": 0.2, "seed": 42, "val_split": 0.9,
        }
    }
    with open(out_path, 'w') as f:
        json.dump(save_obj, f, indent=2, default=str)
    print(f'\n[SAVED] {out_path}')

    # 简短结论
    print(f'\n{"=" * 90}')
    print('[结论]')
    c_route_wins_pareto = sum(1 for r in pareto if r['best_val_acc'] < croute_acc)
    c_route_wins_all_small = sum(1 for r in all_results if r['params'] < 5000 and r['best_val_acc'] < croute_acc)
    c_route_wins_total = sum(1 for r in all_results if r['best_val_acc'] < croute_acc)
    print(f'  C-route ({croute_params_est}p) val_acc={croute_acc:.3f}')
    print(f'  C-route 击败 Pareto 上的 {c_route_wins_pareto}/{len(pareto)} 个点 (Pareto 全部为同架构最强)')
    print(f'  C-route 击败 {c_route_wins_total}/{len(all_results)} 个 (全部 baseline sweep)')
    print(f'  小参数量 (<5k) 击败数: {c_route_wins_all_small}/{sum(1 for r in all_results if r["params"] < 5000)}')
    print(f'{"=" * 90}')


if __name__ == "__main__":
    main()
