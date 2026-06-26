"""
LLM API 实验: Qwen3-235B-A22B (dmxapi) + MiniMax-M3 (MiniMax 官方) vs C-route
- 跑与 POC-7 同样的 roleplay 数据
- 任务: 给定前 N 字符, 续写 K 字符
- 指标: 字符级 top-1 准确率
"""
import os
import sys
import json
import time
import random
import requests

# 关系统代理, 直连
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# 双 API 配置
# ============================================================
QWEN_KEY = os.environ.get("DMXAPI_QWEN_KEY", "")
MM_KEY = os.environ.get("DMXAPI_MM_KEY", "")

QWEN_API = {
    "base": "https://www.dmxapi.cn/v1",
    "endpoint": "/chat/completions",
    "model": "qwen3-235b-a22b",
    "key": QWEN_KEY,
    "timeout": 180,  # qwen 思考模式慢
}

MM_API = {
    "base": "https://api.minimaxi.com",
    "endpoint": "/v1/chat/completions",
    "model": "MiniMax-Text-01",  # 官方推荐
    "key": MM_KEY,
    "timeout": 60,
}

APIS = {
    "Qwen3-235B-A22B (dmxapi)": QWEN_API,
    "MiniMax-M3 (MiniMax 官方)": MM_API,
}

DATA_PATH = r'D:\AI\扭量模型\LMT-twister\datetest\claude4.6_4.7\roleplay_train_no_reasoning.jsonl'

# 参考值
C_ROUTE_ACC = 0.485
C_ROUTE_ACC_STD = 0.003
C_ROUTE_PARAMS = 5900  # 真实 active params (hidden grows 4->9, embed 32, 2-layer LSTM, +meta/personality)
C_ROUTE_NOTE = "POC-7 E_AllPOC6, 3 seeds, 50 epoch"
LSTM_REFS = {
    "LSTM-h8 (1 seed)":  (4033,  0.382, "50 epoch, CPU, 1 seed"),
    "LSTM-h16 (1 seed)": (7817,  0.425, "50 epoch, CPU, 1 seed"),
    "LSTM-h32 (1 seed)": (19993, 0.453, "50 epoch, CPU, 1 seed"),
    "LSTM-h64 (1 seed)": (62777, 0.480, "50 epoch, CPU, 1 seed"),
}


def chat(api_cfg, messages, max_tokens=64, temperature=0.0, max_retries=3):
    url = api_cfg["base"] + api_cfg["endpoint"]
    headers = {
        "Authorization": f"Bearer {api_cfg['key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": api_cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=api_cfg["timeout"])
            dt = time.time() - t0
            if r.status_code != 200:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 退避
                    continue
                return {"error": err, "elapsed_s": dt, "retries": attempt + 1}
            data = r.json()
            content = ""
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                content = json.dumps(data)[:200]
            usage = data.get("usage", {})
            return {
                "content": content,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "elapsed_s": dt,
                "retries": attempt,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return {"error": str(e)[:200], "elapsed_s": time.time() - t0, "retries": attempt + 1}
    return {"error": "max retries", "elapsed_s": 0}


def load_roleplay_corpus():
    pieces = []
    for line in open(DATA_PATH, 'r', encoding='utf-8'):
        d = json.loads(line)
        for msg in d.get('messages', []):
            role = msg.get('role', '')
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(str(x) for x in content)
            if role == 'system':
                pieces.append(f"<|system|>\n{content}")
            elif role == 'user':
                pieces.append(f"<|user|>\n{content}")
            elif role == 'assistant':
                pieces.append(f"<|assistant|>\n{content}<|end|>")
    full = "".join(pieces)
    print(f"  加载 roleplay 文本: {len(full)} 字符, {len(pieces)} pieces")
    return full


def connectivity_test():
    print("=" * 80)
    print("[1/3] 连通性测试 (1 token)")
    print("=" * 80)
    for name, api in APIS.items():
        if not api["key"]:
            print(f"  [{name}] ⚠️  key 为空, 跳过")
            continue
        r = chat(api, [{"role": "user", "content": "回'好'"}], max_tokens=5)
        if "error" in r:
            print(f"  [{name}] ❌ ERR: {r['error'][:200]}")
        else:
            print(f"  [{name}] ✅ reply: '{r['content'][:30]}' "
                  f"(pt={r.get('prompt_tokens')}, ct={r.get('completion_tokens')}, {r['elapsed_s']:.1f}s)")


def ntp_eval(name, api_cfg, full_text, n_samples=15, seq_len=128, n_pred=10, min_success=5):
    print(f"\n[2/3] [{name}] roleplay NTP (n_samples={n_samples}, seq_len={seq_len}, n_pred={n_pred})")
    rng = random.Random(42)
    samples = []
    for _ in range(n_samples):
        i = rng.randint(0, len(full_text) - seq_len - n_pred - 1)
        samples.append((full_text[i:i+seq_len], full_text[i+seq_len:i+seq_len+n_pred]))

    system_prompt = (
        "你是一个角色扮演文本续写器. "
        "给定一段对话, 续写最可能的下一段. "
        "只输出续写内容, 不要任何解释或前缀."
    )

    correct = 0
    total = 0
    detailed = []
    errors = 0
    t_start = time.time()
    for i, (ctx, tgt) in enumerate(samples):
        prompt = f"续写以下对话 (输出约 {n_pred} 字符):\n\n{ctx}\n\n续写:"
        r = chat(
            api_cfg,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max(n_pred * 3, 32),
            temperature=0.0,
        )
        if "error" in r:
            errors += 1
            if errors <= 3 or (errors % 5 == 0):
                print(f"  [{i+1}/{n_samples}] ERR: {r['error'][:80]} (累计 {errors})")
            continue
        out = r.get("content", "").strip()
        clean = out.replace("\n", "").replace(" ", "").replace("\"", "").replace("'", "").replace(":", "")
        ctx_tail = ctx[-15:].replace("\n", "").replace(" ", "")
        if clean.startswith(ctx_tail):
            clean = clean[len(ctx_tail):]
        pred = clean[:n_pred]
        match = sum(1 for a, b in zip(pred, tgt) if a == b)
        correct += match
        total += n_pred
        if len(detailed) < 5:
            print(f"  [{i+1}] tgt='{tgt[:20]}' pred='{pred[:20]}' match={match}/{n_pred} | {r.get('elapsed_s', 0):.1f}s retries={r.get('retries', 0)}")
        detailed.append({
            "ctx_tail": ctx[-30:], "tgt": tgt, "pred": pred, "match": match,
            "elapsed": r.get("elapsed_s"),
        })
        # 达到最小成功就停
        if len(detailed) >= min_success and i > n_samples - 1:
            pass

    acc = correct / total if total else 0
    total_time = time.time() - t_start
    print(f"\n  [{name}] 字符级 NTP acc: {correct}/{total} = {acc:.3f}  (耗时 {total_time:.0f}s, 成功 {len(detailed)}, 错误 {errors})")
    return {
        "model": name,
        "acc": acc,
        "correct": correct,
        "total_chars": total,
        "n_samples": len(detailed),
        "errors": errors,
        "total_time_s": total_time,
        "detailed": detailed,
    }


def main():
    print("API 配置:")
    for name, api in APIS.items():
        print(f"  {name}: model={api['model']}, key={'set' if api['key'] else 'EMPTY'}")
    if not any(api["key"] for api in APIS.values()):
        print("❌ 没有可用 key, 退出")
        return

    connectivity_test()
    full_text = load_roleplay_corpus()

    results = []
    for name, api in APIS.items():
        if not api["key"]:
            continue
        try:
            r = ntp_eval(name, api, full_text, n_samples=15, seq_len=128, n_pred=10)
            results.append(r)
        except Exception as e:
            print(f"  [{name}] FATAL: {e}")
        # 实时保存
        out_path = "results/poc7_llm_api_results.json"
        os.makedirs("results", exist_ok=True)
        save = {
            "c_route_ref": {"acc": C_ROUTE_ACC, "acc_std": C_ROUTE_ACC_STD,
                            "params": C_ROUTE_PARAMS, "note": C_ROUTE_NOTE},
            "lstm_refs": {k: {"params": v[0], "acc": v[1], "note": v[2]} for k, v in LSTM_REFS.items()},
            "llm_results": results,
            "config": {"n_samples": 15, "seq_len": 128, "n_pred": 10, "task": "roleplay 续写 NTP"},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save, f, ensure_ascii=False, indent=2, default=str)
        print(f"  [SAVED] {out_path}")

    # 打印总表
    print(f"\n{'=' * 100}")
    print("[3/3] 总对比表 (C-route + LSTM-baselines + LLM-API)")
    print(f"{'=' * 100}")
    print(f"{'模型':35s} {'类型':>12s} {'params':>12s} {'val_acc':>10s} {'备注':35s}")
    print("-" * 110)
    print(f"{'C-route (POC-7 E)':35s} {'char-LM':>12s} {C_ROUTE_PARAMS:>12d} {C_ROUTE_ACC:>10.3f} {C_ROUTE_NOTE:35s}")
    for n, (p, a, note) in LSTM_REFS.items():
        print(f"{n:35s} {'char-LM':>12s} {p:>12d} {a:>10.3f} {note:35s}")
    for r in results:
        size = "235B" if "Qwen" in r['model'] else ("未知(MM)" if "MiniMax" in r['model'] else "?")
        n = r['n_samples']
        ch = r['total_chars']
        print(f"{r['model']:35s} {'API-LLM':>12s} {size:>12s} {r['acc']:>10.3f} {f'n={n}, chars={ch}':35s}")
    print("=" * 100)


if __name__ == "__main__":
    main()
