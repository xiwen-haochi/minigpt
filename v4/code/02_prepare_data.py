# -*- coding: utf-8 -*-
"""
02_prepare_data.py —— 数据加工：原始文本 -> 训练用二进制文件

产出：
  token_stream/train.bin, val.bin           预训练 token 长河（uint16，小说）
  sft_bin/train_ids.npy, train_labels.npy   SFT 定长样本（int32）
  sft_bin/val_ids.npy,   val_labels.npy

对话模板（ChatML 风格，Qwen 同款）：
  <|im_start|>system\n{人设}<|im_end|>\n
  <|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n{回答}<|im_end|>
SFT 只对"回答"部分计算 loss，"人设+问题"部分 label 置为 -100。

语料说明：本项目 data/sft_raw.jsonl 是 ShareGPT 格式（conversations 列表），
每条带一段很长的 system 人设（"颜柳"），人设也会被编进模板一起训练。
"""
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
# SFT 样本定长（超过截断、不足填充）。
# 注意：本语料每条带 ~300 字的 system 人设，512 装不下会把回答截断，
# 导致模型学不到结尾的 <|im_end|>（不会停止），所以给到 768。
MAX_SFT_LEN = 768
VAL_RATIO = 0.01  # 预训练 1% 做验证集
SFT_VAL_RATIO = 0.02  # SFT 2% 做验证集
random.seed(42)


def load_tokenizer():
    """加载第 1 步训练好的分词器。"""
    if not TOKENIZER_PATH.exists():
        raise SystemExit("[stop] 找不到分词器，请先运行 01_train_tokenizer.py")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def parse_sft_obj(obj):
    """兼容四种常见对话格式，统一返回 (system, 问题, 回答)；失败返回 None。

    支持:
      {"conversations": [{"from": "system/human/gpt", "value": ...}]}  ShareGPT 格式
      {"instruction": ..., "input": ..., "output": ...}                Alpaca 格式
      {"question": ..., "answer": ...}
      {"prompt": ..., "response": ...}
    """
    if "conversations" in obj:  # ShareGPT 格式（本项目语料就是这种）
        system, q, a = "", "", ""
        for turn in obj["conversations"]:
            role, value = turn.get("from", ""), turn.get("value", "").strip()
            if role == "system":
                system = value
            elif role in ("human", "user") and not q:  # 只取第一轮问答（单轮对话）
                q = value
            elif role in ("gpt", "assistant") and not a:
                a = value
        return system, q, a
    if "output" in obj:  # Alpaca 格式
        q = obj.get("instruction", "")
        if obj.get("input"):  # Alpaca 的 input 是补充材料，拼在指令后面
            q = q + "\n" + obj["input"]
        return "", q.strip(), obj["output"].strip()
    if "answer" in obj:
        return "", obj.get("question", "").strip(), obj["answer"].strip()
    if "response" in obj:
        return "", obj.get("prompt", "").strip(), obj["response"].strip()
    return None


def encode_sft(tok, system, q, a):
    """把一条单轮对话编码成 (input_ids, labels)。

    labels 中 prompt（人设+问题）部分为 -100（不算 loss），
    "回答"部分为真实 token id。结尾强制带 <|im_end|>，教模型学会停止。
    """
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    # prompt 部分：system 人设 + 人类问题（都不算 loss）
    prompt_ids = []
    if system:  # 有人设就加 system 段（没有则跳过，兼容无 system 的语料）
        prompt_ids += [im_start] + tok.encode("system\n" + system).ids + [im_end]
        prompt_ids += tok.encode("\n").ids
    prompt_ids += (
        [im_start]
        + tok.encode("user\n" + q).ids
        + [im_end]
        + tok.encode("\nassistant\n").ids
    )
    # answer 部分：模型回答（算 loss）
    answer_ids = tok.encode(a).ids + [im_end]
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return ids[:MAX_SFT_LEN], labels[:MAX_SFT_LEN]


def build_pretrain(tok):
    """小说 -> token 长河（uint16 二进制）。

    按段落攒成约 1000 字的块 -> 打乱 -> 99% 训练 / 1% 验证 ->
    编码后首尾相接存成一条长河，文档之间用 <|endoftext|> 隔开。
    """
    novel_dir = Path("data/novels")
    files = sorted(novel_dir.glob("*.txt"))
    if not files:
        raise SystemExit("[stop] 请把 .txt 小说放进 data/novels/ 目录")

    eos = tok.token_to_id("<|endoftext|>")
    chunks = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        buf = ""
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                continue
            buf += para + "\n"
            if len(buf) >= 1000:  # 块太大浪费显存，太小缺上下文
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        print(f"[pretrain] {p.name}: 累计 {len(chunks)} 块")

    random.shuffle(chunks)
    n_val = max(1, int(len(chunks) * VAL_RATIO))
    groups = {"train": chunks[n_val:], "val": chunks[:n_val]}

    out_dir = Path("token_stream")
    out_dir.mkdir(exist_ok=True)
    for name, group in groups.items():
        ids = []
        for c in group:
            ids.extend(tok.encode(c).ids)
            ids.append(eos)  # 文档边界：教模型识别"一篇文本结束了"
        # uint16 能装下 32000 词表（上限 65535），比 int64 省 4 倍空间
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(out_dir / f"{name}.bin")
        print(f"[pretrain] {name}.bin: {len(arr):,} tokens")


def build_sft(tok):
    """对话 jsonl -> 定长 (ids, labels) 的 .npy 文件。"""
    sft_file = Path("data/sft_raw.jsonl")
    if not sft_file.exists():
        raise SystemExit("[stop] 请准备对话语料 data/sft_raw.jsonl（每行一个 JSON）")

    pad_id = tok.token_to_id("<|endoftext|>")  # 用 eos 做 padding（label 会屏蔽掉）
    samples = []
    with sft_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_sft_obj(json.loads(line))
            # parsed = (system, 问题, 回答)；system 可以为空，问答必须有
            if not parsed or not parsed[1] or not parsed[2]:
                continue
            ids, labels = encode_sft(tok, *parsed)
            pad_n = MAX_SFT_LEN - len(ids)
            ids += [pad_id] * pad_n  # ids 补 pad
            labels += [-100] * pad_n  # labels 补 -100（不参与 loss）
            samples.append((ids, labels))

    if len(samples) < 10:
        raise SystemExit("[stop] 有效对话样本太少，请检查 data/sft_raw.jsonl 格式")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * SFT_VAL_RATIO))
    groups = {"train": samples[n_val:], "val": samples[:n_val]}

    out_dir = Path("sft_bin")
    out_dir.mkdir(exist_ok=True)
    for name, group in groups.items():
        ids = np.array([s[0] for s in group], dtype=np.int32)
        labs = np.array([s[1] for s in group], dtype=np.int32)
        np.save(out_dir / f"{name}_ids.npy", ids)
        np.save(out_dir / f"{name}_labels.npy", labs)
        print(f"[sft] {name}: {len(group)} 条样本")


def main():
    tok = load_tokenizer()
    build_pretrain(tok)
    build_sft(tok)
    print("[done] 数据加工完成")


if __name__ == "__main__":
    main()
