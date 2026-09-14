#!/usr/bin/env python3
"""步骤1：下载 Qwen 分词器 + 清洗数据"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须在 import transformers 之前
import json, re
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = Path("./dataset")
OUT_DIR = Path("./")
(OUT_DIR / "processed").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tokenizer").mkdir(parents=True, exist_ok=True)

# ---------- 1. 加载 Qwen 分词器（首次需联网，之后离线可用）----------
print("📦 加载 Qwen 分词器...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:  # 保险起见
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(str(OUT_DIR / "tokenizer"))
print(f"   ✅ 词表大小: {len(tokenizer):,}")

# ---------- 2. 校验分词器（编码→解码必须还原）----------
test = "今天加班到十点，累死了🤣"
ids = tokenizer.encode(test)
assert tokenizer.decode(ids) == test, "分词器往返校验失败！"
print(f"   🔎 '{test}' -> {ids} -> 还原成功 ✅")

# ---------- 3. 清洗小说 ----------
print("🧹 清洗《神雕侠侣》...")
text = (DATA_DIR / "novel.txt").read_text(encoding="utf-8")
text = re.sub(r"\n+", "\n", text)
text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
text = text.replace("\u3000", " ")
(OUT_DIR / "processed" / "novel_cleaned.txt").write_text(text, encoding="utf-8")
print(f"   ✅ 共 {len(text):,} 字")

# ---------- 4. 校验对话数据 ----------
n = 0
with open(DATA_DIR / "dialogues.jsonl", encoding="utf-8") as fin, open(
    OUT_DIR / "processed" / "dialogues_processed.jsonl", "w", encoding="utf-8"
) as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if "conversations" in d and len(d["conversations"]) >= 2:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
        except json.JSONDecodeError:
            pass
print(f"   ✅ 有效对话: {n} 条")
print("\n🎉 步骤1完成！下一步: python scripts/2_train_cpt.py")
