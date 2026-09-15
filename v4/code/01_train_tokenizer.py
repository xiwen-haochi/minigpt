# -*- coding: utf-8 -*-
"""
01_train_tokenizer.py —— 训练 Byte-Level BPE 分词器（GPT-4 / Qwen 同款方案）

为什么中文不用 jieba：
  现代大模型统一使用 Byte-Level BPE —— 先把文本按 UTF-8 拆成字节，
  再在字节序列上反复合并最高频的相邻片段。中文字符会被自动学成"子词"，
  不需要任何预分词，也永远不会有未登录词（OOV）。

产出：tokenizer/tokenizer.json
"""
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 32000  # 词表大小：小模型 32k 足够，且能用 uint16 存 token 流
SAVE_PATH = Path("tokenizer/tokenizer.json")

# 三个特殊 token：文档结束符 / 对话开始 / 对话结束（固定占词表最前）
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def extract_strings(obj):
    """递归提取 JSON 里的所有字符串。

    兼容嵌套结构：ShareGPT 格式的 conversations 是"列表套字典"，
    直接用 obj.values() 取不到里面的文本，所以要递归。
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):  # 类似 Python 的字典遍历
        for v in obj.values():
            yield from extract_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from extract_strings(v)


def iter_corpus():
    """产出训练分词器用的全部文本（小说 + 对话混合，覆盖面才够）。

    类似 Python 生成器的常规用法：逐文件 yield，避免一次性读入内存。
    """
    novel_dir = Path("data/novels")
    files = sorted(novel_dir.glob("*.txt"))
    if not files:
        raise SystemExit("[stop] 没有找到小说，请把 .txt 小说放进 data/novels/ 目录")
    for p in files:
        print(f"[corpus] 小说: {p}")
        yield p.read_text(encoding="utf-8", errors="ignore")

    sft_file = Path("data/sft_raw.jsonl")
    if sft_file.exists():
        print(f"[corpus] 对话: {sft_file}")
        with sft_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 对话里的所有字符串（含嵌套的 conversations）都喂给分词器
                yield from extract_strings(obj)


def main():
    tokenizer = Tokenizer(models.BPE())  # 空 BPE 模型，等语料来训练
    # ByteLevel 预处理：把文本转成 UTF-8 字节视角，中文无需预分词
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()  # 解码器必须配套，否则中文会变乱码

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,  # 出现少于 2 次的合并不收编，防噪声
    )
    tokenizer.train_from_iterator(iter_corpus(), trainer=trainer)

    SAVE_PATH.parent.mkdir(exist_ok=True)
    tokenizer.save(str(SAVE_PATH))

    print(
        f"[done] 分词器已保存到 {SAVE_PATH}，实际词表大小 {tokenizer.get_vocab_size()}"
    )
    # 试跑：看看中文被切成什么样子
    demo = "你好，今天我们来训练一个中文大模型。"
    print(f"[demo] {demo}")
    print(f"  -> {tokenizer.encode(demo).tokens}")


if __name__ == "__main__":
    main()
