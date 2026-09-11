# -*- coding: utf-8 -*-
"""
train_tokenizer.py —— 小智三代 · BPE 分词器训练脚本
作用：在规范化后的百万行语料（pretrain_train + sft_train）上重训专属分词器，并做三道体检
用法：pip install tokenizers && python train_tokenizer.py
依赖：仅需第三方库 tokenizers（HuggingFace 出品，Rust 内核，支持流式训练）

与 v2 版的关键差异（都是为百万行语料准备的）：
  ① 流式供料——逐行生成文本，Python 侧内存占用与语料规模无关（v2 是全量读进内存）
  ② 抽样训练——BPE 的词频统计在几十万行时早已收敛，喂 40 万行和 217 万行训出的
     词表几乎一样；而训练器内部的词频表是内存大头，限制喂入行数就锁死了内存上限
  ③ min_frequency=2——只合并出现过 2 次以上的片段，抑制大语料里的噪声合并
  ④ 抽样体检——验证集最多取 2000 条算压缩率，不必编码全部几万条
"""

import json  # 读取 jsonl 教材
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ==================== 可调参数区（只改这里） ====================
# 分词器只用"训练集"练（防验证集泄漏）；两个文件都是 normalize_data.py 的产出
TRAIN_FILES = [Path("pretrain_train.jsonl"), Path("sft_train.jsonl")]
VAL_FILES = [Path("pretrain_val.jsonl"), Path("sft_val.jsonl")]  # 体检考场
OUT_DIR = Path("tokenizer")  # 输出目录：tokenizer.json 存在这里
VOCAB_SIZE = 16384  # 词表大小：v3 语料比 v2 大两个量级，词表跟着翻倍（v2 是 8192）

# 抽样训练上限：train 文件在规范化时已洗过牌，取前 N 行即随机样本。
# 40 万行约 500MB 文本，词频统计早已收敛；内存占用因此封顶在几 GB 内。
# 机器内存紧张就调小（如 200_000），宽裕想更精确就调大，词表质量差异极小。
TRAIN_SAMPLE = 400_000
MIN_FREQUENCY = 2  # 至少出现 2 次的片段才允许入词表，过滤一次性噪声
CHECK_SAMPLE = 2000  # 体检抽样条数：验证集有几万条，抽 2000 条估压缩率足够准

# 特殊标记：必须与规范化脚本 normalize_data.py 里的聊天模板一字不差！
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]

# 压缩率健康区间：token/字，数值越低压缩越好（1 个 token 装的字越多）
# 中文语料 + 16K 词表，正常落在 0.3~0.9；超过 1.0 说明大量汉字被拆成多字节，需调大词表
GOOD_RATIO_MIN, GOOD_RATIO_MAX = 0.3, 0.9


# ==================== 四个类，一个类只做一件事 ====================
class CorpusStreamer:
    """职责：只负责供料——逐行流式读取 text 字段，内存占用与语料规模无关"""

    def __init__(self, paths):
        """paths：教材文件路径列表（Path 对象列表）"""
        for p in paths:
            if not p.exists():
                raise SystemExit(f"[stop] 找不到 {p}（请先运行 normalize_data.py）")
        self.paths = paths
        self.fed = 0  # 实际喂出的行数，供报告使用

    def iter_texts(self, limit=None):
        """生成器：每调用一次 next 才读一行，200 万行也不会撑爆内存

        参数：
            limit: 最多产出多少行；None 表示不限。
                   因 train 文件已洗牌，截取前 N 行等价于随机抽样
        返回：
            逐条产出文本字符串的迭代器（类似 Python 的 generator）
        """
        self.fed = 0
        for path in self.paths:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if limit is not None and self.fed >= limit:
                        return  # 到达抽样上限，提前收工
                    try:
                        yield json.loads(line)["text"]
                        self.fed += 1
                    except (json.JSONDecodeError, KeyError):
                        continue  # 坏行跳过，不干扰训练

    def sample_texts(self, n):
        """流式取前 n 条文本（验证集已洗过牌，前 n 条即随机样本）

        参数：
            n: 抽样条数上限
        返回：
            不超过 n 条的文本列表
        """
        return list(self.iter_texts(limit=n))


class BpeTrainerRunner:
    """职责：只负责训练——在抽样语料上流式训练专属 BPE 分词器并保存"""

    def train(self, text_stream):
        """
        参数：text_stream 文本迭代器（CorpusStreamer.iter_texts() 的产出）
        返回：保存路径
        """
        # BPE 模型 + 字节级回退：生僻字/emoji 都能拆成字节兜底，永无未知符
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()  # 保证"编码再解码 = 原文"
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY,  # 只合并高频片段，抑制大语料噪声
            special_tokens=SPECIAL_TOKENS,  # 特殊标记永远占据词表前几个坑位
            # 关键：把 256 个字节全部纳入初始字母表，否则语料里没出现过的字节
            # 在编码时会被静默丢弃（"是谁？"可能只剩半个字）——这是最易踩的坑
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        # train_from_iterator 天生吃迭代器：边读边统计词频，Python 侧全程流式
        tokenizer.train_from_iterator(text_stream, trainer)
        OUT_DIR.mkdir(exist_ok=True)
        out_path = OUT_DIR / "tokenizer.json"
        tokenizer.save(str(out_path))
        return out_path


class CompressionChecker:
    """职责：只负责体检——压缩率、特殊标记、往返一致，三道关卡"""

    def __init__(self, tokenizer):
        """tokenizer：刚训练好的 Tokenizer 对象"""
        self.tok = tokenizer

    def check(self, sample_texts):
        """
        在验证集抽样上执行三道体检；返回报告 dict：
        ratio 压缩率 / special_ok 特殊标记完整性 / roundtrip_ok 往返一致性
        """
        # ① 压缩率：抽样总 token 数 ÷ 总字符数（用没见过的文本考才真实）
        total_tokens = sum(len(self.tok.encode(t).ids) for t in sample_texts)
        total_chars = sum(len(t) for t in sample_texts)
        ratio = total_tokens / max(total_chars, 1)

        # ② 特殊标记：必须各自被编码成"一个" id，被拆碎即格式崩坏
        special_ids = {t: self.tok.encode(t).ids for t in SPECIAL_TOKENS}
        special_ok = all(len(ids) == 1 for ids in special_ids.values())

        # ③ 往返一致：编码成 id 再解码回来，必须与原文一字不差
        # 注意 decode 默认会跳过特殊标记，必须显式关掉，否则模板标记"被解码丢了"
        sample = "<|user|>你是谁？<|assistant|>我叫小智~<|end|>"
        back = self.tok.decode(self.tok.encode(sample).ids, skip_special_tokens=False)
        roundtrip_ok = back == sample

        return {
            "ratio": ratio,
            "special_ids": special_ids,
            "special_ok": special_ok,
            "roundtrip_ok": roundtrip_ok,
            "vocab_real": self.tok.get_vocab_size(),
        }


class TokenizerPipeline:
    """职责：只负责编排——供料 → 训练 → 保存 → 体检 → 给结论"""

    def run(self):
        # ① 供料：训练集流式 + 抽样练分词器，验证集抽样当体检考场
        feeder = CorpusStreamer(TRAIN_FILES)
        train_stream = feeder.iter_texts(limit=TRAIN_SAMPLE)
        val_sample = CorpusStreamer(VAL_FILES).sample_texts(CHECK_SAMPLE)
        print(f"[read] 体检样本 {len(val_sample)} 条；训练语料流式读取中")

        # ② 流式训练 + 保存（语料不进内存，Rust 内核逐行统计词频）
        out_path = BpeTrainerRunner().train(train_stream)
        print(f"[train] 实际喂入 {feeder.fed} 行（抽样上限 {TRAIN_SAMPLE}）")
        print(f"[train] 分词器已保存 → {out_path}")

        # ③ 保存后重新加载一次，验证文件本身可用（防止"存了个坏的"）
        reloaded = Tokenizer.from_file(str(out_path))

        # ④ 三道体检
        report = CompressionChecker(reloaded).check(val_sample)
        print(f"[check] 实际词表大小：{report['vocab_real']}（目标 {VOCAB_SIZE}）")
        print(
            f"[check] 压缩率：{report['ratio']:.2f} token/字"
            f"（健康区间 {GOOD_RATIO_MIN}~{GOOD_RATIO_MAX}）"
        )
        for tok, ids in report["special_ids"].items():
            print(f"[check] 特殊标记 {tok} → id {ids}")
        print(f"[check] 特殊标记独占一格：{'✓' if report['special_ok'] else '✗'}")
        print(f"[check] 编解码往返一致：{'✓' if report['roundtrip_ok'] else '✗'}")

        # ⑤ 给结论：三关全过才放行
        ratio_ok = GOOD_RATIO_MIN <= report["ratio"] <= GOOD_RATIO_MAX
        if ratio_ok and report["special_ok"] and report["roundtrip_ok"]:
            print("[done] 体检全部通过！分词器可交付训练阶段使用")
        else:
            print(
                "[warn] 体检未全过：压缩率偏高请调大 VOCAB_SIZE 后重跑；其余关卡请检查模板与语料"
            )


if __name__ == "__main__":
    TokenizerPipeline().run()
