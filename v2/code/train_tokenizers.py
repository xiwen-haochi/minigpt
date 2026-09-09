# -*- coding: utf-8 -*-
"""
train_tokenizer.py —— 小智二代 · BPE 分词器训练脚本
作用：在清洗后的 train.jsonl 上训练专属 BPE 分词器，并做三道体检
用法：pip install tokenizers && python train_tokenizer.py
依赖：仅需第三方库 tokenizers（HuggingFace 出品，Rust 内核，秒级训练）
"""

import json  # 读取 jsonl 教材
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ==================== 可调参数区（只改这里） ====================
TRAIN_FILE = Path("train.jsonl")  # 分词器只用它练（防验证集泄漏）
VAL_FILE = Path("val.jsonl")  # 体检考场：衡量压缩率的真实水平
OUT_DIR = Path("tokenizer")  # 输出目录：tokenizer.json 存在这里
VOCAB_SIZE = 8192  # 词表大小：小模型配小词表，嵌入层才不吃爆参数预算

# 特殊标记：必须与清洗脚本 clean_split.py 里的聊天模板一字不差！
SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]

# 压缩率健康区间：token/字，数值越低压缩越好（1 token 装的字越多）
# 中文聊天小语料 + 8K 词表，正常落在 0.3~0.9；超过 1.0 说明大量汉字被拆成多字节，需调大词表
GOOD_RATIO_MIN, GOOD_RATIO_MAX = 0.3, 0.9


# ==================== 四个类，一个类只做一件事 ====================
class CorpusExtractor:
    """职责：只负责供料——从 jsonl 里把 text 字段逐条抽出来"""

    def __init__(self, path):
        """path：教材文件路径（Path 对象）"""
        self.path = path

    def all_texts(self):
        """读取全部 text 字段；返回字符串列表，坏行跳过"""
        if not self.path.exists():
            raise SystemExit(f"[stop] 找不到 {self.path}（请先运行 clean_split.py）")
        texts = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue  # 坏行跳过，不干扰训练
        return texts


class BpeTrainer:
    """职责：只负责训练——在你的语料上练一个专属 BPE 分词器并保存"""

    def train(self, texts):
        """
        参数：texts 训练文本列表
        返回：(训练好的 Tokenizer 对象, 保存路径)
        """
        # BPE 模型 + 字节级回退：生僻字/emoji 都能拆成字节兜底，永无未知符
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()  # 保证"编码再解码 = 原文"
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS,  # 特殊标记永远占据词表前几个坑位
            # 关键：把 256 个字节全部纳入初始字母表，否则语料里没出现过的字节
            # 在编码时会被静默丢弃（"是谁？"可能只剩半个字）——这是最易踩的坑
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts, trainer)
        OUT_DIR.mkdir(exist_ok=True)
        out_path = OUT_DIR / "tokenizer.json"
        tokenizer.save(str(out_path))
        return tokenizer, out_path


class CompressionChecker:
    """职责：只负责体检——压缩率、特殊标记、往返一致，三道关卡"""

    def __init__(self, tokenizer):
        """tokenizer：刚训练好的 Tokenizer 对象"""
        self.tok = tokenizer

    def check(self, val_texts):
        """
        在验证集上执行三道体检；返回报告 dict：
        ratio 压缩率 / special_ok 特殊标记完整性 / roundtrip_ok 往返一致性
        """
        # ① 压缩率：验证集总 token 数 ÷ 总字符数（用没见过的文本考才真实）
        total_tokens = sum(len(self.tok.encode(t).ids) for t in val_texts)
        total_chars = sum(len(t) for t in val_texts)
        ratio = total_tokens / max(total_chars, 1)

        # ② 特殊标记：必须各自被编码成"一个" id，被拆碎即格式崩坏
        special_ok = all(len(self.tok.encode(t).ids) == 1 for t in SPECIAL_TOKENS)

        # ③ 往返一致：编码成 id 再解码回来，必须与原文一字不差
        # 注意 decode 默认会跳过特殊标记，必须显式关掉，否则模板标记"被解码丢了"
        sample = "<|user|>你是谁？<|assistant|>我叫小智~<|end|>"
        back = self.tok.decode(self.tok.encode(sample).ids, skip_special_tokens=False)
        roundtrip_ok = back == sample

        return {
            "ratio": ratio,
            "special_ok": special_ok,
            "roundtrip_ok": roundtrip_ok,
            "vocab_real": self.tok.get_vocab_size(),
        }


class TokenizerPipeline:
    """职责：只负责编排——供料 → 训练 → 保存 → 体检 → 给结论"""

    def run(self):
        # ① 供料：训练集练分词器，验证集当体检考场
        train_texts = CorpusExtractor(TRAIN_FILE).all_texts()
        val_texts = CorpusExtractor(VAL_FILE).all_texts()
        print(f"[read] 训练语料 {len(train_texts)} 条，体检语料 {len(val_texts)} 条")

        # ② 训练 + 保存
        tokenizer, out_path = BpeTrainer().train(train_texts)
        print(f"[train] 分词器已保存 → {out_path}")

        # ③ 保存后重新加载一次，验证文件本身可用（防止"存了个坏的"）
        reloaded = Tokenizer.from_file(str(out_path))

        # ④ 三道体检
        report = CompressionChecker(reloaded).check(val_texts)
        print(f"[check] 实际词表大小：{report['vocab_real']}（目标 {VOCAB_SIZE}）")
        print(
            f"[check] 压缩率：{report['ratio']:.2f} token/字"
            f"（健康区间 {GOOD_RATIO_MIN}~{GOOD_RATIO_MAX}）"
        )
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
