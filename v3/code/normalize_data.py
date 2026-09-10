# -*- coding: utf-8 -*-
"""
normalize_data.py —— 小智三代 · 数据规范化脚本
作用：把 MiniMind 开源数据集（预训练纯文本 + SFT 多轮对话）规范化成统一教材
输入：
  dataset/pretrain_t2t_mini.jsonl  每行 {"text": "..."}（约 120 万行）
  dataset/sft_t2t_mini.jsonl       每行 {"conversations": [...]}（约 90 万行）
输出（均写到本脚本所在目录）：
  pretrain_train.jsonl / pretrain_val.jsonl  {"text": "纯文本"}
  sft_train.jsonl / sft_val.jsonl            {"text": "<|user|>问<|assistant|>答<|end|>..."}
用法：python normalize_data.py（纯标准库，无需安装任何依赖）
"""

import hashlib  # 计算整条样本的指纹，用于精确去重
import json  # 读写 jsonl 数据行
import random  # 切分前洗牌，保证训练/验证分布一致
import re  # 归一化文本（去标点空白），用于近似去重
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

# ==================== 可调参数区（只改这里） ====================
DATA_DIR = Path("dataset")  # 原料目录
PRETRAIN_IN = DATA_DIR / "pretrain_t2t_mini.jsonl"  # 预训练原料
SFT_IN = DATA_DIR / "sft_t2t_mini.jsonl"  # SFT 原料

PRETRAIN_TRAIN = Path("pretrain_train.jsonl")  # 输出：预训练训练集
PRETRAIN_VAL = Path("pretrain_val.jsonl")  # 输出：预训练验证集
SFT_TRAIN = Path("sft_train.jsonl")  # 输出：SFT 训练集
SFT_VAL = Path("sft_val.jsonl")  # 输出：SFT 验证集

VAL_RATIO = 0.02  # 验证集占比（98:2 切分）
SEED = 42  # 洗牌随机种子：固定后每次切分结果一致，便于复现

# ==================== 聊天模板（全项目最重要的常量） ====================
# 纪律：训练时用什么模板，推理时就要一字不差地用什么模板
USER_TOKEN = "<|user|>"  # 用户发言起始标记
ASSISTANT_TOKEN = "<|assistant|>"  # 助手发言起始标记
END_TOKEN = "<|end|>"  # 一轮对话结束标记

# ==================== 身份消毒词表 ====================
# MiniMind 数据里残留了它自己的开发者/模型名，必须替换成你自己的人设，
# 否则模型做自我介绍时会报别人的名字。左边是原料里的词，右边是替换结果。
MY_MODEL = "小智"  # 你的模型名字
MY_DEVELOPER = "小智团队"  # 你的开发者署名
IDENTITY_MAP = {
    "jingyaogong": MY_DEVELOPER,  # MiniMind 作者署名
    "nbhhd": MY_DEVELOPER,  # 数据里出现的另一个署名
    "MiniMind": MY_MODEL,  # 模型自称（大写）
    "minimind": MY_MODEL,  # 模型自称（小写）
}

# ==================== 质量过滤规则 ====================
MIN_Q_LEN, MAX_Q_LEN = 2, 300  # 单轮用户发言长度边界（字符数）
MIN_A_LEN, MAX_A_LEN = 2, 1000  # 单轮助手回答长度边界
MAX_TOTAL_CHARS = 1800  # 模板化后整条文本上限（防止超出 max_seq_len=512 个 token）
MIN_PT_LEN, MAX_PT_LEN = 10, 2000  # 预训练文本长度边界
BAN_WORDS = [  # 人设漏出黑名单：出现即整条丢弃
    "作为AI",
    "作为一个人工智能",
    "作为语言模型",
    "作为聊天机器人",
]


# ==================== 工具函数 ====================
def norm_text(text):
    """归一化：去掉所有空白和常见标点，用于识别"换皮重复"

    参数：
        text: 原始字符串
    返回：
        去掉空白与标点后的字符串
    """
    return re.sub(r"[\s，。！？、：；,.!?:;~…·—'\"\"''（）()【】「」《》<>-]", "", text)


def sanitize(text, counter):
    """身份消毒：把原料里的别人名字替换成自己的人设

    参数：
        text: 原始字符串
        counter: dict，累计每个词被替换的次数（用于体检报告）
    返回：
        消毒后的字符串
    """
    for old, new in IDENTITY_MAP.items():
        if old in text:
            counter[old] = counter.get(old, 0) + text.count(old)  # 记录命中次数
            text = text.replace(old, new)
    return text


# ==================== 六个类，一个类只做一件事 ====================
class SftLoader:
    """职责：把 SFT 原料解析成统一的多轮对话列表，同时剥掉思维链"""

    def __init__(self, path):
        """path：SFT 原料文件路径（Path 对象）"""
        self.path = path
        self.reasoning_dropped = 0  # 统计被剥掉的思维链条数

    def load(self):
        """读取全部行；返回 (对话列表, 坏行数)

        每条对话统一为 [(问题1, 回答1), (问题2, 回答2), ...] 的轮次列表。
        解析时完成两件事：
        ① 剥思维链——assistant 消息里的 reasoning_content 字段直接不读；
        ② 结构校验——必须 user 问、assistant 答交替出现，异常即丢弃。
        """
        if not self.path.exists():
            raise SystemExit(f"[stop] 找不到原料文件：{self.path}")
        dialogs, bad = [], 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    msgs = json.loads(line)["conversations"]
                    turns = self._parse(msgs)
                    if turns:
                        dialogs.append(turns)
                    else:
                        bad += 1  # 结构异常的对话按坏行处理
                except (json.JSONDecodeError, KeyError, TypeError):
                    bad += 1
        return dialogs, bad

    def _parse(self, msgs):
        """把 conversations 数组解析成轮次列表；结构异常返回 None

        参数：
            msgs: 原始 conversations 数组（list of dict）
        返回：
            [(问, 答), ...] 或 None
        """
        turns, cur_q = [], None
        for m in msgs:
            role = m.get("role")
            if role == "assistant" and m.get("reasoning_content"):
                self.reasoning_dropped += 1  # 思维链命中计数（内容不读即等于剥掉）
            content = (m.get("content") or "").strip()
            if not content:
                continue  # 空发言无意义，跳过
            if role == "user":
                if cur_q is not None:
                    return None  # 连续两个 user，结构异常
                cur_q = content
            elif role == "assistant":
                if cur_q is None:
                    return None  # 没有提问就回答，结构异常
                turns.append((cur_q, content))
                cur_q = None
            # 其他 role（如 system）不参与训练，直接跳过
        # 末尾未配对的问题（cur_q 残留）随返回值一起丢弃
        return turns or None


class PretrainLoader:
    """职责：把预训练原料解析成纯文本列表"""

    def __init__(self, path):
        """path：预训练原料文件路径（Path 对象）"""
        self.path = path

    def load(self):
        """读取全部行；返回 (文本列表, 坏行数)"""
        if not self.path.exists():
            raise SystemExit(f"[stop] 找不到原料文件：{self.path}")
        texts, bad = [], 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    text = json.loads(line)["text"].strip()
                    if text:
                        texts.append(text)
                    else:
                        bad += 1  # 空文本按坏行处理
                except (json.JSONDecodeError, KeyError, AttributeError):
                    bad += 1
        return texts, bad


class DedupFilter:
    """职责：通用清洗——精确去重 → 近似去重（对两种数据复用同一套纪律）"""

    def dedup(self, fp_of, key_of, items):
        """依次执行两步去重

        参数：
            fp_of: 函数，从一条样本算出精确指纹字符串
            key_of: 函数，从一条样本算出近似去重键
            items: 样本列表
        返回：
            (存活列表, 统计 dict)
        """
        report = {"exact_dup": 0, "near_dup": 0}

        # ① 精确去重：整条样本算 MD5 指纹，完全相同只留一条
        seen_hash, step1 = set(), []
        for it in items:
            fp = hashlib.md5(fp_of(it).encode("utf-8")).hexdigest()
            if fp in seen_hash:
                report["exact_dup"] += 1
                continue
            seen_hash.add(fp)
            step1.append(it)

        # ② 近似去重：归一化后取前缀作键，相同视为换皮重复
        seen_near, kept = set(), []
        for it in step1:
            key = key_of(it)
            if key in seen_near:
                report["near_dup"] += 1
                continue
            seen_near.add(key)
            kept.append(it)
        return kept, report


class SftQualityGate:
    """职责：SFT 质检——长度越界 / 人设漏出 / 模板标记泄漏 → 整条丢弃"""

    def check(self, turns):
        """返回 True 表示合格

        参数：
            turns: [(问, 答), ...] 轮次列表
        """
        for q, a in turns:
            if not (MIN_Q_LEN <= len(q) <= MAX_Q_LEN):
                return False
            if not (MIN_A_LEN <= len(a) <= MAX_A_LEN):
                return False
            if any(w in q + a for w in BAN_WORDS):
                return False
            if "<|" in q + a:  # 原料里不该出现模板标记，出现即泄漏
                return False
        return True


class ChatTemplater:
    """职责：把多轮对话序列化成带特殊标记的训练文本"""

    def render(self, turns):
        """序列化格式：<|user|>问<|assistant|>答<|end|>（多轮首尾相接）

        参数：
            turns: [(问, 答), ...] 轮次列表
        返回：
            拼接好的模板文本
        """
        return "".join(
            f"{USER_TOKEN}{q}{ASSISTANT_TOKEN}{a}{END_TOKEN}" for q, a in turns
        )


class DataSplitter:
    """职责：洗牌后按 98:2 切成训练/验证集并写盘"""

    def split_and_write(self, rows, train_path, val_path):
        """切分永远是流水线最后一步，保证验证集与训练集零重叠

        参数：
            rows: 模板化后的数据行列表
            train_path / val_path: 输出文件路径
        返回：
            (训练集条数, 验证集条数)
        """
        random.Random(SEED).shuffle(rows)  # 固定种子洗牌，结果可复现
        n_val = max(1, int(len(rows) * VAL_RATIO))  # 至少留 1 条验证
        self._write(val_path, rows[:n_val])
        self._write(train_path, rows[n_val:])
        return len(rows) - n_val, n_val

    @staticmethod
    def _write(path, rows):
        """按行写入 jsonl（每行一条 JSON，UTF-8 中文不转义）"""
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class NormalizePipeline:
    """职责：编排两条流水线（SFT 一条、预训练一条），并打印体检报告"""

    def run(self):
        """先规范化 SFT，再规范化预训练；各自独立产出 train/val"""
        self._run_sft()
        self._run_pretrain()
        print("[done] 数据规范化完成！下一站：重训 BPE 分词器")

    # ---------- SFT 流水线：读 → 消毒 → 去重 → 质检 → 模板化 → 切分 ----------
    def _run_sft(self):
        print("=" * 20, "SFT 对话数据", "=" * 20)

        # ① 读原料（解析时已剥掉思维链）
        loader = SftLoader(SFT_IN)
        dialogs, bad = loader.load()
        print(f"[read] 原料 {len(dialogs)} 条（坏行/结构异常 {bad} 条）")
        print(f"[strip] 剥掉思维链 {loader.reasoning_dropped} 段（reasoning_content）")

        # ② 身份消毒：把别人的名字换成自己的人设
        hits = {}
        dialogs = [
            [(sanitize(q, hits), sanitize(a, hits)) for q, a in turns]
            for turns in dialogs
        ]
        print(f"[sanitize] 身份词替换 {sum(hits.values())} 处：{hits}")

        # ③ 去重：精确（整条对话）→ 近似（首个问题归一化前 15 字）
        dialogs, rep = DedupFilter().dedup(
            fp_of=lambda t: "".join(q + a for q, a in t),
            key_of=lambda t: norm_text(t[0][0])[:15],
            items=dialogs,
        )
        print(f"[dedup] 精确去重 -{rep['exact_dup']}，近似去重 -{rep['near_dup']}")

        # ④ 质检 + ⑤ 模板化（超长整条文本一并丢弃）
        gate, templater = SftQualityGate(), ChatTemplater()
        rows, dropped = [], 0
        for turns in dialogs:
            if not gate.check(turns):
                dropped += 1
                continue
            text = templater.render(turns)
            if len(text) > MAX_TOTAL_CHARS:
                dropped += 1
                continue
            turns_n = len(turns)
            rows.append({"text": text, "turns": turns_n})
        print(f"[filter] 质量过滤 -{dropped} → 剩余 {len(rows)} 条")

        # ⑥ 切分（流水线最后一步，防验证集泄漏）
        n_tr, n_va = DataSplitter().split_and_write(rows, SFT_TRAIN, SFT_VAL)
        print(f"[split] 训练集 {n_tr} 条 → {SFT_TRAIN}，验证集 {n_va} 条 → {SFT_VAL}")

        # ---- 轮次分布体检：看看多轮对话占多少 ----
        dist = {}
        for r in rows:
            dist[r["turns"]] = dist.get(r["turns"], 0) + 1
        print(f"[dist] 轮次分布（轮数: 条数）：{dict(sorted(dist.items()))}")

    # ---------- 预训练流水线：读 → 消毒 → 去重 → 质检 → 切分 ----------
    def _run_pretrain(self):
        print("=" * 20, "预训练纯文本", "=" * 20)

        # ① 读原料
        texts, bad = PretrainLoader(PRETRAIN_IN).load()
        print(f"[read] 原料 {len(texts)} 条（坏行 {bad} 条）")

        # ② 身份消毒
        hits = {}
        texts = [sanitize(t, hits) for t in texts]
        print(f"[sanitize] 身份词替换 {sum(hits.values())} 处：{hits}")

        # ③ 去重：精确（全文）→ 近似（归一化前 20 字）
        texts, rep = DedupFilter().dedup(
            fp_of=lambda t: t,
            key_of=lambda t: norm_text(t)[:20],
            items=texts,
        )
        print(f"[dedup] 精确去重 -{rep['exact_dup']}，近似去重 -{rep['near_dup']}")

        # ④ 质检：长度越界 / 黑名单 / 模板泄漏 → 丢弃
        kept = [
            t
            for t in texts
            if MIN_PT_LEN <= len(t) <= MAX_PT_LEN
            and not any(w in t for w in BAN_WORDS)
            and "<|" not in t
        ]
        print(f"[filter] 质量过滤 -{len(texts) - len(kept)} → 剩余 {len(kept)} 条")

        # ⑤ 切分（预训练无需模板化，直接包 {"text": ...}）
        rows = [{"text": t} for t in kept]
        n_tr, n_va = DataSplitter().split_and_write(rows, PRETRAIN_TRAIN, PRETRAIN_VAL)
        print(
            f"[split] 训练集 {n_tr} 条 → {PRETRAIN_TRAIN}，"
            f"验证集 {n_va} 条 → {PRETRAIN_VAL}"
        )


if __name__ == "__main__":
    NormalizePipeline().run()
