# -*- coding: utf-8 -*-
"""
clean_split.py —— 小智二代 · 数据清洗与切分脚本
作用：把蒸馏产出的 distill_data.jsonl 清洗、模板化，并切成 train/val 两份教材
用法：python clean_split.py（纯标准库，无需安装任何依赖）
"""

import hashlib  # 计算整条对话的指纹，用于精确去重
import json  # 读写 jsonl 数据行
import random  # 切分前洗牌，保证训练/验证分布一致
import re  # 归一化文本（去标点空白），用于近似去重
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

# ==================== 可调参数区（只改这里） ====================
IN_FILE = Path("distill_data.jsonl")  # 输入：蒸馏脚本的产出
TRAIN_FILE = Path("train.jsonl")  # 输出：训练集
VAL_FILE = Path("val.jsonl")  # 输出：验证集
VAL_RATIO = 0.02  # 验证集占比（98:2 切分）
SEED = 42  # 洗牌随机种子：固定后每次切分结果一致，便于复现

# ==================== 聊天模板（全项目最重要的常量） ====================
# 纪律：训练时用什么模板，推理时就要一字不差地用什么模板
USER_TOKEN = "<|user|>"  # 用户发言起始标记
ASSISTANT_TOKEN = "<|assistant|>"  # 助手发言起始标记
END_TOKEN = "<|end|>"  # 一轮对话结束标记

# ==================== 质量过滤规则 ====================
MIN_Q_LEN, MAX_Q_LEN = 2, 200  # 用户问题长度边界（字符数）
MIN_A_LEN, MAX_A_LEN = 2, 500  # 助手回答长度边界
BAN_WORDS = [  # 人设漏出黑名单：出现即整条丢弃
    "作为AI",
    "作为一个人工智能",
    "作为语言模型",
    "作为聊天机器人",
]


# ==================== 五个类，一个类只做一件事 ====================
class JsonlReader:
    """职责：只负责读——把原料文件解析成统一的对话列表"""

    def __init__(self, path):
        """path：原料文件路径（Path 对象）"""
        self.path = path

    def load(self):
        """读取全部行；返回 (对话列表, 坏行数)
        对话统一为 {'q': 问题, 'a': 回答, 'type': 类别} 的扁平结构"""
        if not self.path.exists():
            raise SystemExit(
                f"[stop] 找不到原料文件：{self.path}（请先运行 distill.py）"
            )
        items, bad = [], 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                msgs = item["messages"]
                items.append(
                    {
                        "q": msgs[0]["content"].strip(),
                        "a": msgs[1]["content"].strip(),
                        "type": item.get("type", "unknown"),
                    }
                )
            except (json.JSONDecodeError, KeyError, IndexError):
                bad += 1  # 坏行不计入，在报告中体现
        return items, bad


class DataCleaner:
    """职责：只负责洗——精确去重 → 近似去重 → 质量过滤（顺序不能乱）"""

    @staticmethod
    def _norm(text):
        """归一化：去掉所有空白和常见标点，用于识别"换皮重复" """
        return re.sub(
            r"[\s，。！？、：；,.!?:;~…·—'\"\"''（）()【】「」《》<>-]", "", text
        )

    def clean(self, items):
        """
        依次执行三步清洗；返回 (合格列表, 各环节淘汰统计 dict)
        顺序纪律：先去重再过滤——先让数据变"纯"，再让数据变"好"
        """
        report = {"exact_dup": 0, "near_dup": 0, "bad_quality": 0}

        # ① 精确去重：整条问答算指纹，完全相同只留一条
        seen_hash, step1 = set(), []
        for it in items:
            fp = hashlib.md5((it["q"] + it["a"]).encode("utf-8")).hexdigest()
            if fp in seen_hash:
                report["exact_dup"] += 1
                continue
            seen_hash.add(fp)
            step1.append(it)

        # ② 近似去重：问题归一化后取前 15 字，相同视为换皮重复
        seen_near, step2 = set(), []
        for it in step1:
            key = self._norm(it["q"])[:15]
            if key in seen_near:
                report["near_dup"] += 1
                continue
            seen_near.add(key)
            step2.append(it)

        # ③ 质量过滤：长度越界 / 人设漏出 / 模板标记泄漏 → 整条丢弃
        kept = []
        for it in step2:
            q, a = it["q"], it["a"]
            if not (MIN_Q_LEN <= len(q) <= MAX_Q_LEN):
                report["bad_quality"] += 1
                continue
            if not (MIN_A_LEN <= len(a) <= MAX_A_LEN):
                report["bad_quality"] += 1
                continue
            if any(w in q + a for w in BAN_WORDS):
                report["bad_quality"] += 1
                continue
            if "<|" in q + a:  # 原料里不该出现模板标记，出现即泄漏
                report["bad_quality"] += 1
                continue
            kept.append(it)
        return kept, report


class ChatTemplater:
    """职责：只负责模板化——把一问一答序列化成带特殊标记的训练文本"""

    def render(self, q, a):
        """序列化格式：<|user|>问题<|assistant|>回答<|end|>（单轮对话）"""
        return f"{USER_TOKEN}{q}{ASSISTANT_TOKEN}{a}{END_TOKEN}"


class DataSplitter:
    """职责：只负责切分——洗牌后按比例切成训练/验证集并写盘"""

    def split_and_write(self, rows):
        """
        参数：rows 模板化后的数据行列表
        返回：(训练集条数, 验证集条数)
        纪律：切分永远是流水线最后一步，保证验证集与训练集零重叠
        """
        random.Random(SEED).shuffle(rows)  # 固定种子洗牌，结果可复现
        n_val = max(1, int(len(rows) * VAL_RATIO))  # 至少留 1 条验证
        val_rows, train_rows = rows[:n_val], rows[n_val:]
        self._write(TRAIN_FILE, train_rows)
        self._write(VAL_FILE, val_rows)
        return len(train_rows), len(val_rows)

    @staticmethod
    def _write(path, rows):
        """按行写入 jsonl（每行一条 JSON，UTF-8 中文不转义）"""
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class CleanPipeline:
    """职责：只负责编排——读 → 洗 → 模板化 → 切分，并打印体检报告"""

    def run(self):
        # ① 读原料
        items, bad_lines = JsonlReader(IN_FILE).load()
        print(f"[read] 原料 {len(items)} 条（坏行 {bad_lines} 条已跳过）")

        # ② 清洗（顺序：精确去重 → 近似去重 → 质量过滤）
        kept, report = DataCleaner().clean(items)
        print(
            f"[clean] 精确去重 -{report['exact_dup']}，"
            f"近似去重 -{report['near_dup']}，"
            f"质量过滤 -{report['bad_quality']} → 剩余 {len(kept)} 条"
        )

        # ③ 模板化：每条对话序列化成带特殊标记的文本
        templater = ChatTemplater()
        rows = [
            {"text": templater.render(it["q"], it["a"]), "type": it["type"]}
            for it in kept
        ]

        # ④ 切分（流水线最后一步，防验证集泄漏）
        n_train, n_val = DataSplitter().split_and_write(rows)
        print(
            f"[split] 训练集 {n_train} 条 → {TRAIN_FILE}，"
            f"验证集 {n_val} 条 → {VAL_FILE}"
        )

        # ---- 类型分布体检：帮你发现"配方跑偏" ----
        dist = {}
        for it in kept:
            dist[it["type"]] = dist.get(it["type"], 0) + 1
        print(f"[dist] 类型分布：{dist}")
        print("[done] 清洗切分完成！可进入 BPE 分词与训练阶段")


if __name__ == "__main__":
    CleanPipeline().run()
