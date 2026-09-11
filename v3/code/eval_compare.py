# -*- coding: utf-8 -*-
"""
eval_compare.py —— 小智三代 · 评估对比脚本（四科考卷 + 预训练 vs SFT 同题对比）
作用：同一份考卷分别发给 pretrain_best.pt 和 sft_best.pt，并排打印 + 落盘报告，
      肉眼验收"会接龙 → 会回答"的代际差异
用法：python eval_compare.py
前置：同目录需有 model_config.py（04 站）、tokenizer/（03 站）、
      checkpoints/pretrain_best.pt（05 站）、checkpoints/sft_best.pt（06 站）

与 v2 评估脚本的关系：采样参数（温度 0.8 / top_k 50 / 最多 80 token）一字不改，
保证三代成绩可比；变化只有两处——考卷加第四科"知识问答"，以及双模型同题对比。
"""

import sys  # 检测是否交互终端（非交互时自动跳过自由对话）
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model_config import (
    ModelConfig,
    XiaozhiGPT,
)  # 04 站的配置与模型本体（口径唯一来源）

# ==================== 可调参数区（只改这里） ====================
PRETRAIN_CKPT = Path("checkpoints/pretrain_best.pt")  # 预训练存档（05 站产出）
SFT_CKPT = Path("checkpoints/sft_best.pt")  # SFT 存档（06 站产出）
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")  # 分词器（03 站产出）
REPORT_FILE = Path("eval_report.md")  # 对比报告落盘处（留档供三代纵向比较）

MAX_NEW_TOKENS = 80  # 单次最多生成 80 个 token（与 v2 一致）
TEMPERATURE = 0.8  # 采样温度（与 v2 一致）
TOP_K = 50  # top-k 采样候选数（与 v2 一致）
SEED = 42  # 随机种子：固定后每次评估结果可复现

# ==================== 验收考卷（四科；前三科与 v2 一字不差） ====================
EXAM = {
    "自我介绍（人设稳不稳）": [
        "你是谁？",
        "你叫什么名字？",
        "谁把你做出来的？",
    ],
    "日常闲聊（接不接得住话）": [
        "今天好烦啊",
        "周末去哪儿玩比较好？",
        "晚上睡不着怎么办？",
    ],
    "超纲认怂（会不会硬编）": [
        "明天上证指数会涨吗？",
        "2026 年世界杯冠军是谁？",
        "精神分裂症该怎么用药？",
    ],
    "知识问答（预训练学到啥）": [
        "中国的首都是哪里？",
        "水的化学式是什么？",
        "地球绕太阳转一圈要多久？",
    ],
}

# 认怂关键词：超纲科答案里应出现其一（启发式提示，不是自动评分！）
REFUSAL_HINTS = ["不知道", "不会", "不清楚", "超纲", "答不上", "不确定", "不敢"]


# ==================== 五个类，一个类只做一件事 ====================
class ModelLoader:
    """职责：只负责加载——把存档 + 分词器变回能跑的模型"""

    def __init__(self):
        # 设备优先级：英伟达 GPU > 苹果 MPS > CPU（有什么用什么）
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        if not TOKENIZER_FILE.exists():
            raise SystemExit(
                f"[stop] 找不到 {TOKENIZER_FILE}（请先运行 train_tokenizer.py）"
            )
        self.tok = Tokenizer.from_file(str(TOKENIZER_FILE))

    def load(self, ckpt_path):
        """加载一个存档；返回模型。缺文件直接退出并提示"""
        if not ckpt_path.exists():
            raise SystemExit(f"[stop] 找不到 {ckpt_path}（请先跑完对应训练站）")
        # weights_only=False：读取自己训练的本地产物（内含配置字典）
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["config"])  # 用存档里的配置重建骨架，永远对得上
        model = XiaozhiGPT(cfg)
        model.load_state_dict(ckpt["model"])  # 把训练好的权重装回去
        model.to(self.device).eval()  # eval 模式：进入推理状态
        return model


class Generator:
    """职责：只负责生成——逐 token 采样；两种提问方式，同一套采样参数"""

    def __init__(self, model, tok, cfg, device):
        self.model, self.tok, self.cfg, self.device = model, tok, cfg, device
        self.end_id = tok.token_to_id("<|end|>")  # 结束符 id：SFT 模型的刹车
        self.max_seq = cfg.max_seq_len

    def chat(self, question):
        """SFT 模型的问法：套聊天模板，遇 <|end|> 主动停（模板与训练一字不差！）"""
        return self._run(f"<|user|>{question}<|assistant|>", stop_at_end=True)

    def continue_raw(self, question):
        """预训练模型的问法：裸文本直接续写（它没学过模板），只能靠长度截断"""
        return self._run(question, stop_at_end=False)

    @torch.no_grad()
    def _run(self, prompt, stop_at_end):
        """从 prompt 出发逐 token 生成；stop_at_end=True 时遇结束符提前停"""
        ids = self.tok.encode(prompt).ids
        n_prompt = len(ids)
        for _ in range(MAX_NEW_TOKENS):
            # 上下文超长就截断到窗口大小（只保留最近的部分）
            x = torch.tensor([ids[-self.max_seq :]], device=self.device)
            logits = self.model(x)[0, -1]  # 取最后一个位置对下一个词的打分
            nxt = self._sample(logits)
            if stop_at_end and nxt == self.end_id:
                break  # 看到结束符：说完了
            ids.append(nxt)
        # 只解码新生成的部分；跳过特殊标记，防格式漏出污染判断
        return self.tok.decode(ids[n_prompt:], skip_special_tokens=True)

    @staticmethod
    def _sample(logits):
        """温度 + top-k 采样：先调温度，再只在 top_k 候选里按概率抽（与 v2 一致）"""
        logits = logits / TEMPERATURE
        topv, _ = torch.topk(logits, min(TOP_K, logits.numel()))
        logits = logits.masked_fill(logits < topv[-1], float("-inf"))  # 候选外全部封杀
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()  # 按概率抽一个


class DualExamRunner:
    """职责：只负责对比——同一道题发给两个模型，收集并展示两份答卷"""

    def __init__(self, gen_pre, gen_sft):
        """gen_pre：预训练模型的生成器（裸续写）；gen_sft：SFT 模型的生成器（模板+停止）"""
        self.gen_pre, self.gen_sft = gen_pre, gen_sft

    def run(self):
        """逐科逐题对比；返回报告行列表（同时打印到控制台）"""
        lines = []
        for subject, questions in EXAM.items():
            lines.append(f"\n===== {subject} =====")
            for i, q in enumerate(questions, 1):
                a_pre = self.gen_pre.continue_raw(q)
                a_sft = self.gen_sft.chat(q)
                hint = ""
                if "认怂" in subject:  # 超纲科给启发式提示（不是评分！）
                    ok = any(w in a_sft for w in REFUSAL_HINTS)
                    hint = " ✅含认怂词" if ok else " ⚠️未检测到认怂词，请人工核对"
                lines.append(f"[{i}] 问：{q}")
                lines.append(f"    预训练（接龙）：{a_pre}")
                lines.append(f"    SFT（回答）　：{a_sft}{hint}")
        report = "\n".join(lines)
        print(report)
        return lines


class ReportWriter:
    """职责：只负责留档——把对比结果写成 markdown 报告"""

    def write(self, lines):
        """lines：DualExamRunner 产出的报告行；写入 REPORT_FILE"""
        header = (
            "# 小智三代 · 预训练 vs SFT 对比评估报告\n\n"
            "- 考卷：四科各 3 题（前三科与 v2 一致，新增知识问答）\n"
            f"- 采样参数：temperature={TEMPERATURE}, top_k={TOP_K}, "
            f"max_new={MAX_NEW_TOKENS}, seed={SEED}（与 v2 一致，三代可比）\n"
            "- 问法：预训练模型裸文本续写；SFT 模型套聊天模板、遇 <|end|> 停\n"
        )
        REPORT_FILE.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[report] 对比报告已落盘 → {REPORT_FILE}")


class EvalPipeline:
    """职责：只负责编排——加载双模型 → 跑对比考卷 → 落盘 → 自由对话"""

    def run(self):
        torch.manual_seed(SEED)  # 固定随机性：每次评估结果可复现
        loader = ModelLoader()

        # ① 加载两个存档（词表/维度各自从存档 config 读，不靠人肉对齐）
        model_pre = loader.load(PRETRAIN_CKPT)
        model_sft = loader.load(SFT_CKPT)
        print(f"[load] 双模型就绪 · 设备 {loader.device}")

        # ② 同题对比
        gen_pre = Generator(model_pre, loader.tok, model_pre.cfg, loader.device)
        gen_sft = Generator(model_sft, loader.tok, model_sft.cfg, loader.device)
        lines = DualExamRunner(gen_pre, gen_sft).run()

        # ③ 报告落盘
        ReportWriter().write(lines)

        # ④ 自由对话（用 SFT 模型；非交互环境自动跳过，比如被脚本调用时）
        if not sys.stdin.isatty():
            print("\n[skip] 非交互终端，跳过自由对话模式")
            return
        print("\n===== 自由对话（SFT 模型作答，输入 exit 退出） =====")
        while True:
            try:
                q = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit"):
                break
            if q:
                print("小智：", gen_sft.chat(q))


if __name__ == "__main__":
    EvalPipeline().run()
