# -*- coding: utf-8 -*-
"""
train.py —— 小智二代 · 训练脚本（打包流 + 验证集监考 + 早停）
作用：把 train.jsonl 编码成 token 流，训练 XiaozhiGPT，产出最佳存档 checkpoint.pt
用法：python train.py
前置：同目录需有 model_config.py（04 站）、tokenizer/（03 站）、train.jsonl + val.jsonl（02 站）
"""

import json  # 读取 jsonl 教材
import math  # 余弦退火要算 cos
import time  # 统计训练耗时
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model_config import ModelConfig, XiaozhiGPT  # 04 站的配置与模型本体

# ==================== 可调参数区（只改这里） ====================
TRAIN_FILE = Path("train.jsonl")  # 训练集（02 站产出）
VAL_FILE = Path("val.jsonl")  # 验证集（监考卷）
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")  # 分词器（03 站产出）
CKPT_FILE = Path("checkpoint.pt")  # 存档：只存验证 loss 最佳的那一刻

SEQ_LEN = 512  # 训练窗口长度：与模型 max_seq_len 一致
BATCH_SIZE = 32  # 每批抽 32 个窗口
TOTAL_STEPS = 300  # 总步数：试跑改 50，小数据 2000 步足够收敛
LR = 4e-4  # 峰值学习率（比一代低一个量级）
WARMUP_STEPS = 100  # 预热步数：学习率从 0 线性爬到峰值
VAL_EVERY = 100  # 每 200 步用验证集监考一次
VAL_BATCHES = 8  # 每次监考抽 8 批取平均，降低偶然性
PATIENCE = 3  # 连续 3 次监考没进步 → 早停
GRAD_CLIP = 1.0  # 梯度裁剪阈值：防单步梯度爆炸
SEED = 42  # 随机种子：固定后结果可复现


# ==================== 五个类，一个类只做一件事 ====================
class TokenStreamer:
    """职责：只负责预分词与打包——jsonl 文本 → 一条长长的 token 流"""

    def __init__(self, tok_path):
        """加载 03 站训练好的分词器"""
        if not tok_path.exists():
            raise SystemExit(f"[stop] 找不到 {tok_path}（请先运行 train_tokenizer.py）")
        self.tok = Tokenizer.from_file(str(tok_path))

    def encode_file(self, path):
        """把一份 jsonl 教材编码并首尾拼接成 token 流；返回 list[int]"""
        if not path.exists():
            raise SystemExit(f"[stop] 找不到 {path}（请先运行 clean_split.py）")
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError):
                continue  # 坏行跳过
            ids.extend(self.tok.encode(text).ids)  # 直接续接，不浪费一个 token
        return ids


class BatchSampler:
    """职责：只负责组批——从 token 流里随机切窗口，每批内容都不同"""

    def __init__(self, ids, seq_len):
        # 存成 tensor 放内存：小语料切片开销可忽略
        self.data = torch.tensor(ids, dtype=torch.long)
        # 实际窗口自适应缩短：验证集很小时也能监考，不至于崩溃
        # 注意要 -2：窗口本身占 win 个 token，右移一位的答案还要多占 1 个
        self.win = min(seq_len, len(self.data) - 2)
        if self.win < 2:
            raise SystemExit("[stop] token 流太短，凑不出训练窗口（先补数据）")

    def sample(self, batch_size, device):
        """随机抽 batch_size 个窗口；x 是输入，y 是右移一位的标准答案"""
        hi = len(self.data) - self.win - 1
        starts = torch.randint(0, hi, (batch_size,))
        # 类似 Python 的列表推导：逐起点切窗口再堆叠成批
        x = torch.stack([self.data[s : s + self.win] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.win + 1] for s in starts])
        return x.to(device), y.to(device)


class LrScheduler:
    """职责：只负责学习率——预热线性爬升，随后余弦退火到峰值的 1/10"""

    def at(self, step):
        """返回第 step 步应使用的学习率"""
        if step < WARMUP_STEPS:
            return LR * (step + 1) / WARMUP_STEPS  # 预热：防开局大步翻车
        t = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
        return LR * (0.1 + 0.9 * (1 + math.cos(math.pi * t)) / 2)  # 余弦退火


class Trainer:
    """职责：只负责训练循环——前向、反向、监考、早停、存档"""

    def __init__(self, cfg, train_ids, val_ids):
        self.device = self._pick_device()
        self.model = XiaozhiGPT(cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.train_batch = BatchSampler(train_ids, SEQ_LEN)
        self.val_batch = BatchSampler(val_ids, SEQ_LEN)
        self.sched = LrScheduler()

    @staticmethod
    def _pick_device():
        """设备优先级：英伟达 GPU > 苹果 MPS > CPU（有什么用什么）"""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def eval_loss(self):
        """验证集监考：抽固定批数算平均 loss，全程不更新参数"""
        self.model.eval()
        losses = []
        for _ in range(VAL_BATCHES):
            x, y = self.val_batch.sample(BATCH_SIZE, self.device)
            logits = self.model(x)
            # 交叉熵：模型猜下一个词 vs 标准答案，逐 token 算分
            losses.append(
                F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            )
        self.model.train()
        return sum(losses) / len(losses)

    def run(self):
        """主循环：训练 TOTAL_STEPS 步，期间定期监考，不行就早停"""
        best_val, bad_rounds = float("inf"), 0
        t0 = time.time()
        for step in range(TOTAL_STEPS):
            # 学习率每步重算并写回优化器（调度器的正确接法）
            lr = self.sched.at(step)
            for g in self.opt.param_groups:
                g["lr"] = lr

            # ---- 标准五步走：取批 → 前向 → 清零 → 反向 → 更新 ----
            x, y = self.train_batch.sample(BATCH_SIZE, self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.opt.step()

            if step % 50 == 0:
                print(f"[train] step {step:>5} | loss {loss.item():.4f} | lr {lr:.2e}")

            # ---- 定期监考：刷新纪录才存档，连续不进步就早停 ----
            if step % VAL_EVERY == 0 or step == TOTAL_STEPS - 1:
                val = self.eval_loss()
                if val < best_val:
                    best_val, bad_rounds = val, 0
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "config": vars(self.model.cfg),
                        },
                        CKPT_FILE,
                    )
                    print(
                        f"[val] step {step:>5} | val_loss {val:.4f} | best {best_val:.4f} ← 新纪录，已存档"
                    )
                else:
                    bad_rounds += 1
                    print(
                        f"[val] step {step:>5} | val_loss {val:.4f} | best {best_val:.4f}（{bad_rounds}/{PATIENCE}）"
                    )
                    if bad_rounds >= PATIENCE:
                        print(f"[early-stop] 验证集连续 {PATIENCE} 次没进步，提前收工")
                        break
        print(
            f"[done] 训练结束 | 最佳验证 loss {best_val:.4f} | 存档 → {CKPT_FILE} | 用时 {time.time()-t0:.0f}s"
        )


class TrainPipeline:
    """职责：只负责编排——供料 → 组建 → 开训"""

    def run(self):
        torch.manual_seed(SEED)  # 固定随机性：同样的数据同样的开局

        # ① 供料：教材 → token 流
        streamer = TokenStreamer(TOKENIZER_FILE)
        train_ids = streamer.encode_file(TRAIN_FILE)
        val_ids = streamer.encode_file(VAL_FILE)
        print(
            f"[data] 训练流 {len(train_ids):,} tokens · 验证流 {len(val_ids):,} tokens"
        )

        # ② 组建：词表以分词器的实际大小为准（可能没练满 8192）
        cfg = ModelConfig(vocab_size=streamer.tok.get_vocab_size())
        trainer = Trainer(cfg, train_ids, val_ids)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"[model] 参数 {n_params/1e6:.2f}M · 设备 {trainer.device}")

        # ③ 开训
        trainer.run()


if __name__ == "__main__":
    TrainPipeline().run()
