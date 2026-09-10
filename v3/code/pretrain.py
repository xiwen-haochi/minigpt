# -*- coding: utf-8 -*-
"""
pretrain.py —— 小智三代 · 预训练脚本（token 长河 + 随机窗口 + 混合精度 + 早停存档）
作用：把 pretrain_train.jsonl 编码成二进制 token 长河，训练 XiaozhiGPT，
      产出 checkpoints/pretrain_best.pt（只存验证 loss 最佳的那一刻）
用法：python pretrain.py
前置：同目录需有 model_config.py（04 站）、tokenizer/（03 站）、
      pretrain_train.jsonl + pretrain_val.jsonl（02 站产出）

与 v2 版的关键差异（都是为百万行语料准备的）：
  ① token 流不进内存——v2 把全部 id 装进 Python list，120 万行会吃掉十几 GB 内存；
     v3 流式编码成 uint32 二进制文件，训练时用 memmap 按需调页，RAM 占用恒定
  ② 混合精度——cuda 上用 bf16 autocast 训练，兑现 04 站"每参数 16 字节"的显存账；
     bf16 数值范围与 fp32 相同，不需要 GradScaler
  ③ 训练规模——步数从 500 提到约 1 epoch（≈15000 步），batch 32×512
  ④ 梯度检查点——0.1B×512 窗口的激活显存远超 04 站"1~2GB"的乐观估计：
     注意力/前馈中间结果层层堆积会顶穿 16G 显存。开启后前向只存每层入口、
     反向重算层内激活，用约 30% 速度换显存峰值减半以上（在 model_config.py 实现）
"""

import json  # 读取 jsonl 教材
import math  # 余弦退火要算 cos
import time  # 统计训练耗时
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

import numpy as np  # memmap 按需读 token 流（numpy 随 torch 一起装好）
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model_config import (
    ModelConfig,
    XiaozhiGPT,
)  # 04 站的配置与模型本体（口径唯一来源）

# ==================== 可调参数区（只改这里） ====================
TRAIN_FILE = Path("pretrain_train.jsonl")  # 训练集（02 站产出，约 120 万行）
VAL_FILE = Path("pretrain_val.jsonl")  # 验证集（监考卷，02 站产出）
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")  # 分词器（03 站产出）
BIN_DIR = Path("token_stream")  # token 长河落盘处：train.bin / val.bin（uint32）
CKPT_FILE = Path("checkpoints/pretrain_best.pt")  # 存档：只存验证 loss 最佳那一刻

SEQ_LEN = 512  # 训练窗口长度：与模型 max_seq_len 一致
BATCH_SIZE = 32  # 每步喂 32×512 ≈ 1.6 万 tokens；16G 单卡宽裕，想更快可加到 64
TOTAL_STEPS = (
    15000  # 总步数：≈1 epoch（3 亿 tokens ÷ 每步 1.6 万）；按实际 token 数调整
)
LR = 1e-3  # 峰值学习率：0.1B 随机开局的标准档位（规划口径）
WARMUP_STEPS = 300  # 预热步数：大模型开局更脆，给足 300 步线性爬升
MIN_LR_RATIO = 0.1  # 余弦退火的地板：最低降到峰值的 1/10，不归零
VAL_EVERY = 500  # 每 500 步用验证集监考一次
VAL_BATCHES = 16  # 每次监考抽 16 批取平均，降低偶然性
PATIENCE = 5  # 连续 5 次监考没进步 → 早停（语料大，多给几次机会）
GRAD_CLIP = 1.0  # 梯度裁剪阈值：防单步梯度爆炸
SEED = 42  # 随机种子：固定后结果可复现
# 梯度检查点：开 = 显存安全（推荐）；关 = 快约 30% 但 batch 32×512 会 OOM
GRAD_CKPT = True

# 编码落盘时的攒批行数：每攒这么多行写一次磁盘，内存占用恒定
ENCODE_FLUSH_LINES = 10_000


# ==================== 五个类，一个类只做一件事 ====================
class TokenBinBuilder:
    """职责：只负责流式编码——jsonl 逐行进，uint32 二进制流式出，内存恒定"""

    def __init__(self, tok_path):
        """加载 03 站训练好的分词器"""
        if not tok_path.exists():
            raise SystemExit(f"[stop] 找不到 {tok_path}（请先运行 train_tokenizer.py）")
        self.tok = Tokenizer.from_file(str(tok_path))

    def build(self, jsonl_path, bin_path):
        """把一份 jsonl 编码成 token 长河落盘；已存在则跳过（断点友好）。返回 token 总数"""
        if not jsonl_path.exists():
            raise SystemExit(
                f"[stop] 找不到 {jsonl_path}（请先运行 normalize_data.py）"
            )
        if bin_path.exists() and bin_path.stat().st_size > 0:
            n = bin_path.stat().st_size // 4  # uint32 每个 token 占 4 字节
            print(f"[bin] {bin_path} 已存在（{n:,} tokens），跳过编码")
            return n

        n_tokens, n_lines, buf = 0, 0, []
        with open(jsonl_path, encoding="utf-8") as fin, open(bin_path, "wb") as fout:
            for line in fin:
                n_lines += 1
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue  # 坏行跳过
                buf.extend(self.tok.encode(text).ids)  # 首尾相接，不浪费一个 token
                # 攒够一批就落盘清空：buf 长度有界，内存与语料规模无关
                if n_lines % ENCODE_FLUSH_LINES == 0:
                    n_tokens += len(buf)
                    np.asarray(buf, dtype=np.uint32).tofile(fout)
                    buf.clear()
                    print(f"[encode] 已处理 {n_lines:,} 行 / {n_tokens:,} tokens")
            if buf:  # 尾批
                n_tokens += len(buf)
                np.asarray(buf, dtype=np.uint32).tofile(fout)
        print(
            f"[bin] {jsonl_path.name} → {bin_path}：{n_lines:,} 行 / {n_tokens:,} tokens"
        )
        return n_tokens


class WindowSampler:
    """职责：只负责组批——memmap 打开 token 长河，随机切窗口（OS 按需调页，不占 RAM）"""

    def __init__(self, bin_path, seq_len):
        # memmap 类似 Python 的"惰性列表"：访问哪段才从磁盘读哪段
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        # 实际窗口自适应缩短：验证集很小时也能监考，不至于崩溃
        # 注意要 -2：窗口本身占 win 个 token，右移一位的答案还要多占 1 个
        self.win = min(seq_len, len(self.data) - 2)
        if self.win < 2:
            raise SystemExit(f"[stop] {bin_path} 太短，凑不出训练窗口（先补数据）")

    def sample(self, batch_size, device):
        """随机抽 batch_size 个窗口；x 是输入，y 是右移一位的标准答案"""
        hi = len(self.data) - self.win - 1
        starts = torch.randint(0, hi, (batch_size,))
        # memmap 切片是 uint32，转成 int64 才能进 Embedding（类似列表推导逐段取出再堆叠）
        x = torch.stack(
            [
                torch.from_numpy(self.data[s : s + self.win].astype(np.int64))
                for s in starts
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.data[s + 1 : s + self.win + 1].astype(np.int64))
                for s in starts
            ]
        )
        return x.to(device), y.to(device)


class LrScheduler:
    """职责：只负责学习率——预热线性爬升，随后余弦退火到峰值的 1/10"""

    def at(self, step):
        """返回第 step 步应使用的学习率"""
        if step < WARMUP_STEPS:
            return LR * (step + 1) / WARMUP_STEPS  # 预热：防开局大步翻车
        t = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
        return LR * (
            MIN_LR_RATIO + (1 - MIN_LR_RATIO) * (1 + math.cos(math.pi * t)) / 2
        )


class Trainer:
    """职责：只负责训练循环——混合精度前向、反向、监考、早停、存档"""

    def __init__(self, cfg, train_bin, val_bin):
        self.device = self._pick_device()
        self.model = XiaozhiGPT(cfg).to(self.device)
        self.model.grad_ckpt = GRAD_CKPT  # 打开梯度检查点：重算换显存（OOM 的解药）
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.train_batch = WindowSampler(train_bin, SEQ_LEN)
        self.val_batch = WindowSampler(val_bin, SEQ_LEN)
        self.sched = LrScheduler()
        # bf16 autocast 只在 cuda 上开启（mps/cpu 上 fp32 更稳，反正冒烟用）
        self.amp = self.device == "cuda"

    @staticmethod
    def _pick_device():
        """设备优先级：英伟达 GPU > 苹果 MPS > CPU（有什么用什么）"""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _forward_loss(self, x, y):
        """混合精度前向并算交叉熵：autocast 内部自动选 bf16/fp32，外部代码无感"""
        with torch.autocast(
            device_type=self.device, dtype=torch.bfloat16, enabled=self.amp
        ):
            logits = self.model(x)
            # 交叉熵：模型猜下一个词 vs 标准答案，逐 token 算分
            return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    @torch.no_grad()
    def eval_loss(self):
        """验证集监考：抽固定批数算平均 loss，全程不更新参数"""
        self.model.eval()
        losses = []
        for _ in range(VAL_BATCHES):
            x, y = self.val_batch.sample(BATCH_SIZE, self.device)
            losses.append(self._forward_loss(x, y).item())
        self.model.train()
        return sum(losses) / len(losses)

    def _save(self, step, val_loss):
        """存档：剥掉 torch.compile 可能加的前缀，带上配置供 SFT/评估站直接加载"""
        CKPT_FILE.parent.mkdir(exist_ok=True)
        model = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {
                "model": model.state_dict(),
                "config": vars(model.cfg),  # 超参跟着权重走，加载时不靠猜
                "step": step,
                "val_loss": val_loss,
            },
            CKPT_FILE,
        )

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
            loss = self._forward_loss(x, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()  # bf16 无需 GradScaler：数值范围与 fp32 相同
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.opt.step()

            if step % 100 == 0:
                print(
                    f"[train] step {step:>6} | loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | 用时 {time.time()-t0:.0f}s"
                )

            # ---- 定期监考：刷新纪录才存档，连续不进步就早停 ----
            if step % VAL_EVERY == 0 or step == TOTAL_STEPS - 1:
                val = self.eval_loss()
                if val < best_val:
                    best_val, bad_rounds = val, 0
                    self._save(step, val)
                    print(
                        f"[val] step {step:>6} | val_loss {val:.4f} | "
                        f"best {best_val:.4f} ← 新纪录，已存档"
                    )
                else:
                    bad_rounds += 1
                    print(
                        f"[val] step {step:>6} | val_loss {val:.4f} | "
                        f"best {best_val:.4f}（{bad_rounds}/{PATIENCE}）"
                    )
                    if bad_rounds >= PATIENCE:
                        print(f"[early-stop] 验证集连续 {PATIENCE} 次没进步，提前收工")
                        break
        print(
            f"[done] 训练结束 | 最佳验证 loss {best_val:.4f} | "
            f"存档 → {CKPT_FILE} | 用时 {time.time()-t0:.0f}s"
        )


class PretrainPipeline:
    """职责：只负责编排——编码落盘 → 组建 → 开训"""

    def run(self):
        torch.manual_seed(SEED)  # 固定随机性：同样的数据同样的开局
        BIN_DIR.mkdir(exist_ok=True)

        # ① 供料：jsonl → 二进制 token 长河（已编码过会自动跳过）
        builder = TokenBinBuilder(TOKENIZER_FILE)
        train_bin, val_bin = BIN_DIR / "train.bin", BIN_DIR / "val.bin"
        n_train = builder.build(TRAIN_FILE, train_bin)
        n_val = builder.build(VAL_FILE, val_bin)
        print(f"[data] 训练长河 {n_train:,} tokens · 监考长河 {n_val:,} tokens")

        # ② 组建：词表以分词器的实际大小为准（可能没练满 16384）
        cfg = ModelConfig(vocab_size=builder.tok.get_vocab_size())
        trainer = Trainer(cfg, train_bin, val_bin)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        print(
            f"[model] 参数 {n_params/1e6:.2f}M · 设备 {trainer.device} · "
            f"混合精度 {'bf16' if trainer.amp else '关'}"
        )

        # ③ 开训
        trainer.run()


if __name__ == "__main__":
    PretrainPipeline().run()
