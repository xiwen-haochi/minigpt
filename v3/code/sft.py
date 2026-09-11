# -*- coding: utf-8 -*-
"""
sft.py —— 小智三代 · SFT 微调脚本（loss mask + 加载预训练存档 + 半精度自适应）
作用：在 pretrain_best.pt 的基础上，用模板化对话数据教模型"问→答→主动停下"，
      产出 checkpoints/sft_best.pt（只存验证 loss 最佳的那一刻）
用法：python sft.py
前置：同目录需有 model_config.py（04 站）、tokenizer/（03 站）、
      sft_train.jsonl + sft_val.jsonl（02 站产出）、checkpoints/pretrain_best.pt（05 站产出）

与预训练脚本的关键差异：
  ① loss mask——user 段是"题目"不是"作业"，labels 里只保留 assistant 段（含 <|end|>），
     其余位置填 -100，cross_entropy 的 ignore_index 会跳过它们
  ② 定长样本而不是 token 长河——对话样本有边界，不能首尾相接；每条 pad 到 SEQ_LEN+1，
     ids/labs 双数组落盘，训练时 memmap 组批，内存与数据量脱钩（同 05 站思路）
  ③ 起点是预训练权重而不是随机初始化；学习率降到 1/5（2e-4），防"灾难性遗忘"
  ④ 半精度自适应与梯度检查点原样沿用 05 站（T4 等老卡自动 fp16 + GradScaler）
"""

import json  # 读取 jsonl 教材
import math  # 余弦退火要算 cos
import time  # 统计训练耗时
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

import numpy as np  # memmap 按需读定长样本（numpy 随 torch 一起装好）
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model_config import (
    ModelConfig,
    XiaozhiGPT,
)  # 04 站的配置与模型本体（口径唯一来源）

# ==================== 可调参数区（只改这里） ====================
TRAIN_FILE = Path("sft_train.jsonl")  # 训练集（02 站产出，已套聊天模板）
VAL_FILE = Path("sft_val.jsonl")  # 验证集（监考卷，02 站产出）
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")  # 分词器（03 站产出）
PRETRAIN_CKPT = Path(
    "checkpoints/pretrain_best.pt"
)  # 起点：预训练最佳存档（05 站产出）
BIN_DIR = Path("sft_bin")  # 定长样本落盘处：train_ids/labs.bin、val_ids/labs.bin
CKPT_FILE = Path("checkpoints/sft_best.pt")  # 存档：只存验证 loss 最佳那一刻

SEQ_LEN = 512  # 窗口长度：与模型 max_seq_len 一致；超长的样本截断（断尾回答不算分）
BATCH_SIZE = 32  # 与预训练同档；显存有检查点兜底
TOTAL_STEPS = 12000  # SFT 步数宜少不宜多：约 0.4 epoch，过训会洗掉预训练的语言能力
LR = 2e-4  # 峰值学习率：预训练的 1/5（规划口径），微调要"精修"不是"重练"
WARMUP_STEPS = 200  # 预热步数：微调开局更短
MIN_LR_RATIO = 0.1  # 余弦退火地板：最低降到峰值的 1/10
VAL_EVERY = 500  # 每 500 步用验证集监考一次
VAL_BATCHES = 16  # 每次监考抽 16 批取平均
PATIENCE = 4  # 连续 4 次监考没进步 → 早停（微调比预训练更容易过拟合）
GRAD_CLIP = 1.0  # 梯度裁剪阈值
SEED = 42  # 随机种子
GRAD_CKPT = True  # 梯度检查点：与预训练相同的显存兜底

IGNORE_ID = -100  # 不算分位置的标签值：cross_entropy 的 ignore_index 约定
ENCODE_FLUSH_LINES = 10_000  # 编码攒批行数：内存占用恒定


# ==================== 五个类，一个类只做一件事 ====================
class SftBinBuilder:
    """职责：只负责编码与打 mask——模板化文本 → 定长 (ids, labels) 样本落盘"""

    def __init__(self, tok_path):
        """加载分词器，并取出两个特殊标记的 id（它们各自只占一个 token，03 站已体检）"""
        if not tok_path.exists():
            raise SystemExit(f"[stop] 找不到 {tok_path}（请先运行 train_tokenizer.py）")
        self.tok = Tokenizer.from_file(str(tok_path))
        self.a_id = self.tok.token_to_id("<|assistant|>")
        self.e_id = self.tok.token_to_id("<|end|>")
        assert self.a_id is not None and self.e_id is not None, "分词器缺特殊标记"
        self.width = SEQ_LEN + 1  # 每条样本定长：输入取前 512，标签错位取后 512

    def encode_one(self, text):
        """一条模板化文本 → (ids, labs) 两个定长数组；labs 只在 assistant 段有真值"""
        ids = self.tok.encode(text).ids[: self.width]  # 超长截断：断尾回答自然不完整
        labs = [IGNORE_ID] * len(ids)
        # 扫描 assistant 段：从 <|assistant|> 的下一个 token 到 <|end|>（含）才算分
        # 类似 Python 的双指针扫描；截断导致 <|end|> 丢失的段自动整段不算分
        i, supervised = 0, 0
        while i < len(ids):
            if ids[i] == self.a_id:
                j = i + 1
                while j < len(ids) and ids[j] != self.e_id:
                    j += 1
                if j < len(ids):  # 找到配对的 <|end|>，这一段是完整回答
                    for k in range(i + 1, j + 1):
                        labs[k] = ids[k]
                        supervised += 1
                i = j + 1
            else:
                i += 1
        # pad 到定长：ids 补 0（反正不算分），labs 补 IGNORE_ID
        pad = self.width - len(ids)
        ids = ids + [0] * pad
        labs = labs + [IGNORE_ID] * pad
        return ids, labs, supervised

    def build(self, jsonl_path, ids_bin, labs_bin):
        """流式编码整份 jsonl 落盘；已存在则跳过（断点友好）。返回样本数"""
        if not jsonl_path.exists():
            raise SystemExit(
                f"[stop] 找不到 {jsonl_path}（请先运行 normalize_data.py）"
            )
        if ids_bin.exists() and ids_bin.stat().st_size > 0:
            n = ids_bin.stat().st_size // (self.width * 4)  # uint32 每样本 width×4 字节
            print(f"[bin] {ids_bin} 已存在（{n:,} 条样本），跳过编码")
            return n

        n, bad, total_tok, sup_tok = 0, 0, 0, 0
        ids_buf, labs_buf = [], []
        with open(jsonl_path, encoding="utf-8") as fin, open(
            ids_bin, "wb"
        ) as f_ids, open(labs_bin, "wb") as f_labs:
            for line in fin:
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    bad += 1
                    continue  # 坏行跳过
                ids, labs, sup = self.encode_one(text)
                if sup == 0:
                    bad += 1
                    continue  # 没有完整回答的样本不算教材
                ids_buf.extend(ids)
                labs_buf.extend(labs)
                total_tok += len(ids)  # 含 pad 前的有效长度另算无妨，比值仅供肉眼参考
                sup_tok += sup
                n += 1
                if n % ENCODE_FLUSH_LINES == 0:
                    np.asarray(ids_buf, dtype=np.uint32).tofile(f_ids)
                    np.asarray(labs_buf, dtype=np.int32).tofile(f_labs)
                    ids_buf.clear()
                    labs_buf.clear()
                    print(f"[encode] 已处理 {n:,} 条")
            if ids_buf:  # 尾批
                np.asarray(ids_buf, dtype=np.uint32).tofile(f_ids)
                np.asarray(labs_buf, dtype=np.int32).tofile(f_labs)
        ratio = sup_tok / max(total_tok, 1)
        print(
            f"[bin] {jsonl_path.name} → {n:,} 条样本（跳过 {bad} 条）· "
            f"算分 token 占比 {ratio:.1%}"
        )
        return n


class PaddedSampler:
    """职责：只负责组批——memmap 打开定长样本，随机抽条数（OS 按需调页，不占 RAM）"""

    def __init__(self, ids_bin, labs_bin, width):
        # reshape 成 (样本数, width)：类似 Python 的等长二维列表，但数据在磁盘上
        self.ids = np.memmap(ids_bin, dtype=np.uint32, mode="r").reshape(-1, width)
        self.labs = np.memmap(labs_bin, dtype=np.int32, mode="r").reshape(-1, width)
        if len(self.ids) < 1:
            raise SystemExit(f"[stop] {ids_bin} 为空（先补数据）")

    def sample(self, batch_size, device):
        """随机抽 batch_size 条；x 取前 512，y 取错一位的后 512（含 mask）"""
        idx = np.random.randint(0, len(self.ids), size=batch_size)
        # memmap 切片转 int64 才能进 Embedding / cross_entropy
        x = torch.from_numpy(self.ids[idx].astype(np.int64)[:, :-1])
        y = torch.from_numpy(self.labs[idx].astype(np.int64)[:, 1:])
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


class SftTrainer:
    """职责：只负责训练循环——加载预训练权重、半精度训练、监考、早停、存档"""

    def __init__(self, train_bins, val_bins):
        self.device = self._pick_device()
        self.model = self._load_pretrained().to(self.device)
        self.model.grad_ckpt = GRAD_CKPT  # 打开梯度检查点：重算换显存（OOM 的解药）
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)
        w = SEQ_LEN + 1
        self.train_batch = PaddedSampler(train_bins[0], train_bins[1], w)
        self.val_batch = PaddedSampler(val_bins[0], val_bins[1], w)
        self.sched = LrScheduler()
        # 混合精度只在 cuda 上开启（mps/cpu 冒烟用 fp32 更稳）。
        # 半精度自动二选一：算力 8.0（Ampere）起有 bf16 张量核 → bf16；
        # T4 等老卡（算力 7.5）只有 fp16 张量核 → fp16（靠 GradScaler 防梯度下溢）
        self.amp = self.device == "cuda"
        self.bf16 = self.amp and torch.cuda.get_device_capability() >= (8, 0)
        self.dtype = torch.bfloat16 if self.bf16 else torch.float16
        # GradScaler 只有 fp16 需要；bf16 / 不开混合精度时它是关闭态，各方法自动退化为透传
        try:
            # torch≥2.3 的新接口（旧接口已弃用告警）
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=self.amp and not self.bf16
            )
        except (TypeError, AttributeError):
            # torch<2.3 的旧接口兜底
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp and not self.bf16)

    @staticmethod
    def _pick_device():
        """设备优先级：英伟达 GPU > 苹果 MPS > CPU（有什么用什么）"""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_pretrained():
        """从 05 站存档恢复模型：超参从存档里的 config 读，不靠人肉对齐"""
        if not PRETRAIN_CKPT.exists():
            raise SystemExit(f"[stop] 找不到 {PRETRAIN_CKPT}（请先跑完预训练一站）")
        ckpt = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["config"])  # 词表/维度/层数全部沿用存档口径
        model = XiaozhiGPT(cfg)
        model.load_state_dict(ckpt["model"])
        print(
            f"[init] 已加载预训练存档（step {ckpt.get('step')}，"
            f"val_loss {ckpt.get('val_loss'):.4f}）"
        )
        return model

    def _forward_loss(self, x, y):
        """混合精度前向并算交叉熵；ignore_index=-100 让 user 段/pad 不参与打分"""
        with torch.autocast(
            device_type=self.device, dtype=self.dtype, enabled=self.amp
        ):
            logits = self.model(x)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE_ID
            )

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
        """存档：与预训练同款格式（model/config/step/val_loss），评估站直接加载"""
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
            # GradScaler 三件套：放大 loss 防 fp16 梯度下溢 → 还原梯度后裁剪 → 无异常才更新
            # bf16 或不开混合精度时 scaler 是关闭态，下面五行自动退化为普通 backward/clip/step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.scaler.step(self.opt)
            self.scaler.update()

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


class SftPipeline:
    """职责：只负责编排——编码落盘 → 加载预训练模型 → 开训"""

    def run(self):
        torch.manual_seed(SEED)  # 固定随机性
        np.random.seed(SEED)  # 组批用 numpy 随机，一并固定
        BIN_DIR.mkdir(exist_ok=True)

        # ① 供料：模板化 jsonl → 定长 (ids, labs) 样本（已编码过会自动跳过）
        builder = SftBinBuilder(TOKENIZER_FILE)
        train_bins = (BIN_DIR / "train_ids.bin", BIN_DIR / "train_labs.bin")
        val_bins = (BIN_DIR / "val_ids.bin", BIN_DIR / "val_labs.bin")
        n_train = builder.build(TRAIN_FILE, *train_bins)
        n_val = builder.build(VAL_FILE, *val_bins)
        print(f"[data] 训练样本 {n_train:,} 条 · 监考样本 {n_val:,} 条")

        # ② 组建：加载预训练存档（config 从存档读，词表与分词器天然一致）
        trainer = SftTrainer(train_bins, val_bins)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        if not trainer.amp:
            prec = "关"
        elif trainer.bf16:
            prec = "bf16"
        else:
            prec = "fp16 + GradScaler"
        print(
            f"[model] 参数 {n_params/1e6:.2f}M · 设备 {trainer.device} · "
            f"混合精度 {prec}"
        )

        # ③ 开训
        trainer.run()


if __name__ == "__main__":
    SftPipeline().run()
