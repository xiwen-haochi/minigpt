# -*- coding: utf-8 -*-
"""
model.py —— v4 共享模型定义（Llama 风格 Decoder-only Transformer）

现代主流架构三件套：
  1. RMSNorm            比 LayerNorm 更快更稳（Llama / Qwen 同款）
  2. RoPE 旋转位置编码   比绝对位置编码外推性更好
  3. SwiGLU 前馈网络     比 GELU-MLP 表达能力更强

本文件被 03_pretrain.py / 04_sft.py / 05_chat.py 共同 import。
"""
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """模型超参数集合。

    @dataclass 是 Python 标准库装饰器，自动生成 __init__，
    用法类似"带类型的字典"：GPTConfig(n_layer=12, ...)
    """

    vocab_size: int = 32000  # 词表大小（必须和分词器一致）
    max_seq_len: int = 512  # 最大上下文长度（tokens）
    n_layer: int = 12  # Transformer 层数
    n_head: int = 12  # 多头注意力头数
    n_embd: int = 768  # 隐藏层维度
    dropout: float = 0.0  # 预训练一般不开 dropout


class RMSNorm(nn.Module):
    """RMSNorm：LayerNorm 简化版，只做缩放不做平移。

    公式: x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放系数

    def forward(self, x):
        # 在 float32 下算均方根，防止半精度下溢，再转回原精度
        out = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (self.weight * out).type_as(x)


def build_rope_cache(head_dim, max_seq_len, theta=10000.0):
    """预计算 RoPE 的 cos/sin 缓存表，形状均为 [max_seq_len, head_dim // 2]。

    类似 Python 里先把查表数据算好存起来，前向时直接查表。
    """
    # 每一对维度对应一个频率：维度序号越大频率越低
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()  # 位置序号 0,1,2,...
    freqs = torch.outer(t, inv_freq)  # 外积 -> [max_seq_len, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """把 RoPE 旋转应用到 q/k 上。

    x: [B, T, H, D]，cos/sin: [T, D/2]
    把最后一维劈成两半 (x1, x2)，按复数乘法旋转：
      (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x2*cos + x1*sin)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # cos/sin 升维到 [1, T, 1, D/2] 以便广播，并跟随 x 的精度（fp16/fp32）
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    """因果多头自注意力（RoPE + SDPA 融合算子）。"""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        # qkv 合并成一次线性变换（相当于 Python 一次返回三元组）
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        cos, sin = build_rope_cache(self.head_dim, cfg.max_seq_len)
        # register_buffer：跟随 model.to(device) 搬移，但不是可学习参数
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)  # 各 [B, T, C]
        q = q.view(B, T, self.n_head, self.head_dim)  # 拆成多头
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])  # 注入位置信息
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        # scaled_dot_product_attention：PyTorch 官方融合算子，
        # 自动选择 flash / mem-efficient / 数学回退，cuda/mps/cpu 通用
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,  # 因果掩码：不许偷看未来 token
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头
        return self.proj(y)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络（Llama 同款）。

    相当于把普通 MLP 的 GELU(x@W1)@W2 换成：
      (SiLU(x@W_gate) * (x@W_up)) @ W_down
    多一路门控，表达能力更强。
    """

    def __init__(self, cfg):
        super().__init__()
        hidden = int(4 * cfg.n_embd * 2 / 3)  # Llama 惯例：扩 4 倍再打 2/3 折
        hidden = (hidden + 63) // 64 * 64  # 对齐到 64 的倍数，利于 GPU 吞吐
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """一层 Transformer：Pre-Norm 结构（先归一化再进子层，残差直连）。"""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # 残差连接，类似 Python 的 x += f(x)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """完整的 GPT：词嵌入 -> N 层 Block -> RMSNorm -> 输出头。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 权重共享（weight tying）：输入嵌入和输出头共用一张表，省参数且更稳
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: [B, T] 的 token id -> 返回 [B, T, vocab_size] 的 logits。"""
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm_f(x))


# ----------------------------------------------------------------------
# 设备 / 精度工具：cuda（新卡 bf16、老卡 fp16）/ mps / cpu 三平台自动适配
# ----------------------------------------------------------------------


def pick_device():
    """自动选择设备和混合精度策略。

    返回 (device, amp_dtype)：
      CUDA 且支持 bf16 -> (cuda, torch.bfloat16)   新卡（A100 / 30 系+）
      CUDA 不支持 bf16 -> (cuda, torch.float16)    老卡（V100 / 1080Ti 等），配 GradScaler
      Apple Silicon    -> (mps, None)              MPS 上半精度不稳，直接 fp32
      其他             -> (cpu, None)              fp32
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), None
    return torch.device("cpu"), None


def amp_context(device, amp_dtype):
    """生成 autocast 上下文；amp_dtype 为 None 时是空上下文（全程 fp32）。"""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_p=0.9, eos_id=None):
    """自回归生成：温度采样 + top-p（核采样）。

    参数:
        model: GPT 模型
        idx: [1, T] 起始 token（torch.long）
        max_new_tokens: 最多新生成多少个 token
        temperature: 温度，越小越保守；<=0 时退化为贪心（argmax）
        top_p: 核采样阈值，只从累计概率前 p 的词里挑
        eos_id: 遇到该 token 提前停止（None 表示不提前停）
    返回:
        [1, T + n] 的完整 token 序列
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.max_seq_len :]  # 超出上下文就截掉最左边
        logits = model(idx_cond)[:, -1, :]  # 只取最后一个位置的分布
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)  # 贪心
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            # top-p：排序后砍掉累计概率超过 p 的长尾
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return idx
