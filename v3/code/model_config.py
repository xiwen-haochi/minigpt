# -*- coding: utf-8 -*-
"""
model_config.py —— 小智三代 · 0.1B 模型配置与预算自检脚本
作用：定义模型超参 → 纯公式算参数/显存预算 → 与分词器词表对账 → PyTorch 真机建模冒烟
用法：python model_config.py（装了 torch 自动做实机验证；没装也能看预算表）
与 v2 的关系：架构（RMSNorm + RoPE + SwiGLU + 权重共享）一行不改，只换规模和数字
"""

import json  # 读取上一站产出的 tokenizer.json，做词表对账
from dataclasses import dataclass  # 把配置打包成结构体，类似带类型提示的 dict
from pathlib import Path  # 跨平台路径处理（macOS / Linux 通用）

# torch 是可选依赖：有它就真机建模，没有就只看预算表（脚本不会崩）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import (
        checkpoint as grad_ckpt_fn,
    )  # 梯度检查点：重算换显存

    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ==================== 可调参数区（只改这里） ====================
@dataclass
class ModelConfig:
    """全部超参集中在这一处：训练和推理脚本都引用它，保证口径唯一"""

    vocab_size: int = 16384  # 词表大小：必须与上一站 tokenizer.json 的实际词表一致！
    d_model: int = 768  # 隐藏维度：v2 的 64 是笔记本妥协，这一代回归主流小模型档位
    n_layers: int = 12  # Transformer 层数：210 万行语料养得起 12 层
    n_heads: int = 12  # 注意力头数：必须能整除 d_model（768/12=每头 64 维）
    d_ff: int = 2048  # 前馈网络中间维度（SwiGLU 三矩阵），约 2.7×d_model
    max_seq_len: int = 512  # 上下文窗口：与规范化脚本的文本长度上限配套

    def __post_init__(self):
        """创建配置时立刻自检，不合规直接报错（防止带病进入训练）"""
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"


# 预算合格区间：210 万行语料（约 3 亿 tokens）匹配 0.08B~0.12B 的模型
PARAM_MIN, PARAM_MAX = 80_000_000, 120_000_000

# 混合精度训练的显存系数（字节/参数）：
# bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + AdamW 一阶/二阶动量 8 = 16 字节
BYTES_PER_PARAM = 16

# 上一站分词器的产出：存在就拿来做词表对账，不存在就跳过（不强制）
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")


# ==================== 预算计算（纯公式，零依赖） ====================
class BudgetCalculator:
    """职责：只负责算账——用公式预估每个部件的参数和显存，不用真的建模"""

    def __init__(self, cfg):
        """cfg：ModelConfig 实例"""
        self.cfg = cfg

    def report(self):
        """逐项计算并打印预算表；返回 (总参数, 嵌入参数)"""
        c = self.cfg
        embed = c.vocab_size * c.d_model  # 词嵌入（权重共享：输出层白嫖它）
        attn = c.n_layers * 4 * c.d_model**2  # 每层 Q/K/V/O 四个方阵
        ffn = c.n_layers * 3 * c.d_model * c.d_ff  # SwiGLU 每层三个矩阵
        norm = (c.n_layers * 2 + 1) * c.d_model  # RMSNorm 每层两个 + 结尾一个
        total = embed + attn + ffn + norm
        print(f"[budget] 词嵌入(共享)  {embed/1e6:>6.2f}M")
        print(f"[budget] 注意力×{c.n_layers}   {attn/1e6:>6.2f}M")
        print(f"[budget] 前馈×{c.n_layers}     {ffn/1e6:>6.2f}M")
        print(f"[budget] 归一化        {norm/1e3:>6.1f}K")
        print(
            f"[budget] 合计 ≈ {total/1e6:.2f}M（非嵌入口径 {(total-embed)/1e6:.2f}M）"
        )
        # 显存账：固定开销 = 参数+梯度+优化器状态；激活随 batch 浮动，另算
        vram = total * BYTES_PER_PARAM / 1e9
        print(
            f"[vram] 训练固定开销 ≈ {vram:.1f}GB（{BYTES_PER_PARAM} 字节/参数，激活另算）"
        )
        return total, embed


# ==================== 词表对账（与上一站分词器联动） ====================
class VocabAuditor:
    """职责：只负责对账——配置里的 vocab_size 必须等于分词器的真实词表"""

    def check(self, cfg):
        """读 tokenizer.json 数真实词表；不一致直接断言失败（带病不许训练）"""
        if not TOKENIZER_FILE.exists():
            print(
                f"[skip] 未找到 {TOKENIZER_FILE}，跳过词表对账（先跑分词器一站可开启）"
            )
            return
        data = json.loads(TOKENIZER_FILE.read_text(encoding="utf-8"))
        # 真实词表按 token id 去重数：BpeTrainer 会把三个特殊标记直接编进词表，
        # 而 added_tokens 里登记的是同一批——两处直接相加会把它们数两遍
        ids = set(data["model"]["vocab"].values())
        ids.update(t["id"] for t in data.get("added_tokens", []))
        real_vocab = len(ids)
        assert (
            real_vocab == cfg.vocab_size
        ), f"词表对不上！分词器实际 {real_vocab}，配置写的 {cfg.vocab_size}"
        print(f"[vocab] 词表对账通过：分词器 {real_vocab} = 配置 {cfg.vocab_size} ✓")


# ==================== 模型本体（需要 torch；架构与 v2 完全一致） ====================
if TORCH_OK:

    class RMSNorm(nn.Module):
        """均方根归一化：只缩放不减均值，比 LayerNorm 少一步（类似简化版标准化）"""

        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放系数

        def forward(self, x):
            # x / sqrt(mean(x²)) × weight：把每行的"能量"归一后再按权重缩放
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )

    class RotaryEmbedding(nn.Module):
        """RoPE 旋转位置编码：不给位置编号，给位置"转角"（零参数，只存预计算的旋转表）"""

        def __init__(self, head_dim, max_seq_len, base=10000):
            super().__init__()
            # 每个维度对对应一个旋转频率：维度越低转得越慢（类似钟表的长短针）
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            t = torch.arange(max_seq_len).float()  # 位置序号 0..T-1
            freqs = torch.outer(t, inv_freq)  # 外积：(位置, 频率) → 转角表
            # 预先存好 cos/sin 表，forward 时直接查（register_buffer = 不算参数但随模型保存）
            self.register_buffer("cos", freqs.cos(), persistent=False)
            self.register_buffer("sin", freqs.sin(), persistent=False)

        def forward(self, x):
            """x 形状 (B, 头数, T, 头维度)；对每两个相邻维度做一次二维旋转"""
            T = x.size(2)
            cos = self.cos[:T][None, None, :, :]  # 加两个广播维度对齐 x
            sin = self.sin[:T][None, None, :, :]
            x1, x2 = x[..., ::2], x[..., 1::2]  # 偶数位 / 奇数位拆开
            # 二维旋转公式：偶位 = x1·cos − x2·sin；奇位 = x1·sin + x2·cos
            out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return out.flatten(-2)  # 交错拼回原来的维度顺序

    class CausalSelfAttention(nn.Module):
        """因果自注意力：每个位置只能看自己和之前的位置（防作弊，生成必须从左往右）"""

        def __init__(self, cfg):
            super().__init__()
            self.n_heads = cfg.n_heads
            self.head_dim = cfg.d_model // cfg.n_heads
            self.qkv = nn.Linear(
                cfg.d_model, 3 * cfg.d_model, bias=False
            )  # Q/K/V 一次算出
            self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)  # 输出投影
            self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

        def forward(self, x):
            B, T, _ = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)  # 沿最后一维切成三份
            # 调整形状为 (B, 头数, T, 头维度)：每个头独立看一句话
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            q, k = self.rope(q), self.rope(k)  # 只有 Q/K 加位置信息
            # PyTorch 内置的高效注意力：is_causal=True 自动上因果掩码
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)  # 多头结果拼回去
            return self.proj(y)

    class SwiGLU(nn.Module):
        """门控前馈网络：w1 算"门"、w3 算"内容"，相乘后由 w2 降回原维度"""

        def __init__(self, cfg):
            super().__init__()
            self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # 门控支路
            self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # 内容支路
            self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)  # 降维输出

        def forward(self, x):
            # silu(门) × 内容：让网络自己学会"哪些信息放行"
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class Block(nn.Module):
        """一个 Transformer 层：Pre-Norm 结构（先归一化再进子层，残差兜底）"""

        def __init__(self, cfg):
            super().__init__()
            self.norm1 = RMSNorm(cfg.d_model)
            self.attn = CausalSelfAttention(cfg)
            self.norm2 = RMSNorm(cfg.d_model)
            self.ffn = SwiGLU(cfg)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))  # 残差连接：原信息抄近道，防梯度消失
            x = x + self.ffn(self.norm2(x))
            return x

    class XiaozhiGPT(nn.Module):
        """小智本体：嵌入 → N 个 Block → 归一化 → 输出层（与嵌入权重共享）"""

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
            self.norm_f = RMSNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            # 权重共享的关键一行：输出层直接复用嵌入矩阵，白省 1258 万参数
            self.lm_head.weight = self.embed.weight
            # 参数初始化：全部改用小方差正态分布（GPT 惯例 std=0.02）
            # PyTorch 默认初始化方差偏大，会让开局 loss 远超 ln(词表)，前期训练不稳
            self.apply(self._init_weights)
            # 梯度检查点开关：默认关；训练站会打开它——前向只存每层入口，
            # 反向时重算层内激活，用约 30% 速度换激活显存从 O(层数) 降到 O(1)
            self.grad_ckpt = False

        def _init_weights(self, module):
            """对 Linear/Embedding 统一做 std=0.02 的正态初始化（self.apply 会遍历所有子模块）"""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx):
            """idx 形状 (B, T) 的 token id；返回 (B, T, 词表) 的 logits（每个位置的下一个词打分）"""
            x = self.embed(idx)
            for blk in self.blocks:
                # 训练且开了检查点：块内激活不存，反向时重算；推理/冒烟走原路
                if self.training and self.grad_ckpt:
                    x = grad_ckpt_fn(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
            return self.lm_head(self.norm_f(x))


# ==================== 实机验证 ====================
class ForwardTester:
    """职责：只负责真机验证——建模、对账参数、前向+反向冒烟"""

    def run(self, cfg, estimated_total):
        """
        参数：cfg 配置；estimated_total 公式估算的总参数
        校验：实机参数必须等于估算（证明权重共享生效）；前向输出形状正确；反向梯度通畅
        """
        model = XiaozhiGPT(cfg)
        real = sum(p.numel() for p in model.parameters())  # 数真实参数个数
        match = (
            "一致 ✓（权重共享生效）"
            if real == estimated_total
            else f"对不上！差 {real-estimated_total}"
        )
        print(f"[model] 实机参数 {real/1e6:.2f}M，与估算{match}")

        # 前向冒烟：假装 2 条 16 token 的句子喂进去，看输出形状对不对
        idx = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(idx)
        assert logits.shape == (2, 16, cfg.vocab_size), f"输出形状异常：{logits.shape}"
        print(f"[forward] 冒烟通过：logits {tuple(logits.shape)}")

        # 反向冒烟：对 logits 求和反传，确认每个参数都拿到梯度（训练能跑起来的最低证明）
        logits.sum().backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not no_grad, f"这些参数没收到梯度：{no_grad}"
        print("[backward] 冒烟通过：全部参数梯度通畅 ✓")

        # 检查点路径冒烟：训练站实际走的就是这条路（重算式反向），单独验证一次
        model.grad_ckpt = True
        model.train()
        model.zero_grad(set_to_none=True)
        model(idx).sum().backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not no_grad, f"检查点路径下这些参数没收到梯度：{no_grad}"
        print("[backward] 梯度检查点路径冒烟通过 ✓")
        return real


# ==================== 编排 ====================
class ConfigPipeline:
    """职责：只负责编排——配置 → 预算 → 词表对账 → 建模冒烟 → 给结论"""

    def run(self):
        cfg = ModelConfig()
        print(
            f"[cfg] 词表 {cfg.vocab_size} · 维度 {cfg.d_model} · "
            f"{cfg.n_layers} 层 · {cfg.n_heads} 头 · 上下文 {cfg.max_seq_len}"
        )

        # ① 公式算预算（零依赖，永远可跑）
        total, _ = BudgetCalculator(cfg).report()

        # ② 与上一站分词器对账词表（文件不在就跳过）
        VocabAuditor().check(cfg)

        # ③ 真机验证（有 torch 才跑）
        if TORCH_OK:
            ForwardTester().run(cfg, total)
        else:
            print("[skip] 未安装 torch，跳过实机验证（pip install torch 后可跑）")

        # ④ 给结论
        if PARAM_MIN <= total <= PARAM_MAX:
            print(
                f"[done] 预算 {total/1e6:.2f}M 落在 {PARAM_MIN/1e6:.0f}~{PARAM_MAX/1e6:.0f}M 区间，配置达标！"
            )
        else:
            print(
                f"[warn] 预算 {total/1e6:.2f}M 超出区间，请调整 d_model / n_layers / d_ff"
            )


if __name__ == "__main__":
    ConfigPipeline().run()
