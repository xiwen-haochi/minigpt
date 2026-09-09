# -*- coding: utf-8 -*-
"""
model_config.py —— 小智二代 · 0.01B 模型配置与预算自检脚本
作用：定义模型超参 → 纯公式算参数预算 → PyTorch 真机建模 → 前向冒烟测试
用法：python model_config.py（装了 torch 自动做实机验证；没装也能看预算表）
"""

from dataclasses import dataclass  # 把配置打包成结构体，类似带类型提示的 dict

# torch 是可选依赖：有它就真机建模，没有就只看预算表（脚本不会崩）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ==================== 可调参数区（只改这里） ====================
@dataclass
class ModelConfig:
    """全部超参集中在这一处：训练和推理脚本都引用它，保证口径唯一"""

    vocab_size: int = 8192  # 词表大小：必须与 tokenizer.json 的实际词表一致！
    d_model: int = 256  # 隐藏维度：每个 token 用 256 个数表示
    n_layers: int = 8  # Transformer 层数：深比宽更会"讲条理"
    n_heads: int = 8  # 注意力头数：必须能整除 d_model
    d_ff: int = 768  # 前馈网络中间维度（SwiGLU 三矩阵）
    max_seq_len: int = 512  # 上下文窗口：约能装 350~500 个汉字

    def __post_init__(self):
        """创建配置时立刻自检，不合规直接报错（防止带病进入训练）"""
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"


# 预算合格区间：总参数落在 8.5M~12M 算 0.01B 达标
PARAM_MIN, PARAM_MAX = 8_500_000, 12_000_000


# ==================== 预算计算（纯公式，零依赖） ====================
class BudgetCalculator:
    """职责：只负责算账——用公式预估每个部件的参数，不用真的建模"""

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
        return total, embed


# ==================== 模型本体（需要 torch） ====================
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
            # 权重共享的关键一行：输出层直接复用嵌入矩阵，白省 210 万参数
            self.lm_head.weight = self.embed.weight
            # 参数初始化：全部改用小方差正态分布（GPT 惯例 std=0.02）
            # PyTorch 默认初始化方差偏大，会让开局 loss 远超 ln(词表)，前期训练不稳
            self.apply(self._init_weights)

        def _init_weights(self, module):
            """对 Linear/Embedding 统一做 std=0.02 的正态初始化（self.apply 会遍历所有子模块）"""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx):
            """idx 形状 (B, T) 的 token id；返回 (B, T, 词表) 的 logits（每个位置的下一个词打分）"""
            x = self.embed(idx)
            for blk in self.blocks:
                x = blk(x)
            return self.lm_head(self.norm_f(x))


# ==================== 实机验证 ====================
class ForwardTester:
    """职责：只负责真机验证——建模、对账参数、跑一次前向冒烟"""

    def run(self, cfg, estimated_total):
        """
        参数：cfg 配置；estimated_total 公式估算的总参数
        校验：实机参数必须等于估算（证明权重共享生效）；输出形状必须正确
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
        return real


# ==================== 编排 ====================
class ConfigPipeline:
    """职责：只负责编排——配置 → 预算 → 建模对账 → 冒烟 → 给结论"""

    def run(self):
        cfg = ModelConfig()
        print(
            f"[cfg] 词表 {cfg.vocab_size} · 维度 {cfg.d_model} · "
            f"{cfg.n_layers} 层 · {cfg.n_heads} 头 · 上下文 {cfg.max_seq_len}"
        )

        # ① 公式算预算（零依赖，永远可跑）
        total, _ = BudgetCalculator(cfg).report()

        # ② 真机验证（有 torch 才跑）
        if TORCH_OK:
            ForwardTester().run(cfg, total)
        else:
            print("[skip] 未安装 torch，跳过实机验证（pip install torch 后可跑）")

        # ③ 给结论
        if PARAM_MIN <= total <= PARAM_MAX:
            print(
                f"[done] 预算 {total/1e6:.2f}M 落在 {PARAM_MIN/1e6:.1f}~{PARAM_MAX/1e6:.0f}M 区间，配置达标！"
            )
        else:
            print(
                f"[warn] 预算 {total/1e6:.2f}M 超出区间，请调整 d_model / n_layers / d_ff"
            )


if __name__ == "__main__":
    ConfigPipeline().run()
