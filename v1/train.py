# train.py —— 读 corpus.txt，炼出 mini_gpt.pt（M1 上约 10~25 分钟）
import re, time, torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = (
    "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
# 小提示：模型很小，个别机器上 cpu 反而比 mps 快，可自行改成 "cpu" 对比

# ---------- 超参数 ----------
BLOCK, BATCH, STEPS, LR = 64, 32, 2000, 3e-3  # 上下文长/批大小/步数/学习率
D, H, NLAYER = 128, 4, 4  # 模型宽度/注意力头数/层数

# ---------- 读语料：每行一对 "问：… 答：…" ----------
pairs = []
for line in open("corpus.txt", encoding="utf-8"):
    m = re.match(r"问：(.+?)\s*答：(.+)", line.strip())
    if m:
        pairs.append((m.group(1).strip(), m.group(2).strip()))
if not pairs:
    raise SystemExit("corpus.txt 里没读到问答对，请确认它和 train.py 在同一目录")
print(f"读到 {len(pairs)} 条问答对")

text = "".join(f"问：{q}\n答：{a}\n\n" for q, a in pairs)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}  # 字符 → 编号
itos = {i: c for c, i in stoi.items()}  # 编号 → 字符
V = len(chars)
data = torch.tensor([stoi[c] for c in text])  # 整个语料变成一长串编号
print(f"设备 {device} | 词表 {V} | 训练文本共 {len(text)} 字符")


def get_batch():  # 随机抓一小段，输入 x，答案是它右移一格的 y
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i : i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------- 模型：4 层 Transformer ----------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        h = self.ln1(x)
        mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), 1
        )  # 只许看过去
        a, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)  # 字 → 向量
        self.pos = nn.Embedding(BLOCK, D)  # 位置 → 向量
        self.blocks = nn.Sequential(*[Block() for _ in range(NLAYER)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)  # 向量 → 下一个字的概率

    def forward(self, x):
        T = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        return self.head(self.ln(self.blocks(h)))


# ---------- 训练：猜字 → 算误差 → 调参数，循环 2000 次 ----------
model = GPT().to(device)
print(f"参数量 {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
t0 = time.time()
for step in range(1, STEPS + 1):
    x, y = get_batch()
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0 or step == 1:
        print(f"step {step:5d} | loss {loss.item():.4f} | 已用 {time.time()-t0:.0f} 秒")

# ---------- 保存：权重 + 字典，全部打进一个文件 ----------
torch.save({"sd": model.state_dict(), "stoi": stoi, "itos": itos}, "mini_gpt.pt")
print(f"训练完成！已生成 mini_gpt.pt（共耗时 {(time.time()-t0)/60:.1f} 分钟）")
