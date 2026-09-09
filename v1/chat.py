# chat.py —— 只吃 mini_gpt.pt，无需语料、无需重新训练
# 用法1：python3 chat.py mini_gpt.pt 你好     （问一句）
# 用法2：python3 chat.py mini_gpt.pt          （连续聊天，输入 q 退出）
import sys, torch
import torch.nn as nn
import torch.nn.functional as F

device = (
    "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
BLOCK, TEMPERATURE = 64, 0.7  # 温度：越小越稳重，越大越天马行空
D, H, NLAYER = 128, 4, 4  # 必须和 train.py 完全一致


# ---------- 模型结构（与 train.py 相同）----------
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
        )
        a, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(BLOCK, D)
        self.blocks = nn.Sequential(*[Block() for _ in range(NLAYER)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, x):
        T = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        return self.head(self.ln(self.blocks(h)))


# ---------- 加载训练生成的文件 ----------
ckpt = torch.load(sys.argv[1], map_location=device)
stoi, itos = ckpt["stoi"], ckpt["itos"]
model = GPT(len(stoi)).to(device)
model.load_state_dict(ckpt["sd"])
model.eval()
print(f"已加载 {sys.argv[1]} | 设备 {device} | 词表 {len(stoi)}")


@torch.no_grad()
def reply(question, max_new=120):
    # 未学过的字符会被直接忽略——这就是字符级模型的认知边界
    ids = [stoi[c] for c in f"问：{question}\n答：" if c in stoi]
    if not ids:
        return "（这个问题里没有我认识的字，换个问法试试？）"
    out, last = [], ""
    for _ in range(max_new):
        ctx = torch.tensor([ids[-BLOCK:]], device=device)
        p = F.softmax(model(ctx)[0, -1] / TEMPERATURE, dim=-1)
        nxt = torch.multinomial(p, 1).item()  # 按概率抽一个字
        ch = itos[nxt]
        if ch == "\n":
            if not out:  # 开头就换行 → 跳过
                ids.append(nxt)
                continue
            if last == "\n":  # 连续两个换行 = 回答结束
                break
        out.append(ch)
        last = ch
        ids.append(nxt)
    return "".join(out).strip()


if __name__ == "__main__":
    if len(sys.argv) >= 3:  # 问一句就退出
        print(reply(sys.argv[2]))
    else:  # 连续聊天模式
        print("小智已上线！输入 q 退出。")
        while True:
            try:
                q = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit", "退出"):
                break
            if q:
                print("小智：" + reply(q))
