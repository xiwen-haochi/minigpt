#!/usr/bin/env python3
"""步骤2：CPT 预训练 —— 空白模型读《神雕侠侣》"""
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
)

TOKENIZER_DIR = "./tokenizer"
NOVEL_PATH = "./processed/novel_cleaned.txt"
OUTPUT_DIR = "./cpt_checkpoint"
BLOCK, BATCH, ACCUM, LR, EPOCHS = 512, 8, 4, 5e-4, 3


def _pick_device():
    """设备优先级：英伟达 GPU > 苹果 MPS > CPU（有什么用什么）"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
print(f"🖥️  训练设备: {DEVICE}")

# ---------- 1. 分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

# ---------- 2. 整本小说编码成 token 流 ----------
print("📚 编码小说...")
text = Path(NOVEL_PATH).read_text(encoding="utf-8")
all_ids = []
for i in tqdm(range(0, len(text), 500000)):
    all_ids.extend(tokenizer.encode(text[i : i + 500000], add_special_tokens=False))
print(f"   ✅ 共 {len(all_ids):,} 个 token")


# ---------- 3. 切成 512 一段 ----------
class NovelDataset(Dataset):
    def __len__(self):
        return (len(all_ids) - 1) // BLOCK

    def __getitem__(self, idx):
        ids = torch.tensor(all_ids[idx * BLOCK : (idx + 1) * BLOCK], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}  # CPT：全文都算分


dataset = NovelDataset()
print(f"   ✅ 训练样本: {len(dataset)} 段")

# ---------- 4. 全新空白 GPT（Qwen 词表 + 精简主干）----------
config = GPT2Config(
    vocab_size=len(tokenizer),  # Qwen 词表约 15 万
    n_positions=512,
    n_embd=512,
    n_layer=6,
    n_head=8,  # 调小主干，抵消大词表的显存开销
)
model = GPT2LMHeadModel(config).to(DEVICE)  # 显式放到选中的设备上
print(f"   🧠 参数量: {sum(p.numel() for p in model.parameters()):,}")

# ---------- 5. 训练 ----------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=LR,
    warmup_steps=200,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=(DEVICE == "cuda"),  # 混合精度只在 cuda 开；mps/cpu 上 fp32 更稳
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# ---------- 6. 保存 ----------
final = Path(OUTPUT_DIR) / "final_model"
trainer.save_model(final)
tokenizer.save_pretrained(final)
print(f"\n🎉 CPT 完成！模型在 {final}")
print("下一步: python scripts/3_train_sft.py")
