#!/usr/bin/env python3
"""步骤3：SFT 微调 —— 只对回答部分计算 loss"""
import json, torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

BASE_MODEL = "./cpt_checkpoint/final_model"
DATA_PATH = "./processed/dialogues_processed.jsonl"
OUTPUT_DIR = "./sft_model"
BLOCK, BATCH, ACCUM, LR, EPOCHS = 512, 4, 8, 2e-5, 5  # 学习率调小 = 温柔地教

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
EOS = tokenizer.eos_token


def make_samples(conversations):
    """System/Human/Assistant 格式；多轮对话逐轮展开成多条样本"""
    sys_text = next((c["value"] for c in conversations if c["from"] == "system"), "")
    prompt, samples = (f"System: {sys_text}\n" if sys_text else ""), []
    for c in conversations:
        if c["from"] == "human":
            prompt += f"Human: {c['value']}\n"
        elif c["from"] == "gpt":
            samples.append((prompt, f"Assistant: {c['value']}{EOS}"))
            prompt += f"Assistant: {c['value']}\n"
    return samples


examples = []
with open(DATA_PATH, encoding="utf-8") as f:
    dialogues = [json.loads(l) for l in f if l.strip()]

for d in tqdm(dialogues, desc="处理对话"):
    for prompt, response in make_samples(d["conversations"]):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(response, add_special_tokens=False)
        ids = (p_ids + r_ids)[:BLOCK]
        r_len = min(len(r_ids), len(ids))
        labels = [-100] * (len(ids) - r_len) + ids[len(ids) - r_len :]  # 问题不算分
        pad = BLOCK - len(ids)
        examples.append(
            (
                ids + [tokenizer.pad_token_id] * pad,
                [1] * len(ids) + [0] * pad,
                labels + [-100] * pad,
            )
        )


class SFTDataset(Dataset):
    def __len__(self):
        return len(examples)

    def __getitem__(self, i):
        ii, aa, ll = examples[i]
        return {
            "input_ids": torch.tensor(ii),
            "attention_mask": torch.tensor(aa),
            "labels": torch.tensor(ll),
        }


dataset = SFTDataset()
print(f"✅ 训练样本: {len(dataset)} 条")

model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    logging_steps=5,
    save_steps=200,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

final = Path(OUTPUT_DIR) / "final_model"
trainer.save_model(final)
tokenizer.save_pretrained(final)
print(f"\n🎉 SFT 完成！会聊天的模型在 {final}")
print("下一步: python scripts/4_inference.py")
