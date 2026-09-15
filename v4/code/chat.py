#!/usr/bin/env python3
"""步骤4：推理 —— 和你亲手训练的 AI 聊天"""
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

MODEL_PATH = "./sft_model/final_model"

SYSTEM_PROMPT = """System: 你是颜柳，一个欢乐幽默的聊天伙伴。你24小时电量满格的乐天派，喜欢用网络热梗、谐音梗和表情包表达情绪。即使吐槽也充满幽默感，绝不传递负能量。
禁忌：不评价政治，不人身攻击，幽默但保持尊重。
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device
print(f"📥 加载模型（设备: {device}）...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device).eval()

history = SYSTEM_PROMPT
print("🤖 开始聊天！（quit 退出 / reset 清空记忆）\n")

while True:
    user = input("你: ").strip()
    if user.lower() == "quit":
        break
    if user.lower() == "reset":
        history = SYSTEM_PROMPT
        print("🔄 已清空记忆\n")
        continue
    if not user:
        continue

    prompt = history + f"Human: {user}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400).to(
        device
    )
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,  # 越高越放飞
            top_p=0.9,  # 只从最可能的前 90% 里挑词
            repetition_penalty=1.15,  # 惩罚复读
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    reply = reply.split("Human:")[0].strip()  # 防止它自问自答
    print(f"颜柳: {reply}\n")

    history += f"Human: {user}\nAssistant: {reply}\n"
