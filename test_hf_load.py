#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_path = "./myriad-qwen3.5-0.8b-hf"

print(f"🚀 正在以原生 Hugging Face 标准加载: {model_path} ...")

model = AutoModelForCausalLM.from_pretrained("./myriad-qwen3.5-0.8b-hf",
                                             trust_remote_code=True).cuda()

tokenizer = AutoTokenizer.from_pretrained("./myriad-qwen3.5-0.8b-hf",
                                          trust_remote_code=True)

print("✅ 加载成功！开始测试！\n")

prompt = "帮我用 Rust 写一个 epoll 异步环形缓冲区，解决死锁"

# 1. 前额叶决策分发工单 (自动点亮天王与宗门)
dispatch_report = model.dispatch_intent(prompt)
print(
    f"🧠 [HF 内部决策] 激活天王: {dispatch_report['macro']} | 激活宗门: {dispatch_report['clan']}"
)

# 2. 标准 HF 流式生成
messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("🤖 回复:\n")
_ = model.generate(**inputs,
                   max_new_tokens=512,
                   streamer=streamer,
                   temperature=0.7,
                   top_p=0.9)
