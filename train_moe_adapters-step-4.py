#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 1共享 + 8天王 + 256小崽子 满血定点定向巡航微调脚本 (完善排雷版)
- 底座: Qwen3.5-0.8B (冻结 Base 权重)
- 嫁接层: Layer 08 ~ 19 (共 12 层)
- 适配器:
    * 1 共享常驻核 (Rank-32, 全局吃数据)
    * 8 大天王 (Rank-16, 靶向吃对应天王数据)
    * 256 微专家 (Rank-8, 靶向吃对应专精数据)
- 吞吐: 纯血 BF16, 单样本直通 (B=1, ACCUM=16), 5090 D 满速巡航
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


# 🌟 1. 显存零拷贝极简低秩适配单元
class MicroLoRA(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 rank,
                 scale=1.0,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.A = nn.Linear(in_dim,
                           rank,
                           bias=False,
                           device=device,
                           dtype=dtype)
        self.B = nn.Linear(rank,
                           out_dim,
                           bias=False,
                           device=device,
                           dtype=dtype)
        # 初始化：A 随机高斯，B 全零 (保证初始增量为 0，不破坏底座)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


# 🌟 2. 12 层嫁接核心 Wrapper (全员设备锁死 + 动态定向通道)
class MyriadGraftedLayer(nn.Module):

    def __init__(self,
                 original_mlp,
                 hidden_dim=1024,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.base_mlp = original_mlp  # 冻结的原版 MLP
        self.current_macro = None
        self.current_micro = None

        # 1. 共享常驻核心 (Rank-32)
        self.shared_lora = MicroLoRA(hidden_dim,
                                     hidden_dim,
                                     rank=32,
                                     scale=1.0,
                                     device=device,
                                     dtype=dtype)

        # 2. 八大天王核心池 (8 个独立的 Rank-16)
        self.macro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim,
                      hidden_dim,
                      rank=16,
                      scale=1.0,
                      device=device,
                      dtype=dtype) for _ in range(8)
        ])

        # 3. 256 小崽子微专家池 (256 个独立的 Rank-8)
        self.micro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim,
                      hidden_dim,
                      rank=8,
                      scale=0.5,
                      device=device,
                      dtype=dtype) for _ in range(256)
        ])

    def forward(self, x, macro_id=None, micro_id=None):
        base_out = self.base_mlp(x)
        shared_delta = self.shared_lora(x)

        # 读取定向通道
        m_id = macro_id if macro_id is not None else self.current_macro
        u_id = micro_id if micro_id is not None else self.current_micro

        # 靶向激活天王
        macro_delta = self.macro_loras[m_id](x) if m_id is not None else 0.0

        # 靶向激活小崽子
        micro_delta = self.micro_loras[u_id](x) if u_id is not None else 0.0

        return base_out + shared_delta + macro_delta + micro_delta


# 🌟 3. 数据集读取与 Mask 构建 (防 NaN 截断防御)
class MoEDataset(Dataset):

    def __init__(self, data_file, tokenizer, max_len=1024):
        self.items = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"[*] 正在载入并解析 {data_file} ...")
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        print(f"✅ 载入完成，共 {len(self.items):,} 条全监督样本！")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        inst = item["instruction"]
        resp = item["response"]

        # 计算全局小崽子唯一 ID: clan_id (0~15) * 16 + micro_id (0~15) -> 0~255
        global_micro_id = item["clan_id"] * 16 + item["micro_id"]
        macro_id = item["macro_id"]

        prompt = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{resp}<|im_end|>"

        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        f_ids = self.tokenizer.encode(full_text,
                                      add_special_tokens=False)[:self.max_len]

        input_ids = torch.tensor(f_ids, dtype=torch.long)
        labels = input_ids.clone()

        # 截断 Prompt 区域不计算梯度
        mask_len = min(len(p_ids), len(f_ids))
        labels[:mask_len] = -100

        # 🌟 核心防 NaN 保护：若全部被遮蔽，强制最后一个 token 计算 loss
        if (labels != -100).sum() == 0:
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "macro_id": macro_id,
            "micro_id": global_micro_id
        }


def main():
    torch.manual_seed(3407)
    data_path = "moe_training_dataset.jsonl"
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件 {data_path}，请先执行 step-3 数据生成！")
        sys.exit(1)

    print("=" * 75)
    print("🚀 【Myriad-MoE: 1共享 + 8天王 + 256小崽子】5090 满血定向训练")
    print(f"📦 设备: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    dtype = torch.bfloat16
    base_model_id = "Qwen/Qwen3.5-0.8B"

    print("[1/5] 载入分词器与基座 Qwen3.5-0.8B (纯血 BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # 🌟 修复点 1：消除 torch_dtype 弃用警告
    model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")

    # 彻底冻结底座全部参数
    for param in model.parameters():
        param.requires_grad = False

    print("[2/5] 在 Layer 08 ~ 19 (共12层) 嫁接树状适配结构...")
    GRAFTED_LAYERS = list(range(8, 20))  # 8~19
    hidden_dim = model.config.hidden_size

    grafted_modules = []
    for idx in GRAFTED_LAYERS:
        # 🌟 修复点 2：明确传入 device="cuda:0" 并强制链式 .to("cuda:0") 杜绝设备不一致
        wrapper = MyriadGraftedLayer(model.model.layers[idx].mlp,
                                     hidden_dim=hidden_dim,
                                     device="cuda:0",
                                     dtype=dtype).to("cuda:0")

        model.model.layers[idx].mlp = wrapper
        grafted_modules.append(wrapper)

    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(
        f"✅ 嫁接完毕！可训练参数量: {trainable_params:,} (约 {trainable_params/1e6:.1f} M)"
    )

    print("[3/5] 构建 Dataset 并配置定向通道...")
    dataset = MoEDataset(data_path, tokenizer, max_len=1024)
    # 🌟 修复点 3：开启 pin_memory 加速主机到 GPU 的直通
    loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    ACCUM_STEPS = 16  # 等效 Batch Size = 16
    EPOCHS = 1
    TOTAL_STEPS = len(loader) // ACCUM_STEPS

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4,
        weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    TOTAL_STEPS * 0.05),
                                                num_training_steps=TOTAL_STEPS)

    print("\n" + "═" * 75)
    print(
        f"🔥 [4/5] 引擎点火！总步数: {TOTAL_STEPS} Steps (等效 Batch Size: {ACCUM_STEPS})"
    )
    print("═" * 75)

    model.train()
    step_loss = 0.0
    accum_steps_count = 0
    global_step = 0
    t0 = time.time()

    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        for i, batch in enumerate(loader):
            # non_blocking=True 配合 pin_memory 实现极限传输
            input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
            labels = batch["labels"].to("cuda:0", non_blocking=True)
            macro_id = batch["macro_id"].item()
            micro_id = batch["micro_id"].item()

            # 靶向注入：动态告诉 12 层当前样本归属
            for wrapper in grafted_modules:
                wrapper.current_macro = macro_id
                wrapper.current_micro = micro_id

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / ACCUM_STEPS
            loss.backward()
            step_loss += loss.item() * ACCUM_STEPS

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_steps_count += 1

                # 🌟 修复点 4：精准的 Loss 平均值统计
                if global_step % 20 == 0 or global_step == 1:
                    lr = scheduler.get_last_lr()[0]
                    speed = (i + 1) / (time.time() - t0)
                    avg_loss = step_loss / accum_steps_count
                    print(f"Step {global_step:4d}/{TOTAL_STEPS} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Speed: {speed:.1f} samples/s")
                    step_loss = 0.0
                    accum_steps_count = 0

    print(f"\n🎉 训练冲线！总耗时: {(time.time()-t0)/60:.2f} 分钟！")

    # 🌟 5. 安全保存权重 (纯净 CPU 解绑 + 元数据)
    save_path = "myriad_moe_adapters.pt"
    print(f"\n[5/5] 正在将全套天王和小崽子权重提炼保存至: {save_path} ...")

    save_dict = {
        "metadata": {
            "base_model": base_model_id,
            "hidden_dim": hidden_dim,
            "grafted_layers": GRAFTED_LAYERS,
            "shared_rank": 32,
            "macro_rank": 16,
            "micro_rank": 8,
            "num_macros": 8,
            "num_micros": 256,
        },
        "adapters": {}
    }

    for idx in GRAFTED_LAYERS:
        wrapper = model.model.layers[idx].mlp
        save_dict["adapters"][f"layer_{idx}_shared"] = {
            k: v.detach().cpu()
            for k, v in wrapper.shared_lora.state_dict().items()
        }
        save_dict["adapters"][f"layer_{idx}_macros"] = {
            k: v.detach().cpu()
            for k, v in wrapper.macro_loras.state_dict().items()
        }
        save_dict["adapters"][f"layer_{idx}_micros"] = {
            k: v.detach().cpu()
            for k, v in wrapper.micro_loras.state_dict().items()
        }

    torch.save(save_dict, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ 权重保存完毕！全套适配器仅占: {file_size_mb:.2f} MB")
    print(f"📦 结构包含: 12 层 x (1 共享常驻 + 8 天王 + 256 小崽子)")
    print("═" * 75)
    print("🎉 恭喜！八大天王与二百五十六小崽子已满血毕业！随时准备大合体！")


if __name__ == "__main__":
    main()
