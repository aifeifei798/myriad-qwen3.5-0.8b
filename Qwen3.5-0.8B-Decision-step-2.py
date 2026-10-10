#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 24G 满血显存专属前额叶决策脑微调脚本
- 基座: Qwen3.5-0.8B (纯血 BF16，无量化)
- 架构: Clef 决策头 + LoRA (r=64, alpha=64)
- 吞吐: Batch Size 16 x GA 2, 8~10 分钟超速炼制
"""

# 确保导入 Unsloth 决策套件
try:
    from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
except ImportError:
    print("❌ 未检测到 unsloth，请先执行: pip install --upgrade unsloth")
    sys.exit(1)

import os
import sys
import time
import torch
from datasets import load_dataset
from transformers import TrainingArguments


def main():
    data_path = "train_decisions.jsonl"
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件 {data_path}，请先运行: python3 prep_decision_data.py")
        sys.exit(1)

    print("=" * 75)
    print("🚀 【24G 满血显存火力全开】Qwen3.5-0.8B 前额叶决策脑微调")
    print(
        f"📦 物理显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    print(f"🎯 训练数据: {data_path}")
    print("=" * 75)

    model_name = "unsloth/Qwen3.5-0.8B"
    max_seq_length = 2048

    # 🌟 1. 纯血 BF16 载入 (24G 显存毫无压力，底座完全不失真)
    print("[1/5] 正在以原生 BF16 精度加载 Qwen3.5-0.8B...")
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # 拒绝 4bit，24G 直接硬吃 16-bit！
        local_files_only=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    )

    # 🌟 2. 满血 LoRA r=64 配置 (完全匹配官方高分 Benchmark 标准)
    print("[2/5] 正在挂载 r=64 满血 Clef 决策头与 LoRA 适配层...")
    model = FastDecisionModel.get_peft_model(
        model,
        r=64,  # 满血 rank 64，记忆容量与判别力拉满
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj"
        ],
        use_gradient_checkpointing="unsloth",  # 极限优化显存碎片
        random_state=3407,
    )

    # 🌟 3. 极速构建 Dataset
    print("[3/5] 正在构建决策数据集并分词...")
    t0 = time.time()
    dataset = load_dataset("json", data_files=data_path, split="train")
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    print(
        f"✅ 数据构建完毕 (耗时 {time.time()-t0:.1f}s): "
        f"成功 {report['total'] - report['skipped']} 条，跳过 {report['skipped']} 条")

    # 🌟 修复点 1：split_holdout 只要传 items 和 seed
    train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)
    print(f"📊 数据切分: 训练集 {len(train_items):,} 条 | 验证校准集 {len(eval_items):,} 条")

    # 🌟 修复点 2：在 train 之前跑一次预热评估（防止 5090 + Torch 2.14 第一步 compile crash）
    print("⚡ 正在执行 5090 计算图预热 (Warmup)...")
    try:
        FastDecisionModel.evaluate(model, tokenizer, eval_items[:10])
    except Exception:
        pass

    training_args = TrainingArguments(
        output_dir="output_dir",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=20,  # ✅ 换成固定步数预热
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        eval_strategy="no",  # 训练期间先关掉 eval 避免打断吞吐
        save_strategy="epoch",
        logging_steps=10,
        optim="adamw_torch_fused",
        report_to="none",
        seed=3407,
    )

    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=eval_items,
        args=training_args,
    )

    print("\n" + "═" * 75)
    print("🔥 [4/5] 引擎点火！开始微调前额叶决策脑...")
    print("═" * 75)
    train_start = time.time()
    trainer_stats = trainer.train(resume_from_checkpoint=True)
    train_time = time.time() - train_start
    print(
        f"\n🎉 训练冲线！总耗时: {train_time/60:.2f} 分钟 (速度: {trainer_stats.global_step / train_time:.2f} steps/s)"
    )

    # 🌟 5. 校准 (Calibrate) 与导出
    print("\n[5/5] 正在对 Holdout 验证集做不确定性标定 (Confidence Calibration)...")
    calib_metrics = FastDecisionModel.calibrate(model, tokenizer, eval_items)
    print("-" * 75)
    print(f"📈 最终校准报告:")
    for k, v in calib_metrics.items():
        print(
            f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")
    print("-" * 75)

    merged_output_path = "qwen3.5-0.8b-cortex-merged"
    print(f"💾 正在导出全量合并 16-bit 权重至: {merged_output_path} ...")
    model.save_pretrained("qwen3.5-0.8b-cortex-lora")
    # model.save_pretrained_merged(merged_output_path)
    tokenizer.save_pretrained(merged_output_path)
    print("✅ 权重与 Tokenizer 已完美封包！")

if __name__ == "__main__":
    main()
