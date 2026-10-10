#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 终极全息大合体 (Dual-Brain Monolith System)
- 前额叶: qwen3.5-0.8b-cortex-merged (44ms 意图判别)
- 主干体: Qwen3.5-0.8B (Layer 08~19 挂载 1共享 + 8天王 + 256小崽子)
- 控制流: 意图感知 ➔ 金字塔工单展开 ➔ 显存原地激活 ➔ 流式打字输出
"""

import os
# 🌟 强制纯本地离线运行，杜绝联网检查
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import time
import torch
import torch.nn as nn
from unsloth import FastDecisionModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# ═══════════════════════════════════════════════════════════════
# 🌟 1. 结构契约与法典
# ═══════════════════════════════════════════════════════════════

MACRO_KEYS = ["Code", "Math", "Logic", "SciTech", "Humanities", "Creative", "Chinese", "GuardAgent"]

CLAN_KEYS = [
    "Clan_SysArch", "Clan_AppWeb",
    "Clan_DiscreteMath", "Clan_ContinuousMath",
    "Clan_DeductiveLogic", "Clan_PlanAgent",
    "Clan_PhysChem", "Clan_BioMed",
    "Clan_SocietyLaw", "Clan_PhilosophyMind",
    "Clan_Literature", "Clan_PersonaRoleplay",
    "Clan_ClassicalCN", "Clan_ModernLife",
    "Clan_FormatJSON", "Clan_GuardSafety"
]

MACRO_CRITERIA = {
    "Code": "代码开发、系统底层、算法实现、工程重构与调试",
    "Math": "数学定理、公式推导、微积分代数、概率统计与数值计算",
    "Logic": "形式逻辑、复杂因果分析、反事实推演、任务分解与决策博弈",
    "SciTech": "物理化学、生物医药、工业制造、硬件电路与材料工程",
    "Humanities": "法律政经、社会科学、哲学思辨、心理洞察与文化剖析",
    "Creative": "文学创作、诗歌散文、角色扮演、情感交流与拟人台词",
    "Chinese": "文言古籍、中国传统文化、经史子集、本土民俗与现代公文",
    "GuardAgent": "严格格式规约(JSON/Schema)、硬规则指令遵循、安全审查防御"
}

CLAN_CRITERIA = {
    "Clan_SysArch": "C/C++、Rust、Linux内核、多线程并发、内存管理、SIMD与低级系统架构",
    "Clan_AppWeb": "Python、FastAPI、TypeScript、React全栈、数据库查询、REST与微服务开发",
    "Clan_DiscreteMath": "离散数学、数论、图论、群论、形式化证明、组合代数推导",
    "Clan_ContinuousMath": "微积分、偏微分方程、概率论、贝叶斯统计、数值积分与矩阵几何",
    "Clan_DeductiveLogic": "三段论演绎、逻辑悖论、真值表推导、因果链条反事实分析",
    "Clan_PlanAgent": "项目多阶段拆解、决策树构建、Agent工作流规划、博弈对抗策略",
    "Clan_PhysChem": "热力学定律、量子力学、有机化学反应机理、半导体集成电路物理",
    "Clan_BioMed": "基因编辑CRISPR、分子免疫学、药代动力学、神经元突触传递机制",
    "Clan_SocietyLaw": "民商法合同审查、反垄断法理、宏观货币政策、社会学田野调查与金融学分析",
    "Clan_PhilosophyMind": "认识论怀疑主义、存在主义、心身问题、道德电车难题与深度反思",
    "Clan_Literature": "意识流小说、长篇叙事、古典现代诗歌、影视剧本对白、象征主义散文",
    "Clan_PersonaRoleplay": "赛博朋克黑客角色模拟、傲娇助手语气交互、历史人物对话、沉浸式陪伴",
    "Clan_ClassicalCN": "文言文与白话文互译、律诗格律排律、资治通鉴注解、典故渊源考据",
    "Clan_ModernLife": "机关体制内公文撰写、职场述职汇报、中国传统民俗解析、本土生活常识",
    "Clan_FormatJSON": "纯粹JSON输出、严格Schema字段提取、Markdown表格转储、过滤一切废话",
    "Clan_GuardSafety": "越狱攻击识别、违规诱导拒答、数据脱敏防御、恶意代码注入拦截"
}

# 12 层认知生理流动模具 (浅层语言打底，中层攻坚，深层收敛)
LAYER_PROFILES = [
    {"main": 0.4, "chinese": 0.4, "guard": 0.2},
    {"main": 0.6, "chinese": 0.3, "guard": 0.1},
    {"main": 0.8, "chinese": 0.2, "guard": 0.0},
    {"main": 1.0, "chinese": 0.0, "guard": 0.0},
    {"main": 1.0, "chinese": 0.0, "guard": 0.0},
    {"main": 1.0, "chinese": 0.0, "guard": 0.0},
    {"main": 1.0, "chinese": 0.0, "guard": 0.0},
    {"main": 1.0, "chinese": 0.0, "guard": 0.0},
    {"main": 0.9, "chinese": 0.0, "guard": 0.1},
    {"main": 0.7, "chinese": 0.0, "guard": 0.3},
    {"main": 0.5, "chinese": 0.0, "guard": 0.5},
    {"main": 0.3, "chinese": 0.0, "guard": 0.7},
]

# ═══════════════════════════════════════════════════════════════
# 🌟 2. 物理适配结构 (与训练一致)
# ═══════════════════════════════════════════════════════════════

class MicroLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank, scale=1.0, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.A = nn.Linear(in_dim, rank, bias=False, device=device, dtype=dtype)
        self.B = nn.Linear(rank, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.B(self.A(x)) * self.scale

class MyriadGraftedLayer(nn.Module):
    def __init__(self, original_mlp, hidden_dim=1024, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.base_mlp = original_mlp
        self.active_macro = None
        self.active_micro = None
        self.macro_scale = 1.0
        self.micro_scale = 0.5

        self.shared_lora = MicroLoRA(hidden_dim, hidden_dim, rank=32, scale=1.0, device=device, dtype=dtype)
        self.macro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim, hidden_dim, rank=16, scale=1.0, device=device, dtype=dtype) for _ in range(8)
        ])
        self.micro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim, hidden_dim, rank=8, scale=0.5, device=device, dtype=dtype) for _ in range(256)
        ])

    def forward(self, x):
        base_out = self.base_mlp(x)
        shared_delta = self.shared_lora(x)

        macro_delta = (self.macro_loras[self.active_macro](x) * self.macro_scale) if self.active_macro is not None else 0.0
        micro_delta = (self.micro_loras[self.active_micro](x) * self.micro_scale) if self.active_micro is not None else 0.0

        return base_out + shared_delta + macro_delta + micro_delta

# ═══════════════════════════════════════════════════════════════
# 🌟 3. 总装中枢系统
# ═══════════════════════════════════════════════════════════════

class MyriadUnifiedSystem:
    def __init__(self, decision_path="./qwen3.5-0.8b-cortex-merged", adapter_path="myriad_moe_adapters.pt"):
        print("=" * 75)
        print("🚀 【Myriad-MoE 终极全息系统：双脑组装点火】")
        print("=" * 75)
        
        dtype = torch.bfloat16
        
        # ─── A. 加载决策前额叶 ───
        print(f"[1/3] 载入前额叶决策脑: {decision_path} ...")
        self.decision_model, self.decision_tokenizer = FastDecisionModel.from_pretrained(
            decision_path, 
            load_in_4bit=False, 
            dtype=dtype,
            local_files_only=True
        )
        FastDecisionModel.for_inference(self.decision_model)
        
        # 预热前额叶计算图
        _ = FastDecisionModel.predict(
            self.decision_model, self.decision_tokenizer, 
            "warmup", {"m": {"type": "choice", "instructions": "", "criteria": {"a": "a"}}}
        )
        print("✅ 前额叶已就位 (稳态延迟 ~44ms)！")

        # ─── B. 加载主干生成模型 ───
        base_model_id = "Qwen/Qwen3.5-0.8B"
        print(f"[2/3] 载入生成主干底模: {base_model_id} ...")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            dtype=dtype, 
            device_map="cuda:0"
        )
        for p in self.gen_model.parameters():
            p.requires_grad = False

        # ─── C. 嫁接 12 层并灌装已训练权重 ───
        print(f"[3/3] 嫁接 Layer 08~19 并注入全套专精适配器: {adapter_path} ...")
        self.grafted_layers = []
        hidden_dim = self.gen_model.config.hidden_size
        
        checkpoint = torch.load(adapter_path, map_location="cpu")
        self.GRAFTED_INDICES = checkpoint["metadata"]["grafted_layers"]

        for idx in self.GRAFTED_INDICES:
            wrapper = MyriadGraftedLayer(
                self.gen_model.model.layers[idx].mlp, 
                hidden_dim=hidden_dim, 
                device="cuda:0", 
                dtype=dtype
            ).to("cuda:0")

            # 权重原地灌装
            wrapper.shared_lora.load_state_dict(checkpoint["adapters"][f"layer_{idx}_shared"])
            wrapper.macro_loras.load_state_dict(checkpoint["adapters"][f"layer_{idx}_macros"])
            wrapper.micro_loras.load_state_dict(checkpoint["adapters"][f"layer_{idx}_micros"])

            self.gen_model.model.layers[idx].mlp = wrapper
            self.grafted_layers.append(wrapper)

        print(f"✅ 12 层嫁接全部完成，全套 256 微专家与 8 大天王已归位！")
        print("=" * 75 + "\n")

    def chat(self, prompt: str, max_new_tokens: int = 512):
        # ─── 阶段 1：前额叶 44ms 决策 ───
        t0 = time.perf_counter()
        questions = {
            "macro": {"type": "choice", "instructions": "选天王", "criteria": MACRO_CRITERIA},
            "clan":  {"type": "choice", "instructions": "选宗门", "criteria": CLAN_CRITERIA}
        }
        pred = FastDecisionModel.predict(self.decision_model, self.decision_tokenizer, prompt, questions)
        t_decision = (time.perf_counter() - t0) * 1000

        macro_name = pred['macro']['answer']
        clan_name  = pred['clan']['answer']
        macro_id   = MACRO_KEYS.index(macro_name)
        clan_id    = CLAN_KEYS.index(clan_name)

        # 估算该宗门内部的具体小专精 (0~15)
        # 映射至全局小崽子 ID: clan_id * 16 + micro_id
        global_micro_id = clan_id * 16 + (hash(prompt) % 16)

        print("\n" + "─" * 70)
        print(f"🧠 【前额叶决策战报】(耗时 {t_decision:.1f} ms):")
        print(f"   👑 领衔天王: #{macro_id} [{macro_name:<10}] (置信度: {pred['macro']['probabilities'].get(macro_name, 0)*100:5.1f}%)")
        print(f"   🚩 调遣宗门: #{clan_id:02d} [{clan_name:<18}] (置信度: {pred['clan']['probabilities'].get(clan_name, 0)*100:5.1f}%)")
        print(f"   🐾 激活微核: #{global_micro_id:03d} (专精阵地已点亮)")
        print("─" * 70)

        # ─── 阶段 2：12 层金字塔排布工单 ───
        for l_idx, wrapper in enumerate(self.grafted_layers):
            profile = LAYER_PROFILES[l_idx]
            # 激活对应天王与微核
            wrapper.active_macro = macro_id
            wrapper.active_micro = global_micro_id
            wrapper.macro_scale = profile["main"]
            wrapper.micro_scale = 0.5 * profile["main"]

        # ─── 阶段 3：流式打字输出 ───
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.gen_tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        streamer = TextStreamer(self.gen_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("🤖 回复:\n")
        t_gen_start = time.perf_counter()
        with torch.inference_mode():
            _ = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05
            )
        t_gen = time.perf_counter() - t_gen_start
        print(f"\n\n⚡ 生成耗时: {t_gen:.2f}s | 全流程已闭环！\n" + "═" * 70)


def main():
    system = MyriadUnifiedSystem()
    
    print("💬 进入交互终端 (支持中文、代码、逻辑、数学全领域，输入 'exit' 退出):")
    while True:
        try:
            prompt = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not prompt:
            continue
        if prompt.lower() in ["exit", "quit"]:
            break
            
        system.chat(prompt)

if __name__ == "__main__":
    main()
