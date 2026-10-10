#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 100% 纯血 Hugging Face 双脑全自闭环 (零 Unsloth 依赖)
- 前额叶: 纯 PyTorch (ClefModel + systemone) 50ms 决策
- 主干体: Qwen3.5-0.8B + 12层 MoE (myriad_moe_adapters.pt) 流式生成
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, AutoTokenizer, TextStreamer, AutoModelForCausalLM
from peft import PeftModel

# 导入纯血本地决策实现
import sys

sys.path.append("./myriad-qwen3.5-0.8b-hf-v2")
from joint_schema_model import JointSchemaHead, ClefModel, systemone

MACRO_KEYS = [
    "Code", "Math", "Logic", "SciTech", "Humanities", "Creative", "Chinese",
    "GuardAgent"
]
CLAN_KEYS = [
    "Clan_SysArch", "Clan_AppWeb", "Clan_DiscreteMath", "Clan_ContinuousMath",
    "Clan_DeductiveLogic", "Clan_PlanAgent", "Clan_PhysChem", "Clan_BioMed",
    "Clan_SocietyLaw", "Clan_PhilosophyMind", "Clan_Literature",
    "Clan_PersonaRoleplay", "Clan_ClassicalCN", "Clan_ModernLife",
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

LAYER_PROFILES = [{
    "main": 0.4
}, {
    "main": 0.6
}, {
    "main": 0.8
}, {
    "main": 1.0
}, {
    "main": 1.0
}, {
    "main": 1.0
}, {
    "main": 1.0
}, {
    "main": 1.0
}, {
    "main": 0.9
}, {
    "main": 0.7
}, {
    "main": 0.5
}, {
    "main": 0.3
}]


# ─── 物理 MoE 结构 ───
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

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class MyriadGraftedLayer(nn.Module):

    def __init__(self,
                 original_mlp,
                 hidden_dim=1024,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.base_mlp = original_mlp
        self.active_macro = None
        self.active_micro = None
        self.macro_scale = 1.0
        self.micro_scale = 0.5

        self.shared_lora = MicroLoRA(hidden_dim,
                                     hidden_dim,
                                     rank=32,
                                     scale=1.0,
                                     device=device,
                                     dtype=dtype)
        self.macro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim,
                      hidden_dim,
                      rank=16,
                      scale=1.0,
                      device=device,
                      dtype=dtype) for _ in range(8)
        ])
        self.micro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim,
                      hidden_dim,
                      rank=8,
                      scale=0.5,
                      device=device,
                      dtype=dtype) for _ in range(256)
        ])

    def forward(self, x):
        base_out = self.base_mlp(x)
        shared_delta = self.shared_lora(x)
        macro_delta = (
            self.macro_loras[self.active_macro](x) *
            self.macro_scale) if self.active_macro is not None else 0.0
        micro_delta = (
            self.micro_loras[self.active_micro](x) *
            self.micro_scale) if self.active_micro is not None else 0.0
        return base_out + shared_delta + macro_delta + micro_delta


def main():
    print("=" * 75)
    print("🚀 【纯血 Hugging Face 双脑大合体点火】(100% 零 Unsloth 依赖)")
    print("=" * 75)

    base_model_id = "Qwen/Qwen3.5-0.8B"
    lora_dir = "./myriad-qwen3.5-0.8b-hf-v2/cortex_lora"
    adapter_path = "./myriad-qwen3.5-0.8b-hf-v2/myriad_moe_adapters.pt"
    dtype = torch.bfloat16

    # 🌟 1. 载入纯血前额叶决策脑
    print("[1/3] 载入前额叶纯血决策脑 (PEFT + Clef)...")
    path = Path(lora_dir)
    cortex_base = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_id, dtype=dtype, device_map={"": "cuda:0"})
    cortex_backbone = PeftModel.from_pretrained(cortex_base, str(path)).eval()

    head_config = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"),
                         strict=True)
    head = head.to(device="cuda:0", dtype=dtype).eval()

    cortex_processor = AutoProcessor.from_pretrained(str(path))
    decision_engine = ClefModel(cortex_backbone, head).eval()
    print("✅ 纯血前额叶就绪！")

    # 🌟 2. 载入生成主干并嫁接 12 层武器库
    # ✅ 改为 AutoModelForCausalLM 并从元数据读 hidden_dim：
    gen_tokenizer = AutoTokenizer.from_pretrained(base_model_id)  # 👈 把这一行补上！
    gen_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                     dtype=dtype,
                                                     device_map="cuda:0")
    for p in gen_model.parameters():
        p.requires_grad = False

    checkpoint = torch.load(adapter_path, map_location="cpu")
    GRAFTED_LAYERS = checkpoint["metadata"]["grafted_layers"]
    hidden_dim = checkpoint["metadata"]["hidden_dim"]  # 🌟 直接读元数据，绝对不报错
    grafted_modules = []
    for idx in GRAFTED_LAYERS:
        wrapper = MyriadGraftedLayer(gen_model.model.layers[idx].mlp,
                                     hidden_dim=hidden_dim,
                                     device="cuda:0",
                                     dtype=dtype).to("cuda:0")
        wrapper.shared_lora.load_state_dict(
            checkpoint["adapters"][f"layer_{idx}_shared"])
        wrapper.macro_loras.load_state_dict(
            checkpoint["adapters"][f"layer_{idx}_macros"])
        wrapper.micro_loras.load_state_dict(
            checkpoint["adapters"][f"layer_{idx}_micros"])
        gen_model.model.layers[idx].mlp = wrapper
        grafted_modules.append(wrapper)

    print("✅ 全套 12 层 MoE 武器库已就位！")
    print("=" * 75)

    # 🌟 3. 实机双脑联动作战测试！
    prompt = "请帮我写一段针对 Linux epoll 高并发调优的 Rust 零拷贝网络库代码，要求解决死锁"
    print(f"\n👤 提问: {prompt}\n")

    # A. 纯 PyTorch 前额叶决策
    t0 = time.perf_counter()
    request = {
        "model": "default",
        "state": prompt,
        "questions": {
            "macro": {
                "type": "choice",
                "instructions": "选天王",
                "criteria": MACRO_CRITERIA
            },
            "clan": {
                "type": "choice",
                "instructions": "选宗门",
                "criteria": CLAN_CRITERIA
            }
        }
    }
    ans = systemone(decision_engine, cortex_processor, request)
    t_decision = (time.perf_counter() - t0) * 1000

    answers = ans.get("answers", ans)
    macro_name = answers["macro"]["choice"]
    clan_name = answers["clan"]["choice"]
    macro_id = MACRO_KEYS.index(macro_name)
    clan_id = CLAN_KEYS.index(clan_name)
    global_micro_id = clan_id * 16 + (hash(prompt) % 16)

    print("─" * 70)
    print(f"🧠 【前额叶决策战报】(纯原生耗时: {t_decision:.1f} ms):")
    print(f"   👑 领衔天王: #{macro_id} [{macro_name}]")
    print(f"   🚩 调遣宗门: #{clan_id:02d} [{clan_name}]")
    print(f"   🐾 激活微核: #{global_micro_id:03d}")
    print("─" * 70)

    # B. 12 层金字塔排布工单
    for l_idx, wrapper in enumerate(grafted_modules):
        profile = LAYER_PROFILES[l_idx]
        wrapper.active_macro = macro_id
        wrapper.active_micro = global_micro_id
        wrapper.macro_scale = profile["main"]
        wrapper.micro_scale = 0.5 * profile["main"]

    # C. 生成主干开火！现场写代码！
    messages = [{"role": "user", "content": prompt}]
    prompt_text = gen_tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
    inputs = gen_tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

    streamer = TextStreamer(gen_tokenizer,
                            skip_prompt=True,
                            skip_special_tokens=True)

    print("🤖 回复:\n")
    t_gen_start = time.perf_counter()
    with torch.inference_mode():
        _ = gen_model.generate(
            **inputs,
            max_new_tokens=512,
            streamer=streamer,
            do_sample=True,  # 🌟 加上这行，警告彻底消失，采样更灵动！
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )
    t_gen = time.perf_counter() - t_gen_start
    print(f"\n⚡ 生成耗时: {t_gen:.2f}s | 纯血端到端完整闭环！\n" + "═" * 70)


if __name__ == "__main__":
    main()
