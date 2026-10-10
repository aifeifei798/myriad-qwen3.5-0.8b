#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 100% 纯血 Hugging Face 双脑全自闭环 + 全链路内窥探测套件 (黄金配比调优版)
- 前额叶: 纯 PyTorch (ClefModel + systemone) 决策心率与置信度探测
- 嫁接层: 12 层金字塔 MoE 能量透视 (阻尼调校: Shared / Macro / Micro 黄金介入比)
- 消融仓: 5 模态消融对照实验 (全真对比)
- 主干体: 真实 Rust 逻辑与代码块流式输出
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    AutoTokenizer,
    TextStreamer,
    AutoModelForCausalLM
)
from peft import PeftModel

# 导入本地前额叶实现
sys.path.append("./myriad-qwen3.5-0.8b-hf-v2")
try:
    from joint_schema_model import JointSchemaHead, ClefModel, systemone
except ImportError:
    pass

# ─── 领域元数据配置 ───
MACRO_KEYS = [
    "Code", "Math", "Logic", "SciTech",
    "Humanities", "Creative", "Chinese", "GuardAgent"
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

# 🌟 重新调校的金字塔能量 Profile (主峰从 1.0 降至 0.35 黄金带)
LAYER_PROFILES = [
    {"main": 0.15}, {"main": 0.20}, {"main": 0.28},
    {"main": 0.35}, {"main": 0.35}, {"main": 0.35},
    {"main": 0.35}, {"main": 0.32}, {"main": 0.28},
    {"main": 0.22}, {"main": 0.16}, {"main": 0.10}
]

# ─── 物理 MoE 结构 + 能量探针集成 ───
class MicroLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank, scale=1.0, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.A = nn.Linear(in_dim, rank, bias=False, device=device, dtype=dtype)
        self.B = nn.Linear(rank, out_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class MyriadGraftedLayerWithProbe(nn.Module):
    def __init__(self, original_mlp, hidden_dim=1024, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.base_mlp = original_mlp
        self.active_macro = None
        self.active_micro = None
        
        # 🌟 核心阻尼参数 (关键降火点)
        self.shared_scale = 0.15   # 彻底把 Shared 能量从 58% 压到 8% 左右
        self.macro_scale = 0.35    # 宏观天王主导比例
        self.micro_scale = 0.28    # 微核细节比例

        # 遥测探针控制与寄存器
        self.probing_enabled = False
        self.last_probe_stats = {}

        self.shared_lora = MicroLoRA(hidden_dim, hidden_dim, rank=32, scale=1.0, device=device, dtype=dtype)
        self.macro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim, hidden_dim, rank=16, scale=1.0, device=device, dtype=dtype)
            for _ in range(8)
        ])
        self.micro_loras = nn.ModuleList([
            MicroLoRA(hidden_dim, hidden_dim, rank=8, scale=1.0, device=device, dtype=dtype)
            for _ in range(256)
        ])

    def forward(self, x):
        base_out = self.base_mlp(x)
        
        # 🌟 乘以 shared_scale 阻尼，彻底驯服 Shared LoRA
        shared_delta = self.shared_lora(x) * self.shared_scale
        macro_delta = (self.macro_loras[self.active_macro](x) * self.macro_scale) if self.active_macro is not None else torch.zeros_like(base_out)
        micro_delta = (self.micro_loras[self.active_micro](x) * self.micro_scale) if self.active_micro is not None else torch.zeros_like(base_out)

        out = base_out + shared_delta + macro_delta + micro_delta

        # 实时能量比探测 (Norm Probing)
        if self.probing_enabled:
            with torch.no_grad():
                b_norm = torch.linalg.norm(base_out, dim=-1).mean().item()
                s_norm = torch.linalg.norm(shared_delta, dim=-1).mean().item()
                m_norm = torch.linalg.norm(macro_delta, dim=-1).mean().item()
                u_norm = torch.linalg.norm(micro_delta, dim=-1).mean().item()
                total_delta = shared_delta + macro_delta + micro_delta
                tot_norm = torch.linalg.norm(total_delta, dim=-1).mean().item()
                eps = 1e-8

                self.last_probe_stats = {
                    "base_norm": b_norm,
                    "shared_pct": (s_norm / (b_norm + eps)) * 100,
                    "macro_pct": (m_norm / (b_norm + eps)) * 100,
                    "micro_pct": (u_norm / (b_norm + eps)) * 100,
                    "injection_ratio": (tot_norm / (b_norm + eps)) * 100,
                }

        return out


# ─── 探测套件函数 ───
def probe_frontal_entropy(logits, class_keys, top_k=3):
    """前额叶置信度与决策熵探测"""
    if logits is None:
        return None
    probs = torch.softmax(logits.float(), dim=-1).squeeze(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    max_entropy = math.log(len(class_keys))
    confusion = (entropy / max_entropy) * 100

    top_v, top_i = torch.topk(probs, k=min(top_k, len(class_keys)))
    leaderboard = [
        {"name": class_keys[idx.item()], "prob": val.item() * 100}
        for val, idx in zip(top_v, top_i)
    ]
    return {"entropy": entropy, "confusion": confusion, "leaderboard": leaderboard}


def print_energy_table(grafted_modules, grafted_layer_ids):
    """打印 12 层器官能量分布表"""
    print("\n" + "═" * 32 + " 🔬 12层嫁接器官能量透视表 " + "═" * 32)
    print(f"{'层号':<7} | {'基底模长':<10} | {'Shared%':<9} | {'Macro%':<9} | {'Micro%':<9} | {'总渗透率(Δ/Base)':<16} | 状态评估")
    print("-" * 88)
    for l_idx, wrapper in enumerate(grafted_modules):
        s = wrapper.last_probe_stats
        inj = s.get("injection_ratio", 0.0)
        
        # 更加科学的健康度评级标准
        if inj < 1.0:
            status = "💤 介入过低(几乎无感知)"
        elif inj > 25.0:
            status = "🔥 严重过载(易乱码/幻觉)"
        elif 4.0 <= inj <= 18.0:
            status = "🟢 黄金介入区(精准引导)"
        else:
            status = "🟡 轻度偏离"

        print(
            f"L{grafted_layer_ids[l_idx]:02d}    | "
            f"{s.get('base_norm', 0.0):<10.2f} | "
            f"{s.get('shared_pct', 0.0):>6.2f}%   | "
            f"{s.get('macro_pct', 0.0):>6.2f}%   | "
            f"{s.get('micro_pct', 0.0):>6.2f}%   | "
            f"{inj:>10.2f}%    | {status}"
        )
    print("═" * 88)


def run_ablation_probes(gen_model, gen_tokenizer, inputs, grafted_modules, macro_id, micro_id):
    """多模态消融与错配探测 (Ablation Lens)"""
    print("\n" + "═" * 28 + " 🧪 消融与抗噪对抗探测 (Logit Lens) " + "═" * 28)
    
    chaos_macro_id = (macro_id + 4) % len(MACRO_KEYS)
    chaos_micro_id = (micro_id + 128) % 256

    modes = {
        "全装上阵 (Full MoE)":       {"shared": 0.15, "m": macro_id,        "u": micro_id,        "m_s": 0.35, "u_s": 0.28},
        "阉割微核 (No Micro)":       {"shared": 0.15, "m": macro_id,        "u": None,            "m_s": 0.35, "u_s": 0.00},
        "阉割天王 (No Macro)":       {"shared": 0.15, "m": None,            "u": micro_id,        "m_s": 0.00, "u_s": 0.28},
        "纯血底模 (All Off)":        {"shared": 0.00, "m": None,            "u": None,            "m_s": 0.00, "u_s": 0.00},
        "赛博精神分裂 (Chaos Clash)": {"shared": 0.15, "m": chaos_macro_id,  "u": chaos_micro_id,  "m_s": 0.35, "u_s": 0.28},
    }

    for name, cfg in modes.items():
        for wrapper in grafted_modules:
            wrapper.shared_scale = cfg["shared"]
            wrapper.active_macro = cfg["m"]
            wrapper.active_micro = cfg["u"]
            wrapper.macro_scale = cfg["m_s"]
            wrapper.micro_scale = cfg["u_s"]

        with torch.inference_mode():
            out = gen_model(**inputs)
            last_token_logits = out.logits[:, -1, :]
            top1_id = torch.argmax(last_token_logits, dim=-1).item()
            token_str = gen_tokenizer.decode([top1_id]).replace("\n", "\\n")
            top1_prob = torch.softmax(last_token_logits.float(), dim=-1)[0, top1_id].item() * 100

        print(f"👉 模式 [{name:<20}] 首个产出 Token: '{token_str}' (置信度: {top1_prob:.1f}%)")
    print("═" * 84)


# ─── 主程序入口 ───
def main():
    print("=" * 80)
    print("🚀 【Myriad-MoE 双脑合体全开】带全功能深度内窥遥测引擎 (黄金调教版)")
    print("=" * 80)

    base_model_id = "Qwen/Qwen3.5-0.8B"
    lora_dir = "./myriad-qwen3.5-0.8b-hf-v2/cortex_lora"
    adapter_path = "./myriad-qwen3.5-0.8b-hf-v2/myriad_moe_adapters.pt"
    dtype = torch.bfloat16

    # 1. 载入纯血前额叶决策脑
    print("[1/4] 载入纯血前额叶决策脑 (PEFT + Clef)...")
    path = Path(lora_dir)
    cortex_base = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map={"": "cuda:0"}
    )
    cortex_backbone = PeftModel.from_pretrained(cortex_base, str(path)).eval()

    head_config = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    head = head.to(device="cuda:0", dtype=dtype).eval()

    cortex_processor = AutoProcessor.from_pretrained(str(path))
    decision_engine = ClefModel(cortex_backbone, head).eval()
    print("✅ 纯血前额叶就绪！")

    # 2. 载入生成主干并嫁接 12 层带探测功能武器库
    print("[2/4] 载入生成主干，实施 12 层器官外科移植与探测探针接入...")
    gen_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    gen_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=dtype,
        device_map="cuda:0"
    )
    for p in gen_model.parameters():
        p.requires_grad = False

    checkpoint = torch.load(adapter_path, map_location="cpu")
    GRAFTED_LAYERS = checkpoint["metadata"]["grafted_layers"]
    hidden_dim = checkpoint["metadata"]["hidden_dim"]

    grafted_modules = []
    for idx in GRAFTED_LAYERS:
        wrapper = MyriadGraftedLayerWithProbe(
            gen_model.model.layers[idx].mlp,
            hidden_dim=hidden_dim,
            device="cuda:0",
            dtype=dtype
        ).to("cuda:0")

        wrapper.shared_lora.load_state_dict(checkpoint["adapters"][f"layer_{idx}_shared"])
        wrapper.macro_loras.load_state_dict(checkpoint["adapters"][f"layer_{idx}_macros"])
        wrapper.micro_loras.load_state_dict(checkpoint["adapters"][f"layer_{idx}_micros"])

        gen_model.model.layers[idx].mlp = wrapper
        grafted_modules.append(wrapper)

    print("✅ 12 层带遥测探针的 MoE 器官全部嫁接锁死！")

    # 3. 双脑联动作战提问
    prompt = "请帮我写一段针对 Linux epoll 高并发调优的 Rust 零拷贝网络库代码，要求解决死锁"
    print(f"\n👤 提问目标: \"{prompt}\"")

    # ─── A. 前额叶决策与内窥 ───
    t0 = time.perf_counter()
    request = {
        "model": "default",
        "state": prompt,
        "questions": {
            "macro": {"type": "choice", "instructions": "选天王", "criteria": MACRO_CRITERIA},
            "clan": {"type": "choice", "instructions": "选宗门", "criteria": CLAN_CRITERIA}
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

    print("\n" + "═" * 30 + " 🧠 前额叶决策战报 " + "═" * 30)
    print(f"⏱️ 决策耗时: {t_decision:.2f} ms")
    print(f"👑 选定天王: #{macro_id} [{macro_name}]")
    print(f"🚩 选定宗门: #{clan_id:02d} [{clan_name}]")
    print(f"🐾 激活微核: #{global_micro_id:03d} (基于 Prompt 哈希槽注入)")

    if "logits" in ans:
        clan_probe = probe_frontal_entropy(ans["logits"].get("clan"), CLAN_KEYS)
        if clan_probe:
            print(f"📊 宗族决策困惑指数: {clan_probe['confusion']:.1f}% " +
                  ("(极其笃定)" if clan_probe['confusion'] < 20 else "(存在竞合摇摆)"))
            print("🏁 前额叶竞争态势 Top-3:")
            for item in clan_probe["leaderboard"]:
                print(f"   - {item['name']:<20}: {item['prob']:.2f}%")
    print("═" * 78)

    # ─── B. 能量探测与消融实验 ───
    messages = [{"role": "user", "content": prompt}]
    prompt_text = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

    # 1. 开启能量探针，按照重新校准的金字塔配比配置参数
    for l_idx, wrapper in enumerate(grafted_modules):
        profile = LAYER_PROFILES[l_idx]
        wrapper.active_macro = macro_id
        wrapper.active_micro = global_micro_id
        wrapper.shared_scale = 0.15                  # 👈 锁定 15% 阻尼
        wrapper.macro_scale = profile["main"]        # 👈 匹配平缓金字塔 (0.15~0.35)
        wrapper.micro_scale = 0.8 * profile["main"]  # 👈 强化微核发言权
        wrapper.probing_enabled = True

    with torch.inference_mode():
        _ = gen_model(inputs["input_ids"][:, :min(32, inputs["input_ids"].shape[1])])

    # 2. 打印能量分布表
    print_energy_table(grafted_modules, GRAFTED_LAYERS)

    # 3. 关闭能量计算，转入消融对抗探针
    for wrapper in grafted_modules:
        wrapper.probing_enabled = False

    run_ablation_probes(gen_model, gen_tokenizer, inputs, grafted_modules, macro_id, global_micro_id)

    # ─── C. 重新对齐金字塔权重，开始实战端到端流式生成 ───
    for l_idx, wrapper in enumerate(grafted_modules):
        profile = LAYER_PROFILES[l_idx]
        wrapper.active_macro = macro_id
        wrapper.active_micro = global_micro_id
        wrapper.shared_scale = 0.15
        wrapper.macro_scale = profile["main"]
        wrapper.micro_scale = 0.8 * profile["main"]

    streamer = TextStreamer(gen_tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("\n🤖 [主干体全功率开火回答]:\n")
    t_gen_start = time.perf_counter()
    with torch.inference_mode():
        _ = gen_model.generate(
            **inputs,
            max_new_tokens=600,
            streamer=streamer,
            do_sample=True,
            temperature=0.6,    # 略微降低发散度，提高代码结构严谨性
            top_p=0.9,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        )
    t_gen = time.perf_counter() - t_gen_start
    print(f"\n⚡ 生成纯耗时: {t_gen:.2f}s | 端到端全链路探测与生成闭环完成！\n" + "═" * 80)


if __name__ == "__main__":
    main()