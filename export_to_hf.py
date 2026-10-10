#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 一键封装为 Hugging Face 标准格式仓库
产出目录: ./myriad-qwen3.5-0.8b-hf
支持: AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
"""

import os
import json
import shutil
import torch
from transformers import AutoTokenizer

OUTPUT_DIR = "./myriad-qwen3.5-0.8b-hf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("📦 【Myriad-MoE: 封装为 Hugging Face 标准开源格式】")
print(f"📁 输出目录: {OUTPUT_DIR}")
print("=" * 75)

# ─── 1. 生成 configuration_myriad.py ───
config_py_content = '''# -*- coding: utf-8 -*-
from transformers import PretrainedConfig

class MyriadConfig(PretrainedConfig):
    model_type = "myriad_qwen"
    
    def __init__(
        self,
        base_model_name_or_path="Qwen/Qwen3.5-0.8B",
        decision_model_path="./qwen3.5-0.8b-cortex-merged",
        adapter_file="myriad_moe_adapters.pt",
        grafted_layers=list(range(8, 20)),
        macro_keys=None,
        clan_keys=None,
        layer_profiles=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.decision_model_path = decision_model_path
        self.adapter_file = adapter_file
        self.grafted_layers = grafted_layers
        self.macro_keys = macro_keys or [
            "Code", "Math", "Logic", "SciTech", "Humanities", "Creative", "Chinese", "GuardAgent"
        ]
        self.clan_keys = clan_keys or [
            "Clan_SysArch", "Clan_AppWeb",
            "Clan_DiscreteMath", "Clan_ContinuousMath",
            "Clan_DeductiveLogic", "Clan_PlanAgent",
            "Clan_PhysChem", "Clan_BioMed",
            "Clan_SocietyLaw", "Clan_PhilosophyMind",
            "Clan_Literature", "Clan_PersonaRoleplay",
            "Clan_ClassicalCN", "Clan_ModernLife",
            "Clan_FormatJSON", "Clan_GuardSafety"
        ]
        self.layer_profiles = layer_profiles or [
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
'''

with open(f"{OUTPUT_DIR}/configuration_myriad.py", "w", encoding="utf-8") as f:
    f.write(config_py_content)
print("✅ [1/5] 已写入 configuration_myriad.py")


# ─── 2. 生成 modeling_myriad.py ───
modeling_py_content = '''# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from unsloth import FastDecisionModel
from .configuration_myriad import MyriadConfig

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

class MyriadQwenForCausalLM(PreTrainedModel):
    config_class = MyriadConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        dtype = torch.bfloat16
        
        # 1. 载入主干
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            dtype=dtype, 
            device_map="cuda:0"
        )
        for p in self.gen_model.parameters():
            p.requires_grad = False
            
        # 2. 嫁接 12 层
        self.grafted_layers = []
        hidden_dim = self.gen_model.config.hidden_size
        
        # 尝试从模型本地路径或全局路径读取适配器
        adapter_file = os.path.join(getattr(config, "_name_or_path", "./"), config.adapter_file)
        if not os.path.exists(adapter_file):
            adapter_file = config.adapter_file
            
        checkpoint = torch.load(adapter_file, map_location="cpu")
        
        for idx in config.grafted_layers:
            wrapper = MyriadGraftedLayer(
                self.gen_model.model.layers[idx].mlp, 
                hidden_dim=hidden_dim, 
                device="cuda:0", 
                dtype=dtype
            ).to("cuda:0")

            wrapper.shared_lora.load_state_dict(checkpoint["adapters"][f"layer_{idx}_shared"])
            wrapper.macro_loras.load_state_dict(checkpoint["adapters"][f"layer_{idx}_macros"])
            wrapper.micro_loras.load_state_dict(checkpoint["adapters"][f"layer_{idx}_micros"])

            self.gen_model.model.layers[idx].mlp = wrapper
            self.grafted_layers.append(wrapper)

        # 3. 载入决策前额叶
        self.decision_model, self.decision_tokenizer = FastDecisionModel.from_pretrained(
            config.decision_model_path, 
            load_in_4bit=False, 
            dtype=dtype,
            local_files_only=True
        )
        FastDecisionModel.for_inference(self.decision_model)

    def dispatch_intent(self, prompt_text: str):
        """前额叶毫秒级意图决策并动态排布 12 层金字塔工单"""
        questions = {
            "macro": {"type": "choice", "instructions": "选天王", "criteria": MACRO_CRITERIA},
            "clan":  {"type": "choice", "instructions": "选宗门", "criteria": CLAN_CRITERIA}
        }
        pred = FastDecisionModel.predict(self.decision_model, self.decision_tokenizer, prompt_text, questions)
        
        macro_name = pred['macro']['answer']
        clan_name  = pred['clan']['answer']
        macro_id   = self.config.macro_keys.index(macro_name)
        clan_id    = self.config.clan_keys.index(clan_name)
        global_micro_id = clan_id * 16 + (hash(prompt_text) % 16)
        
        # 动态排布 12 层
        for l_idx, wrapper in enumerate(self.grafted_layers):
            profile = self.config.layer_profiles[l_idx]
            wrapper.active_macro = macro_id
            wrapper.active_micro = global_micro_id
            wrapper.macro_scale = profile["main"]
            wrapper.micro_scale = 0.5 * profile["main"]
            
        return {
            "macro": macro_name,
            "clan": clan_name,
            "micro_id": global_micro_id,
            "macro_probs": pred['macro']['probabilities'],
            "clan_probs": pred['clan']['probabilities']
        }

    def generate(self, *args, **kwargs):
        """标准生成接口，自动穿透执行"""
        return self.gen_model.generate(*args, **kwargs)
'''

with open(f"{OUTPUT_DIR}/modeling_myriad.py", "w", encoding="utf-8") as f:
    f.write(modeling_py_content)
print("✅ [2/5] 已写入 modeling_myriad.py")


# ─── 3. 生成 config.json (核心在于 auto_map) ───
config_json = {
    "architectures": ["MyriadQwenForCausalLM"],
    "model_type": "myriad_qwen",
    "auto_map": {
        "AutoConfig": "configuration_myriad.MyriadConfig",
        "AutoModelForCausalLM": "modeling_myriad.MyriadQwenForCausalLM"
    },
    "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
    "decision_model_path": "./qwen3.5-0.8b-cortex-merged",
    "adapter_file": "myriad_moe_adapters.pt",
    "grafted_layers": list(range(8, 20)),
    "macro_keys": ["Code", "Math", "Logic", "SciTech", "Humanities", "Creative", "Chinese", "GuardAgent"],
    "clan_keys": [
        "Clan_SysArch", "Clan_AppWeb",
        "Clan_DiscreteMath", "Clan_ContinuousMath",
        "Clan_DeductiveLogic", "Clan_PlanAgent",
        "Clan_PhysChem", "Clan_BioMed",
        "Clan_SocietyLaw", "Clan_PhilosophyMind",
        "Clan_Literature", "Clan_PersonaRoleplay",
        "Clan_ClassicalCN", "Clan_ModernLife",
        "Clan_FormatJSON", "Clan_GuardSafety"
    ],
    "layer_profiles": [
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
        {"main": 0.3, "chinese": 0.0, "guard": 0.7}
    ]
}

with open(f"{OUTPUT_DIR}/config.json", "w", encoding="utf-8") as f:
    json.dump(config_json, f, indent=2, ensure_ascii=False)
print("✅ [3/5] 已写入 config.json (附带 auto_map 动态类注册)")


# ─── 4. 拷贝适配器权重文件 ───
src_adapter = "myriad_moe_adapters.pt"
if os.path.exists(src_adapter):
    shutil.copy(src_adapter, f"{OUTPUT_DIR}/{src_adapter}")
    print(f"✅ [4/5] 已复制全套权重文件: {src_adapter}")
else:
    print(f"⚠️ 未找到 {src_adapter}，请确保权重在当前目录下")


# ─── 5. 导出标准分词器 ───
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ [5/5] 已导出完整 Tokenizer 文件群")

print("=" * 75)
print(f"🎉 导出完成！标准 HF 模型仓库已生成至: {OUTPUT_DIR}")
print("=" * 75)
