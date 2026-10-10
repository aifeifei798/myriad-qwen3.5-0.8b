#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 🌟 绝无 unsloth！纯正的 PyTorch + Transformers + PEFT
import json
import time
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

# 导入刚拷过来的纯血实现
import sys

sys.path.append("./myriad-qwen3.5-0.8b-hf-v2")
from joint_schema_model import JointSchemaHead, ClefModel, systemone

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


def load_pure_cortex(lora_dir: str,
                     base_model_id="Qwen/Qwen3.5-0.8B",
                     device="cuda:0"):
    print(f"🚀 [纯血 HF] 正在组装前额叶 (零 Unsloth 依赖)...")
    path = Path(lora_dir)
    dtype = torch.bfloat16

    # 1. 载入原生基座并挂上 PEFT 纯 LoRA (~50MB)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_id, dtype=dtype, device_map={"": str(device)})
    backbone = PeftModel.from_pretrained(base, str(path)).eval()

    # 2. 用原生 PyTorch 挂载打分头 (joint_head)
    head_config = json.loads((path / "joint_head_config.json").read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"),
                         strict=True)
    head = head.to(device=device, dtype=dtype).eval()

    # 3. 载入官方 Processor
    processor = AutoProcessor.from_pretrained(str(path))

    # 封装为 ClefModel
    return ClefModel(backbone, head).eval(), processor


def main():
    lora_dir = "./myriad-qwen3.5-0.8b-hf-v2/cortex_lora"
    model, processor = load_pure_cortex(lora_dir)
    print("✅ 纯血前额叶组装完毕！开始测试：\n")

    prompt = "请帮我写一段针对 Linux epoll 高并发调优的 Rust 零拷贝网络库代码，要求解决死锁"

    request = {
        "model": "default",  # 🌟 补上这一个字段即可！
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

    t0 = time.perf_counter()
    # 🌟 调用纯 PyTorch 的原生 systemone 函数决策！
    ans = systemone(model, processor, request)
    cost = (time.perf_counter() - t0) * 1000

    print("═" * 70)
    print(f"📝 提问: {prompt}")
    print(f"⚡ 耗时: {cost:.2f} ms (纯原生 PyTorch + PEFT 前向)")
    
    # 🌟 兼容读取 SystemOne 标准返回格式
    answers = ans.get("answers", ans)
    macro_res = answers["macro"]
    clan_res = answers["clan"]
    
    macro_choice = macro_res.get("choice", macro_res)
    clan_choice = clan_res.get("choice", clan_res)

    print(f"👑 天王命中: {macro_choice}")
    print(f"🚩 宗门命中: {clan_choice}")
    if "probabilities" in macro_res:
        print(f"   天王置信度: {macro_res['probabilities']}")
    print("═" * 70)


if __name__ == "__main__":
    main()
