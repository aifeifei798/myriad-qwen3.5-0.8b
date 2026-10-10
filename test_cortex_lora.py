#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
# 🌟 纯离线运行，严防后台偷摸下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 🌟 必须在最顶层导入 unsloth
import unsloth
import time
import torch
from unsloth import FastDecisionModel

# 8 天王与 16 宗门标准法典
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


def test_prompt(model, tokenizer, prompt_text):
    questions = {
        "macro": {
            "type": "choice",
            "instructions": "该指令应当派发给哪位天王大核？",
            "criteria": MACRO_CRITERIA,
        },
        "clan": {
            "type": "choice",
            "instructions": "该指令应当派发给哪个专属宗门？",
            "criteria": CLAN_CRITERIA,
        }
    }

    t0 = time.perf_counter()
    pred = FastDecisionModel.predict(model, tokenizer, prompt_text, questions)
    cost_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "═" * 70)
    print(f"📝 测试 Prompt: {prompt_text}")
    print(f"⚡ 决策耗时: {cost_ms:.2f} ms")
    print(f"👑 命中天王: {pred['macro']['answer']}")
    print(f"   Top-2 天王置信度:")
    top_macros = sorted(pred['macro']['probabilities'].items(),
                        key=lambda x: x[1],
                        reverse=True)[:2]
    for k, v in top_macros:
        print(f"      - {k:<12}: {v*100:5.1f}%")

    print(f"🚩 命中宗门: {pred['clan']['answer']}")
    top_clans = sorted(pred['clan']['probabilities'].items(),
                       key=lambda x: x[1],
                       reverse=True)[:2]
    print(f"   Top-2 宗门置信度:")
    for k, v in top_clans:
        print(f"      - {k:<20}: {v*100:5.1f}%")
    print("═" * 70)


def main():
    lora_path = "./qwen3.5-0.8b-cortex-lora"
    print(f"🚀 正在加载轻量级纯 LoRA 前额叶卡带: {lora_path} ...")

    # 🌟 Unsloth 会自动读取 adapter_config.json 找到本地基模，挂载这 50MB 适配器！
    model, tokenizer = FastDecisionModel.from_pretrained(lora_path,
                                                         load_in_4bit=False,
                                                         dtype=torch.bfloat16,
                                                         local_files_only=True)
    FastDecisionModel.for_inference(model)
    print("✅ LoRA 挂载成功！开始测试毫秒级决策能力：\n")

    # 1. 深度系统架构题 (测试 Code + Clan_SysArch)
    test_prompt(model, tokenizer,
                "请帮我写一段针对 Linux epoll 高并发调优的 Rust 零拷贝网络库代码，要求解决死锁")

    # 2. 强格式约束题 (测试 GuardAgent + Clan_FormatJSON)
    test_prompt(model, tokenizer, "请把这段混乱的地址文本解析成标准 JSON 格式，严格不要带任何解释和前缀")

    # 3. 古典文学题 (测试 Chinese + Clan_ClassicalCN)
    test_prompt(model, tokenizer, "请用太史公笔法写一篇关于当代程序员深夜攻克 bug 的史记列传")


if __name__ == "__main__":
    main()
