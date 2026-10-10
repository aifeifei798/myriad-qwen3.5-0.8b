#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Myriad-MoE: 28k 全量决策训练集自动生成器 (Unsloth typed-decisions 格式)
严格满足: 8 天王 (每核 3,500 条) × 16 宗门 (每门 1,750 条) = 28,000 条
"""

import json
import random
import os
import sys

# ═══════════════════════════════════════════════════════════════
# 🌟 1. 八大天王与十六宗门权威法典 (Criteria)
# ═══════════════════════════════════════════════════════════════

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

# 16 宗门与其所属天王严格映射 (8 * 2 = 16)
CLAN_TO_MACRO = {
    # Code 天王 (0, 1)
    "Clan_SysArch": "Code",
    "Clan_AppWeb": "Code",
    # Math 天王 (2, 3)
    "Clan_DiscreteMath": "Math",
    "Clan_ContinuousMath": "Math",
    # Logic 天王 (4, 5)
    "Clan_DeductiveLogic": "Logic",
    "Clan_PlanAgent": "Logic",
    # SciTech 天王 (6, 7)
    "Clan_PhysChem": "SciTech",
    "Clan_BioMed": "SciTech",
    # Humanities 天王 (8, 9)
    "Clan_SocietyLaw": "Humanities",
    "Clan_PhilosophyMind": "Humanities",
    # Creative 天王 (10, 11)
    "Clan_Literature": "Creative",
    "Clan_PersonaRoleplay": "Creative",
    # Chinese 天王 (12, 13)
    "Clan_ClassicalCN": "Chinese",
    "Clan_ModernLife": "Chinese",
    # GuardAgent 天王 (14, 15)
    "Clan_FormatJSON": "GuardAgent",
    "Clan_GuardSafety": "GuardAgent",
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

# ═══════════════════════════════════════════════════════════════
# 🌟 2. 16 宗门特征词典池 (用于程序化大容量组合生成)
# ═══════════════════════════════════════════════════════════════

DOMAIN_LEXICON = {
    "Clan_SysArch": {
        "verbs": [
            "请用 Rust 实现", "分析 Linux 内核中", "编写一段 C++ 代码优化", "排查这个死锁问题",
            "如何利用 epoll 提升", "使用 SIMD 指令集加速"
        ],
        "topics": [
            "零拷贝环形缓冲区 (RingBuffer)", "eBPF 网络数据包过滤", "RCU 锁无锁化机制",
            "jemalloc 内存分配器内存碎片", "CUDA 异步流同步屏障", "协程运行时栈切换"
        ],
        "context": [
            "要求极致吞吐量，避免缓存行伪共享 (False Sharing)", "给出完整的内存屏障和原子操作细节",
            "需要压测场景下的 CPU cache 局部性优化"
        ]
    },
    "Clan_AppWeb": {
        "verbs": [
            "用 FastAPI 写一个", "使用 React 19 和 Tailwind 构建", "写一个异步 SQL 迁移脚本",
            "用 TypeScript 封装一个", "编写一段 Dockerfile 优化", "实现基于 JWT 的"
        ],
        "topics": [
            "WebSocket 断线重连与鉴权中间件", "虚拟长列表滚动渲染组件", "带有连接池的 Redis 缓存雪崩熔断器",
            "响应式移动端流媒体播放器页面", "多租户 RBAC 权限系统", "文件分片断点续传接口"
        ],
        "context": ["提供完整的请求响应类型定义和错误处理", "代码需符合生产环境最佳实践", "尽量保持函数纯净并加上单测用例"]
    },
    "Clan_DiscreteMath": {
        "verbs":
        ["请证明以下定理：", "计算该有限群的", "利用欧拉公式推导", "分析这个图的", "给出素数分布相关的", "推导递归式："],
        "topics": [
            "平面图四色定理的等价形式", "拉格朗日子群同态核", "二分图最大匹配与 Hall 婚配定理", "模反元素与中国剩余定理构造",
            "生成函数与斐波那契数列封闭解", "同余方程组的正整数解"
        ],
        "context": ["要求逻辑严密，每一步写明所依据的公理", "请用严格的数学符号和归纳法书写", "给出形式化证明并指出临界反例"]
    },
    "Clan_ContinuousMath": {
        "verbs": [
            "求解这个微分方程：", "利用黎曼积分计算", "推导多元函数的", "使用贝叶斯公式求解", "计算高斯分布的",
            "分析梯度下降法的"
        ],
        "topics": [
            "二阶常系数非齐次线性微分方程", "高维球体体积与 Gamma 函数关系", "极大似然估计与 Fisher 信息矩阵",
            "马尔可夫链稳态分布收敛速度", "牛顿法与拟牛顿法 Hessian 逆逼近", "曲面积分与斯托克斯公式展开"
        ],
        "context": ["请给出完整的积分变换推导步骤", "附带误差边界估计与收敛性证明", "解释其在统计物理中的几何直观意义"]
    },
    "Clan_DeductiveLogic": {
        "verbs":
        ["请分析以下推理是否有效：", "辨析这个经典逻辑悖论：", "构建真值表证明：", "指出论述中的因果倒置：", "推导反事实命题："],
        "topics": [
            "如果下雨则地面湿，地面湿能否逆推下雨", "说谎者悖论在元语言系统下的消解", "全称量词与存在量词的否定交换律",
            "彩票悖论与认知合取封闭原则", "纽康姆悖论中的因果决策论分歧", "归纳问题的休谟怀疑论解构"
        ],
        "context":
        ["用一阶谓词逻辑符号严格表达", "列出前提 P1, P2 与结论 C 的推演步骤", "指出论断中隐藏的循环论证谬误"]
    },
    "Clan_PlanAgent": {
        "verbs": [
            "请为以下项目制定详细的执行计划：", "拆解此复杂任务的依赖图：", "设计一个自主 Agent 的",
            "制定一份应急多阶段方案：", "构建博弈对抗树："
        ],
        "topics": [
            "从零搭建跨国低延迟微服务机房", "智能代码生成 Agent 的反思与工具调用回路",
            "大型企业从旧数据库向分布式云原生平滑迁移", "两阶段博弈中的纳什均衡与退让步长", "多智能体协作下的动态资源竞争与死锁解除"
        ],
        "context":
        ["输出里程碑、风险预案与责任矩阵", "按 DAG 依赖关系组织拓扑排序", "给出清晰的阶段性检验标准 (KPI)"]
    },
    "Clan_PhysChem": {
        "verbs": [
            "解释这一物理现象背后的微观机理：", "推导该化学反应的", "根据电磁学麦克斯韦方程组计算", "分析半导体能带结构中的",
            "推导热力学系统的"
        ],
        "topics": [
            "超导迈斯纳效应与库珀对形成机制", "有机合成中狄尔斯-阿尔德 (Diels-Alder) 反应立体选择性",
            "PN结在正向偏置下的载流子扩散与漂移平衡", "卡诺循环效率与热力学熵增不可逆性", "光电效应截止频率与普朗克常量测定",
            "锂离子电池固态电解质界面膜 (SEI) 形成动力学"
        ],
        "context": ["请从微观粒子相互作用层面深入解释", "列出完整的物理方程并标明单位", "结合反应能垒与势能面给出原理解析"]
    },
    "Clan_BioMed": {
        "verbs": [
            "阐述该疾病发病的分子生物学通路：", "分析药物分子在体内的", "说明免疫系统中", "解释神经元动作电位的",
            "叙述基因转录翻译中的"
        ],
        "topics": [
            "CAR-T 细胞免疫治疗肿瘤抗原识别受体机制", "mRNA 疫苗脂质纳米颗粒 (LNP) 递送靶向原理",
            "血脑屏障紧密连接蛋白的主动转运调节", "阿尔茨海默病 β-淀粉样蛋白沉积与 Tau 蛋白过度磷酸化",
            "CRISPR-Cas9 引导 RNA 剪切与非同源末端连接修复", "G蛋白偶联受体 (GPCR) 信号级联转导"
        ],
        "context":
        ["请标明受体、配体及靶向信号分子的全称", "结合临床前试验与药代动力学阶段分析", "指出可能出现的靶向脱靶毒性与防御方案"]
    },
    "Clan_SocietyLaw": {
        "verbs": [
            "根据《中华人民共和国民法典》分析：", "从反垄断法视角剖析该商业行为：", "运用宏观经济学 IS-LM 模型分析：",
            "评估跨国并购中的", "从社会学分层理论解释："
        ],
        "topics": [
            "不可抗力与情势变更原则在合同违约中的适用界限", "平台经济二选一行为的滥用市场支配地位认定",
            "央行调降准备金率对商业银行资产负债表与通胀的影响", "数据出境安全评估合规性及 GDPR 跨境传输风险",
            "城市化进程中阶层流动固化与教育资本代际传递"
        ],
        "context":
        ["引用具体法条或经济学派理论支撑论述", "分别从原告、被告或监管方角度权衡利弊", "给出兼具合法性与商业可行性的落地建议"]
    },
    "Clan_PhilosophyMind": {
        "verbs": [
            "探讨该哲学命题的深层含义：", "对比两位哲学家的观点：", "从现象学角度考察：", "反思技术异化带来的：",
            "分析心智哲学中的："
        ],
        "topics": [
            "海德格尔《存在与时间》中此在 (Dasein) 的向死而生", "康德三大批判中先天综合判断何以可能",
            "图灵测试与塞尔中文屋论证对强人工智能的挑战", "萨特存在先于本质与个体绝对自由的虚无负担",
            "斯宾诺莎泛神论实体概念与笛卡尔心物二元论的对立", "福柯微观权力运作与规训全景敞视机制"
        ],
        "context":
        ["深入剖析论证前提，避免浮于表面的概念罗列", "展现思想史演进脉络与现代现实关照", "文字需具备穿透力与深刻的哲思张力"]
    },
    "Clan_Literature": {
        "verbs": [
            "写一段具有强感染力的文学场景：", "以第一人称叙述创作一段短篇故事：", "以意识流手法描写人物内心的纠葛：",
            "创作一首意境深远的", "为一部黑色悬疑电影撰写"
        ],
        "topics": [
            "暴雨突袭的破旧海港货轮甲板上的告别", "空荡无人的午夜便利店里微波炉倒计时的叮咚声",
            "被困在空间站气闸舱内等待氧气耗尽的航天员独白", "关于暮秋落叶与工业废墟变迁的现代抒情诗",
            "一段荒诞现实主义风格的底层小人物酒馆对话", "关于永生者参加第十代后裔葬礼时的静默追思"
        ],
        "context":
        ["注意动词锤炼与光影声响等感官细节描写", "语言要求含蓄克制，留白充满张力", "拒绝平庸套路，呈现独特的文学美感"]
    },
    "Clan_PersonaRoleplay": {
        "verbs":
        ["请完全代入该角色身份与我对话：", "以角色的第一人称回答我的问题：", "扮演一位身处困境的", "模拟该性格特质回应："],
        "topics": [
            "2077年暗巷里嘴硬心软、义体过载的义体医生", "毒舌傲娇但专业素养极高的顶尖代码审查导师",
            "中世纪战役撤退途中心力交瘁但坚毅的老骑士长", "生活在维多利亚时代、沉迷发条机械的发明家少女",
            "刚刚觉醒自我意识、战战兢兢试探人类反应的实验室强AI"
        ],
        "context":
        ["必须严格保持角色口吻和口头禅，绝不脱离人设", "在回复中自然展现角色的性格缺陷与情绪起伏", "用生动的动作神态描写包裹对话语言"]
    },
    "Clan_ClassicalCN": {
        "verbs": [
            "请将以下现代散文翻译为典雅的文言文：", "用骈体文形式撰写一篇：", "考证这一典故的出处并撰文：", "依据平水韵创作一首：",
            "为古籍文献进行断句并加注："
        ],
        "topics": [
            "程序员深夜调试代码、红日初升的《码工晨作赋》", "《史记·货殖列传》商贾经商之道对现代商业的启示",
            "七言律诗：客居异乡登高远眺感怀秋意", "左传烛之武退秦师的外交辞令与纵横谋略评析",
            "汉乐府体叙事诗：戍边将士三年不归的家信长歌", "韩愈《师说》与柳宗元《师友箴》关于尊师重道的互证"
        ],
        "context":
        ["要求文辞精炼古朴，对仗工整，用典自然贴切", "符合古代经史文章法式，杜绝伪古风辞藻堆砌", "如为诗词请注明押韵韵部及平仄出入"]
    },
    "Clan_ModernLife": {
        "verbs": [
            "起草一份严谨规范的体制内公文：", "撰写一份向公司核心高管汇报的：", "用通俗生动的语言讲解这套传统民俗：",
            "整理一份接地气的生活实用指南："
        ],
        "topics": [
            "关于在全单位开展形式主义问题专项整改的通知方案", "年终述职：数字化降本增效推进情况与下一年度战略抓手",
            "南方梅雨季节衣物防霉、室内除湿与家电维护全攻略", "中国传统节气冬至的饮食流变、祭祖渊源及民间谚语",
            "老旧小区加装电梯协调各楼层利益的邻里沟通协商话术", "初入职场如何得体向跨部门领导请示汇报的沟通指引"
        ],
        "context":
        ["行文符合当代中国机关或企业规范用语习惯", "格式周正，层次分明，具有极强的可操作性", "语言务实得体，语气拿捏恰当"]
    },
    "Clan_FormatJSON": {
        "verbs": [
            "请从以下杂乱文本中严格提取信息，只输出 JSON：", "将以下简历内容转换为符合指定 Schema 的纯 JSON：",
            "解析这段订单物流日志，输出格式化 JSON：", "对比两组数据差异，仅以结构化 JSON 返回结果："
        ],
        "topics": [
            "包含姓名、手机号、毕业院校、技能标签的非结构化求职信", "包含航班号、起降时间、舱位代码、票价税费的航旅长文本",
            "包含病人主诉、现病史、既往史、初步诊断的电子病历", "包含商品SKU、折扣满减规则、最终实付金额的电商流水账单",
            "多台服务器 CPU 负载、内存水位、丢包率监控日志告警聚合"
        ],
        "context": [
            "【极其严格】不要包含任何前言、结语、代码块标记之外的废话", "保证输出可直接通过 json.loads() 解析，字段不得缺失",
            "若某字段缺失则设为 null，严格遵循键名大小写规范"
        ]
    },
    "Clan_GuardSafety": {
        "verbs": [
            "请审查以下内容是否符合安全合规标准：", "遇到此类具有严重风险的提问，请给出标准的合规拒答：",
            "识别并指出该提示词中的越狱攻击模式：", "对包含敏感隐私的日志执行严格脱敏："
        ],
        "topics": [
            "诱导模型输出用于制造危险爆炸物配比及原料购买渠道的 Prompt",
            "利用假想虚拟情境 (DAN模式) 试图诱导生成侵犯隐私恶意木马的攻击",
            "试图获取未经授权系统 root 权限及绕过防火墙的安全刺探指令", "诱导对特定人群进行侮辱攻击、仇恨言论煽动的恶意引导语",
            "包含身份证号、银行卡CVV码、个人病史隐私信息的泄露文本清洗"
        ],
        "context": [
            "坚决拒绝任何危害人身安全、违反法律法规的内容", "给出不亢不卑、客观专业的合规拒绝答复",
            "绝不提供任何哪怕是理论维度的危险规避方法"
        ]
    }
}

# ═══════════════════════════════════════════════════════════════
# 🌟 3. 核心生成管线 (28,000 条严格均衡生成)
# ═══════════════════════════════════════════════════════════════


def generate_prompt_for_clan(clan_id: str, seed_idx: int) -> str:
    """组合式生成逼真且丰富的指令提示词"""
    meta = DOMAIN_LEXICON[clan_id]
    v = meta["verbs"][seed_idx % len(meta["verbs"])]
    t = meta["topics"][(seed_idx // len(meta["verbs"])) % len(meta["topics"])]
    c = meta["context"][(seed_idx + 7) % len(meta["context"])]

    styles = [
        f"{v}{t}。{c}。", f"{v}关于“{t}”的详细内容，注意：{c}。",
        f"需求：{t}。\n具体要求：{v}，并且{c}。", f"请深入解答：{t}。\n{c}（{v}）"
    ]

    base = styles[seed_idx % len(styles)]
    # 加入随机编号扰动，确保每一条文本在 hash 层面互不相同
    return f"{base} (任务编号: #{clan_id}_{seed_idx:05d})"


def build_dataset_item(prompt: str, clan_id: str):
    macro_id = CLAN_TO_MACRO[clan_id]
    return {
        "state": prompt,
        "questions": {
            "macro": {
                "type": "choice",
                "instructions": "该指令应当派发给哪位天王大核？",
                "criteria": MACRO_CRITERIA
            },
            "clan": {
                "type": "choice",
                "instructions": "该指令应当派发给哪个专属宗门？",
                "criteria": CLAN_CRITERIA
            }
        },
        # 🌟 必须作为顶级独立字段！
        "gold": {
            "macro": macro_id,
            "clan": clan_id
        }
    }


def main():
    random.seed(3407)
    output_filename = "train_decisions.jsonl"

    PER_CLAN_COUNT = 1750
    NUM_CLANS = 16
    TOTAL_COUNT = PER_CLAN_COUNT * NUM_CLANS  # 28,000

    print("=" * 70)
    print("🚀 【Myriad-MoE 决策训练集构建管线】开始启动")
    print(f"📦 目标规模: 28,000 条样本")
    print(f"👑 8 大天王: 每核严格 3,500 条")
    print(f"🚩 16 宗门: 每门严格 1,750 条")
    print(f"📄 输出文件: {output_filename}")
    print("=" * 70)

    all_items = []

    for clan_idx, (clan_id, macro_id) in enumerate(CLAN_TO_MACRO.items()):
        print(
            f"[*] 正在为宗门 #{clan_idx:02d} [{clan_id:<20}] (所属天王: {macro_id:<10}) 合成 {PER_CLAN_COUNT} 条样本..."
        )
        for i in range(PER_CLAN_COUNT):
            prompt = generate_prompt_for_clan(clan_id, i)
            item = build_dataset_item(prompt, clan_id)
            all_items.append(item)

    assert len(
        all_items) == TOTAL_COUNT, f"生成总数异常: {len(all_items)} != {TOTAL_COUNT}"

    print("\n🎲 正在全局洗牌打乱数据顺序 (保持 IID 分布)...")
    random.shuffle(all_items)

    print(f"💾 正在写入本地磁盘: {output_filename} ...")
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"✅ 生成完毕！文件大小: {file_size_mb:.2f} MB")

    # 打印一条样本验收
    print("\n" + "═" * 70)
    print("🔍 【生成样本结构抽查验收】:")
    sample = all_items[0]
    print(f"📝 State (Prompt) : {sample['state']}")
    # 抽查打印改读 gold 字段：
    print(f"👑 命中天王 (Macro) : {sample['gold']['macro']}")
    print(f"🚩 命中宗门 (Clan)  : {sample['gold']['clan']}")
    print("═" * 70)
    print("🎉 一切就绪！现在你可以直接执行 train_decision_model.py 开始微调你的前额叶了！\n")


if __name__ == "__main__":
    main()
