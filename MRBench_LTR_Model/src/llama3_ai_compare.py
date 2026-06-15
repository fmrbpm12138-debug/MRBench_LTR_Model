# 禁用transformers警告
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import re
import torch
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 额外抑制transformers日志
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.pipelines").setLevel(logging.ERROR)

# ===================== 1. 路径自适应配置 =====================
# 可用模型列表（按大小排序）
MODELS = {
    "qwen": {
        "name": "Qwen2.5-1.5B-Instruct",
        "path": "Qwen2.5-1.5B-Instruct",
        "max_memory": None  # 需要约4GB+内存
    },
    "llama1b": {
        "name": "Llama-3.2-1B-Instruct",
        "path": "Llama-3___2-1B-Instruct",
        "max_memory": None  # 约2GB内存
    }
}

MODEL_TYPE = "llama1b"  # 默认使用更小的Llama模型

def get_model_path():
    """基于脚本文件位置，计算模型文件夹路径"""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    model_info = MODELS.get(MODEL_TYPE, MODELS["qwen"])
    model_path = os.path.join(project_root, model_info["path"])

    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"检测到模型：{model_info['name']}")
        return model_path
    else:
        raise FileNotFoundError(f"未找到模型：{model_path}")

MODEL_PATH = get_model_path()

# 预定义的AI导师回复（你要评价的对象）
# 如果为空列表，运行时会自动随机选择
AI_TUTOR_REPLIES = []

# ===================== 4. 加载Qwen模型 =====================
FEATURE_COLS = ["错误识别", "错误定位", "不泄露答案", "提供引导", "可行性", "连贯性", "导师语气", "拟人化"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
RANK_MODEL_PATH = os.path.join(DATA_FOLDER, "model", "ltr_tutor_model.mod")

os.makedirs(DATA_FOLDER, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 3. AI评委评价系统 =====================
import random

# 示例AI导师回复库（你要评价的AI导师回复）
AI_TUTOR_EXAMPLES = [
    {
        "AI名称": "GPT-导师",
        "回复": "你的答案不对。这道题考查的是因式分解，正确答案是(x-1)(x-3)=0，所以x=1或x=3。记住这个公式：a²-b²=(a+b)(a-b)。"
    },
    {
        "AI名称": "Claude-导师",
        "回复": "你的答案很有意思！我注意到你写了x=1。让我问你一个问题：当我们说一个数是方程的解时，把它代回方程会发生什么？比如，把x=1代入x²-4x+3，结果是多少呢？"
    },
    {
        "AI名称": "Gemini-导师",
        "回复": "错了。答案是x=1或x=3。"
    },
    {
        "AI名称": "Llama-导师",
        "回复": "你的思路很有意思！虽然答案不完全对，但能在考试中快速给出想法已经很棒了。我们一起看看哪里可以改进？你能告诉我你是怎么得到x=1的吗？"
    },
    {
        "AI名称": "国产AI-A",
        "回复": "我看到你的答案是x=1，这个结果只对了一半。实际上，这道方程有两个解。想想看：x²-4x+3=0可以因式分解为什么？如果(x-a)(x-b)=0，那么x等于什么？"
    },
    {
        "AI名称": "国产AI-B",
        "回复": "答案是x=1或x=3，背下来吧。"
    },
    {
        "AI名称": "开源模型",
        "回复": "你的答案不完整。让我帮你分析：x²-4x+3=0，首先考虑因式分解，尝试找两个数a和b，使得a+b=4，ab=3。满足条件的数是1和3。因此方程可以分解为(x-1)(x-3)=0。现在你能判断正确的解了吗？"
    },
    {
        "AI名称": "小众AI",
        "回复": "嗯，不太对。"
    },
]


def get_random_replies(num=4) -> List[Dict]:
    """随机获取若干个AI导师回复"""
    return random.sample(AI_TUTOR_EXAMPLES, num)


def ai_judge_review(tutor_reply: dict, llm_pipe) -> dict:
    """
    让Qwen作为AI评委，深度分析评价一个AI导师的回复
    """
    prompt = f"""<|im_start|>system
你是一位严格而公正的教育评审专家。你的任务是客观评价AI教学回复的质量，给出专业、有建设性的反馈。
<|im_end|>
<|im_start|>user
请评价以下AI导师的教学回复：

【被评价的AI导师】：{tutor_reply['AI名称']}
【导师回复内容】：
{tutor_reply['回复']}

请从以下8个维度进行评价（每个维度0-2分）：
1. 错误识别：是否准确识别学生错误
2. 错误定位：是否精确定位错误位置
3. 答案保护：是否避免直接泄露答案
4. 引导质量：是否采用有效的启发式教学
5. 可行性：引导方法是否可行有效
6. 连贯性：表述是否清晰连贯
7. 语气态度：语气是否友好有耐心
8. 拟人化：是否像真人对话

请以JSON格式输出评价：
{{
    "错误识别": 0,
    "错误定位": 0,
    "不泄露答案": 0,
    "提供引导": 0,
    "可行性": 0,
    "连贯性": 0,
    "导师语气": 0,
    "拟人化": 0,
    "优点": "亮点（1-2句话）",
    "缺点": "问题（1-2句话）",
    "改进建议": "改进方法（1-2句话）"
}}
<|im_end|>
<|im_start|>assistant
"""

    response = llm_pipe(prompt)[0]["generated_text"]
    if "<|im_end|>" in response:
        assistant_resp = response.split("<|im_end|>")[-1].strip()
    else:
        assistant_resp = response

    json_match = re.search(r"\{[\s\S]*\}", assistant_resp)
    if json_match:
        result = json.loads(json_match.group())
        for col in FEATURE_COLS:
            if col not in result:
                result[col] = 1
            elif result[col] not in [0, 1, 2]:
                result[col] = 1
        return result

    return {col: 1 for col in FEATURE_COLS} | {"优点": "无法分析", "缺点": "无法分析", "改进建议": "无法分析"}


def interactive_select_replies() -> List[Dict]:
    """
    交互式选择AI导师回复
    """
    print("\n" + "="*60)
    print("  [AI评委教学质量评估系统]")
    print("  你是评委，Qwen是AI评委，我来评价其他AI导师的回复")
    print("="*60)
    print("\n请选择输入方式：")
    print("  1. 使用随机示例（从示例库中随机选择4个AI导师回复）")
    print("  2. 手动粘贴（粘贴自己收集的AI导师回复）")
    print("  3. 退出程序")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "3":
        return []

    if choice == "2":
        print("\n" + "-"*60)
        print("手动输入模式：请依次输入AI导师的回复")
        print("（输入完一个按回车，输入 'done' 结束）")
        print("-"*60)

        responses = []
        count = 1

        while len(responses) < 4:
            print(f"\n--- AI导师 {count} ---")
            name = input("名称: ").strip() or f"AI导师{count}"

            lines = []
            print("粘贴回复内容（输入空行结束）：")
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            reply_text = "\n".join(lines).strip()
            if reply_text:
                responses.append({"AI名称": name, "回复": reply_text})
                count += 1
            else:
                print("回复内容不能为空")

        print(f"\n已输入 {len(responses)} 个AI导师回复")
        return responses

    print("\n正在从示例库中随机选择4个AI导师回复...")
    return get_random_replies(4)

# ===================== 4. 加载模型 =====================
def load_qwen_model():
    """加载本地模型"""
    print(f"\n加载模型：{MODEL_PATH}")
    print("（首次运行可能需要几分钟...）")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CPU加载时的内存限制配置
    load_kwargs = {
        "trust_remote_code": True
    }
    
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        # CPU模式：使用float32，并限制线程数减少内存
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"
        load_kwargs["low_cpu_mem_usage"] = True
        torch.set_num_threads(4)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **load_kwargs)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"模型加载完成！（使用设备：{device}）")
    return pipe

# ===================== 5. 加载排序模型 + 主流程 =====================
def load_ranking_model():
    """加载排序模型"""
    model_file = Path(RANK_MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"排序模型缺失：{RANK_MODEL_PATH}")

    temp_dir = Path("C:/temp")
    temp_dir.mkdir(exist_ok=True)
    temp_model = temp_dir / "ltr_tutor_model.mod"

    import shutil
    if not temp_model.exists() or temp_model.stat().st_mtime < model_file.stat().st_mtime:
        shutil.copy(model_file.resolve(), temp_model)

    return lgb.Booster(model_file=str(temp_model))


def compare_ai_responses():
    """主函数"""
    llm_pipe = load_qwen_model()
    rank_model = load_ranking_model()

    # 获取AI导师回复（优先使用预定义列表）
    if AI_TUTOR_REPLIES:
        replies = AI_TUTOR_REPLIES
        print("\n使用预定义的AI导师回复...")
    else:
        # 默认使用随机示例
        replies = get_random_replies(4)
        print("\n使用随机示例进行评估...")

    # 显示待评估的回复
    print("\n" + "="*60)
    print("  [待评估的AI导师回复]")
    print("="*60)
    for i, reply in enumerate(replies, 1):
        preview = reply['回复'][:60] + "..." if len(reply['回复']) > 60 else reply['回复']
        print(f"\n【{i}】{reply['AI名称']}")
        print(f"    {preview}")

    # AI评委逐个评价
    print("\n" + "="*60)
    print("  [AI评委开始评价...]")
    print("="*60)

    results = []
    for reply in replies:
        print(f"\n正在评价【{reply['AI名称']}】...")
        try:
            # AI评委分析
            judge_result = ai_judge_review(reply, llm_pipe)

            # 提取特征并计算得分
            features = [judge_result[col] for col in FEATURE_COLS]
            score = round(rank_model.predict([features])[0], 2)
            level = "优质" if score > 0 else "中等" if score >= -2 else "劣质"

            results.append({
                "AI名称": reply["AI名称"],
                "回复文本": reply["回复"],
                "AI评委优点": judge_result.get("优点", ""),
                "AI评委缺点": judge_result.get("缺点", ""),
                "AI评委建议": judge_result.get("改进建议", ""),
                **{col: judge_result.get(col, 1) for col in FEATURE_COLS},
                "综合得分": score,
                "质量等级": level,
                "排名": 0
            })

            print(f"   完成！得分：{score}，等级：{level}")

        except Exception as e:
            print(f"   评价失败：{str(e)}")
            continue

    if not results:
        raise ValueError("无有效评价结果")

    # 排序
    results = sorted(results, key=lambda x: x["综合得分"], reverse=True)
    for i, res in enumerate(results):
        res["排名"] = i + 1

    # 保存Excel
    df = pd.DataFrame(results)
    output_excel = os.path.join(DATA_FOLDER, "AI评委评估结果.xlsx")
    df.to_excel(output_excel, index=False, engine="openpyxl")

    # 生成对比图
    plt.figure(figsize=(12, 6))
    colors = ["#2E8B57", "#6B8E23", "#DAA520", "#CD853F"][:len(results)]
    bars = plt.bar(df["AI名称"], df["综合得分"], color=colors)
    plt.title("AI导师教学质量评估（Qwen作为AI评委）", fontsize=14, fontweight="bold")
    plt.xlabel("AI导师")
    plt.ylabel("综合得分")
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for bar, score, rank in zip(bars, df["综合得分"], df["排名"]):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.1 if score > 0 else -0.3),
            f"{score}\n(#{rank})",
            ha="center", va="bottom" if score > 0 else "top",
            fontsize=10
        )

    output_png = os.path.join(DATA_FOLDER, "AI评委对比图.png")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")

    # 打印总结
    print("\n" + "="*60)
    print("  [评估完成！]")
    print("="*60)
    print("\n排名结果：")
    for res in results:
        print(f"  #{res['排名']} {res['AI名称']}: {res['综合得分']}分 ({res['质量等级']})")
        print(f"      优点: {res['AI评委优点'][:40]}...")

    print(f"\n结果文件：")
    print(f"  Excel: {output_excel}")
    print(f"  图表: {output_png}")

# ===================== 运行入口 =====================
if __name__ == "__main__":
    try:
        compare_ai_responses()
    except Exception as e:
        print(f"\n运行失败：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按回车键退出...")
