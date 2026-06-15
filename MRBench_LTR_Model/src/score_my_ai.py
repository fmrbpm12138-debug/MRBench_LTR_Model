# -*- coding: utf-8 -*-
"""
粘贴 AI 导师回复，自动评分工具
使用方法：粘贴 AI 的回复文本，系统引导你在 8 个维度打分并给出总分
"""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import lightgbm as lgb
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(DATA_FOLDER, "model", "regression_model.txt")

FEATURE_COLS = ["错误识别", "错误定位", "不泄露答案", "提供引导",
                "可行性", "连贯性", "导师语气", "拟人化"]

FEATURE_HINTS = {
    "错误识别": "AI 有没有发现学生错在哪里？",
    "错误定位": "AI 有没有指出错误在第几行/哪一步？",
    "不泄露答案": "AI 有没有直接告诉学生正确答案？",
    "提供引导": "AI 有没有用提示、提问来引导学生思考？",
    "可行性": "AI 的建议是否实际、可操作？",
    "连贯性": "AI 的回复是否通顺、逻辑清晰？",
    "导师语气": "AI 语气是否友好、耐心、鼓励？",
    "拟人化": "AI 说话是否自然流畅，像真人在对话？"
}

GRADE_MAP = {4: ("S", "优秀"), 3: ("A", "良好"), 2: ("B", "一般"), 1: ("C", "较差"), 0: ("D", "很差")}


def get_grade(score: float) -> tuple:
    s = int(score)
    if s >= 13: return GRADE_MAP[4]
    if s >= 10: return GRADE_MAP[3]
    if s >= 7:  return GRADE_MAP[2]
    if s >= 4:  return GRADE_MAP[1]
    return GRADE_MAP[0]


def load_model():
    temp = Path("C:/temp/regression_model.txt")
    model_file = str(temp) if temp.exists() else MODEL_PATH
    if not os.path.exists(model_file):
        print("\n[错误] 找不到模型！请先从主菜单选 [3] 训练模型。")
        return None
    return lgb.Booster(model_file=model_file)


def score_and_show(features: list) -> float:
    df = pd.DataFrame([features], columns=FEATURE_COLS)
    score = model.predict(df)[0]
    grade_char, grade_text = get_grade(score)

    print("\n" + "=" * 60)
    print(f"  总分：{score:.1f} / 16   等级：[ {grade_char} ] {grade_text}")
    print("=" * 60)

    total = sum(features)
    for i, (name, val) in enumerate(zip(FEATURE_COLS, features)):
        bar = "█" * val + "░" * (2 - val)
        hint = FEATURE_HINTS[name]
        print(f"  {i+1}. {name}  [{val}/2] {bar}   ← {hint}")

    print("=" * 60)

    # 质量建议
    weak = [FEATURE_COLS[i] for i, v in enumerate(features) if v < 2]
    if weak:
        print(f"\n  建议提升：{', '.join(weak)}")
    else:
        print("\n  各维度表现均衡，继续保持！")
    return score


def main():
    print("\n" + "=" * 60)
    print("  AI 导师教学质量评分工具")
    print("  输入 AI 回复 → 在 8 个维度打分 → 得出综合评分")
    print("=" * 60)

    print("\n请粘贴 AI 导师的回复内容（粘贴完成后按 Ctrl+Z 回车结束）：")
    print("-" * 60)
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    response_text = "\n".join(lines).strip()

    if not response_text:
        print("\n[提示] 未输入内容，退出。")
        return

    # 显示字数
    word_count = len(response_text)
    print(f"\n[OK] 已收到回复（{word_count} 字），现在开始在 8 个维度打分。\n")
    print("每个维度：0 = 没做到，1 = 部分做到，2 = 完全做到\n")

    features = []
    for name in FEATURE_COLS:
        hint = FEATURE_HINTS[name]
        while True:
            try:
                val = int(input(f"  {name}（{hint}）："))
                if val in (0, 1, 2):
                    features.append(val)
                    break
                print("  → 请输入 0、1 或 2")
            except ValueError:
                print("  → 请输入数字 0、1 或 2")

    score_and_show(features)

    # 存档选项
    print("\n是否保存此次评分记录？")
    save = input("  输入文件名保存（直接回车跳过）：").strip()
    if save:
        row = {"AI回复": response_text[:200]}
        row.update(dict(zip(FEATURE_COLS, features)))
        out_path = os.path.join(DATA_FOLDER, f"score_record_{save}.csv")
        existing = pd.DataFrame() if not os.path.exists(out_path) else pd.read_csv(out_path)
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"  已保存到 {out_path}")


if __name__ == "__main__":
    print("[*] 正在加载模型...")
    model = load_model()
    if model is None:
        input("\n按回车退出...")
        sys.exit(1)
    print("[OK] 模型加载成功！\n")
    main()
    input("\n按回车返回主菜单...")
