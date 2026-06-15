# -*- coding: utf-8 -*-
"""
AI Tutor Quality Ranking Tool
AI导师教学质量排序工具
"""
import sys
import io
import os

# 强制UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ============ 路径设置 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(DATA_FOLDER, "model", "regression_model.txt")

# ============ 中文特征定义 ============
FEATURE_COLS = ["错误识别", "错误定位", "不泄露答案", "提供引导", 
                "可行性", "连贯性", "导师语气", "拟人化"]

FEATURE_DESCS = {
    "错误识别": "AI有没有发现学生的错误",
    "错误定位": "AI有没有指出错误位置",
    "不泄露答案": "AI有没有直接给答案",
    "提供引导": "AI有没有给提示让学生思考",
    "可行性": "AI的建议是否实际可行",
    "连贯性": "AI回复是否通顺有条理",
    "导师语气": "AI语气是否友好鼓励",
    "拟人化": "AI说话是否像真人"
}

def load_model():
    """加载模型"""
    temp_model = Path("C:/temp/regression_model.txt")
    model_file = str(temp_model) if temp_model.exists() else MODEL_PATH
    
    if not os.path.exists(model_file):
        print("\n[错误] 找不到模型！")
        print("请先从主菜单选择 [3] 训练模型")
        return None
    
    return lgb.Booster(model_file=model_file)

def score_response(features, model):
    """根据8个维度评分"""
    input_df = pd.DataFrame([features], columns=FEATURE_COLS)
    return model.predict(input_df)[0]

def get_grade(score):
    """根据分数返回等级"""
    if score >= 13:
        return "S", "优秀"
    elif score >= 10:
        return "A", "良好"
    elif score >= 7:
        return "B", "一般"
    elif score >= 4:
        return "C", "较差"
    else:
        return "D", "很差"

def print_result(排名, 名称, 特征, 分数, 等级字符, 等级文字):
    """打印单个AI的评分结果"""
    print(f"\n  第{排名}名  {名称}")
    print(f"  分数: {分数:.1f}/16  等级: [{等级字符}]{等级文字}")
    print(f"  " + "-"*50)
    
    for i, (特征名, 值) in enumerate(zip(FEATURE_COLS, 特征)):
        描述 = FEATURE_DESCS[特征名]
        进度条 = "█" * 值 + "░" * (2 - 值)
        print(f"    {i+1}. {特征名} ({描述}): [{值}/2] {进度条}")

def interactive_ranking(model):
    """交互式排名模式"""
    print("\n" + "="*55)
    print("          AI导师教学质量排序")
    print("="*55)
    
    try:
        n = int(input("\n要比较几个AI？(2-10): "))
        if not 2 <= n <= 10:
            print("请输入2-10之间的数字")
            return
    except:
        print("输入无效")
        return
    
    ais = []
    for i in range(n):
        print(f"\n--- AI {i+1}/{n} ---")
        
        名称 = input("  AI名称 (例如: ChatGPT, Claude): ").strip()
        if not 名称:
            名称 = f"AI-{i+1}"
        
        print(f"\n  每个维度打分 (0=否, 1=部分, 2=是):\n")
        
        特征 = []
        for 特征名 in FEATURE_COLS:
            描述 = FEATURE_DESCS[特征名]
            while True:
                try:
                    值 = int(input(f"  {特征名} - {描述}: "))
                    if 值 in [0, 1, 2]:
                        特征.append(值)
                        break
                    print("  请输入 0, 1 或 2")
                except:
                    print("  请输入 0, 1 或 2")
        
        分数 = score_response(特征, model)
        ais.append({"名称": 名称, "特征": 特征, "分数": 分数})
    
    ais.sort(key=lambda x: x["分数"], reverse=True)
    
    print(f"\n\n{'='*55}")
    print("              AI排名结果")
    print(f"{'='*55}")
    
    for i, ai in enumerate(ais):
        g, gtext = get_grade(ai["分数"])
        print_result(i+1, ai["名称"], ai["特征"], ai["分数"], g, gtext)
    
    print(f"\n{'='*55}")
    print("                 总结")
    print(f"{'='*55}")
    最好 = ais[0]
    最差 = ais[-1]
    平均 = np.mean([a["分数"] for a in ais])
    
    print(f"\n  最佳选择: {最好['名称']} ({最好['分数']:.1f} 分)")
    print(f"  需要改进: {最差['名称']} ({最差['分数']:.1f} 分)")
    print(f"  平均分:   {平均:.1f} 分")

def demo_ranking(model):
    """演示模式"""
    print("\n" + "="*55)
    print("         演示: 对比3个AI")
    print("="*55)
    
    demos = [
        {"名称": "Claude", "特征": [2, 2, 2, 2, 1, 2, 2, 1]},
        {"名称": "GPT-4", "特征": [2, 1, 2, 1, 2, 2, 1, 2]},
        {"名称": "通义", "特征": [1, 1, 1, 2, 1, 1, 2, 0]}
    ]
    
    for d in demos:
        d["分数"] = score_response(d["特征"], model)
    
    demos.sort(key=lambda x: x["分数"], reverse=True)
    
    for i, d in enumerate(demos):
        g, gtext = get_grade(d["分数"])
        print_result(i+1, d["名称"], d["特征"], d["分数"], g, gtext)

def main():
    """主函数"""
    print("\n" + "="*55)
    print("          AI导师教学质量排序")
    print("="*55)
    
    print("\n选择模式:")
    print("  [1] 手动输入 - 对比我自己的AI")
    print("  [2] 演示模式 - 查看示例对比")
    print("  [0] 退出")
    
    choice = input("\n选择 (0-2): ").strip()
    
    if choice == "1":
        interactive_ranking(model)
    elif choice == "2":
        demo_ranking(model)
    else:
        return

if __name__ == "__main__":
    print("[*] 正在加载模型...")
    model = load_model()
    if model is None:
        input("\n按回车键退出...")
        sys.exit(1)
    print("[OK] 模型加载成功！\n")
    main()
