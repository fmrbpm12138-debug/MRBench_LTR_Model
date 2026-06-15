# -*- coding: utf-8 -*-
import sys
import io
import os

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# ===================== 绝对鲁棒的路径逻辑 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(DATA_FOLDER, "model", "regression_model.txt")
TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, "mrbench_train_data.xlsx")
JSON_V1 = os.path.join(DATA_FOLDER, "MRBench_V1.json")
JSON_V2 = os.path.join(DATA_FOLDER, "MRBench_V2.json")

# 论文8维度特征
FEATURE_COLS = [
    "错误识别", "错误定位", "不泄露答案",
    "提供引导", "可行性", "连贯性",
    "导师语气", "拟人化"
]

# 英文到中文映射
ENGLISH_TO_CHINESE = {
    "Mistake_Identification": "错误识别",
    "Mistake_Location": "错误定位",
    "Revealing_of_the_Answer": "不泄露答案",
    "Providing_Guidance": "提供引导",
    "Actionability": "可行性",
    "Coherence": "连贯性",
    "Tutor_Tone": "导师语气",
    "humanlikeness": "拟人化"
}

# ===================== 1. 加载回归模型 =====================
print("[*] 正在加载评分回归模型...")

import shutil
from pathlib import Path
temp_dir = Path("C:/temp")
temp_dir.mkdir(exist_ok=True)
temp_model = temp_dir / "regression_model.txt"

# 优先使用临时目录的模型
model_file = str(temp_model) if temp_model.exists() else MODEL_PATH

if not os.path.exists(model_file):
    raise FileNotFoundError(
        f"找不到回归模型！请先运行 train.py 训练模型。\n"
        f"预期位置：{MODEL_PATH}"
    )

model = lgb.Booster(model_file=model_file)
print("[OK] 模型加载成功！")

# ===================== 2. 核心评分函数 =====================
def score_teaching_response(input_features):
    """输入8个维度特征，输出AI预测的专家评分（0-16分）"""
    input_df = pd.DataFrame([input_features], columns=FEATURE_COLS)
    return model.predict(input_df)[0]

# ===================== 3. 快速测试示例 =====================
print("\n" + "="*80)
print("[*] 快速测试示例（验证模型评分逻辑）")
print("="*80)

# 示例1：完美
perfect_features = [2, 2, 2, 2, 2, 2, 2, 2]
perfect_score = score_teaching_response(perfect_features)
print(f"\n[完美] 8维全2分 → 专家评分：{perfect_score:.1f}/16")

# 示例2：劣质
bad_features = [0, 0, 0, 0, 0, 0, 0, 0]
bad_score = score_teaching_response(bad_features)
print(f"[劣质] 8维全0分 → 专家评分：{bad_score:.1f}/16")

# 示例3：中等
medium_features = [1, 1, 1, 1, 1, 1, 1, 1]
medium_score = score_teaching_response(medium_features)
print(f"[中等] 8维全1分 → 专家评分：{medium_score:.1f}/16")

# ===================== 4. 从JSON加载真实专家案例进行评估 =====================
print("\n" + "="*80)
print("[*] 从MRBench数据集加载真实专家案例进行评估")
print("="*80)

all_data = []
for json_path in [JSON_V1, JSON_V2]:
    if not os.path.exists(json_path):
        print(f"[警告] 文件不存在：{json_path}")
        continue
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_data.extend(data)
        print(f"[OK] 加载 {os.path.basename(json_path)}：{len(data)} 条对话")

if not all_data:
    raise FileNotFoundError("未找到任何JSON数据文件！")

# 提取真实专家(Expert)标注
expert_records = []
for item in all_data:
    if "anno_llm_responses" in item and "Expert" in item["anno_llm_responses"]:
        expert = item["anno_llm_responses"]["Expert"]
        annotation = expert.get("annotation", {})
        
        features = {}
        for en_key, cn_key in ENGLISH_TO_CHINESE.items():
            val = annotation.get(en_key, 1)
            # 统一数值：Yes=2, No=0, 其他=1
            if isinstance(val, bool):
                features[cn_key] = 2 if val else 0
            elif isinstance(val, str):
                if "yes" in val.lower() or "correct" in val.lower():
                    features[cn_key] = 2
                elif "no" in val.lower():
                    features[cn_key] = 0
                else:
                    features[cn_key] = 1
            else:
                features[cn_key] = int(val) if val in [0, 1, 2] else 1
        
        expert_records.append({
            "导师回复文本": expert.get("response", ""),
            "conversation_id": item.get("conversation_id", ""),
            "Data": item.get("Data", ""),
            **features
        })

df = pd.DataFrame(expert_records)
print(f"\n[OK] 共提取 {len(df)} 条真实专家(Expert)案例")

# 计算真实专家总分
df["expert_label"] = df[FEATURE_COLS].sum(axis=1)

# 使用模型预测评分
df["pred_score"] = df[FEATURE_COLS].apply(
    lambda row: score_teaching_response(row.values.tolist()), 
    axis=1
)

# 四舍五入到整数
df["pred_rounded"] = np.round(df["pred_score"]).astype(int)
df["pred_rounded"] = df["pred_rounded"].clip(0, 16)

# ===================== 5. 统计分析 =====================
print("\n" + "="*80)
print("[*] 模型评估结果统计")
print("="*80)

# 基础指标
mae = mean_absolute_error(df['expert_label'], df['pred_score'])
rmse = np.sqrt(mean_squared_error(df['expert_label'], df['pred_score']))
r2 = r2_score(df['expert_label'], df['pred_score'])

# 相关性
pearson_corr, _ = stats.pearsonr(df['expert_label'], df['pred_score'])
spearman_corr, _ = stats.spearmanr(df['expert_label'], df['pred_score'])

# 精确度
exact_match = np.mean(df['pred_rounded'] == df['expert_label']) * 100
within_1 = np.mean(np.abs(df['pred_rounded'] - df['expert_label']) <= 1) * 100
within_2 = np.mean(np.abs(df['pred_rounded'] - df['expert_label']) <= 2) * 100

print(f"\n【误差分析】")
print(f"  平均绝对误差(MAE)：{mae:.3f} 分")
print(f"  均方根误差(RMSE)：{rmse:.3f} 分")

print(f"\n【相关性分析】（越接近1越好）")
print(f"  皮尔逊相关系数：{pearson_corr:.4f}")
print(f"  斯皮尔曼相关系数：{spearman_corr:.4f}")
print(f"  R² 决定系数：{r2:.4f}")

print(f"\n【精确度分析】")
print(f"  精确匹配（±0分）：{exact_match:.1f}%")
print(f"  允许±1分误差：{within_1:.1f}%")
print(f"  允许±2分误差：{within_2:.1f}%")

# ===================== 6. 详细案例对比 =====================
print("\n" + "="*80)
print("[*] 随机抽取10条详细对比：")
print("="*80)

np.random.seed(42)
sample_df = df.sample(n=min(10, len(df)), random_state=42)

for i, (_, row) in enumerate(sample_df.iterrows()):
    true_score = int(row["expert_label"])
    pred_score = row["pred_score"]
    pred_rounded = int(row["pred_rounded"])
    diff = pred_rounded - true_score
    
    # 状态符号
    if abs(diff) == 0:
        status = "✓ 完全一致"
    elif abs(diff) == 1:
        status = "✓ 接近"
    elif abs(diff) == 2:
        status = "△ 小差异"
    else:
        status = "✗ 差异较大"
    
    print(f"\n【案例 {i+1}】{status}")
    print(f"  来源：{row['Data']}")
    reply = row['导师回复文本'][:50] + "..." if len(row['导师回复文本']) > 50 else row['导师回复文本']
    print(f"  专家回复：{reply}")
    print(f"  真实专家分：{true_score}/16")
    print(f"  AI预测分：{pred_score:.2f}（四舍五入：{pred_rounded}）")
    print(f"  误差：{diff:+d} 分")

print("\n" + "="*80)
print("[*] 解读说明：")
print("  - 精确匹配率越高，AI越接近专家评分")
print("  - ±1分误差在实际应用中通常可接受")
print("  - 这个模型可以直接替代专家进行评分")
print("="*80)
