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
TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, "mrbench_train_data.xlsx")
MODEL_SAVE_PATH = os.path.join(DATA_FOLDER, "model", "regression_model.txt")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 论文8维度特征
FEATURE_COLS = [
    "错误识别", "错误定位", "不泄露答案",
    "提供引导", "可行性", "连贯性",
    "导师语气", "拟人化"
]

# ===================== 1. 加载训练数据 =====================
print("[*] 正在加载训练数据...")
if not os.path.exists(TRAIN_DATA_PATH):
    # 如果没有预处理数据，尝试从JSON提取
    print("[*] 未找到预处理数据，尝试从JSON提取...")
    try:
        from preprocess_dataset import preprocess_mrbench_dataset
        preprocess_mrbench_dataset()
    except Exception as e:
        print(f"[错误] 无法自动预处理：{e}")
        print("[提示] 请先运行 python preprocess_dataset.py")
        raise FileNotFoundError("找不到训练数据！")

df = pd.read_excel(TRAIN_DATA_PATH)
df["expert_label"] = df[FEATURE_COLS].sum(axis=1)
print(f"[OK] 加载训练数据成功！共{len(df)}条样本")

# ===================== 2. 拆分训练/测试集 =====================
all_query_ids = df["query_id"].unique()
np.random.seed(42)
train_query_ids = np.random.choice(all_query_ids, size=int(len(all_query_ids)*0.8), replace=False)
test_query_ids = np.setdiff1d(all_query_ids, train_query_ids)

train_df = df[df["query_id"].isin(train_query_ids)]
test_df = df[df["query_id"].isin(test_query_ids)]

X_train = train_df[FEATURE_COLS].values
y_train = train_df["expert_label"].values
X_test = test_df[FEATURE_COLS].values
y_test = test_df["expert_label"].values

print(f"[OK] 数据集拆分完成！")
print(f"[*] 训练集：{len(train_df)}条 | 测试集：{len(test_df)}条")

# ===================== 3. 训练回归模型（直接预测专家分数）=====================
print("\n[*] 开始训练回归模型...")

# 创建数据集
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# 回归训练参数
train_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',           # 回归任务：直接预测分数
    'metric': ['mae', 'rmse'],           # 使用MAE和RMSE评估
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

model = lgb.train(
    train_params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_eval],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50)
    ]
)

# ===================== 4. 详细评估 =====================
print("\n" + "="*80)
print("[*] 模型评估结果")
print("="*80)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 基础指标
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 相关性
pearson_corr, _ = stats.pearsonr(y_test, y_pred)
spearman_corr, _ = stats.spearmanr(y_test, y_pred)

# 精确度指标（预测值四舍五入后与真实值一致的比例）
y_pred_rounded = np.round(y_pred).astype(int)
y_pred_rounded = np.clip(y_pred_rounded, 0, 16)  # 限制在0-16范围
exact_match = np.mean(y_pred_rounded == y_test) * 100

# 允许±1分误差的准确率
within_1 = np.mean(np.abs(y_pred_rounded - y_test) <= 1) * 100
within_2 = np.mean(np.abs(y_pred_rounded - y_test) <= 2) * 100

print(f"\n【误差分析】")
print(f"  平均绝对误差(MAE)：{mae:.3f} 分")
print(f"  均方根误差(RMSE)：{rmse:.3f} 分")

print(f"\n【相关性分析】")
print(f"  皮尔逊相关系数：{pearson_corr:.4f}")
print(f"  斯皮尔曼相关系数：{spearman_corr:.4f}")
print(f"  R² 决定系数：{r2:.4f}")

print(f"\n【精确度分析】（四舍五入到整数后）")
print(f"  精确匹配率（±0分）：{exact_match:.1f}%")
print(f"  允许±1分误差：{within_1:.1f}%")
print(f"  允许±2分误差：{within_2:.1f}%")

print(f"\n【预测示例对比】")
# 取几个典型例子
indices = [0, len(test_df)//2, len(test_df)-1]
for idx in indices:
    true_val = y_test[idx]
    pred_val = y_pred[idx]
    diff = int(round(pred_val)) - true_val
    status = "✓" if abs(diff) <= 1 else "△" if abs(diff) <= 2 else "✗"
    print(f"  案例{idx+1}: 专家={true_val:2d}分 | AI预测={pred_val:5.2f}分 | 差异={diff:+d}分 [{status}]")

# ===================== 5. 保存模型 =====================
import shutil
from pathlib import Path
temp_dir = Path("C:/temp")
temp_dir.mkdir(exist_ok=True)
temp_model_path = temp_dir / "regression_model.txt"

model.save_model(str(temp_model_path))
if os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
    shutil.copy(temp_model_path, MODEL_SAVE_PATH)

print(f"\n[OK] 模型训练完成！")
print(f"[*] 最优迭代轮数：{model.best_iteration}")
print(f"[*] 模型已保存到：{temp_model_path}")

# ===================== 6. 特征重要性 =====================
print("\n[*] 特征重要性排名：")
importance = model.feature_importance()
feat_imp = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(feat_imp, 1):
    bar = "█" * int(imp / max(importance) * 20)
    print(f"  {i}. {feat}: {imp} {bar}")

print("\n" + "="*80)
print("[*] 训练完成！运行 inference.py 测试模型效果")
print("="*80)
