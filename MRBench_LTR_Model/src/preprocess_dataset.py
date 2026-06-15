import os
import json
import pandas as pd

# ===================== 【核心修正】绝对鲁棒的路径逻辑，基于脚本文件位置 =====================
# 获取当前脚本所在的src文件夹路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（src的上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# 数据文件夹的绝对路径
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

# ===================== 【核心修正】和JSON真实结构完全匹配的维度映射 =====================
DIMENSION_MAP = {
    "错误识别": "Mistake_Identification",
    "错误定位": "Mistake_Location",
    "不泄露答案": "Revealing_of_the_Answer",
    "提供引导": "Providing_Guidance",
    "可行性": "Actionability",
    "连贯性": "Coherence",
    "导师语气": "Tutor_Tone",
    "拟人化": "humanlikeness",
}

# ===================== 【核心修正】匹配JSON真实标注值的打分函数 =====================
def annotation_to_score(dim_name, annotation):
    """
    完全匹配JSON里的Yes/No/To some extent标注规则
    分数越高，越符合专家偏好的教学策略
    """
    if not isinstance(annotation, str):
        return 0
    # 标准化标注文本，去空格、转小写
    annotation_clean = annotation.strip().lower()

    # 错误识别、错误定位 维度
    if dim_name in ["错误识别", "错误定位"]:
        if annotation_clean == "yes":
            return 2
        elif "to some extent" in annotation_clean:
            return 1
        else:
            return 0
    # 不泄露答案 维度（反向规则：不泄露才是高分）
    elif dim_name == "不泄露答案":
        if annotation_clean == "no":
            return 2
        elif "to some extent" in annotation_clean:
            return 1
        else:
            return 0
    # 提供引导、可行性、连贯性 维度
    elif dim_name in ["提供引导", "可行性", "连贯性"]:
        if annotation_clean == "yes":
            return 2
        elif "to some extent" in annotation_clean:
            return 1
        else:
            return 0
    # 导师语气 维度（鼓励式语气最优）
    elif dim_name == "导师语气":
        if "encouraging" in annotation_clean:
            return 2
        elif "neutral" in annotation_clean:
            return 1
        else:
            return 0
    # 拟人化 维度
    elif dim_name == "拟人化":
        return 2 if annotation_clean == "yes" else 0
    # 兜底
    else:
        return 0

# ===================== 核心预处理函数 =====================
def preprocess_mrbench_dataset(dataset_json_path=None, output_excel_path=None):
    try:
        # 自动补全路径
        if dataset_json_path is None:
            dataset_json_path = os.path.join(DATA_FOLDER, "MRBench_V1.json")
        if output_excel_path is None:
            output_excel_path = os.path.join(DATA_FOLDER, "mrbench_train_data.xlsx")
        
        print(f"[*] 正在读取数据文件：{dataset_json_path}")
        print(f"[*] 输出文件路径：{output_excel_path}")

        # 校验文件存在
        if not os.path.exists(dataset_json_path):
            raise FileNotFoundError(f"数据文件不存在！请确认文件在：{dataset_json_path}")
        
        # 兼容所有编码读取
        try:
            with open(dataset_json_path, "r", encoding="utf-8-sig") as f:
                raw_dataset = json.load(f)
        except UnicodeDecodeError:
            with open(dataset_json_path, "r", encoding="gbk", errors="ignore") as f:
                raw_dataset = json.load(f)

        # 解析数据，带完整容错
        train_data = []
        skip_count = 0
        total_conversation = len(raw_dataset)

        for conversation in raw_dataset:
            # 缺核心字段直接跳过
            if "conversation_id" not in conversation or "anno_llm_responses" not in conversation:
                skip_count +=1
                continue
            
            # 同一个对话的conversation_id作为排序学习的query_id
            query_id = conversation["conversation_id"]
            conversation_history = conversation.get("conversation_history", "")
            data_source = conversation.get("Data", "unknown")

            # 遍历所有LLM/专家的回复+标注
            for tutor_name, response_data in conversation["anno_llm_responses"].items():
                # 缺回复/标注直接跳过
                if "response" not in response_data or "annotation" not in response_data:
                    skip_count +=1
                    continue
                
                # 构建单条样本
                row = {
                    "query_id": query_id,
                    "对话历史": conversation_history,
                    "导师回复文本": response_data["response"],
                    "数据来源": data_source,
                    "导师类型": tutor_name
                }

                # 【核心】正确转换8个维度的标注为数值分数
                annotation_data = response_data["annotation"]
                for chinese_dim, json_key in DIMENSION_MAP.items():
                    dim_annotation = annotation_data.get(json_key, "no")
                    row[chinese_dim] = annotation_to_score(chinese_dim, dim_annotation)
                
                train_data.append(row)

        # 校验有效数据
        if len(train_data) == 0:
            raise ValueError("没有解析到有效数据！请检查JSON格式是否匹配")

        # 保存为训练用Excel
        df = pd.DataFrame(train_data)
        df.to_excel(output_excel_path, index=False)

        # 打印结果，确认有分数差异
        df["专家总分"] = df[DIMENSION_MAP.keys()].sum(axis=1)
        print(f"\n[OK] 数据集预处理完成！")
        print(f"[*] 原始总对话数：{total_conversation} 个")
        print(f"[OK] 有效训练样本数：{len(df)} 条回复")
        print(f"[!] 跳过无效样本数：{skip_count} 条")
        print(f"[*] 独立对话数（query_id）：{df['query_id'].nunique()} 个")
        print(f"[*] 专家总分最小值：{df['专家总分'].min()}")
        print(f"[*] 专家总分最大值：{df['专家总分'].max()}")
        print(f"[*] 专家总分平均值：{df['专家总分'].mean():.2f}")
        print(f"[*] 训练数据已保存到：{output_excel_path}")
        return df

    except Exception as e:
        print(f"\n[X] 预处理出错！错误原因：{str(e)}")
        return None

# ===================== 一键运行 =====================
if __name__ == "__main__":
    preprocess_mrbench_dataset()