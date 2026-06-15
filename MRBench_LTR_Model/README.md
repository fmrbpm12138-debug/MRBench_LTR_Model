# AI导师回复质量评估工具

基于 Qwen2.5-1.5B-Instruct 模型 + LightGBM 排序模型的AI导师回复质量自动评估工具。

## 使用方法

### 1. 安装依赖

双击运行 `install_requirements.bat` 或在命令行执行：

```bash
pip install -r requirements.txt
```

### 2. 修改待评估的AI回复

用记事本打开 `src/llama3_ai_compare.py`，找到 `AI_RESPONSES` 部分，修改你想对比的AI回复：

```python
AI_RESPONSES = [
    {
        "AI名称": "AI导师A",
        "回复文本": "你这道题的错误在于没考虑边界条件..."
    },
    {
        "AI名称": "AI导师B",
        "回复文本": "这道题答案是5，你直接记下来就行..."
    },
    # 添加更多AI...
]
```

### 3. 运行评估

双击运行 `run_ai_compare.bat`，等待评估完成。

## 输出结果

评估结果会保存在 `data` 文件夹中：

- `多AI对比评估结果.xlsx` - 详细评分表格
- `多AI对比图.png` - 可视化对比图表

## 项目结构

```
MRBench_LTR_Model/
├── Qwen2.5-1.5B-Instruct/    # Qwen模型文件夹（约3GB）
├── data/                      # 数据和模型文件夹
│   └── model/
│       └── ltr_tutor_model.mod # 排序模型
├── src/
│   └── llama3_ai_compare.py   # 主程序
├── requirements.txt           # 依赖列表
├── run_ai_compare.bat         # 一键运行脚本
├── install_requirements.bat   # 安装依赖脚本
└── README.md                  # 使用说明（本文件）
```

## 系统要求

- Python 3.8+
- Windows 10/11
- 内存 8GB+（CPU运行）

## 注意事项

- 首次运行需要几分钟加载模型，请耐心等待
- 如果有NVIDIA显卡，会自动使用GPU加速
- 无显卡则使用CPU运行（速度较慢但效果相同）
- 模型文件较大（约3GB），首次运行前请确保网络正常
