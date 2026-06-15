# -*- coding: utf-8 -*-
import sys
import os

# 在导入任何其他模块之前先设置编码！
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import traceback
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

# 禁用transformers警告
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 现在导入主模块
try:
    from llama3_ai_compare import compare_ai_responses, get_random_replies
    # 默认使用随机示例，自动选择4个AI导师回复
    import llama3_ai_compare
    llama3_ai_compare.AI_TUTOR_REPLIES = get_random_replies(4)
    compare_ai_responses()
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    input("\n按回车键退出...")
