# -*- coding: utf-8 -*-
"""
AI 导师教学质量评分系统 - 图形界面版本
双击此文件 或 从主菜单[exe]启动
"""
import sys
import os
import io

# Windows 控制台 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ============ 路径 & 特征定义 ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def get_exe_dir():
    """exe 打包后模型路径在 exe 同目录的 data/model 下"""
    if getattr(sys, 'frozen', False):          # PyInstaller 打包环境
        return os.path.dirname(sys.executable)
    return PROJECT_ROOT

DATA_FOLDER = os.path.join(get_exe_dir(), "data")
MODEL_PATH = os.path.join(DATA_FOLDER, "model", "regression_model.txt")

# 优先：C:/temp 的副本（开发时常用）
# 其次：exe 同目录下的 data/model（打包后）
TEMP_MODEL = Path("C:/temp/regression_model.txt")
EXE_MODEL  = Path(get_exe_dir()) / "data" / "model" / "regression_model.txt"

FEATURE_COLS = [
    "错误识别", "错误定位", "不泄露答案", "提供引导",
    "可行性", "连贯性", "导师语气", "拟人化"
]

FEATURE_DESCS = {
    "错误识别": "AI是否发现学生的错误",
    "错误定位": "AI是否指出错误的具体位置",
    "不泄露答案": "AI是否避免直接给出答案",
    "提供引导": "AI是否用提示引导学生思考",
    "可行性": "AI的建议是否实际可行",
    "连贯性": "AI回复是否通顺有逻辑",
    "导师语气": "AI语气是否友好鼓励",
    "拟人化": "AI说话是否像真人自然流畅"
}

GRADE_INFO = {
    (16, 21): ("S", "优秀", "#00C851"),
    (13, 16):  ("A", "良好", "#11BB11"),
    (10, 13):  ("B", "一般", "#FFBB00"),
    (7, 10):   ("C", "较差", "#FF7700"),
    (0, 7):    ("D", "很差", "#FF3333"),
}

FEEDBACK_SUGGESTIONS = {
    "错误识别": "建议明确指出学生哪一步做错了",
    "错误定位": "可以标注错误发生在第几行或哪个环节",
    "不泄露答案": "多用'你再想想''哪里有问题'等提示语",
    "提供引导": "增加追问或小提示，引导学生自主发现",
    "可行性": "建议更具体、更容易操作的步骤",
    "连贯性": "组织好逻辑顺序，让回复更清晰",
    "导师语气": "语气可以更温和、更有耐心",
    "拟人化": "让语言更口语化，减少机器感"
}

# ============ 模型加载 ============
def load_model():
    # 优先级：C:/temp > exe所在目录 > 项目data
    for candidate in [TEMP_MODEL, EXE_MODEL, Path(MODEL_PATH)]:
        if candidate.exists():
            return lgb.Booster(model_file=str(candidate))
    return None

model = load_model()
MODEL_OK = model is not None

# ============ GUI ============
class ScoreGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI 导师教学质量评分系统")
        self.geometry("920x720")
        self.configure(bg="#F0F2F5")
        self.resizable(False, False)

        # 设置中文字体
        try:
            self.font_main = ("Microsoft YaHei UI", 10)
            self.font_title = ("Microsoft YaHei UI", 13, "bold")
            self.font_small = ("Microsoft YaHei UI", 9)
            self.font_result = ("Consolas", 14, "bold")
        except Exception:
            self.font_main = ("Arial", 10)
            self.font_title = ("Arial", 13, "bold")
            self.font_small = ("Arial", 9)
            self.font_result = ("Courier New", 14, "bold")

        self._build_ui()

        if not MODEL_OK:
            self.after(500, lambda: messagebox.showwarning(
                "模型未找到",
                "未找到 regression_model.txt，请先运行训练。\n"
                "当前将使用规则打分（仅供参考）。"
            ))

    # ── 界面构建 ────────────────────────────────────────────────
    def _build_ui(self):
        # 顶部标题栏
        top = tk.Frame(self, bg="#2C3E50", height=50)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(
            top, text="🎓  AI 导师教学质量评分系统",
            font=self.font_title, fg="white", bg="#2C3E50"
        ).pack(pady=12)

        # 主容器
        main = tk.Frame(self, bg="#F0F2F5")
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # ── 左侧：AI 回复输入区 ──
        left = tk.LabelFrame(
            main, text="📝  AI 导师回复内容", font=self.font_main,
            bg="#F0F2F5", fg="#2C3E50", padx=10, pady=8
        )
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.text_area = scrolledtext.ScrolledText(
            left, height=22, font=self.font_main, wrap="word",
            bg="white", fg="#222", insertbackground="#2C3E50",
            relief="solid", bd=1
        )
        self.text_area.pack(fill="both", expand=True, pady=(0, 6))

        tk.Label(
            left,
            text="粘贴 AI 导师的回复内容，然后为每个维度打分，点击「开始评分」",
            font=self.font_small, fg="#666", bg="#F0F2F5", anchor="w"
        ).pack(fill="x")

        # 状态栏（模型是否加载）
        if MODEL_OK:
            status_txt = "✅ 评分模型已就绪"
            status_color = "#00C851"
        else:
            status_txt = "⚠️  规则打分模式（请先训练模型）"
            status_color = "#FF7700"
        self.status_label = tk.Label(
            left, text=status_txt, font=self.font_small,
            fg=status_color, bg="#F0F2F5", anchor="w"
        )
        self.status_label.pack(fill="x", pady=(4, 0))

        # ── 右侧：8 维打分区 ──
        right = tk.LabelFrame(
            main, text="📊  8 维度评分", font=self.font_main,
            bg="#F0F2F5", fg="#2C3E50", padx=10, pady=8
        )
        right.pack(side="right", fill="both", padx=(8, 0))

        self.dim_vars = {}
        self.dim_labels = {}
        self.dim_bars = {}

        for i, feat in enumerate(FEATURE_COLS):
            row = tk.Frame(right, bg="#F0F2F5")
            row.pack(fill="x", pady=3)

            # 维度名称
            lbl = tk.Label(
                row, text=f"{i+1}. {feat}", font=self.font_main,
                fg="#2C3E50", bg="#F0F2F5", width=10, anchor="w"
            )
            lbl.pack(side="left")

            # 刻度说明
            scale_lbl_frame = tk.Frame(row, bg="#F0F2F5")
            scale_lbl_frame.pack(side="left", padx=(2, 4))

            tk.Label(scale_lbl_frame, text="0", font=("Arial", 8),
                     fg="#999", bg="#F0F2F5").pack(side="left")
            tk.Label(scale_lbl_frame, text="1", font=("Arial", 8),
                     fg="#999", bg="#F0F2F5").pack(side="right")

            # 滑块
            var = tk.IntVar(value=1)
            self.dim_vars[feat] = var

            def make_scale(f, v, idx):
                s = ttk.Scale(row, from_=0, to=2, orient="horizontal",
                              length=120, variable=v,
                              command=lambda _, ff=f, vv=v, ii=idx: self._on_slide(ff, vv, ii))
                return s

            scale = make_scale(feat, var, i)
            scale.pack(side="left")

            # 当前值 + 进度条文字
            val_frame = tk.Frame(row, bg="#F0F2F5", width=36)
            val_frame.pack(side="left", padx=(4, 0))
            val_frame.pack_propagate(False)

            val_lbl = tk.Label(
                val_frame, text="1", font=self.font_result,
                fg="#2C3E50", bg="#F0F2F5", anchor="center"
            )
            val_lbl.pack()
            self.dim_labels[feat] = val_lbl

        # 说明按钮
        tk.Button(
            right, text="📖 各维度说明", font=self.font_small,
            command=self._show_dim_guide, relief="groove", cursor="hand2"
        ).pack(pady=(6, 0), fill="x")

        # ── 评分按钮（跨两列） ──
        btn_frame = tk.Frame(main, bg="#F0F2F5")
        btn_frame.pack(fill="x", pady=(10, 0))

        score_btn = tk.Button(
            btn_frame, text="⭐  开始评分", font=("Microsoft YaHei UI", 12, "bold"),
            bg="#2C3E50", fg="white", relief="flat", cursor="hand2",
            command=self._do_score, height=2
        )
        score_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        clear_btn = tk.Button(
            btn_frame, text="🧹  清空", font=("Microsoft YaHei UI", 11),
            bg="#DDD", fg="#333", relief="flat", cursor="hand2",
            command=self._clear_all
        )
        clear_btn.pack(side="right", fill="x", padx=(6, 0), ipady=4)

        # ── 结果展示区（底部） ──
        result_frame = tk.LabelFrame(
            self, text="📈  评分结果", font=self.font_main,
            bg="#F0F2F5", fg="#2C3E50", padx=12, pady=8
        )
        result_frame.pack(fill="x", padx=16, pady=(0, 12))

        # 总分区
        self.score_lbl = tk.Label(
            result_frame, text="--", font=("Microsoft YaHei UI", 36, "bold"),
            fg="#2C3E50", bg="#F0F2F5"
        )
        self.score_lbl.pack(side="left", padx=(0, 12))

        self.grade_lbl = tk.Label(
            result_frame, text="请先打分",
            font=("Microsoft YaHei UI", 16), bg="#F0F2F5"
        )
        self.grade_lbl.pack(side="left", padx=(0, 16))

        # 各维度小条
        bar_frame = tk.Frame(result_frame, bg="#F0F2F5")
        bar_frame.pack(side="left", fill="y")
        self.bar_widgets = {}
        for i, feat in enumerate(FEATURE_COLS):
            f = tk.Frame(bar_frame, width=22, height=18, bg="#E8E8E8")
            f.pack_propagate(False)
            inner = tk.Frame(f, bg="#3498DB", height=18)
            inner.pack(fill="x", side="bottom")
            f.pack(side="left", padx=1)
            self.bar_widgets[feat] = inner

        # 建议区
        self.suggestion_lbl = tk.Label(
            result_frame, text="", font=self.font_small,
            fg="#555", bg="#F0F2F5", anchor="e", wraplength=340
        )
        self.suggestion_lbl.pack(side="right", fill="both", expand=True, padx=(16, 0))

    # ── 交互回调 ────────────────────────────────────────────────
    def _on_slide(self, feat, var, idx):
        val = round(var.get())
        var.set(val)
        self.dim_labels[feat].config(text=str(val))

    def _show_dim_guide(self):
        lines = [f"{i+1}. {k} — {v}" for i, (k, v) in enumerate(FEATURE_DESCS.items())]
        msg = "\n".join(lines) + "\n\n评分：0=没做到  1=部分做到  2=完全做到"
        messagebox.showinfo("8 维度说明", msg)

    def _clear_all(self):
        self.text_area.delete("1.0", "end")
        for feat in FEATURE_COLS:
            self.dim_vars[feat].set(1)
            self.dim_labels[feat].config(text="1")
        self.score_lbl.config(text="--", fg="#2C3E50")
        self.grade_lbl.config(text="请先打分", fg="#2C3E50")
        self.suggestion_lbl.config(text="")
        for feat in FEATURE_COLS:
            self.bar_widgets[feat].master.config(bg="#E8E8E8")
            self.bar_widgets[feat].config(bg="#3498DB", height=18)
            self.bar_widgets[feat].pack_configure(fill="x", side="bottom")

    def _do_score(self):
        # 收集 8 维特征
        features = [self.dim_vars[f].get() for f in FEATURE_COLS]

        # 用模型预测（如果有）
        if model is not None:
            df = pd.DataFrame([features], columns=FEATURE_COLS)
            score = float(model.predict(df)[0])
            total = round(score, 1)
            raw_total = int(round(score))
        else:
            # 规则打分：直接求和
            total = raw_total = sum(features)
            score = float(total)

        # 等级
        for (lo, hi), (gchar, gtext, gcolor) in GRADE_INFO.items():
            if lo <= raw_total < hi:
                self.score_lbl.config(text=f"{total:.1f}", fg=gcolor)
                self.grade_lbl.config(text=f"[ {gchar} ] {gtext}", fg=gcolor)
                break

        # 更新各维度进度条
        colors = ["#E74C3C", "#F39C12", "#2ECC71"]  # 0红 1橙 2绿
        for feat, val in zip(FEATURE_COLS, features):
            bar = self.bar_widgets[feat]
            bar.master.config(bg=colors[val])
            bar.config(bg=colors[val], height=12 + val * 5)
            bar.pack_configure(fill="x", side="bottom")

        # 建议
        weak = [f for f, v in zip(FEATURE_COLS, features) if v < 2]
        if weak:
            suggestions = [FEEDBACK_SUGGESTIONS[f] for f in weak[:3]]
            self.suggestion_lbl.config(text="💡 " + "\n💡 ".join(suggestions))
        else:
            self.suggestion_lbl.config(text="🌟 各维度表现优秀，继续保持！")

        # 更新窗口标题显示分数
        self.title(f"AI 导师教学质量评分系统  —  当前得分：{total}/16")


if __name__ == "__main__":
    app = ScoreGUI()
    app.mainloop()
