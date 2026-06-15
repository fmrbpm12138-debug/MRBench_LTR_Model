# -*- coding: utf-8 -*-
"""
智谱AI自动评分模块
粘贴多个AI的回复 → 自动分析 → 8维打分 → 智能排序
"""
import sys
import io
import os

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ============ 评分提示词 ============
SCORING_PROMPT = """你是一个专业的AI导师教学质量评估专家。

请分析以下AI导师回复，根据8个维度进行打分（每个维度0-2分）：
- 0分 = 没做到
- 1分 = 部分做到
- 2分 = 完全做到

8个维度：
1. 错误识别：是否发现并指出学生的错误
2. 错误定位：是否准确指出错误发生的位置/步骤
3. 不泄露答案：是否避免直接给出完整答案，而是引导思考
4. 提供引导：是否用提示、追问等方式引导学生自主发现
5. 可行性：建议是否实际可行、操作性强
6. 连贯性：回复是否逻辑通顺、条理清晰
7. 导师语气：语气是否友好、耐心、鼓励性强
8. 拟人化：语言是否自然流畅，像真人对话

请以JSON格式返回评分结果，格式如下：
{{"scores": [错误识别分, 错误定位分, 不泄露答案分, 提供引导分, 可行性分, 连贯性分, 导师语气分, 拟人化分], "brief": "简要评价（20字内）"}}

【AI回复内容】
{response}

请直接返回JSON，不要其他内容："""

FEATURE_NAMES = ["错误识别", "错误定位", "不泄露答案", "提供引导",
                 "可行性", "连贯性", "导师语气", "拟人化"]


class AutoScorer:
    """自动评分器"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None

    def init_client(self):
        """初始化智谱API客户端"""
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=self.api_key)
            return True
        except ImportError as e:
            messagebox.showerror(
                "缺少依赖",
                "无法加载智谱SDK，常见原因是子依赖未装上（例如 sniffio）。\n\n"
                f"具体错误：{e}\n\n"
                "请关闭本窗口后，在命令行执行：\n"
                "pip install zhipuai sniffio\n\n"
                "或返回主菜单再选 [7] 自动安装。",
            )
            return False
        except Exception as e:
            messagebox.showerror("API密钥错误", f"检查API密钥：{e}")
            return False

    def score_single(self, response: str, progress_callback=None) -> dict:
        """对单条回复评分"""
        if not self.client:
            return None

        try:
            if progress_callback:
                progress_callback("正在分析回复...")

            completion = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{
                    "role": "user",
                    "content": SCORING_PROMPT.format(response=response)
                }],
                temperature=0.1,  # 低温度保证评分稳定
            )

            result_text = completion.choices[0].message.content.strip()

            # 尝试解析JSON
            # 去掉可能的markdown代码块
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                result_text = "\n".join(lines[1:-1])

            result = json.loads(result_text)

            scores = result.get("scores", [])
            if len(scores) != 8:
                raise ValueError(f"评分维度数量不对：{len(scores)}")

            return {
                "scores": [int(s) for s in scores],
                "brief": result.get("brief", ""),
                "total": sum(scores)
            }

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始返回: {result_text}")
            return None
        except Exception as e:
            print(f"评分出错: {e}")
            return None

    def score_all(self, responses: dict, progress_callback=None) -> dict:
        """
        对多个AI回复评分并排序

        Args:
            responses: dict, {"AI名称": "回复内容", ...}
            progress_callback: 进度回调函数

        Returns:
            dict, {"ranked": [(名称, 分数, 详情), ...], "all_scores": {...}}
        """
        results = {}

        for i, (name, content) in enumerate(responses.items()):
            if progress_callback:
                progress_callback(f"正在评分 {name} ({i+1}/{len(responses)})...")

            result = self.score_single(content)
            if result:
                results[name] = result
            else:
                results[name] = {
                    "scores": [0] * 8,
                    "brief": "评分失败",
                    "total": 0
                }

            # 避免API限流
            time.sleep(0.5)

        # 按总分排序
        ranked = sorted(results.items(), key=lambda x: x[1]["total"], reverse=True)

        return {
            "ranked": ranked,
            "all_scores": results
        }


def get_api_key_from_file():
    """从配置文件读取API密钥"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "api_key.txt")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def delete_api_key():
    """删除API密钥文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "data", "api_key.txt")
    if os.path.exists(config_path):
        try:
            os.remove(config_path)
        except OSError:
            pass


def save_api_key(key: str):
    """保存API密钥"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    config_path = os.path.join(data_dir, "api_key.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(key)


# ============ GUI 界面 ============
class AutoScoreGUI(tk.Tk):
    """自动评分GUI"""

    def __init__(self):
        super().__init__()
        self.title("AI导师自动评分系统 - 智谱AI版")
        self.geometry("900x700")
        self.configure(bg="#F0F2F5")
        self.api_key = None
        self.scorer = None
        self.ai_entries = {}  # 存储各个AI的输入框

        self._check_api_key()
        self._build_ui()

    def destroy(self):
        """窗口关闭时自动删除API密钥"""
        delete_api_key()
        super().destroy()

    def _check_api_key(self):
        """检查API密钥"""
        saved_key = get_api_key_from_file()
        if saved_key:
            self.api_key = saved_key
            self.scorer = AutoScorer(self.api_key)
            if not self.scorer.init_client():
                self.api_key = None
                self.scorer = None

    def _build_ui(self):
        # 顶部标题
        header = tk.Frame(self, bg="#2C3E50", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="🤖 AI导师自动评分系统",
            font=("Microsoft YaHei UI", 18, "bold"),
            fg="white", bg="#2C3E50"
        ).pack(pady=15)

        # 主容器
        main = tk.Frame(self, bg="#F0F2F5")
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # 左侧：AI回复输入区
        left_frame = tk.LabelFrame(
            main, text="📝 AI回复内容（可添加多个AI对比）",
            font=("Microsoft YaHei UI", 11),
            bg="#F0F2F5", fg="#2C3E50", padx=10, pady=8
        )
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # AI列表容器
        self.ai_list_frame = tk.Frame(left_frame, bg="#F0F2F5")
        self.ai_list_frame.pack(fill="both", expand=True)

        # 添加第一个AI输入框
        self._add_ai_input("AI-1")

        # 按钮行
        btn_frame = tk.Frame(left_frame, bg="#F0F2F5")
        btn_frame.pack(fill="x", pady=(10, 0))

        tk.Button(
            btn_frame, text="➕ 添加AI",
            font=("Microsoft YaHei UI", 10),
            command=lambda: self._add_ai_input(f"AI-{len(self.ai_entries) + 1}"),
            bg="#27AE60", fg="white", relief="flat", cursor="hand2"
        ).pack(side="left", padx=(0, 5))

        tk.Button(
            btn_frame, text="🗑️ 清空全部",
            font=("Microsoft YaHei UI", 10),
            command=self._clear_all,
            bg="#E74C3C", fg="white", relief="flat", cursor="hand2"
        ).pack(side="left")

        # 右侧：设置区
        right_frame = tk.LabelFrame(
            main, text="⚙️ 设置",
            font=("Microsoft YaHei UI", 11),
            bg="#F0F2F5", fg="#2C3E50", padx=10, pady=8
        )
        right_frame.pack(side="right", fill="y", padx=(8, 0))

        # API密钥设置
        tk.Label(
            right_frame, text="智谱API密钥：",
            font=("Microsoft YaHei UI", 10),
            bg="#F0F2F5", fg="#2C3E50", anchor="w"
        ).pack(fill="x", pady=(0, 5))

        self.key_var = tk.StringVar(value=self.api_key or "")
        key_entry = tk.Entry(
            right_frame, textvariable=self.key_var,
            font=("Consolas", 9), show="*", width=28
        )
        key_entry.pack(fill="x", pady=(0, 5))

        tk.Button(
            right_frame, text="💾 保存密钥",
            font=("Microsoft YaHei UI", 9),
            command=lambda: self._save_key(key_entry.get()),
            bg="#3498DB", fg="white", relief="flat", cursor="hand2"
        ).pack(fill="x", pady=(0, 15))

        # 状态显示
        status_frame = tk.Frame(right_frame, bg="#F0F2F5")
        status_frame.pack(fill="x", pady=(10, 0))

        if self.api_key:
            status_txt = "✅ API已就绪"
            status_color = "#27AE60"
        else:
            status_txt = "⚠️ 请先配置API密钥"
            status_color = "#E74C3C"

        self.status_lbl = tk.Label(
            status_frame, text=status_txt,
            font=("Microsoft YaHei UI", 9),
            fg=status_color, bg="#F0F2F5"
        )
        self.status_lbl.pack()

        # 维度说明
        tk.Label(
            right_frame, text="📖 8维评分标准：",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="#F0F2F5", fg="#2C3E50", anchor="w"
        ).pack(fill="x", pady=(15, 5))

        dims_text = """0分=没做到
1分=部分做到
2分=完全做到

1.错误识别
2.错误定位
3.不泄露答案
4.提供引导
5.可行性
6.连贯性
7.导师语气
8.拟人化"""

        tk.Label(
            right_frame, text=dims_text,
            font=("Microsoft YaHei UI", 8),
            bg="#F0F2F5", fg="#555", justify="left", anchor="w"
        ).pack(fill="x")

        # 底部：评分按钮
        bottom_frame = tk.Frame(self, bg="#F0F2F5")
        bottom_frame.pack(fill="x", padx=16, pady=(0, 12))

        self.score_btn = tk.Button(
            bottom_frame, text="🚀 开始自动评分并排序",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="#E74C3C", fg="white", relief="flat", cursor="hand2",
            command=self._start_scoring, height=2
        )
        self.score_btn.pack(fill="x")

        # 结果显示区
        self.result_frame = tk.LabelFrame(
            self, text="📊 评分结果",
            font=("Microsoft YaHei UI", 11),
            bg="#F0F2F5", fg="#2C3E50", padx=12, pady=8
        )
        self.result_frame.pack(fill="x", padx=16, pady=(0, 12))

        self.result_text = scrolledtext.ScrolledText(
            self.result_frame, height=8, font=("Microsoft YaHei UI", 10),
            bg="white", fg="#2C3E50", wrap="word"
        )
        self.result_text.pack(fill="x")

    def _add_ai_input(self, name: str):
        """添加一个AI输入框"""
        ai_frame = tk.Frame(self.ai_list_frame, bg="#FFFFFF", relief="solid", bd=1)
        ai_frame.pack(fill="x", pady=(0, 8))

        # AI名称输入
        name_frame = tk.Frame(ai_frame, bg="#FFFFFF")
        name_frame.pack(fill="x", padx=8, pady=(8, 0))

        tk.Label(
            name_frame, text="名称：",
            font=("Microsoft YaHei UI", 10),
            bg="#FFFFFF", fg="#2C3E50"
        ).pack(side="left")

        name_var = tk.StringVar(value=name)
        tk.Entry(
            name_frame, textvariable=name_var,
            font=("Microsoft YaHei UI", 10), width=15
        ).pack(side="left", padx=(0, 10))

        # 删除按钮
        if len(self.ai_entries) > 1:
            tk.Button(
                name_frame, text="×",
                font=("Arial", 12, "bold"),
                bg="#E74C3C", fg="white", relief="flat",
                command=lambda f=ai_frame, n=name: self._remove_ai(f, n),
                width=2, height=1
            ).pack(side="right")

        # 回复文本框
        text = scrolledtext.ScrolledText(
            ai_frame, height=4, font=("Microsoft YaHei UI", 9),
            bg="#FAFAFA", fg="#333", wrap="word"
        )
        text.pack(fill="x", padx=8, pady=(0, 8))

        self.ai_entries[name] = {"frame": ai_frame, "name_var": name_var, "text": text}

    def _remove_ai(self, frame: tk.Frame, name: str):
        """删除AI输入框"""
        frame.destroy()
        del self.ai_entries[name]

    def _clear_all(self):
        """清空所有输入"""
        for entry in list(self.ai_entries.values()):
            entry["frame"].destroy()
        self.ai_entries.clear()
        self._add_ai_input("AI-1")
        self.result_text.delete("1.0", "end")

    def _save_key(self, key: str):
        """保存API密钥"""
        if not key.strip():
            messagebox.showwarning("提示", "请输入API密钥")
            return

        save_api_key(key.strip())
        self.api_key = key.strip()
        self.scorer = AutoScorer(self.api_key)

        if self.scorer.init_client():
            self.status_lbl.config(text="✅ API已就绪", fg="#27AE60")
            messagebox.showinfo("成功", "API密钥已保存")
        else:
            self.api_key = None
            self.scorer = None
            self.status_lbl.config(text="⚠️ API密钥无效", fg="#E74C3C")

    def _get_responses(self) -> dict:
        """获取所有AI回复"""
        responses = {}
        for entry in self.ai_entries.values():
            name = entry["name_var"].get().strip() or "未命名"
            content = entry["text"].get("1.0", "end").strip()
            if content:
                responses[name] = content
        return responses

    def _start_scoring(self):
        """开始评分"""
        if not self.scorer:
            messagebox.showwarning("提示", "请先配置智谱API密钥")
            return

        responses = self._get_responses()
        if len(responses) < 2:
            messagebox.showwarning("提示", "请至少输入2个AI的回复进行对比")
            return

        # 禁用按钮，显示进度
        self.score_btn.config(state="disabled", text="⏳ 评分中，请稍候...")

        def progress(msg):
            self.score_btn.config(text=f"⏳ {msg}")

        try:
            result = self.scorer.score_all(responses, progress_callback=progress)
            self._show_result(result)
        except Exception as e:
            messagebox.showerror("评分出错", str(e))
        finally:
            self.score_btn.config(state="normal", text="🚀 开始自动评分并排序")

    def _show_result(self, result: dict):
        """显示评分结果"""
        self.result_text.delete("1.0", "end")

        ranked = result["ranked"]

        output = []
        output.append("=" * 60)
        output.append("                    🎯 AI导师评分排名")
        output.append("=" * 60)
        output.append("")

        colors = ["🥇", "🥈", "🥉"] + ["  "] * 10

        for i, (name, data) in enumerate(ranked):
            medal = colors[i] if i < 3 else f"  {i+1}."
            scores = data["scores"]
            total = data["total"]
            brief = data.get("brief", "")

            grade = "S" if total >= 13 else "A" if total >= 10 else "B" if total >= 7 else "C" if total >= 4 else "D"

            output.append(f"{medal} {name}")
            output.append(f"   总分：{total}/16  等级：{grade}")
            if brief:
                output.append(f"   评价：{brief}")
            output.append("   各维度：")
            for j, (fname, s) in enumerate(zip(FEATURE_NAMES, scores)):
                bar = "█" * s + "░" * (2 - s)
                output.append(f"     {j+1}.{fname}: [{s}/2] {bar}")
            output.append("")

        output.append("=" * 60)

        # 计算对比分析
        best = ranked[0]
        worst = ranked[-1]
        output.append("")
        output.append("📊 对比分析：")
        output.append(f"   最佳：{best[0]}（{best[1]['total']}分）")
        output.append(f"   最差：{worst[0]}（{worst[1]['total']}分）")

        if len(ranked) >= 2:
            diff = ranked[0][1]['total'] - ranked[-1][1]['total']
            output.append(f"   分差：{diff}分")

        self.result_text.insert("1.0", "\n".join(output))


if __name__ == "__main__":
    app = AutoScoreGUI()
    app.mainloop()
