# -*- coding: utf-8 -*-
"""一键启动菜单：中文界面，避免 .bat 乱码。"""
import io
import os
import subprocess
import sys

# Windows 控制台尽量用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def run_script(name: str) -> None:
    path = os.path.join(SRC, name)
    subprocess.run([sys.executable, path], cwd=SRC, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def show_help() -> None:
    print(
        """
------------------------------------------------------------
  评分说明（8 个维度，每个 0～2 分，总分 0～16）
------------------------------------------------------------
  1 错误识别   是否发现学生错在哪
  2 错误定位   是否指出错误位置
  3 不泄露答案 是否避免直接给答案
  4 提供引导   是否用提示引导学生思考
  5 可行性     建议是否现实可行
  6 连贯性     回复是否通顺有条理
  7 导师语气   是否友好、鼓励
  8 拟人化     是否像真人说话

  打分：0=没做到  1=部分做到  2=完全做到

  等级参考：S 13～16  A 10～12  B 7～9  C 4～6  D 0～3
------------------------------------------------------------
"""
    )
    input("按回车返回主菜单…")


def main() -> None:
    while True:
        os.system("cls" if sys.platform == "win32" else "clear")
        print(
            """
==========================================================
  AI 导师教学质量评分工具
==========================================================

  请选数字后回车：

  [1] 验证模型 — 用数据里的案例跑一遍，看预测是否接近专家分
  [2] 多 AI 排序（手动） — 自己给多个模型打分，自动算分并排先后
  [3] 训练模型 — 用 data 里的数据重新训练（会更新 regression_model）
  [4] 说明 — 8 个维度分别是什么意思、怎么打分
  [5] GUI 评分界面 — 图形窗口，点滑块打分（比赛演示用）
  [6] 命令行评分 — 粘贴回复，逐条手动输入维度

  ── 新功能 ─────────────────────────────────────
  [7] 🚀 智能自动评分 — AI自动分析回复，一键排序（需要智谱API）

  [0] 退出

==========================================================
"""
        )
        choice = input("请输入 0～7：").strip()

        if choice == "0":
            print("再见。")
            break
        if choice == "1":
            run_script("inference.py")
            input("\n按回车返回主菜单…")
        elif choice == "2":
            run_script("rank_ai.py")
            input("\n按回车返回主菜单…")
        elif choice == "3":
            # 仅装评分/训练所需包，避免 requirements.txt 里的 torch 等大包装满磁盘
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--prefer-binary",
                    "-q",
                    "lightgbm",
                    "pandas",
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "openpyxl",
                ],
                cwd=ROOT,
            )
            run_script("train.py")
            print("\n训练完成。模型一般在：data\\model\\regression_model.txt")
            print("接下来可选 [1] 验证，或 [2] 给多个 AI 排序。")
            input("\n按回车返回主菜单…")
        elif choice == "4":
            show_help()
        elif choice == "5":
            run_script("gui.py")
        elif choice == "6":
            run_script("score_my_ai.py")
            input("\n按回车返回主菜单…")
        elif choice == "7":
            print("\n正在下载智谱AI SDK 及依赖，请稍候...")
            # zhipuai 在部分环境下会漏装子依赖（如 sniffio），导致 pip 成功但 import 失败
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--prefer-binary",
                    "zhipuai",
                    "sniffio",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                print("\n❌ 安装失败！错误信息：")
                print(result.stderr or result.stdout)
                input("\n按回车返回主菜单...")
                continue
            chk = subprocess.run(
                [sys.executable, "-c", "from zhipuai import ZhipuAI"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if chk.returncode != 0:
                print("\n❌ 包已安装但仍无法加载智谱SDK，请把下面信息复制保存：")
                print(chk.stderr or chk.stdout)
                input("\n按回车返回主菜单...")
                continue
            print("✅ 依赖就绪，正在启动智能自动评分...")
            run_script("auto_score.py")
        else:
            input("无效输入，按回车重试…")


if __name__ == "__main__":
    main()
