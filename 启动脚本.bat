@echo off
echo 正在启动 Gemini WebUI...
echo 请勿关闭此窗口...

:: 检查是否安装了 gradio
pip show gradio >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在首次安装 Gradio 库...
    pip install gradio
)

:: 运行 Python 脚本
:: 注意：如果你有多个 Python 版本，这里可能需要改成 python3 或者指定绝对路径
python webui.py

pause