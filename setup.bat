@echo off
chcp 65001 >nul
echo ========================================
echo   whisper-workspace Windows 初始化
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo         下载地址: https://www.python.org/downloads/
    echo         ⚠️  安装时务必勾选 "Add Python to PATH"
    pause
    exit /b 1
)
python --version
echo.

:: 检查 ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 未找到 ffmpeg，请安装:
    echo           1. 下载: https://www.gyan.dev/ffmpeg/builds/
    echo           2. 解压到 C:\ffmpeg
    echo           3. 添加 C:\ffmpeg\bin 到系统 PATH
    echo.
)
echo [OK] ffmpeg 就绪
echo.

:: 创建虚拟环境
echo [1/3] 创建虚拟环境...
python -m venv venv
echo.

:: 安装依赖
echo [2/3] 安装 Python 依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt -q
echo.

:: 验证 CUDA
echo [3/3] 检查 CUDA 是否可用...
python -c "from faster_whisper import WhisperModel; m = WhisperModel('tiny', device='cuda', compute_type='float16'); print('[OK] CUDA GPU 加速可用!')" 2>nul
if errorlevel 1 (
    echo [WARNING] CUDA 不可用，将使用 CPU 模式（慢）
    echo          如果您的显卡是 NVIDIA，请更新驱动: https://www.nvidia.com/drivers
) else (
    echo [OK] RTX 4070 GPU 加速就绪! 🚀
)

echo.
echo ========================================
echo   初始化完成！
echo.
echo   常用命令:
echo     python run.py 1          处理第1条
echo     python run.py            处理全部
echo     python run.py --dry-run  预览条目
echo ========================================
pause
