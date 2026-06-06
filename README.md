# whisper-workspace

faster-whisper 音频转录 → 中文文档，标准流程工具。

## 环境要求

| 工具 | 用途 | 安装 |
|------|------|------|
| Python 3.10+ | 运行脚本 | [python.org](https://www.python.org/downloads/) |
| ffmpeg | 音频解码 | macOS: `brew install ffmpeg` / Windows: [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) |
| NVIDIA GPU（可选） | CUDA 加速 | 驱动 ≥ 525，无需装 CUDA Toolkit |

## 快速开始

### macOS / Linux

```bash
make install       # 创建 venv + 安装依赖
make one           # 处理第 1 条
make run           # 处理全部
make all N=3       # 处理前 3 条
make dry           # 预览条目
```

### Windows

```powershell
setup.bat          # 一键初始化（自动检测 CUDA）
python run.py 1    # 处理第 1 条
python run.py      # 处理全部
python run.py --dry-run
```

### GPU 加速

```bash
# macOS/Linux
make gpu-one                      # 强制 GPU
python run.py 1 --device cuda --compute-type float16

# Windows（auto 会自动选 CUDA）
python run.py 1
```

## 目录结构

```
whisper-workspace/
├── resource/          # 元数据 JSON（约定目录，脚本自动扫描）
├── downloads/         # 下载缓存（不提交 git）
├── output/            # 转录输出（不提交 git）
├── run.py             # 标准流程：下载 → 转录 → 输出
├── transcribe.py      # 独立转录工具（零散文件用）
├── setup.bat          # Windows 初始化
├── Makefile           # macOS/Linux 快捷命令
└── requirements.txt
```

## 元数据格式

在 `resource/` 目录放入 JSON 文件，格式如下：

```json
[
  {
    "url": "https://example.com/audio.m4a",
    "title": "节目标题",
    "description": "节目简介",
    "channel": "频道名"
  }
]
```

支持单个 `.json` 文件或 JSON 数组。脚本会自动合并 `resource/` 下所有文件。

## run.py 参数

```
python run.py [条数] [-f 指定文件] [--device auto|cuda|cpu] [--model 模型] [--dry-run]

  条数           正数=前N条，负数=后N条，不传=全部
  -f, --file     指定元数据文件（不传则扫描 resource/）
  --model        模型大小：tiny / base / small / medium / large-v3（默认 small）
  --device       计算设备（默认 auto）
  --compute-type 精度（GPU 建议 float16，CPU 建议 int8）
  --dry-run      预览条目，不执行
```

## 模型说明

| 模型 | 大小 | 速度 | 质量 | 适用场景 |
|------|------|------|------|----------|
| tiny | ~150MB | 极快 | 一般 | 快速测试 |
| base | ~300MB | 快 | 尚可 | 简单英文 |
| small | ~500MB | 中等 | 良好 | **日常推荐** |
| medium | ~1.5GB | 较慢 | 很好 | 高质量需求 |
| large-v3 | ~3GB | 慢 | 最佳 | 专业级精度 |

首次运行会自动从 HuggingFace 下载模型。

## 性能参考

| 设备 | 15分钟音频耗时 |
|------|---------------|
| Mac CPU (Apple Silicon) | 5-10 分钟 |
| Windows RTX 4070 CUDA | 10-30 秒 |
