# whisper-workspace Makefile (macOS/Linux)
# Windows 用户请直接使用 python run.py，或参考 setup.bat

VENV := venv
PYTHON := $(VENV)/bin/python3
RUN := $(PYTHON) run.py

.PHONY: help install run test all clean

help:
	@echo "whisper-workspace 标准流程"
	@echo ""
	@echo "  make install     安装依赖"
	@echo "  make run         处理 resource/ 下全部条目"
	@echo "  make one         处理第 1 条（最常用）"
	@echo "  make test        同 make one"
	@echo "  make last        处理最后 1 条"
	@echo "  make all N=3     处理前 3 条"
	@echo "  make gpu-one     处理第 1 条（强制 GPU）"
	@echo "  make dry         预览全部条目（不执行）"
	@echo "  make clean       清理输出和下载缓存"
	@echo ""
	@echo "  Windows: 直接运行 python run.py 1"

install: $(VENV)/bin/activate
	$(PYTHON) -m pip install -r requirements.txt -q
	@echo "✅ 依赖就绪"

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

# --- 标准流程 (auto 设备) ---

run:
	@test -d resource && ls resource/*.json >/dev/null 2>&1 || { echo "❌ resource/ 目录下没有 JSON 文件"; exit 1; }
	$(RUN)

one:
	$(RUN) 1

test: one

last:
	$(RUN) -1

all:
	$(RUN) $(N)

# --- GPU 加速（适用于有 NVIDIA 显卡的 Mac/Linux） ---

gpu:
	$(RUN) --device cuda --compute-type float16

gpu-one:
	$(RUN) --device cuda --compute-type float16 1

gpu-all:
	$(RUN) --device cuda --compute-type float16 $(N)

# --- 工具 ---

dry:
	$(RUN) --dry-run

list: dry

clean-dl:
	rm -rf downloads
	@echo "✅ 下载缓存已清理"

clean-out:
	rm -rf output
	@echo "✅ 输出文件已清理"

clean: clean-dl clean-out
	@echo "✅ 全部清理完成"
