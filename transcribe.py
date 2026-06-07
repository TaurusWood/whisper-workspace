#!/usr/bin/env python3
"""
使用 faster-whisper 将 m4a 音频文件转录并翻译为中文文档。

用法:
    python transcribe.py audio.m4a                          # 单个文件
    python transcribe.py a.m4a b.m4a                        # 多个文件
    python transcribe.py *.m4a --model medium               # 使用更大的模型
    python transcribe.py audio.m4a --no-translate            # 只转录不翻译
"""

import argparse
import os
import shutil
import site
import sys
import time
from pathlib import Path


def _ensure_cuda_dlls() -> None:
    """Windows: copy CUDA 12 runtime DLLs from nvidia-* wheels to ctranslate2 dir."""
    if sys.platform != "win32":
        return
    needed = [
        "cublas64_12.dll",
        "cublasLt64_12.dll",
    ]
    for sp in site.getsitepackages():
        nvidia_root = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if not os.path.isdir(bin_dir):
                continue
            for dll in needed:
                src = os.path.join(bin_dir, dll)
                if not os.path.exists(src):
                    continue
                dst = os.path.join(sp, "ctranslate2", dll)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)


_ensure_cuda_dlls()

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator


# ---------------------------------------------------------------------------
# 转录
# ---------------------------------------------------------------------------

def transcribe_audio(model: WhisperModel, audio_path: str) -> tuple[str, str]:
    """
    转录音频文件，返回 (完整文本, 检测到的语言代码)。
    """
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"  检测到语言: {info.language} (概率: {info.language_probability:.2f})")
    print(f"  时长: {info.duration:.1f} 秒\n")

    lines: list[str] = []
    for seg in segments:
        timestamp = f"[{seg.start:>7.1f}s → {seg.end:>7.1f}s]"
        print(f"  {timestamp} {seg.text}")
        lines.append(seg.text)

    return "\n".join(lines), info.language


# ---------------------------------------------------------------------------
# 翻译
# ---------------------------------------------------------------------------

def translate_to_chinese(text: str, chunk_size: int = 4000) -> str:
    """
    将文本分块翻译为中文，返回完整的中文译文。

    长文本会被按段落边界切分，翻译过程中会显示进度。
    """
    translator = GoogleTranslator(source="auto", target="zh-CN")

    # 按段落切分，在 chunk_size 附近按段落边界合并
    paragraphs = text.split("\n")
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if len(buf) + len(para) < chunk_size:
            buf += para + "\n"
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = para + "\n"

    if buf.strip():
        chunks.append(buf.strip())

    if not chunks:
        return ""

    translated_chunks: list[str] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"  翻译中... ({i + 1}/{total})", end=" ", flush=True)
        try:
            result = translator.translate(chunk)
            translated_chunks.append(result)
            print("✓")
        except Exception as exc:
            print(f"✗ ({exc})")
            translated_chunks.append(f"[翻译失败]\n{chunk}")
        # 避免请求过于频繁
        if i < total - 1:
            time.sleep(0.3)

    return "\n\n".join(translated_chunks)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 faster-whisper 转录音频并翻译为中文文档"
    )
    parser.add_argument(
        "input", nargs="+",
        help="输入的音频文件（支持 m4a / mp3 / wav 等格式）"
    )
    parser.add_argument(
        "--model", default="small",
        choices=["tiny", "tiny.en", "base", "base.en",
                 "small", "small.en", "medium", "medium.en",
                 "large-v2", "large-v3", "turbo"],
        help="Whisper 模型大小 (默认: small)。含 .en 后缀仅支持英语。"
    )
    parser.add_argument(
        "--output-dir", default="./output",
        help="输出目录 (默认: ./output)"
    )
    parser.add_argument(
        "--no-translate", action="store_true",
        help="只转录，不翻译为中文"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备 (默认: auto — 优先 GPU)"
    )
    parser.add_argument(
        "--compute-type", default="auto",
        help="计算精度 (默认: auto；可指定 int8 / float16 / int8_float16)"
    )

    args = parser.parse_args()

    # 检查输入文件
    for f in args.input:
        if not os.path.exists(f):
            print(f"❌ 文件不存在: {f}")
            sys.exit(1)

    # 准备输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型（首次会从 HuggingFace 下载）
    print(f"\n⏳ 正在加载模型 '{args.model}' ...")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    print("✅ 模型加载完成\n")

    # 逐个处理文件
    for idx, audio_path in enumerate(args.input):
        print(f"{'=' * 60}")
        print(f"[{idx + 1}/{len(args.input)}] 处理: {audio_path}")
        print(f"{'=' * 60}\n")

        # --- 转录 ---
        print("🎙️  转录中...")
        transcript, language = transcribe_audio(model, audio_path)

        base_name = Path(audio_path).stem

        # 保存原始转录
        txt_path = os.path.join(args.output_dir, f"{base_name}_原文本.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\n📄 原始转录 → {txt_path}")

        # --- 翻译 ---
        if not args.no_translate:
            print(f"\n🌐 正在翻译为中文 ...")
            chinese = translate_to_chinese(transcript)

            md_path = os.path.join(args.output_dir, f"{base_name}_中文.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {base_name}\n\n")
                f.write(f"> 源语言: {language} | ")
                f.write(f"模型: {args.model} | ")
                f.write(f"翻译引擎: Google Translate\n\n")
                f.write("---\n\n")
                f.write(chinese)
            print(f"📄 中文翻译 → {md_path}")

        print()

    print("✅ 全部完成！")


if __name__ == "__main__":
    main()
