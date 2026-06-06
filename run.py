#!/usr/bin/env python3
"""
标准流程脚本：读取元数据 → 下载音频 → 转录 → 输出文档。

用法:
    python run.py                          # 遍历 resource/ 下所有 JSON 的全部条目
    python run.py 1                        # 只处理第 1 条（测试用）
    python run.py 3                        # 只处理前 3 条
    python run.py -f resource/xxx.json     # 指定文件，全部处理
    python run.py -f resource/xxx.json 1   # 指定文件 + 第 1 条

目录约定:
    resource/   ← 放入元数据 JSON 文件（必填，脚本自动扫描）
    downloads/  ← 下载的音频缓存（自动创建，已下载的不会重复下载）
    output/     ← 转录输出的文档（自动创建）
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# 路径约定
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = BASE_DIR / "resource"
DOWNLOAD_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_MODEL = "small"


# ---------------------------------------------------------------------------
# 元数据加载
# ---------------------------------------------------------------------------

def load_metadata(file_path: str | None = None) -> list[dict]:
    """
    加载元数据条目。

    如果 file_path 为 None，自动扫描 RESOURCE_DIR 下所有 .json 文件，
    按文件名排序后合并所有数组条目。支持单个 JSON 文件和 JSON 数组两种格式。
    """
    if file_path:
        paths = [Path(file_path)]
        for p in paths:
            if not p.exists():
                print(f"❌ 文件不存在: {p}")
                sys.exit(1)
    else:
        paths = sorted(RESOURCE_DIR.glob("*.json"))
        if not paths:
            print(f"❌ resource/ 目录下没有找到 JSON 文件")
            sys.exit(1)

    entries: list[dict] = []
    for p in paths:
        print(f"📋 加载元数据: {p.name}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            entries.extend(data)
        elif isinstance(data, dict):
            entries.append(data)
        else:
            print(f"⚠️  跳过 {p.name}: 不支持的 JSON 格式（需为对象或数组）")

    return entries


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------

def download_file(url: str, dest_dir: str) -> str:
    """
    下载文件到目标目录。
    若本地已存在（基于 URL hash），跳过下载。
    返回本地文件路径。
    """
    os.makedirs(dest_dir, exist_ok=True)

    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1] or ".m4a"
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    filename = f"{url_hash}{ext}"
    local_path = os.path.join(dest_dir, filename)

    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  ⏭️  已缓存 ({size_mb:.1f} MB)")
        return local_path

    print(f"  ⬇️  正在下载...", end="", flush=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)

    size_mb = downloaded / 1024 / 1024
    print(f"\r  ✅ 下载完成 ({size_mb:.1f} MB)")
    return local_path


# ---------------------------------------------------------------------------
# 转录
# ---------------------------------------------------------------------------

def transcribe_audio(model: WhisperModel, audio_path: str) -> tuple[str, str]:
    """
    转录音频，返回 (完整文本, 语言代码)。
    """
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"  🎙️  语言: {info.language} | 概率: {info.language_probability:.2f} | "
          f"时长: {info.duration:.0f}s")

    lines: list[str] = []
    for seg in segments:
        ts = f"[{seg.start:>6.0f}s]"
        print(f"  {ts} {seg.text}")
        lines.append(seg.text)

    return "\n".join(lines), info.language


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def safe_filename(s: str) -> str:
    """清理文件名中的非法字符。"""
    return re.sub(r'[\\/:*?"<>|]', "：", s)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="标准流程：读取元数据 → 下载音频 → 转录 → 输出文档"
    )
    parser.add_argument(
        "count", nargs="?", default=None,
        help="条数：正数=前 N 条，负数=后 N 条。不传=全部。例: 1 / -1 / 3"
    )
    parser.add_argument(
        "-f", "--file",
        help="指定元数据 JSON 文件。不传则自动扫描 resource/ 目录"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Whisper 模型 (默认: {DEFAULT_MODEL})。可选: tiny, base, small, medium, large-v3"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda"],
        help="计算设备 (默认: auto — 自动选 GPU > CPU)"
    )
    parser.add_argument(
        "--compute-type", default="auto",
        help="精度 (默认: auto)。GPU 建议 float16，CPU 建议 int8"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="跳过下载阶段（音频已存在于 downloads/ 时使用）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只列出待处理条目，不执行实际操作"
    )

    args = parser.parse_args()

    # --- 解析条数 ---
    # 正数 = 前 N 条，负数 = 后 N 条，None = 全部
    limit: int | None = None
    from_end: bool = False
    if args.count is not None:
        try:
            limit = int(args.count)
        except ValueError:
            print(f"❌ 条数必须是数字，收到: {args.count}")
            sys.exit(1)
        if limit == 0:
            print("❌ 条数不能为 0")
            sys.exit(1)
        if limit < 0:
            from_end = True
            limit = abs(limit)

    # --- 加载元数据 ---
    entries = load_metadata(args.file)
    total = len(entries)

    if limit is not None:
        if from_end:
            entries = entries[-limit:]
        else:
            entries = entries[:limit]

    print(f"📋 共处理 {len(entries)}/{total} 条\n")

    if args.dry_run:
        for i, e in enumerate(entries):
            print(f"  [{i + 1}] {e.get('title', 'N/A')}")
            print(f"       URL: {e.get('url', 'N/A')[:80]}...")
            print(f"       频道: {e.get('channel', 'N/A')}")
        return

    # --- 加载模型 ---
    device = args.device
    compute_type = args.compute_type
    print(f"⏳ 加载模型 '{args.model}' (device={device}, compute={compute_type})...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    print("✅ 模型就绪\n")

    # --- 逐条处理 ---
    for idx, entry in enumerate(entries):
        url = entry.get("url", "")
        title = entry.get("title", f"untitled_{idx}")
        channel = entry.get("channel", "")
        description = entry.get("description", "")

        if not url:
            print(f"⚠️  跳过 [{idx + 1}]: 缺少 url 字段")
            continue

        print(f"{'=' * 60}")
        print(f"[{idx + 1}/{len(entries)}] {title}")
        print(f"{'=' * 60}")

        # 1. 下载
        if not args.skip_download:
            try:
                audio_path = download_file(url, str(DOWNLOAD_DIR))
            except Exception as e:
                print(f"  ❌ 下载失败: {e}")
                continue
        else:
            parsed = urlparse(url)
            ext = os.path.splitext(parsed.path)[1] or ".m4a"
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            audio_path = str(DOWNLOAD_DIR / f"{url_hash}{ext}")
            if not os.path.exists(audio_path):
                print(f"  ❌ 音频文件不存在: {audio_path}")
                continue

        # 2. 转录
        print("  🎙️  转录中...")
        try:
            transcript, language = transcribe_audio(model, audio_path)
        except Exception as e:
            print(f"  ❌ 转录失败: {e}")
            continue

        # 3. 保存输出
        os.makedirs(str(OUTPUT_DIR), exist_ok=True)
        safe_title = safe_filename(title)

        # 纯文本版
        txt_path = OUTPUT_DIR / f"{safe_title}_转录.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n")
            f.write(f"频道: {channel}\n")
            f.write(f"检测语言: {language}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(transcript)
        print(f"  📄 TXT → {txt_path.name}")

        # Markdown 版（含 description）
        md_path = OUTPUT_DIR / f"{safe_title}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(f"> 频道: {channel} | 音频语言: {language}\n\n")
            f.write("## 转录文本\n\n")
            f.write(transcript)
            if description:
                f.write("\n\n---\n\n")
                f.write("## 原始简介\n\n")
                f.write(description)
        print(f"  📄 MD  → {md_path.name}")

        print()

    print(f"✅ 全部完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
