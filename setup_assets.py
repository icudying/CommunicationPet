#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CommunicationPet 资源准备脚本

本项目使用 OpenAI 兼容 API 作为 LLM，并使用外部 GPT-SoVITS api_v2 服务作为 TTS。因此需要
本地保留的模型只剩：

  1. VAD 模型  silero_vad.onnx  -> model/vad/silero_vad.onnx  （本脚本下载）
  2. ASR 模型  paraformer       -> 复用 GPT-SoVITS 自带快照    （本脚本仅校验）

ASR 复用说明：funasr 1.0.27（GPT-SoVITS runtime 自带版本）与 ModelScope 上
的 SenseVoiceSmall 快照不兼容（checkpoint 缺 dims 元数据，ModelDimensions
报缺 10 个必填参数），因此本项目改用 GPT-SoVITS 自带且日常在用的
paraformer-large 中文 ASR 快照。该快照位于 GPT-SoVITS 安装目录下的
tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch，
由 GPT-SoVITS 自身的 ASR 工具负责下载，本脚本只校验其是否就位。

用法:
    python setup_assets.py            # 下载/校验缺失资源
    python setup_assets.py --verify   # 仅校验是否就绪
    python setup_assets.py --force     # 强制重新下载 VAD

注: silero 优先走 GitHub 直链，失败回退 HF 镜像与加速镜像。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VAD_PATH = REPO_ROOT / "model" / "vad" / "silero_vad.onnx"

_GSV_DIR_PREFIX = "GPT-SoVITS-v2pro-"
_PARAFORMER_REL = Path(
    "tools", "asr", "models",
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)

# paraformer 加载所需的文件（与 funasr AutoModel 实际加载所需对齐）。
PARAFORMER_REQUIRED_FILES = [
    "model.pt",      # 权重
    "config.yaml",   # 模型配置
    "am.mvn",        # 输入均值/方差归一化
    "tokens.json",   # 分词器词表
]

# 默认走国内 HF 镜像，避免 huggingface.co 直连 401/超时；用户已设则尊重用户。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# silero_vad.onnx 真实模型（V4 约 1.8MB，匹配上游仓库 LFS 指针
# 的 size=1807524）。仓库里的占位文件可能只是 Git-LFS 指针（132 字节），
# 不是真实模型，必须重新下载。
#
# 注意：仓库 owner 是 `snakers4`（不是 `snakers`），文件在 `files/` 子目录下，
# 不是仓库根目录。V5 模型固定 512 样本窗口，与本项目 vad_session.py 的
# chunk_window=1024 不兼容，因此用 V4。
SILERO_URLS = [
    # 官方 V4 tag（最准）—— 文件路径 files/silero_vad.onnx
    "https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/silero_vad.onnx",
    # master 分支 data 目录（也是 V4 兼容文件，备用）
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    # HuggingFace 镜像（国内可达）
    "https://hf-mirror.com/snakers4/silero-vad/resolve/main/silero_vad.onnx",
    "https://huggingface.co/snakers4/silero-vad/resolve/main/silero_vad.onnx",
    # GitHub 加速镜像（顺序放后，避免某些镜像不稳定）
    "https://ghfast.top/https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/silero_vad.onnx",
    "https://gh-proxy.com/https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/silero_vad.onnx",
]


def find_gsv_root() -> Path | None:
    """定位 GPT-SoVITS 安装根目录（与 realtime/local_modules.py 同逻辑）。

    解析顺序：MINIMIND_GSV_ROOT 环境变量 -> <repo>/vendor/GPT-SoVITS ->
    <repo>/../VoiceChat 下 GPT-SoVITS-v2pro-*。
    """
    env_root = os.environ.get("MINIMIND_GSV_ROOT")
    if env_root and Path(env_root).is_dir():
        return Path(env_root)

    vendor_root = REPO_ROOT / "vendor" / "GPT-SoVITS"
    if (vendor_root / "runtime" / "python.exe").is_file():
        return vendor_root

    voicechat = REPO_ROOT.parent / "VoiceChat"
    if voicechat.is_dir():
        for candidate in sorted(voicechat.glob(_GSV_DIR_PREFIX + "*"), reverse=True):
            if (candidate / "runtime" / "python.exe").is_file():
                return candidate
    return None


def paraformer_dir() -> Path:
    """paraformer 快照路径：优先 MINIMIND_ASR_MODEL_PATH，否则 GSV 自带路径。"""
    env_asr = os.environ.get("MINIMIND_ASR_MODEL_PATH")
    if env_asr and Path(env_asr).is_dir():
        return Path(env_asr)
    root = find_gsv_root()
    if root is not None:
        return root / _PARAFORMER_REL
    return REPO_ROOT / "vendor" / "GPT-SoVITS" / _PARAFORMER_REL


def _is_lfs_pointer(path: Path) -> bool:
    """上游仓库里的 silero_vad.onnx 可能只是 LFS 指针（132 字节）。"""
    try:
        if path.stat().st_size < 1024:
            head = path.read_text(encoding="utf-8", errors="ignore").strip()
            return head.startswith("version https://git-lfs")
    except OSError:
        return True
    return False


def _download(url: str, dest: Path, timeout: int = 120) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  下载 {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CommunicationPet-setup/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, OSError) as exc:
        print(f"    失败: {exc}")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False


def _required_files_ok(directory: Path, required: list[str]) -> bool:
    """检查必需文件是否都存在且非空（阈值 100 字节过滤 0 字节占位与 LFS 指针）。"""
    for name in required:
        p = directory / name
        if not p.is_file():
            return False
        if p.stat().st_size < 100:
            return False
    return True


def ensure_silero(force: bool, verify_only: bool) -> bool:
    print("[VAD] silero_vad.onnx")
    if VAD_PATH.exists() and not _is_lfs_pointer(VAD_PATH) and not force:
        print(f"  已存在 ({VAD_PATH.stat().st_size} bytes)，跳过")
        return True
    if verify_only:
        print("  缺失或为 LFS 指针，需下载")
        return False
    for url in SILERO_URLS:
        if _download(url, VAD_PATH):
            print(f"  成功 -> {VAD_PATH} ({VAD_PATH.stat().st_size} bytes)")
            return True
    print("  所有源均失败，请手动下载 silero_vad.onnx 放到 model/vad/")
    return False


def ensure_paraformer(force: bool, verify_only: bool) -> bool:
    """校验 GPT-SoVITS 自带的 paraformer 快照是否就位。

    该快照不属于本项目自身资产，由 GPT-SoVITS 的 ASR 工具负责下载，
    因此这里只做存在性校验，不主动下载（避免重复维护一份 900MB 模型）。
    """
    print("[ASR] paraformer (复用 GPT-SoVITS 自带)")
    asr_dir = paraformer_dir()
    print(f"  路径: {asr_dir}")
    gsv_root = find_gsv_root()
    if gsv_root is None:
        print("  [警告] 未找到 GPT-SoVITS 安装目录 (VoiceChat\\GPT-SoVITS-v2pro-*)")
        print("         ASR 模型无法定位。请确认 GPT-SoVITS 已安装在项目同级目录，")
        print("         或设置环境变量 MINIMIND_GSV_ROOT / MINIMIND_ASR_MODEL_PATH。")
        return False

    if _required_files_ok(asr_dir, PARAFORMER_REQUIRED_FILES) and not force:
        print(f"  已就位 ({asr_dir})，跳过")
        return True
    if verify_only:
        print("  缺失或文件不全，需在 GPT-SoVITS 侧准备")
        return False

    # paraformer 由 GPT-SoVITS 自动下载，这里无法代劳，只给出指引。
    print("  paraformer 快照缺失。请在 GPT-SoVITS 中运行一次其 ASR 工具触发自动下载：")
    print("    1. 启动 GPT-SoVITS WebUI")
    print("    2. 进入 '工具' -> '语音伴奏/ASR' 执行一次（会自动拉取 paraformer 等模型）")
    print(f"    3. 或确认目录已存在: {asr_dir}")
    print("  完成后重新运行本脚本校验。")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="CommunicationPet 资源准备（VAD 下载 + ASR 校验）")
    parser.add_argument("--verify", action="store_true", help="仅校验是否就绪，不下载")
    parser.add_argument("--force", action="store_true", help="强制重新下载 VAD")
    args = parser.parse_args()

    print("=" * 56)
    print("  CommunicationPet setup_assets")
    print("=" * 56)
    ok_vad = ensure_silero(args.force, args.verify)
    ok_asr = ensure_paraformer(args.force, args.verify)
    print("-" * 56)
    print(f"VAD  silero_vad.onnx : {'OK' if ok_vad else 'MISSING'}")
    print(f"ASR  paraformer       : {'OK' if ok_asr else 'MISSING'}")
    if ok_vad and ok_asr:
        print("全部就绪。")
        return 0
    print("仍有缺失，详见上方日志。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
