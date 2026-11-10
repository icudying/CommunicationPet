from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# 本项目复用 GPT-SoVITS 的 runtime (torch/funasr/CUDA DLL)，
# 因此 ASR 也直接复用 GPT-SoVITS 自带的 paraformer 快照，而不是下载一个与本
# 项目所绑 funasr 版本不兼容的 SenseVoiceSmall (见 asr_service.py 的兼容性说明)。
# GSV 安装目录名带版本日期，这里按前缀扫描以适应版本升级。
_GSV_DIR_PREFIX = "GPT-SoVITS-v2pro-"
_PARAFORMER_REL = Path(
    "tools", "asr", "models",
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)


def find_gsv_root() -> Path | None:
    """定位 GPT-SoVITS 安装根目录。

    解析顺序：
      1. MINIMIND_GSV_ROOT 环境变量（显式覆盖）
      2. <repo>/vendor/GPT-SoVITS
      3. <repo>/../VoiceChat 下的 GPT-SoVITS-v2pro-* 目录（兼容旧布局）
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


def default_asr_model_path() -> Path:
    """ASR 模型默认路径：复用 GPT-SoVITS 自带的 paraformer 快照。

    该默认值仅用于未显式传 --asr-model-path / MINIMIND_ASR_MODEL_PATH 的场合；
    启动器与 main.js 仍可通过环境变量覆盖。
    """
    root = find_gsv_root()
    if root is not None:
        return root / _PARAFORMER_REL
    # 即使 GSV 未就位也返回一个有意义的路径，方便报错信息指向
    return REPO_ROOT / "vendor" / "GPT-SoVITS" / _PARAFORMER_REL


def _prepend_if_missing(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def ensure_repo_root_on_path() -> Path:
    _prepend_if_missing(REPO_ROOT)
    return REPO_ROOT


def ensure_llm_module_on_path() -> Path:
    ensure_repo_root_on_path()
    src_path = REPO_ROOT / "llm-module" / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"llm-module src path not found: {src_path}")
    _prepend_if_missing(src_path)
    return src_path


def ensure_mem0_on_path() -> Path | None:
    ensure_repo_root_on_path()
    src_path = REPO_ROOT / "mem0"
    if not src_path.exists():
        return None
    _prepend_if_missing(src_path)
    return src_path


# NOTE: the VoxCPM streaming TTS module (voxcpm-tts-streaming-module/) was
# removed from this project — TTS is now served by the external GPT-SoVITS
# api_v2 server via realtime/gpt_sovits_tts_service.py. The former
# ensure_voxcpm_on_path() helper is intentionally gone; if a stale caller
# ever needs a shim, route it to the GPT-SoVITS service instead.
