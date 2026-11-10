from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTSRuntimeConfig:
    model_path: str | None = None
    lora_root: str | None = None
    load_denoiser: bool = False
    optimize: bool = True
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    min_len: int = 2
    max_len: int = 4096
    normalize: bool = False
    denoise: bool = False


@dataclass
class TTSRequestOptions:
    model_version: str | None = None
    lora_selection: str | None = None
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    # aux_ref_audio_paths mirrors GPT-SoVITS api_v2's parameter of the same
    # name: a list of auxiliary reference audios used for multi-reference tone
    # fusion. Only the main ref_audio_path needs a matching prompt_text; the
    # aux entries are used by the model for timbre fusion without their own
    # prompt text.
    aux_ref_audio_paths: list[str] | None = None
    # Fine-tuned GPT-SoVITS checkpoints. When present the service switches the
    # server's globally loaded weights before synthesizing; None/empty means
    # keep the base pretrained weights (zero-shot personas).
    gpt_weights_path: str | None = None
    sovits_weights_path: str | None = None
    reference_wav_path: str | None = None
    cfg_value: float | None = None
    inference_timesteps: int | None = None
    seed: int | None = None
