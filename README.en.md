# CommunicationPet

[简体中文](README.md) | English

A local-first realtime voice companion for Windows. Microphone → VAD → ASR → LLM → GPT-SoVITS TTS, with the desktop UI built in Electron.

## Features

- Realtime microphone conversation with VAD segmentation and barge-in
- Chinese ASR (reuses the paraformer snapshot bundled with GPT-SoVITS)
- Any OpenAI-compatible chat-completions API (DeepSeek by default)
- Streaming GPT-SoVITS TTS with reference audio and optional fine-tuned weights
- Persona switching via `personas/<id>/SKILL.md` + `voice.json`
- Optional SQLite history and Markdown memory mirror
- Optional `mem0` long-term memory
- Optional VRM avatar DLC (disabled by default)

## Requirements

- Windows 10/11
- Node.js 18+
- Python 3.9–3.11 (the GPT-SoVITS portable runtime works well)
- NVIDIA GPU + CUDA
- [GPT-SoVITS v2Pro](https://github.com/RVC-Boss/GPT-SoVITS) (external TTS service)
- Git LFS

## Quick start

```bash
git lfs install
git clone <your-repo-url> CommunicationPet
cd CommunicationPet
git lfs pull
```

Put GPT-SoVITS at `vendor/GPT-SoVITS` (or set `MINIMIND_GSV_ROOT`), then:

```powershell
runtime\python.exe -m pip install -r requirements.txt   # from the GPT-SoVITS runtime
python setup_assets.py                                  # downloads the VAD model
```

Configure the LLM key either in **Settings → LLM** or with environment variables:

```powershell
$env:MINIMIND_LLM_API_KEY = "your-key"
$env:MINIMIND_LLM_BASE_URL = "https://api.deepseek.com"
$env:MINIMIND_LLM_MODEL = "deepseek-chat"
```

Start GPT-SoVITS (`runtime\python.exe -I api_v2.py`), then:

```powershell
npm install --prefix electron-app
npm run dev
```

On Windows you can also double-click `一键启动.bat`.

See [README.md](README.md) for the full configuration table, persona format, privacy notes and license information.

## License

Project code: Apache-2.0. Third-party components and voice assets: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
