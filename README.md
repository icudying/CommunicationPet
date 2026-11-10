# CommunicationPet

本地运行的实时语音对话桌面伙伴。麦克风 → VAD → ASR → LLM → GPT-SoVITS 语音合成，一条链路全部跑在自己的电脑上（LLM 可接任意 OpenAI 兼容 API）。

> 界面：Electron 暗色玻璃拟态；后端：Python FastAPI + WebSocket；语音合成：外部 GPT-SoVITS `api_v2` 服务。

## 功能

- 实时麦克风对话，支持 VAD 自动断句与打断
- 中文 ASR（复用 GPT-SoVITS 自带的 paraformer 快照）
- OpenAI 兼容 LLM（默认 DeepSeek，可在设置里改成任意兼容服务）
- GPT-SoVITS 流式语音合成，支持参考音频 + 微调权重
- 人格切换：`personas/<id>/SKILL.md` + `voice.json`
- 可选的 SQLite 会话历史与 Markdown 记忆镜像
- 可选的 `mem0` 长期记忆
- 可选化身 DLC（VRM 动作/口型），默认关闭，见 [docs/AVATAR_DLC.md](docs/AVATAR_DLC.md)

## 目录结构

```text
CommunicationPet/
  electron-app/          Electron 桌面端
    main.js              主进程：服务编排、IPC、配置
    renderer/            界面
  realtime/              Python 实时后端
    backend_main.py      FastAPI + WebSocket 入口
    asr_service.py       paraformer / SenseVoice ASR
    gpt_sovits_tts_service.py
    persona_loader.py
    history_*.py         会话历史与记忆
  llm-module/            精简的 OpenAI 兼容客户端
  mem0/                  可选长期记忆（vendored，Apache-2.0）
  personas/              人格与语音资产
    niya/                妮娅 Niya
    peri/                佩莉 Peri
  model/vad/             Silero VAD
  setup_assets.py        下载/校验 VAD
  一键启动.bat           Windows 一键启动
```

## 环境要求

- Windows 10/11（当前主要在 Windows 上开发和验证）
- Node.js 18+
- Python 3.9–3.11（推荐直接复用 GPT-SoVITS 的便携 runtime）
- NVIDIA GPU + CUDA（ASR/TTS 默认走 CUDA；CPU 需要自行改配置）
- [GPT-SoVITS v2Pro](https://github.com/RVC-Boss/GPT-SoVITS)（外部 TTS 服务，MIT）
- Git LFS（语音权重通过 LFS 分发）

## 快速开始

### 1. 克隆仓库（必须带 LFS）

```bash
git lfs install
git clone <your-repo-url> CommunicationPet
cd CommunicationPet
git lfs pull
```

> 仓库包含约 630MB 的语音权重。GitHub 免费 LFS 额度是 1GB 存储 / 1GB 月流量；如果你的额度不够，可以把 `personas/*/weights/` 换成自己的权重或改用零样本参考音频。

### 2. 准备 GPT-SoVITS

把 GPT-SoVITS v2Pro 放到以下任一位置：

```text
CommunicationPet/vendor/GPT-SoVITS/        # 推荐
CommunicationPet/../VoiceChat/GPT-SoVITS-v2pro-20250604/   # 兼容旧布局
```

或设置环境变量：

```powershell
$env:MINIMIND_GSV_ROOT = "D:\GPT-SoVITS"
```

确认存在 `runtime\python.exe` 与 `tools\asr\models\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`。

### 3. 安装 Python 依赖

```powershell
# 推荐用 GPT-SoVITS 的 runtime，省掉重复安装 torch
& "$env:MINIMIND_GSV_ROOT\runtime\python.exe" -m pip install -r requirements.txt
```

如果你使用自己的 Python 环境：

```powershell
conda create -n communicationpet python=3.11 -y
conda activate communicationpet
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. 准备 VAD 模型

```powershell
python setup_assets.py
```

该脚本只下载 `model/vad/silero_vad.onnx`，并校验 GPT-SoVITS 自带的 paraformer 是否就位。

### 5. 配置 LLM API Key

两种方式任选其一：

- 启动应用后，在右上角 **Settings → LLM** 填入 API Key、Base URL、模型名；
- 或使用环境变量：

```powershell
$env:MINIMIND_LLM_API_KEY = "你的 key"
$env:MINIMIND_LLM_BASE_URL = "https://api.deepseek.com"
$env:MINIMIND_LLM_MODEL = "deepseek-chat"
```

也可以复制 `.env.example` 作为参考。**不要把真实 key 提交到仓库。**

### 6. 启动

一键启动（会拉起 GPT-SoVITS 与 Electron）：

```powershell
.\一键启动.bat
```

手动启动：

```powershell
# 终端 1：GPT-SoVITS
cd vendor\GPT-SoVITS
runtime\python.exe -I api_v2.py

# 终端 2：桌面端（会自动拉起 realtime 后端）
npm install --prefix electron-app
npm run dev
```

首次点击绿色的“开始语音”按钮即可对话。顶部下拉可切换人格。

## 配置项

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `MINIMIND_LLM_API_KEY` | LLM API Key | 空（请在设置里填） |
| `MINIMIND_LLM_BASE_URL` | OpenAI 兼容 API 地址 | `https://api.deepseek.com` |
| `MINIMIND_LLM_MODEL` | 模型名 | `deepseek-chat` |
| `MINIMIND_PYTHON` | 后端使用的 Python | 自动探测 |
| `MINIMIND_GSV_ROOT` | GPT-SoVITS 根目录 | `vendor/GPT-SoVITS` |
| `MINIMIND_ASR_MODEL_PATH` | paraformer 模型目录 | GPT-SoVITS 自带路径 |
| `MINIMIND_VAD_MODEL_PATH` | Silero VAD 路径 | `model/vad/silero_vad.onnx` |
| `MINIMIND_GPTSOVITS_URL` | TTS 服务地址 | `http://127.0.0.1:9880` |
| `MINIMIND_MEMORY_ENABLED` | 启用 mem0 记忆 | `0` |
| `MINIMIND_MOTION_ENABLED` | 启用化身动作解析 | `0`（无化身时保持关闭） |
| `MINIMIND_HAST_ENABLED` | 启用化身 DLC | `0` |

## 人格与语音

每个人格是一个目录：

```text
personas/<id>/
  SKILL.md       人格设定与表达方式
  voice.json     显示名、参考音频、提示文本、微调权重
  reference.wav  参考音频
  weights/       GPT-SoVITS 微调权重（可选）
```

`voice.json` 中的路径支持相对路径（相对于人格目录）或绝对路径。新增人格时复制一个目录、改 `id` / `name` / 音频即可。

> 仓库自带的人格语音资产仅用于演示。请确认你拥有相应声音与角色素材的使用权；相关权利不随本项目许可证转让。

## 隐私说明

- 会话历史默认写在 `runtime/data/`，Markdown 记忆镜像写在 `runtime/memory_vault/`，两者都已加入 `.gitignore`。
- API Key 只保存在本机 Electron user-data 目录或环境变量中，不会写入仓库。
- 本仓库不包含任何真实 API Key、聊天记录或用户路径。
- 如果曾经把 key 写进公开文件，请立即到服务商后台撤销并重新生成。

## 化身 DLC

主仓库只包含语音对话。VRM 化身、骨骼动作、口型与表情由可选的外部服务提供，默认关闭。启用方式与许可说明见 [docs/AVATAR_DLC.md](docs/AVATAR_DLC.md)。

## 许可证与第三方

- 本项目代码：Apache-2.0，见 [LICENSE](LICENSE)。
- 第三方组件与素材许可：见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [NOTICE](NOTICE)。
- `mem0/` 为上游 Apache-2.0 代码的 vendored 副本，修改说明见 `mem0/LOCAL_CHANGES.md`。
- GPT-SoVITS、Silero VAD、FunASR 等为外部依赖，遵循各自许可。

## 致谢

本项目的实时语音框架源自 Whispera（Apache-2.0），长期记忆来自 [mem0](https://github.com/mem0ai/mem0)，语音合成依赖 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)。
