@echo off
chcp 65001 >nul
title CommunicationPet Launcher
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

REM ============================================================
REM  CommunicationPet 一键启动
REM  依赖：Node.js、GPT-SoVITS v2Pro（外部 TTS 服务）
REM  GPT-SoVITS 查找顺序：
REM    1. 环境变量 MINIMIND_GSV_ROOT
REM    2. %ROOT%\vendor\GPT-SoVITS
REM    3. %ROOT%\..\VoiceChat\GPT-SoVITS-v2pro-20250604
REM ============================================================

set "GSV=%MINIMIND_GSV_ROOT%"
if not defined GSV if exist "%ROOT%\vendor\GPT-SoVITS\runtime\python.exe" set "GSV=%ROOT%\vendor\GPT-SoVITS"
if not defined GSV if exist "%ROOT%\..\VoiceChat\GPT-SoVITS-v2pro-20250604\runtime\python.exe" set "GSV=%ROOT%\..\VoiceChat\GPT-SoVITS-v2pro-20250604"

if not defined GSV (
    echo [错误] 未找到 GPT-SoVITS。
    echo        请把它放到 vendor\GPT-SoVITS，或设置环境变量 MINIMIND_GSV_ROOT。
    pause
    exit /b 1
)

set "GSV_PY=%GSV%\runtime\python.exe"
if not exist "%GSV_PY%" (
    echo [错误] 未找到 GPT-SoVITS 的 Python: %GSV_PY%
    pause
    exit /b 1
)

REM 把 GPT-SoVITS runtime 加入 PATH，python 才能找到 torch/CUDA 等 DLL。
set "PATH=%GSV%\runtime;%PATH%"

REM 把本项目路径写入 GPT-SoVITS runtime 的 python39._pth，
REM 否则 python -m realtime.app 找不到 realtime 包。
powershell -NoProfile -Command "$pth='%GSV%\runtime\python39._pth'; $root='%ROOT%'; if((Test-Path $pth) -and ((Get-Content $pth -Raw) -notlike ('*'+$root+'*'))){ (Get-Content $pth) + $root | Set-Content -Path $pth -Encoding ASCII; Write-Host '[setup] 已把项目根目录写入 _pth' }"

echo ============================================================
echo   CommunicationPet 一键启动
echo ============================================================

set "MINIMIND_PYTHON=%GSV_PY%"
set "MINIMIND_GSV_ROOT=%GSV%"
set "MINIMIND_ASR_MODEL_PATH=%GSV%\tools\asr\models\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
set "MINIMIND_VAD_MODEL_PATH=%ROOT%\model\vad\silero_vad.onnx"

echo [1/4] 校验 VAD/ASR 模型...
REM GPT-SoVITS runtime 是 Python 3.9，后端使用 3.10+ 的 X | Y 类型注解，
REM 需要 eval_type_backport 补丁。
"%GSV_PY%" -c "import eval_type_backport" 2>nul
if errorlevel 1 (
    echo       安装 eval_type_backport...
    "%GSV_PY%" -m pip install eval_type_backport
)
"%GSV_PY%" "%ROOT%\setup_assets.py" --verify
if errorlevel 1 (
    echo       检测到模型缺失，开始下载（首次约 5~15 分钟）...
    "%GSV_PY%" "%ROOT%\setup_assets.py"
    if errorlevel 1 (
        echo [错误] 模型下载失败，请查看上方日志。
        pause
        exit /b 1
    )
)

echo [2/4] 检查 GPT-SoVITS TTS 服务（端口 9880）...
powershell -NoProfile -Command "try{$c=New-Object System.Net.Sockets.TcpClient;$c.Connect('127.0.0.1',9880);$c.Close();exit 0}catch{exit 1}"
if not errorlevel 1 (
    echo       端口 9880 已在监听，跳过 TTS 启动
    goto tts_ready
)

echo       启动 GPT-SoVITS TTS 服务（新窗口）...
start "GPT-SoVITS TTS" /D "%GSV%" "%GSV_PY%" -I api_v2.py

echo       等待模型加载（首次约 30~120s）...
set "tts_try=0"
:tts_wait
powershell -NoProfile -Command "try{$c=New-Object System.Net.Sockets.TcpClient;$c.Connect('127.0.0.1',9880);$c.Close();exit 0}catch{exit 1}"
if not errorlevel 1 goto tts_done
set /a tts_try+=1
if %tts_try% GEQ 60 (
    echo [警告] 等待 3 分钟后仍未就绪，继续启动。
    echo        请查看 "GPT-SoVITS TTS" 窗口是否有报错。
    goto tts_done
)
ping -n 3 127.0.0.1 >nul
goto tts_wait
:tts_done
echo [TTS] 就绪（重试次数: %tts_try%）

:tts_ready
echo [3/4] 检查前端依赖...
where npm >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 npm，请先安装 Node.js。
    pause
    exit /b 1
)
if not exist "%ROOT%\electron-app\node_modules" (
    echo       首次运行，安装前端依赖...
    set "ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/"
    call npm install --prefix "%ROOT%\electron-app" --registry=https://registry.npmmirror.com
    if errorlevel 1 (
        echo [错误] npm install 失败，请检查网络后重试。
        pause
        exit /b 1
    )
)

echo [4/4] 启动 CommunicationPet...
echo       启动后点击绿色"开始语音"按钮；首次使用请在设置里填入自己的 LLM API Key。
set "PYTHONUTF8=1"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\ensure_electron.ps1"
if errorlevel 1 (
    echo [错误] Electron 二进制缺失且自动补装失败。
    pause
    exit /b 1
)

call npm run dev --prefix "%ROOT%\electron-app"
pause
