const { app, BrowserWindow, ipcMain, shell, dialog } = require("electron");
const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const BACKEND_HTTP_BASE = process.env.MINIMIND_BACKEND_HTTP_BASE || "http://127.0.0.1:8011";
const BACKEND_WS_URL = process.env.MINIMIND_BACKEND_WS_URL || "ws://127.0.0.1:8011/ws/realtime";
const LLAMA_HTTP_BASE = process.env.MINIMIND_LLM_BASE_URL || "http://127.0.0.1:8080";
const MEMORY_ENABLED = !/^(0|false|no)$/i.test(process.env.MINIMIND_MEMORY_ENABLED || "0");
const MEMORY_EMBEDDER_HTTP_BASE = process.env.MINIMIND_MEMORY_EMBEDDER_BASE_URL || "http://127.0.0.1:8081";
const DEV_REPO_ROOT = path.resolve(__dirname, "..");
const PACKAGED_BUNDLE_ROOT = path.join(process.resourcesPath, "app-bundle");
const PACKAGED_APP_ROOT = path.dirname(process.execPath);
const REPO_ROOT = app.isPackaged ? PACKAGED_BUNDLE_ROOT : DEV_REPO_ROOT;
const APP_ROOT = app.isPackaged ? PACKAGED_APP_ROOT : DEV_REPO_ROOT;
const LLM_MODULE_ROOT = path.join(REPO_ROOT, "llm-module");
const DEFAULT_VAD_MODEL_PATH = path.join(REPO_ROOT, "model", "vad", "silero_vad.onnx");
const configuredStartTimeoutMs = Number(process.env.MINIMIND_SERVICE_START_TIMEOUT_MS || 180000);
const DEFAULT_START_TIMEOUT_MS = Number.isFinite(configuredStartTimeoutMs) ? configuredStartTimeoutMs : 180000;
const WINDOW_ICON_PATH = path.join(__dirname, "build", "icon.ico");

// 化身（可选 DLC）：一个独立的本机 Node 服务，提供 VRM 渲染、骨骼解算、表情与口型。
// 主仓库不包含化身服务与 VRM 模型，默认关闭；需要时把服务放到 vendor/avatar，
// 或设置 MINIMIND_HAST_ROOT 指向服务目录，并设置 MINIMIND_HAST_ENABLED=1。
// 动作意图由后端 LLM 产出（见 realtime/motion_intent.py）。
const HAST_PORT = Number(process.env.MINIMIND_HAST_PORT || 8769);
const HAST_HTTP_BASE = `http://127.0.0.1:${HAST_PORT}`;
// embed=1 让页面隐藏自带聊天界面、关闭自主聊天与浏览器朗读，只当渲染窗口用。
// 模型 key 与化身服务的 configs/avatars.json 保持一致。
const AVATAR_MODELS = Object.freeze([
  { key: "red", label: "Avatar · Red" },
  { key: "blue", label: "Avatar · Blue" }
]);
const DEFAULT_AVATAR_MODEL = "red";
let selectedAvatarModel = null;
const HAST_ENABLED = /^(1|true|yes)$/i.test(process.env.MINIMIND_HAST_ENABLED || "0");

const CONFIG_DIR = path.join(app.getPath("userData"), "config");
const CONFIG_FILE = path.join(CONFIG_DIR, "app-config.json");

// Assets live in a single root that can be overridden at runtime. Resolution order:
//   1. MINIMIND_ASSETS_ROOT env var (explicit override, highest priority)
//   2. the folder chosen in Settings and stored in app-config.json (if it still exists)
//   3. the bundled/default "assets" folder next to the app
// Everything below is resolved on demand so a Settings change takes effect without a
// full restart (running services still need a restart to pick up new paths).
function getDefaultAssetsRoot() {
  return path.join(APP_ROOT, "assets");
}

function getAssetsRoot() {
  if (process.env.MINIMIND_ASSETS_ROOT) {
    return process.env.MINIMIND_ASSETS_ROOT;
  }
  const configured = loadAssetsRoot();
  if (configured && pathExists(configured, "directory")) {
    return configured;
  }
  return getDefaultAssetsRoot();
}

// Report where the active assets root came from, so the UI can explain the
// current state ("env" / "config" / "default") without duplicating the logic.
function getAssetsRootSource() {
  if (process.env.MINIMIND_ASSETS_ROOT) {
    return "env";
  }
  const configured = loadAssetsRoot();
  if (configured && pathExists(configured, "directory")) {
    return "config";
  }
  return "default";
}

function getLlamaServerPath() {
  if (process.env.MINIMIND_LLAMA_SERVER) {
    return process.env.MINIMIND_LLAMA_SERVER;
  }
  return path.join(getAssetsRoot(), "llama-bin", process.platform === "win32" ? "llama-server.exe" : "llama-server");
}

function getLlmModelsRoot() {
  return path.join(getAssetsRoot(), "llm");
}

// 化身服务（可选 DLC）。解析顺序：
//   1. MINIMIND_HAST_ROOT 环境变量
//   2. <repo>/vendor/avatar
//   3. 与仓库同级的 HAST-SEC-main（兼容旧布局）
function getHastRoot() {
  if (process.env.MINIMIND_HAST_ROOT) {
    return process.env.MINIMIND_HAST_ROOT;
  }
  const vendorRoot = path.join(REPO_ROOT, "vendor", "avatar");
  if (fs.existsSync(vendorRoot)) {
    return vendorRoot;
  }
  return path.join(REPO_ROOT, "..", "HAST-SEC-main");
}

function resolveNodeExecutable() {
  return process.env.MINIMIND_NODE || "node";
}

function getMemoryModelsRoot() {
  return path.join(getAssetsRoot(), "embedding");
}

// GPT-SoVITS 是外部 TTS 服务。解析顺序：
//   1. MINIMIND_GSV_ROOT 环境变量
//   2. <repo>/vendor/GPT-SoVITS
//   3. 与仓库同级的 VoiceChat/GPT-SoVITS-*（兼容旧布局）
// GSV 安装目录名带版本日期，按前缀扫描以适应升级。
function getGsvRoot() {
  if (process.env.MINIMIND_GSV_ROOT && fs.existsSync(process.env.MINIMIND_GSV_ROOT)) {
    return process.env.MINIMIND_GSV_ROOT;
  }
  const vendorRoot = path.join(REPO_ROOT, "vendor", "GPT-SoVITS");
  if (fs.existsSync(vendorRoot)) {
    return vendorRoot;
  }
  const voicechat = path.join(REPO_ROOT, "..", "VoiceChat");
  if (fs.existsSync(path.join(voicechat))) {
    try {
      const candidates = fs
        .readdirSync(voicechat)
        .filter((name) => name.startsWith("GPT-SoVITS-v2pro-"))
        .sort()
        .reverse();
      for (const name of candidates) {
        const py = path.join(voicechat, name, "runtime", "python.exe");
        if (fs.existsSync(py)) {
          return path.join(voicechat, name);
        }
      }
    } catch (_) {
      // ignore readdir errors
    }
  }
  return path.join(voicechat, "GPT-SoVITS-v2pro-20250604");
}

const PARAFORMER_REL = path.join(
  "tools", "asr", "models",
  "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
);

function getAsrModelPath() {
  if (process.env.MINIMIND_ASR_MODEL_PATH) {
    return process.env.MINIMIND_ASR_MODEL_PATH;
  }
  return path.join(getGsvRoot(), PARAFORMER_REL);
}

function getTtsModelPathCandidates() {
  const assetsRoot = getAssetsRoot();
  return [
    path.join(assetsRoot, "tts", "openbmb__VoxCPM2"),
    path.join(assetsRoot, "tts", "openbmb__VoxCPM1.5")
  ];
}

function getDefaultTtsModelPath() {
  const candidates = getTtsModelPathCandidates();
  return candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0];
}

function getLoraRoot() {
  return process.env.MINIMIND_TTS_LORA_ROOT || path.join(getAssetsRoot(), "lora");
}

const DEFAULT_LLM_CONFIG = {
  // Any OpenAI-compatible chat-completions API works. The default points at
  // DeepSeek, but the key is intentionally empty: enter your own key in
  // Settings (or set MINIMIND_LLM_API_KEY) before the first conversation.
  mode: "api",
  modelPath: null,
  // 话痨程度档位 key (minimal | concise | normal | talkative | chatty)。
  // 仅这一个字段用专用 IPC (llm:set-talkativeness) 更新，避免触碰 mode 合并逻辑。
  talkativeness: "normal",
  api: {
    baseUrl: "https://api.deepseek.com", // /v1 is appended downstream by the LLM client
    apiKey: "",
    model: "deepseek-chat"
  }
};

const DEFAULT_TTS_CONFIG = {
  // GPT-SoVITS voice cloning is driven entirely by the reference audio + prompt
  // text of the active persona. The VoxCPM-era fields below are inert and kept
  // only so older saved configs / renderer code don't break on read.
  model_version: "1.5",
  lora_selection: null,
  prompt_audio_path: null,
  prompt_text: null,
  cfg_value: 2.0,
  inference_timesteps: 10,
  seed: -1
};

const DEFAULT_PERSONA_CONFIG = {
  id: "niya" // active persona; personas live in <repo>/personas/<id>/{SKILL.md,voice.json}
};

function ensureConfigDir() {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
}

function loadConfig() {
  try {
    ensureConfigDir();
    if (fs.existsSync(CONFIG_FILE)) {
      const data = fs.readFileSync(CONFIG_FILE, "utf8");
      return JSON.parse(data);
    }
  } catch (error) {
    console.error("Failed to load config:", error);
  }
  return { llm: { ...DEFAULT_LLM_CONFIG }, tts: { ...DEFAULT_TTS_CONFIG }, persona: { ...DEFAULT_PERSONA_CONFIG } };
}

function getPersonasDir() {
  return path.join(REPO_ROOT, "personas");
}

// Enumerate persona folders (each has SKILL.md + voice.json) so the renderer
// can populate the persona switcher without parsing SKILL.md on the main side.
function listPersonas() {
  const dir = getPersonasDir();
  if (!fs.existsSync(dir)) {
    return [];
  }
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const personas = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const voiceFile = path.join(dir, entry.name, "voice.json");
    const skillFile = path.join(dir, entry.name, "SKILL.md");
    if (!fs.existsSync(voiceFile) || !fs.existsSync(skillFile)) {
      continue;
    }
    try {
      const voice = JSON.parse(fs.readFileSync(voiceFile, "utf8"));
      personas.push({
        id: String(voice.id || entry.name),
        name: String(voice.name || entry.name),
        source: String(voice.source || "")
      });
    } catch (error) {
      console.error(`Failed to read persona ${entry.name}:`, error);
    }
  }
  return personas;
}

function loadPersonaConfig() {
  const config = loadConfig();
  return { ...DEFAULT_PERSONA_CONFIG, ...(config.persona || {}) };
}

function savePersonaConfig(personaConfig) {
  const config = loadConfig();
  config.persona = personaConfig;
  return saveConfig(config);
}

function saveConfig(config) {
  try {
    ensureConfigDir();
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), "utf8");
    return true;
  } catch (error) {
    console.error("Failed to save config:", error);
    return false;
  }
}

function getSelectedAvatarModel() {
  if (selectedAvatarModel) {
    return selectedAvatarModel;
  }
  const stored = loadConfig().avatarModel;
  selectedAvatarModel = AVATAR_MODELS.some((model) => model.key === stored)
    ? stored
    : DEFAULT_AVATAR_MODEL;
  return selectedAvatarModel;
}

function setSelectedAvatarModel(key) {
  const model = AVATAR_MODELS.find((item) => item.key === key);
  if (!model) {
    throw new Error(`未知的角色模型: ${key}`);
  }
  selectedAvatarModel = model.key;
  const config = loadConfig();
  config.avatarModel = model.key;
  saveConfig(config);
  return model.key;
}

function getAvatarUrl() {
  return `${HAST_HTTP_BASE}/language-lab?embed=1&model=${encodeURIComponent(getSelectedAvatarModel())}`;
}

// 话痨程度档位白名单，与后端 TALKATIVENESS_PROFILES 对齐。
const TALKATIVENESS_LEVELS = new Set(["minimal", "concise", "normal", "talkative", "chatty"]);

function loadLlmConfig() {
  const config = loadConfig();
  const stored = config.llm || {};
  // Default to "api" mode: the local llama-server + GGUF stack was removed
  // in this build, so only honor "local" when the user explicitly chose it
  // (and has actually placed a llama-server binary). Anything else — first run
  // with no config, an older config that only had modelPath, an unknown value
  // — falls back to "api" so the realtime backend can start at all.
  const merged = {
    ...DEFAULT_LLM_CONFIG,
    ...stored,
    mode: stored.mode === "local" ? "local" : "api",
    // 话痨程度必须落在已知档位内，否则回退到默认「普通」。
    talkativeness: TALKATIVENESS_LEVELS.has(stored.talkativeness) ? stored.talkativeness : "normal",
    api: { ...DEFAULT_LLM_CONFIG.api, ...(stored.api || {}) }
  };
  // Environment variables win over the saved config, so CI / power users can
  // start the app without writing an API key into the Electron user-data dir.
  if (process.env.MINIMIND_LLM_BASE_URL) {
    merged.api.baseUrl = process.env.MINIMIND_LLM_BASE_URL;
  }
  if (process.env.MINIMIND_LLM_API_KEY) {
    merged.api.apiKey = process.env.MINIMIND_LLM_API_KEY;
  }
  if (process.env.MINIMIND_LLM_MODEL) {
    merged.api.model = process.env.MINIMIND_LLM_MODEL;
  }
  return merged;
}

function saveLlmConfig(llmConfig) {
  const config = loadConfig();
  config.llm = llmConfig;
  return saveConfig(config);
}

// 仅更新话痨程度档位，保留其余 LLM 配置（mode/api/modelPath）不变，避免
// llm:set-config 的 mode 合并逻辑把当前模式误改成 local。
function saveLlmTalkativeness(level) {
  const config = loadConfig();
  const stored = config.llm && typeof config.llm === "object" ? config.llm : { ...DEFAULT_LLM_CONFIG };
  stored.talkativeness = TALKATIVENESS_LEVELS.has(level) ? level : "normal";
  config.llm = stored;
  return saveConfig(config);
}

function loadTtsConfig() {
  const config = loadConfig();
  return config.tts || { ...DEFAULT_TTS_CONFIG };
}

function saveTtsConfig(ttsConfig) {
  const config = loadConfig();
  config.tts = ttsConfig;
  return saveConfig(config);
}

function loadAssetsRoot() {
  const config = loadConfig();
  const stored = config.assetsRoot;
  return typeof stored === "string" && stored.trim() ? stored.trim() : null;
}

function saveAssetsRoot(assetsRoot) {
  const config = loadConfig();
  if (assetsRoot && String(assetsRoot).trim()) {
    config.assetsRoot = String(assetsRoot).trim();
  } else {
    delete config.assetsRoot;
  }
  return saveConfig(config);
}

// Editable Markdown mirror of the conversation history + long-term memory.
// SQLite stays the source of truth; this folder is what the user reads/edits.
// Defaults to <repo>/runtime/memory_vault but can point at any folder (e.g. an
// Obsidian vault). Takes effect the next time the realtime backend starts.
function getDefaultHistoryVaultRoot() {
  return path.join(REPO_ROOT, "runtime", "memory_vault");
}

function loadHistoryVaultRoot() {
  const config = loadConfig();
  const stored = config.historyVaultRoot;
  return typeof stored === "string" && stored.trim() ? stored.trim() : null;
}

function saveHistoryVaultRoot(vaultRoot) {
  const config = loadConfig();
  if (vaultRoot && String(vaultRoot).trim()) {
    config.historyVaultRoot = String(vaultRoot).trim();
  } else {
    delete config.historyVaultRoot;
  }
  return saveConfig(config);
}

function getHistoryVaultRoot() {
  return process.env.MINIMIND_VAULT_PATH || loadHistoryVaultRoot() || getDefaultHistoryVaultRoot();
}

// SQLite file holding the authoritative conversation/memory records. The UI only
// picks a folder; the file name stays fixed so a chosen folder is unambiguous.
const HISTORY_DB_FILENAME = "whispera.db";

function getDefaultHistoryDbPath() {
  return path.join(REPO_ROOT, "runtime", "data", HISTORY_DB_FILENAME);
}

function loadHistoryDbPath() {
  const config = loadConfig();
  const stored = config.historyDbPath;
  return typeof stored === "string" && stored.trim() ? stored.trim() : null;
}

function saveHistoryDbPath(dbPath) {
  const config = loadConfig();
  if (dbPath && String(dbPath).trim()) {
    config.historyDbPath = String(dbPath).trim();
  } else {
    delete config.historyDbPath;
  }
  return saveConfig(config);
}

function getHistoryDbPath() {
  return process.env.MINIMIND_DB_PATH || loadHistoryDbPath() || getDefaultHistoryDbPath();
}

function pathExists(targetPath, kind = "any") {
  try {
    const stats = fs.statSync(targetPath);
    if (kind === "file") {
      return stats.isFile();
    }
    if (kind === "directory") {
      return stats.isDirectory();
    }
    return true;
  } catch (_) {
    return false;
  }
}

function listGgufModels(rootDir) {
  if (!pathExists(rootDir, "directory")) {
    return [];
  }

  const matches = [];
  const stack = [rootDir];

  while (stack.length) {
    const currentDir = stack.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(currentDir, { withFileTypes: true });
    } catch (_) {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (entry.isFile() && entry.name.toLowerCase().endsWith(".gguf")) {
        matches.push(fullPath);
      }
    }
  }

  return matches.sort((a, b) => a.localeCompare(b));
}

function resolveLlmModelSelection() {
  const config = loadLlmConfig();
  const configuredPath = config.modelPath || null;
  const envPath = process.env.MINIMIND_LLM_MODEL || null;
  const ggufCandidates = listGgufModels(getLlmModelsRoot());

  if (configuredPath && pathExists(configuredPath, "file")) {
    return {
      configuredPath,
      resolvedPath: configuredPath,
      source: "config",
      candidates: ggufCandidates
    };
  }

  if (envPath && pathExists(envPath, "file")) {
    return {
      configuredPath,
      resolvedPath: envPath,
      source: "env",
      candidates: ggufCandidates
    };
  }

  if (ggufCandidates.length === 1) {
    return {
      configuredPath,
      resolvedPath: ggufCandidates[0],
      source: "assets.auto",
      candidates: ggufCandidates
    };
  }

  return {
    configuredPath,
    resolvedPath: null,
    source: ggufCandidates.length > 1 ? "assets.multiple" : "missing",
    candidates: ggufCandidates
  };
}

function resolveMemoryModelSelection() {
  const envPath = process.env.MINIMIND_MEMORY_EMBEDDER_MODEL_PATH || null;
  const ggufCandidates = listGgufModels(getMemoryModelsRoot());

  if (envPath && pathExists(envPath, "file")) {
    return {
      resolvedPath: envPath,
      source: "env",
      candidates: ggufCandidates
    };
  }

  if (ggufCandidates.length === 1) {
    return {
      resolvedPath: ggufCandidates[0],
      source: "assets.auto",
      candidates: ggufCandidates
    };
  }

  return {
    resolvedPath: null,
    source: ggufCandidates.length > 1 ? "assets.multiple" : "missing",
    candidates: ggufCandidates
  };
}

function getResourceStatus() {
  const llmSelection = resolveLlmModelSelection();
  const memorySelection = resolveMemoryModelSelection();
  const llamaServerPath = process.env.MINIMIND_LLAMA_SERVER || getLlamaServerPath();
  const asrModelPath = process.env.MINIMIND_ASR_MODEL_PATH || getAsrModelPath();
  const vadModelPath = process.env.MINIMIND_VAD_MODEL_PATH || DEFAULT_VAD_MODEL_PATH;
  const ttsModelPath = process.env.MINIMIND_TTS_MODEL_PATH || getDefaultTtsModelPath();
  const assetsRoot = getAssetsRoot();

  return {
    appRoot: APP_ROOT,
    assetsRoot,
    assetsRootSource: getAssetsRootSource(),
    assetsRootExists: pathExists(assetsRoot, "directory"),
    llama: {
      path: llamaServerPath,
      exists: pathExists(llamaServerPath, "file")
    },
    llm: {
      directory: getLlmModelsRoot(),
      configuredPath: llmSelection.configuredPath,
      resolvedPath: llmSelection.resolvedPath,
      exists: Boolean(llmSelection.resolvedPath && pathExists(llmSelection.resolvedPath, "file")),
      source: llmSelection.source,
      candidateCount: llmSelection.candidates.length
    },
    memory: {
      enabled: MEMORY_ENABLED,
      directory: getMemoryModelsRoot(),
      embedderHttpBase: MEMORY_EMBEDDER_HTTP_BASE,
      resolvedPath: memorySelection.resolvedPath,
      exists: !MEMORY_ENABLED || Boolean(memorySelection.resolvedPath && pathExists(memorySelection.resolvedPath, "file")),
      source: memorySelection.source,
      candidateCount: memorySelection.candidates.length
    },
    asr: {
      path: asrModelPath,
      exists: pathExists(asrModelPath, "directory")
    },
    vad: {
      path: vadModelPath,
      exists: pathExists(vadModelPath, "file")
    },
    tts: {
      path: ttsModelPath,
      exists: pathExists(ttsModelPath, "directory")
    },
    lora: {
      path: getLoraRoot(),
      exists: pathExists(getLoraRoot(), "directory")
    },
    summary: {
      llmReady: pathExists(llamaServerPath, "file") && Boolean(llmSelection.resolvedPath && pathExists(llmSelection.resolvedPath, "file")),
      memoryReady: !MEMORY_ENABLED || Boolean(memorySelection.resolvedPath && pathExists(memorySelection.resolvedPath, "file")),
      asrReady: pathExists(asrModelPath, "directory"),
      vadReady: pathExists(vadModelPath, "file"),
      ttsReady: pathExists(ttsModelPath, "directory")
    }
  };
}

function getPathInfo(targetPath) {
  if (!targetPath) {
    return {
      exists: false,
      type: null,
      path: null,
    };
  }

  try {
    const resolvedPath = path.resolve(String(targetPath));
    const stats = fs.statSync(resolvedPath);
    return {
      exists: true,
      type: stats.isDirectory() ? "directory" : stats.isFile() ? "file" : "other",
      path: resolvedPath,
      sizeBytes: stats.size,
    };
  } catch (error) {
    return {
      exists: false,
      type: null,
      path: path.resolve(String(targetPath)),
      error: error.message,
    };
  }
}

function scanLoraDirectory() {
  const loraRoot = path.resolve(getLoraRoot());
  if (!fs.existsSync(loraRoot)) {
    return [];
  }

  const checkpoints = [];

  function scanDir(dir, relativePath = "") {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;

        const fullPath = path.join(dir, entry.name);
        const relPath = relativePath ? `${relativePath}/${entry.name}` : entry.name;

        const hasSafetensors = fs.existsSync(path.join(fullPath, "lora_weights.safetensors"));
        const hasCkpt = fs.existsSync(path.join(fullPath, "lora_weights.ckpt"));

        if (hasSafetensors || hasCkpt) {
          let baseModel = null;
          const configPath = path.join(fullPath, "lora_config.json");
          if (fs.existsSync(configPath)) {
            try {
              const configData = JSON.parse(fs.readFileSync(configPath, "utf8"));
              baseModel = configData.base_model || null;
            } catch (e) {
              // Ignore config read errors
            }
          }
          checkpoints.push({
            path: relPath,
            label: relPath,
            base_model: baseModel
          });
        }

        scanDir(fullPath, relPath);
      }
    } catch (e) {
      // Ignore directory read errors
    }
  }

  scanDir(loraRoot);
  return checkpoints.sort((a, b) => (b.path || "").localeCompare(a.path || ""));
}

let mainWindow = null;
// 化身现在是主窗口内嵌的 <webview>（guest WebContents），不再是独立窗口。
// 由 mainWindow 的 did-attach-webview 捕获，页面侧自行创建与显隐。
let avatarGuest = null;
let avatarReady = false;
// 化身服务是否已经在线（自己拉起的或外部已在跑的）。渲染层的化身按钮据此启用，
// 避免服务没起来时点开一个错误页。
let hastAvailable = false;
let serviceStartPromise = null;
let serviceStopPromise = null;

const managedServices = {
  llama: null,
  memory: null,
  realtime: null,
  hast: null
};
const serviceLogFiles = {
  llama: null,
  memory: null,
  realtime: null,
  hast: null
};
const serviceLogStreams = {
  llama: null,
  memory: null,
  realtime: null,
  hast: null
};
const serviceLaunchInfo = {
  llama: null,
  memory: null,
  realtime: null,
  hast: null
};
const serviceExits = {
  llama: null,
  memory: null,
  realtime: null,
  hast: null
};
const serviceLogs = {
  llama: [],
  memory: [],
  realtime: [],
  hast: []
};
// 默认开启：把 spawn 出来的子进程 stdout/stderr 落盘到 service-logs，
// 这样 backend 启动崩溃时能在 <userData>/service-logs/<ts>-realtime.log 直接看到 traceback，
// 而不是只进内存（waitForReady 抛错时仅带最近 30 行，且用户无感）。
let runtimeLoggingEnabled = true;

function parseHttpEndpoint(url, fallbackHost, fallbackPort) {
  try {
    const parsed = new URL(url);
    return {
      host: parsed.hostname || fallbackHost,
      port: Number(parsed.port || fallbackPort),
      origin: parsed.origin
    };
  } catch (_) {
    return { host: fallbackHost, port: fallbackPort, origin: `http://${fallbackHost}:${fallbackPort}` };
  }
}

const llamaEndpoint = parseHttpEndpoint(LLAMA_HTTP_BASE, "127.0.0.1", 8080);
const memoryEndpoint = parseHttpEndpoint(MEMORY_EMBEDDER_HTTP_BASE, "127.0.0.1", 8081);
const backendEndpoint = parseHttpEndpoint(BACKEND_HTTP_BASE, "127.0.0.1", 8011);

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getServiceLogDir() {
  const logDir = path.join(app.getPath("userData"), "service-logs");
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  return logDir;
}

function getTurnCaptureDir() {
  const captureDir = path.join(getServiceLogDir(), "session-captures");
  if (!fs.existsSync(captureDir)) {
    fs.mkdirSync(captureDir, { recursive: true });
  }
  return captureDir;
}

function makeLogTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function closeServiceLog(name) {
  if (serviceLogStreams[name]) {
    serviceLogStreams[name].end();
    serviceLogStreams[name] = null;
  }
}

function openServiceLog(name, command, args, cwd) {
  closeServiceLog(name);
  const logFile = path.join(getServiceLogDir(), `${makeLogTimestamp()}-${name}.log`);
  const stream = fs.createWriteStream(logFile, { flags: "a", encoding: "utf8" });
  serviceLogFiles[name] = logFile;
  serviceLogStreams[name] = stream;
  stream.write(`[${new Date().toISOString()}] ${name} start\n`);
  stream.write(`cwd: ${cwd}\n`);
  stream.write(`command: ${command} ${args.join(" ")}\n\n`);
  return logFile;
}

function ensureServiceLog(name) {
  if (!runtimeLoggingEnabled) {
    return null;
  }
  if (serviceLogStreams[name]) {
    return serviceLogFiles[name];
  }
  const launchInfo = serviceLaunchInfo[name];
  if (!launchInfo) {
    return null;
  }
  return openServiceLog(name, launchInfo.command, launchInfo.args, launchInfo.cwd);
}

function writeServiceLog(name, level, text) {
  if (!runtimeLoggingEnabled || !text) {
    return;
  }
  ensureServiceLog(name);
  const stream = serviceLogStreams[name];
  if (!stream) {
    return;
  }
  for (const line of text.replace(/\r\n/g, "\n").split("\n")) {
    if (line) {
      stream.write(`[${new Date().toISOString()}] [${level}] ${line}\n`);
    }
  }
}

function sendServiceEvent(type, payload = {}) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  mainWindow.webContents.send("voice-services:event", {
    type,
    at: new Date().toISOString(),
    ...payload
  });
}

function getVoiceLoggingState() {
  return {
    enabled: runtimeLoggingEnabled,
    logsDir: getServiceLogDir(),
    turnCaptureDir: getTurnCaptureDir(),
  };
}

function setVoiceLoggingEnabled(enabled) {
  runtimeLoggingEnabled = Boolean(enabled);
  if (!runtimeLoggingEnabled) {
    closeServiceLog("llama");
    closeServiceLog("memory");
    closeServiceLog("realtime");
  } else {
    if (isChildRunning(managedServices.llama)) {
      ensureServiceLog("llama");
    }
    if (isChildRunning(managedServices.memory)) {
      ensureServiceLog("memory");
    }
    if (isChildRunning(managedServices.realtime)) {
      ensureServiceLog("realtime");
    }
  }
  sendServiceEvent("logging.capture_changed", getVoiceLoggingState());
  return getVoiceLoggingState();
}

// 按优先级解析 Python 可执行文件。每一项都要 pathExists 校验后再返回 ——
// 之前 MINIMIND_PYTHON 不校验存在性，一旦 bat 的环境变量在 npm→electron 传递链中
// 丢失或路径写错，就会直接返回一个不存在的路径，spawn 立即 ENOENT 崩溃，
// 而前端会把这个崩溃当成 "Service Start Failed"。同时打印实际选中的来源，
// 让用户在终端一眼看到 backend 到底用的是哪个 python（GSV runtime / conda / 系统）。
function resolvePythonExecutable() {
  const candidates = [];
  if (process.env.MINIMIND_PYTHON) {
    candidates.push({ label: "MINIMIND_PYTHON", value: process.env.MINIMIND_PYTHON });
  }
  const bundledPython = process.platform === "win32"
    ? path.join(REPO_ROOT, "runtime", "python", "python.exe")
    : path.join(REPO_ROOT, "runtime", "python", "bin", "python3");
  candidates.push({ label: "bundled", value: bundledPython });
  if (process.env.CONDA_PREFIX) {
    candidates.push({
      label: "CONDA_PREFIX",
      value: path.join(process.env.CONDA_PREFIX, process.platform === "win32" ? "python.exe" : "bin/python")
    });
  }
  if (process.env.PYTHON) {
    candidates.push({ label: "PYTHON", value: process.env.PYTHON });
  }
  for (const c of candidates) {
    if (c.value && fs.existsSync(c.value)) {
      console.log(`[VOICE] python resolved via ${c.label}: ${c.value}`);
      return c.value;
    }
  }
  const fallback = process.env.PYTHON || "python";
  console.log(`[VOICE] python resolved via fallback (no candidate existed on disk): ${fallback}`);
  return fallback;
}

function checkMem0RuntimeReady(pythonExecutable) {
  if (!MEMORY_ENABLED) {
    return;
  }

  const result = spawnSync(
    pythonExecutable,
    [
      "-X",
      "utf8",
      "-c",
      "from mem0 import Memory; import openai, posthog, qdrant_client, sqlalchemy; print(Memory.__name__)"
    ],
    {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
      windowsHide: true,
      encoding: "utf8",
    }
  );

  if (result.error) {
    throw new Error(`Memory dependency check failed for ${pythonExecutable}: ${result.error.message}`);
  }

  if (result.status !== 0) {
    const detail = String(result.stderr || result.stdout || `exit code ${result.status}`).trim();
    throw new Error(
      `Memory is enabled but mem0 dependencies are unavailable in ${pythonExecutable}. ` +
      `Run: ${pythonExecutable} -m pip install -r requirements-mem0.txt. ` +
      `Details: ${detail.slice(-500)}`
    );
  }
}

function isChildRunning(child) {
  return Boolean(child && child.exitCode === null && !child.killed);
}

function attachProcessEvents(name, child) {
  const pipeLog = (stream, level) => {
    stream.on("data", (chunk) => {
      const rawText = chunk.toString("utf8");
      writeServiceLog(name, level, rawText);
      const text = rawText.trim();
      if (text) {
        // Backend diagnostic channel. Two kinds of lines are forwarded to the
        // main-process terminal so we can diagnose the realtime backend
        // independent of the service-log file (which is gated by
        // runtimeLoggingEnabled and may be turned off by the renderer at
        // startup). Sits alongside [VOICE]/[VOICE-FE].
        //   1. lines tagged [VOICE-BE] (explicit diagnostics from backend code)
        //   2. ALL stderr lines (so startup tracebacks / import errors are
        //      always visible even when logging is disabled)
        for (const line of text.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed.startsWith("[VOICE-BE]")) {
            console.log(`${trimmed}`);
          } else if (level === "stderr") {
            console.log(`[VOICE-BE:${name}] ${trimmed}`);
          }
        }
        serviceLogs[name].push(`[${level}] ${text}`);
        serviceLogs[name] = serviceLogs[name].slice(-30);
        if (runtimeLoggingEnabled) {
          sendServiceEvent("process.output", {
            service: name,
            level,
            text: text.slice(-4000),
            logFile: serviceLogFiles[name]
          });
        }
      }
    });
  };

  pipeLog(child.stdout, "stdout");
  pipeLog(child.stderr, "stderr");

  child.on("exit", (code, signal) => {
    serviceExits[name] = {
      code,
      signal,
      logs: [...serviceLogs[name]]
    };
    sendServiceEvent("process.exit", { service: name, code, signal });
    closeServiceLog(name);
    if (managedServices[name] === child) {
      managedServices[name] = null;
    }
  });

  child.on("error", (error) => {
    serviceExits[name] = {
      code: null,
      signal: null,
      error: error.message,
      logs: [...serviceLogs[name]]
    };
    sendServiceEvent("process.error", { service: name, message: error.message });
    closeServiceLog(name);
  });
}

async function fetchJson(url, timeoutMs = 3000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      return { ok: false, status: response.status, data: null };
    }
    return { ok: true, status: response.status, data: await response.json() };
  } catch (error) {
    return { ok: false, status: 0, data: null, error: error.message };
  } finally {
    clearTimeout(timeout);
  }
}

async function postJson(url, payload, timeoutMs = DEFAULT_START_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload ?? {}),
      signal: controller.signal
    });
    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      data = null;
    }
    return { ok: response.ok, status: response.status, data };
  } catch (error) {
    return { ok: false, status: 0, data: null, error: error.message };
  } finally {
    clearTimeout(timeout);
  }
}

async function checkLlamaReady() {
  const result = await fetchJson(`${llamaEndpoint.origin}/v1/models`);
  return result.ok ? result : null;
}

async function checkMemoryReady() {
  if (!MEMORY_ENABLED) {
    return { ok: true, status: 200, data: { disabled: true } };
  }
  const result = await fetchJson(`${memoryEndpoint.origin}/v1/models`);
  return result.ok ? result : null;
}

async function checkRealtimeReady() {
  const result = await fetchJson(`${backendEndpoint.origin}/health`);
  if (!result.ok) {
    return null;
  }
  if (result.data && result.data.upstream_ok === false) {
    return null;
  }
  return result;
}

async function checkHastReady() {
  const result = await fetchJson(`${HAST_HTTP_BASE}/v13-api/status`);
  if (!result.ok || !result.data || !result.data.runtime) {
    return null;
  }
  return result;
}

// 旧版化身服务的路由表在启动时就固定了，后来新增的模型文件不会出现在
// 旧进程里。这里确认服务不仅在线，而且确实提供当前版本的两套模型。
async function checkHastAvatarLibrary() {
  const result = await fetchJson(`${HAST_HTTP_BASE}/configs/avatars.json`);
  const models = result.ok && result.data && result.data.models;
  return Boolean(models && AVATAR_MODELS.every((model) => models[model.key]));
}

function findListeningPid(port) {
  if (process.platform === "win32") {
    const result = spawnSync("netstat", ["-ano", "-p", "tcp"], { encoding: "utf8", windowsHide: true });
    if (result.status !== 0 || !result.stdout) {
      return null;
    }
    const match = result.stdout.match(new RegExp(`:${port}\\s+\\S+\\s+LISTENING\\s+(\\d+)`));
    return match ? Number(match[1]) : null;
  }
  const result = spawnSync("lsof", ["-ti", `tcp:${port}`], { encoding: "utf8" });
  if (result.status !== 0 || !result.stdout) {
    return null;
  }
  const pid = Number(result.stdout.trim().split(/\s+/)[0]);
  return Number.isInteger(pid) && pid > 0 ? pid : null;
}

// 只在前一步已经确认「该端口上确实是 HAST 服务」之后调用，避免误杀其它程序。
async function stopStaleHastService() {
  const pid = findListeningPid(HAST_PORT);
  if (!pid) {
    return false;
  }
  console.warn(`[AVATAR] stale HAST service on port ${HAST_PORT} (pid ${pid}); restarting`);
  if (process.platform === "win32") {
    spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
  } else {
    try {
      process.kill(pid, "SIGTERM");
    } catch (_) {
      // already gone
    }
  }
  for (let i = 0; i < 20; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    if (!(await checkHastReady())) {
      return true;
    }
  }
  if (process.platform !== "win32") {
    try {
      process.kill(pid, "SIGKILL");
    } catch (_) {
      // ignore
    }
  }
  return false;
}

async function warmupRealtimeModels(ttsOptions = null) {
  sendServiceEvent("warmup.begin", { service: "realtime", stage: "models", url: `${backendEndpoint.origin}/warmup` });
  const result = await postJson(`${backendEndpoint.origin}/warmup`, {
    force: false,
    tts_options: ttsOptions || undefined,
  });
  if (!result.ok || !result.data || result.data.ok === false) {
    const detail = result.data && (result.data.detail || result.data.error);
    const errorMessage = detail || result.error || `warmup failed with status ${result.status}`;
    sendServiceEvent("warmup.error", { service: "realtime", stage: "models", message: errorMessage });
    throw new Error(errorMessage);
  }
  sendServiceEvent("warmup.completed", {
    service: "realtime",
    stage: "models",
    cached: Boolean(result.data.cached),
    asr_model_load_ms: result.data.asr_model_load_ms,
    asr_inference_ms: result.data.asr_inference_ms,
    tts_model_load_ms: result.data.tts_model_load_ms,
    tts_inference_ms: result.data.tts_inference_ms,
    total_ms: result.data.total_ms,
  });
  return result.data;
}

function formatRecentServiceLogs(name) {
  const logs = serviceLogs[name] || [];
  if (!logs.length) {
    return "";
  }
  return ` Recent ${name} logs: ${logs.slice(-8).join(" | ")}`;
}

async function waitForReady(service, name, checkReady, childGetter, timeoutMs = DEFAULT_START_TIMEOUT_MS) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < timeoutMs) {
    const exit = serviceExits[name];
    if (exit) {
      const exitDetail = exit.error || `code ${exit.code}${exit.signal ? `, signal ${exit.signal}` : ""}`;
      throw new Error(`${service} exited early with ${exitDetail}.${formatRecentServiceLogs(name)}`);
    }

    const child = childGetter();
    if (child && child.exitCode !== null) {
      throw new Error(`${service} exited early with code ${child.exitCode}.${formatRecentServiceLogs(name)}`);
    }

    const result = await checkReady();
    if (result) {
      return result;
    }

    lastError = `${service} is not ready yet`;
    await delay(1000);
  }

  throw new Error(`${lastError || service} after ${Math.round(timeoutMs / 1000)}s`);
}

function spawnManagedService(name, command, args, cwd, extraEnv = null) {
  serviceExits[name] = null;
  serviceLogs[name] = [];
  serviceLaunchInfo[name] = { command, args: [...args], cwd };
  const logFile = runtimeLoggingEnabled ? openServiceLog(name, command, args, cwd) : null;
  sendServiceEvent("process.start", { service: name, command, args, cwd, logFile });
  const childEnv = {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
    ...(extraEnv || {}),
  };
  const child = spawn(command, args, {
    cwd,
    env: childEnv,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"]
  });
  managedServices[name] = child;
  attachProcessEvents(name, child);
  return child;
}

function buildPythonInvocation(pythonExecutable, moduleArgs) {
  return {
    command: pythonExecutable,
    args: ["-X", "utf8", ...moduleArgs],
  };
}

function buildLlamaArgs(pythonExecutable) {
  const args = ["scripts/start_llama_server.py", "--host", llamaEndpoint.host, "--port", String(llamaEndpoint.port)];
  const llamaServerPath = getLlamaServerPath();
  if (!pathExists(llamaServerPath, "file")) {
    throw new Error(`llama-server executable not found: ${llamaServerPath}`);
  }
  args.push("--server", llamaServerPath);

  const llmSelection = resolveLlmModelSelection();
  const modelPath = llmSelection.resolvedPath;
  if (modelPath) {
    args.push("--model", modelPath);
  } else {
    throw new Error(`No GGUF model is available. Put one in ${getLlmModelsRoot()} or choose a local GGUF file in Settings.`);
  }

  const memoryInferEnabled = !/^(0|false|no)$/i.test(process.env.MINIMIND_MEMORY_INFER || "1");
  const defaultLlamaCtxSize = MEMORY_ENABLED && memoryInferEnabled ? "16384" : "8192";
  args.push("--ctx-size", process.env.MINIMIND_LLAMA_CTX_SIZE || defaultLlamaCtxSize);
  args.push("--n-gpu-layers", process.env.MINIMIND_LLAMA_N_GPU_LAYERS || "99");

  return buildPythonInvocation(pythonExecutable, args);
}

function buildMemoryEmbeddingArgs(pythonExecutable) {
  const args = ["scripts/start_llama_server.py", "--host", memoryEndpoint.host, "--port", String(memoryEndpoint.port)];
  const llamaServerPath = getLlamaServerPath();
  if (!pathExists(llamaServerPath, "file")) {
    throw new Error(`llama-server executable not found: ${llamaServerPath}`);
  }
  args.push("--server", llamaServerPath);

  const memorySelection = resolveMemoryModelSelection();
  const modelPath = memorySelection.resolvedPath;
  if (modelPath) {
    args.push("--model", modelPath);
  } else {
    throw new Error(`Memory is enabled but no embedding GGUF model is available. Set MINIMIND_MEMORY_EMBEDDER_MODEL_PATH or put one GGUF in ${getMemoryModelsRoot()}.`);
  }

  args.push("--ctx-size", process.env.MINIMIND_MEMORY_CTX_SIZE || "512");
  args.push("--n-gpu-layers", process.env.MINIMIND_MEMORY_N_GPU_LAYERS || "99");
  args.push("--extra", "--embedding", "--pooling", process.env.MINIMIND_MEMORY_POOLING || "mean");

  return buildPythonInvocation(pythonExecutable, args);
}

// Decide what LLM endpoint the realtime backend should talk to. In "local"
// mode it uses the bundled llama-server origin; in "api" mode it uses the
// user-entered base URL / model, and returns the api key so the caller can
// inject it into the child process env (never a CLI arg — keeps it out of logs).
function resolveRealtimeLlmOptions() {
  const llmConfig = loadLlmConfig();
  if (llmConfig.mode === "api") {
    const baseUrl = String(llmConfig.api.baseUrl || "").trim();
    const model = String(llmConfig.api.model || "").trim();
    const apiKey = String(llmConfig.api.apiKey || "").trim();
    if (!baseUrl) {
      throw new Error("API 模式已启用，但未填写请求地址。请在 LLM 设置中填写并应用配置。");
    }
    if (!model) {
      throw new Error("API 模式已启用，但未填写模型名。请在 LLM 设置中填写并应用配置。");
    }
    return { mode: "api", baseUrl: baseUrl.replace(/\/+$/, ""), model, apiKey };
  }
  return { mode: "local", baseUrl: llamaEndpoint.origin, model: null, apiKey: "" };
}

function buildRealtimeArgs(pythonExecutable, llmOptions) {
  const asrModelPath = getAsrModelPath();
  if (!pathExists(asrModelPath, "directory")) {
    throw new Error(`ASR model directory not found: ${asrModelPath}`);
  }
  const vadModelPath = process.env.MINIMIND_VAD_MODEL_PATH || DEFAULT_VAD_MODEL_PATH;
  if (!pathExists(vadModelPath, "file")) {
    throw new Error(`VAD model file not found: ${vadModelPath}`);
  }

  const args = [
    "-m",
    "realtime.app",
    "--host",
    backendEndpoint.host,
    "--port",
    String(backendEndpoint.port),
    "--asr-model-path",
    asrModelPath,
    "--asr-device",
    process.env.MINIMIND_ASR_DEVICE || "cuda",
    "--vad-path",
    vadModelPath,
    "--llm-base-url",
    llmOptions.baseUrl
  ];
  if (llmOptions.model) {
    args.push("--llm-model", llmOptions.model);
  }

  const disableTts = /^(1|true|yes)$/i.test(process.env.MINIMIND_DISABLE_TTS || "");
  // GPT-SoVITS is an external HTTP service (started separately by the launcher),
  // so TTS is on by default and no local VoxCPM model stack is required.
  const gptSovitsUrl = process.env.MINIMIND_GPTSOVITS_URL || "http://127.0.0.1:9880";
  const personasDir = getPersonasDir();
  const personaConfig = loadPersonaConfig();

  if (disableTts) {
    args.push("--disable-tts");
  } else {
    args.push("--enable-tts");
    args.push("--gpt-sovits-url", gptSovitsUrl);
    if (fs.existsSync(personasDir)) {
      args.push("--persona-root", personasDir);
    }
    if (personaConfig.id) {
      args.push("--persona", personaConfig.id);
    }
  }
  args.push("--debug-output-dir", getTurnCaptureDir());
  if (/^(1|true|yes)$/i.test(process.env.MINIMIND_DEBUG_TURNS || "")) {
    args.push("--debug-turns");
  }
  if (process.env.MINIMIND_DEBUG_OUTPUT_DIR) {
    args.push("--debug-output-dir", process.env.MINIMIND_DEBUG_OUTPUT_DIR);
  }
  if (MEMORY_ENABLED) {
    args.push("--enable-memory");
  }

  // Conversation history mirror root (SQLite lives under runtime/data).
  args.push("--history-vault-root", getHistoryVaultRoot());
  args.push("--history-db-path", getHistoryDbPath());

  const invocation = buildPythonInvocation(pythonExecutable, args);
  // API key travels via env, not argv, so it never appears in launch logs.
  invocation.env = llmOptions.apiKey ? { MINIMIND_LLM_API_KEY: llmOptions.apiKey } : null;
  return invocation;
}

// 拉起化身服务（可选 DLC）。它是个纯 Node HTTP 服务（无 WS、严格同源校验），
// 因此化身窗口必须直接导航到它的 origin，而不是从 file:// 去 fetch 它的 API。
async function startHastService() {
  const hastRoot = getHastRoot();
  if (!pathExists(path.join(hastRoot, "service", "start.mjs"), "file")) {
    throw new Error(`化身服务未找到: ${hastRoot}。请设置 MINIMIND_HAST_ROOT 指向化身服务目录。`);
  }
  if (!pathExists(path.join(hastRoot, "node_modules", "three"), "directory")) {
    throw new Error(`化身依赖未安装。请先在 ${hastRoot} 执行 npm ci。`);
  }

  const existing = await checkHastReady();
  if (existing) {
    // 若旧实例没有当前模型库（例如代码/资源更新后遗留的进程），重启它。
    // 否则页面会一直回退到默认的红色形态，表现就是「切蓝色没反应」。
    if (await checkHastAvatarLibrary()) {
      hastAvailable = true;
      sendServiceEvent("service.ready", { service: "hast", mode: "external", url: HAST_HTTP_BASE });
      return;
    }
    await stopStaleHastService();
    managedServices.hast = null;
  }

  if (!isChildRunning(managedServices.hast)) {
    spawnManagedService("hast", resolveNodeExecutable(), ["service/start.mjs"], hastRoot, {
      PORT: String(HAST_PORT),
    });
  }
  await waitForReady("avatar server", "hast", checkHastReady, () => managedServices.hast);
  hastAvailable = true;
  sendServiceEvent("service.ready", { service: "hast", mode: "managed", url: HAST_HTTP_BASE });
}

// 把消息转发给内嵌的化身页面。页面还没就绪时静默丢弃, 不阻塞对话主链路。
function sendToAvatar(channel, payload) {
  if (!avatarGuest || avatarGuest.isDestroyed()) {
    return false;
  }
  if (!avatarReady) {
    // 页面还在加载 VRM 与策略。这里必须留痕：否则"角色不动"会被误判成动作解析失败。
    console.warn(`[AVATAR] drop ${channel}: page not ready yet`);
    return false;
  }
  avatarGuest.send(channel, payload);
  return true;
}

async function startVoiceServices(ttsOptions = null) {
  if (serviceStartPromise) {
    return serviceStartPromise;
  }

  serviceStartPromise = (async () => {
    try {
      const pythonExecutable = resolvePythonExecutable();
      checkMem0RuntimeReady(pythonExecutable);
      sendServiceEvent("start.begin", { python: pythonExecutable });
      console.log(`[VOICE] start.begin python=${pythonExecutable}`);

      const llmOptions = resolveRealtimeLlmOptions();
      console.log(`[VOICE] llm mode=${llmOptions.mode} baseUrl=${llmOptions.baseUrl} model=${llmOptions.model}`);

      if (llmOptions.mode === "api") {
        // API mode: the realtime backend talks directly to an external
        // OpenAI-compatible endpoint, so no local llama-server is started.
        sendServiceEvent("service.ready", { service: "llama", mode: "api", url: llmOptions.baseUrl });
      } else {
        let llamaReady = await checkLlamaReady();
        if (llamaReady) {
          sendServiceEvent("service.ready", { service: "llama", mode: "external", url: llamaEndpoint.origin });
        } else {
          if (!isChildRunning(managedServices.llama)) {
            const { command, args } = buildLlamaArgs(pythonExecutable);
            spawnManagedService("llama", command, args, LLM_MODULE_ROOT);
          }
          llamaReady = await waitForReady("llama-server", "llama", checkLlamaReady, () => managedServices.llama);
          sendServiceEvent("service.ready", { service: "llama", mode: "managed", url: llamaEndpoint.origin });
        }
      }

      if (MEMORY_ENABLED) {
        let memoryReady = await checkMemoryReady();
        if (memoryReady) {
          sendServiceEvent("service.ready", { service: "memory", mode: "external", url: memoryEndpoint.origin });
        } else {
          if (!isChildRunning(managedServices.memory)) {
            const { command, args } = buildMemoryEmbeddingArgs(pythonExecutable);
            spawnManagedService("memory", command, args, LLM_MODULE_ROOT);
          }
          memoryReady = await waitForReady("memory embedding server", "memory", checkMemoryReady, () => managedServices.memory);
          sendServiceEvent("service.ready", { service: "memory", mode: "managed", url: memoryEndpoint.origin });
        }
      }

      if (HAST_ENABLED) {
        // 化身是增强能力：它起不来不应该挡住语音对话，因此只报告不抛出。
        try {
          await startHastService();
          // 内嵌化身由渲染层按需创建（见 app.js 的 ensureAvatarView）。这里只广播
          // 就绪，让化身按钮可以点。
          sendServiceEvent("service.ready", { service: "avatar", mode: "embedded", url: getAvatarUrl() });
        } catch (error) {
          console.error("[VOICE] hast start failed:", error.message);
          sendServiceEvent("service.error", { service: "hast", message: error.message });
        }
      }

      let realtimeReady = await checkRealtimeReady();
      if (realtimeReady) {
        sendServiceEvent("service.ready", { service: "realtime", mode: "external", url: backendEndpoint.origin });
      } else {
        if (!isChildRunning(managedServices.realtime)) {
          const { command, args, env } = buildRealtimeArgs(pythonExecutable, llmOptions);
          console.log(`[VOICE] spawn realtime: ${command} ${args.join(" ")}`);
          spawnManagedService("realtime", command, args, REPO_ROOT, env);
        } else {
          console.log("[VOICE] realtime child already running, skip spawn");
        }
        realtimeReady = await waitForReady("realtime backend", "realtime", checkRealtimeReady, () => managedServices.realtime);
        console.log("[VOICE] realtime ready:", Boolean(realtimeReady));
        sendServiceEvent("service.ready", { service: "realtime", mode: "managed", url: backendEndpoint.origin });
      }

      console.log("[VOICE] warming up realtime models...");
      await warmupRealtimeModels(ttsOptions);
      console.log("[VOICE] warmup ok");

      const status = getVoiceServicesStatus();
      sendServiceEvent("start.completed", status);
      return status;
    } catch (error) {
      console.error("[VOICE] start.error:", error.message);
      sendServiceEvent("start.error", { message: error.message });
      await stopVoiceServices().catch((stopError) => sendServiceEvent("stop.error", { message: stopError.message }));
      throw error;
    }
  })();

  try {
    return await serviceStartPromise;
  } finally {
    serviceStartPromise = null;
  }
}

function waitForChildExit(child, timeoutMs = 10000) {
  if (!child || child.exitCode !== null) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function killProcessTree(name, child) {
  if (!isChildRunning(child)) {
    return;
  }

  sendServiceEvent("process.stop", { service: name, pid: child.pid });
  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore"
    });
    await new Promise((resolve) => killer.once("exit", resolve));
  } else {
    child.kill("SIGTERM");
  }

  await waitForChildExit(child);
  closeServiceLog(name);
}

async function stopVoiceServices() {
  if (serviceStopPromise) {
    return serviceStopPromise;
  }

  serviceStopPromise = (async () => {
    sendServiceEvent("stop.begin");
    await killProcessTree("realtime", managedServices.realtime);
    managedServices.realtime = null;
    await killProcessTree("memory", managedServices.memory);
    managedServices.memory = null;
    await killProcessTree("llama", managedServices.llama);
    managedServices.llama = null;
    await killProcessTree("hast", managedServices.hast);
    managedServices.hast = null;
    avatarReady = false;
    hastAvailable = false;
    // guest 随主窗口存活, 这里只清引用; 渲染层会在下次进入化身模式时重建。
    avatarGuest = null;
    const status = getVoiceServicesStatus();
    sendServiceEvent("stop.completed", status);
    return status;
  })();

  try {
    return await serviceStopPromise;
  } finally {
    serviceStopPromise = null;
  }
}

function getVoiceServicesStatus() {
  return {
    backend: {
      httpBase: BACKEND_HTTP_BASE,
      wsUrl: BACKEND_WS_URL
    },
    llm: {
      httpBase: llamaEndpoint.origin,
      managed: isChildRunning(managedServices.llama),
      pid: managedServices.llama ? managedServices.llama.pid : null,
      logFile: serviceLogFiles.llama
    },
    memory: {
      enabled: MEMORY_ENABLED,
      httpBase: memoryEndpoint.origin,
      managed: isChildRunning(managedServices.memory),
      pid: managedServices.memory ? managedServices.memory.pid : null,
      logFile: serviceLogFiles.memory
    },
    realtime: {
      httpBase: backendEndpoint.origin,
      managed: isChildRunning(managedServices.realtime),
      pid: managedServices.realtime ? managedServices.realtime.pid : null,
      logFile: serviceLogFiles.realtime
    },
    avatar: {
      enabled: HAST_ENABLED,
      available: hastAvailable,
      httpBase: HAST_HTTP_BASE,
      url: getAvatarUrl(),
      model: getSelectedAvatarModel(),
      models: AVATAR_MODELS,
      managed: isChildRunning(managedServices.hast),
      pid: managedServices.hast ? managedServices.hast.pid : null,
      logFile: serviceLogFiles.hast,
      // windowOpen 指窗口对象是否还在; visible 才是用户看得见与否 —— 渲染层的
      // 开关按钮必须用 visible, 否则隐藏之后会一直以为窗口还开着。
      attached: Boolean(avatarGuest && !avatarGuest.isDestroyed()),
      ready: avatarReady
    },
    logs: {
      dir: getServiceLogDir()
    }
  };
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    frame: false,
    titleBarStyle: "hidden",
    backgroundColor: "#121417",
    icon: pathExists(WINDOW_ICON_PATH, "file") ? WINDOW_ICON_PATH : undefined,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      // 口型由本窗口的音频分析器驱动（setInterval 约 30Hz）。主窗口被遮挡或
      // 最小化时若被 Chromium 降频到 1Hz，化身会僵在半张的嘴上，故关闭节流。
      backgroundThrottling: false,
      // 化身以 <webview> 内嵌在本窗口里（取代原来的着色器球体背景）。
      webviewTag: true
    }
  });

  // <webview> 的 preload 只能在挂载前由主进程注入（HTML 属性里写不了运行时路径），
  // 同时在这里把可加载的地址限制在本机化身服务，避免内嵌层被指向任意站点。
  mainWindow.webContents.on("will-attach-webview", (event, webPreferences, params) => {
    delete webPreferences.preloadURL;
    webPreferences.preload = path.join(__dirname, "hast-preload.js");
    webPreferences.contextIsolation = true;
    webPreferences.nodeIntegration = false;
    if (!String(params.src || "").startsWith(HAST_HTTP_BASE)) {
      console.error(`[AVATAR] blocked webview src: ${params.src}`);
      event.preventDefault();
    }
  });

  // 记下内嵌化身页面的 guest WebContents, 之后所有动作/口型都发到它。
  mainWindow.webContents.on("did-attach-webview", (_event, guest) => {
    avatarGuest = guest;
    avatarReady = false;
    console.log(`[AVATAR] webview attached id=${guest.id}`);
    guest.on("destroyed", () => {
      if (avatarGuest === guest) {
        avatarGuest = null;
        avatarReady = false;
      }
    });
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  // Forward renderer console logs to the main-process terminal so frontend
  // audio/WebSocket behavior is visible without opening DevTools. Tagged
  // [VOICE-FE] to sit alongside the [VOICE] logs from startVoiceServices.
  mainWindow.webContents.on("console-message", (_event, level, message, line, sourceId) => {
    const tag = level === 3 ? "[VOICE-FE!]" : "[VOICE-FE]";
    const where = sourceId ? ` ${sourceId}:${line}` : "";
    console.log(`${tag}${where} ${message}`);
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
    avatarGuest = null;
    avatarReady = false;
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(() => {
  ipcMain.handle("window:minimize", () => {
    if (mainWindow) {
      mainWindow.minimize();
    }
  });

  ipcMain.handle("window:toggle-maximize", () => {
    if (!mainWindow) {
      return false;
    }
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
      return false;
    }
    mainWindow.maximize();
    return true;
  });

  ipcMain.handle("window:close", () => {
    if (mainWindow) {
      mainWindow.close();
    }
  });

  ipcMain.handle("window:is-maximized", () => {
    return mainWindow ? mainWindow.isMaximized() : false;
  });

  ipcMain.handle("app:get-info", () => {
    return {
      name: app.getName(),
      version: app.getVersion()
    };
  });

  ipcMain.handle("backend:get-config", () => {
    return {
      httpBase: BACKEND_HTTP_BASE,
      wsUrl: BACKEND_WS_URL
    };
  });
  ipcMain.handle("assets:get-resource-status", () => getResourceStatus());
  ipcMain.handle("path:get-info", (_event, targetPath) => getPathInfo(targetPath));

  ipcMain.handle("voice-services:start", (_event, ttsOptions) => startVoiceServices(ttsOptions));
  ipcMain.handle("voice-services:stop", () => stopVoiceServices());
  ipcMain.handle("voice-services:get-status", () => getVoiceServicesStatus());
  ipcMain.handle("voice-logging:get-state", () => getVoiceLoggingState());
  ipcMain.handle("voice-logging:set-enabled", (_event, enabled) => setVoiceLoggingEnabled(enabled));
  ipcMain.handle("voice-services:open-log-folder", () => shell.openPath(getServiceLogDir()));

  // 内嵌化身与主渲染层之间的桥。渲染层负责把后端的 assistant.motion 与音频口型
  // 推过来，这里原样转发给 <webview> 的 guest；guest 侧由 hast-preload.js 接收。
  // 化身页面的创建与显隐完全由渲染层掌握（见 app.js 的 ensureAvatarView），
  // 所以这里不再有 show/hide/reload。
  ipcMain.handle("avatar:get-status", () => getVoiceServicesStatus().avatar);
  // 切换角色模型：已加载的化身页面就地换模型；页面还没就绪时记到 URL/配置，
  // 下次创建或 ready 后自动应用。
  ipcMain.handle("avatar:set-model", (_event, key) => {
    const model = setSelectedAvatarModel(key);
    if (avatarGuest && !avatarGuest.isDestroyed() && avatarReady) {
      avatarGuest.send("avatar:model", model);
    }
    return { model, models: AVATAR_MODELS, url: getAvatarUrl() };
  });
  ipcMain.on("avatar:ready", () => {
    avatarReady = true;
    const model = getSelectedAvatarModel();
    if (avatarGuest && !avatarGuest.isDestroyed()) {
      avatarGuest.send("avatar:model", model);
    }
    sendServiceEvent("avatar.ready", { url: getAvatarUrl(), model, models: AVATAR_MODELS });
  });
  ipcMain.on("avatar:log", (_event, payload) => {
    writeServiceLog("hast", "page", String(payload && payload.message ? payload.message : payload));
  });
  ipcMain.on("motion:plan", (_event, payload) => {
    sendToAvatar("avatar:plan", payload);
  });
  ipcMain.on("motion:framing", (_event, mode) => {
    sendToAvatar("avatar:framing", mode === "bust" ? "bust" : "full");
  });
  ipcMain.on("motion:visemes", (_event, values) => {
    sendToAvatar("avatar:visemes", values);
  });
  ipcMain.on("motion:speaking", (_event, speaking) => {
    sendToAvatar("avatar:speaking", Boolean(speaking));
  });
  ipcMain.on("motion:listening", (_event, listening) => {
    sendToAvatar("avatar:listening", Boolean(listening));
  });
  ipcMain.on("motion:reset", () => {
    sendToAvatar("avatar:reset", null);
  });

  ipcMain.handle("llm:show-model-file-dialog", async () => {
    if (!mainWindow) {
      return { canceled: true, filePaths: [] };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择 LLM 模型文件",
      filters: [
        { name: "GGUF 模型", extensions: ["gguf"] },
        { name: "所有文件", extensions: ["*"] }
      ],
      properties: ["openFile"]
    });
    return result;
  });

  ipcMain.handle("llm:get-model-path", () => {
    const config = loadLlmConfig();
    return config.modelPath || null;
  });

  ipcMain.handle("llm:set-model-path", (_event, modelPath) => {
    const config = loadLlmConfig();
    config.modelPath = modelPath || null;
    return saveLlmConfig(config);
  });

  ipcMain.handle("llm:get-config", () => loadLlmConfig());

  ipcMain.handle("llm:set-config", (_event, incoming) => {
    const current = loadLlmConfig();
    const next = incoming && typeof incoming === "object" ? incoming : {};
    const merged = {
      ...current,
      ...next,
      mode: next.mode === "api" ? "api" : "local",
      api: { ...current.api, ...(next.api || {}) }
    };
    return saveLlmConfig(merged);
  });

  // 话痨程度专用更新通道：只写 talkativeness 字段，不动 mode/api，避免
  // 拖动滑块时意外重置 LLM 运行模式。
  ipcMain.handle("llm:set-talkativeness", (_event, level) => {
    return saveLlmTalkativeness(level);
  });

  ipcMain.handle("llm:get-model-info", (_event, modelPath) => {
    if (!modelPath || !fs.existsSync(modelPath)) {
      return { exists: false, name: null, size: null };
    }
    try {
      const stats = fs.statSync(modelPath);
      const name = path.basename(modelPath);
      const sizeMB = Math.round(stats.size / (1024 * 1024));
      const sizeGB = (stats.size / (1024 * 1024 * 1024)).toFixed(2);
      return {
        exists: true,
        name,
        size: sizeMB >= 1024 ? `${sizeGB} GB` : `${sizeMB} MB`,
        sizeBytes: stats.size
      };
    } catch (error) {
      return { exists: false, name: null, size: null, error: error.message };
    }
  });

  ipcMain.handle("lora:scan-local", () => {
    try {
      return { models: scanLoraDirectory() };
    } catch (error) {
      console.error("Failed to scan LoRA directory:", error);
      return { models: [], error: error.message };
    }
  });

  ipcMain.handle("tts:show-audio-file-dialog", async () => {
    if (!mainWindow) {
      return { canceled: true, filePaths: [] };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择参考音频文件",
      filters: [
        { name: "音频文件", extensions: ["wav", "mp3", "flac", "ogg", "m4a", "aac"] },
        { name: "所有文件", extensions: ["*"] }
      ],
      properties: ["openFile"]
    });
    return result;
  });

  ipcMain.handle("tts:get-config", () => {
    return loadTtsConfig();
  });

  ipcMain.handle("tts:save-config", (_event, ttsConfig) => {
    return saveTtsConfig(ttsConfig);
  });

  ipcMain.handle("persona:list", () => {
    return listPersonas();
  });

  ipcMain.handle("persona:get-config", () => {
    return loadPersonaConfig();
  });

  ipcMain.handle("persona:set-config", (_event, personaConfig) => {
    return savePersonaConfig(personaConfig);
  });

  ipcMain.handle("assets:get-root", () => {
    const configured = loadAssetsRoot();
    const resolved = getAssetsRoot();
    return {
      configuredPath: configured,
      resolvedPath: resolved,
      source: getAssetsRootSource(),
      exists: pathExists(resolved, "directory"),
      defaultPath: getDefaultAssetsRoot(),
      envOverride: process.env.MINIMIND_ASSETS_ROOT || null
    };
  });

  ipcMain.handle("assets:choose-root", async () => {
    if (!mainWindow) {
      return { canceled: true };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择 assets 资源文件夹",
      properties: ["openDirectory"]
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return { canceled: true };
    }
    const chosen = result.filePaths[0];
    saveAssetsRoot(chosen);
    return { canceled: false, path: chosen };
  });

  ipcMain.handle("assets:set-root", (_event, assetsRoot) => {
    return saveAssetsRoot(assetsRoot);
  });

  ipcMain.handle("assets:clear-root", () => {
    return saveAssetsRoot(null);
  });

  ipcMain.handle("history:get-config", () => {
    const configured = loadHistoryVaultRoot();
    const resolved = getHistoryVaultRoot();
    return {
      configuredPath: configured,
      resolvedPath: resolved,
      defaultPath: getDefaultHistoryVaultRoot(),
      exists: pathExists(resolved, "directory"),
      envOverride: process.env.MINIMIND_VAULT_PATH || null
    };
  });

  ipcMain.handle("history:choose-vault-root", async () => {
    if (!mainWindow) {
      return { canceled: true };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择历史记录 Markdown 目录",
      properties: ["openDirectory", "createDirectory"]
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return { canceled: true };
    }
    const chosen = result.filePaths[0];
    saveHistoryVaultRoot(chosen);
    return { canceled: false, path: chosen };
  });

  ipcMain.handle("history:set-vault-root", (_event, vaultRoot) => {
    return saveHistoryVaultRoot(vaultRoot);
  });

  ipcMain.handle("history:clear-vault-root", () => {
    return saveHistoryVaultRoot(null);
  });

  ipcMain.handle("history:open-vault-root", async () => {
    const resolved = getHistoryVaultRoot();
    try {
      if (!fs.existsSync(resolved)) {
        fs.mkdirSync(resolved, { recursive: true });
      }
    } catch (error) {
      return { ok: false, error: String(error) };
    }
    const error = await shell.openPath(resolved);
    return error ? { ok: false, error } : { ok: true, path: resolved };
  });

  ipcMain.handle("history:get-db-config", () => {
    const configured = loadHistoryDbPath();
    const resolved = getHistoryDbPath();
    return {
      configuredPath: configured,
      resolvedPath: resolved,
      defaultPath: getDefaultHistoryDbPath(),
      fileName: HISTORY_DB_FILENAME,
      exists: pathExists(resolved, "file"),
      envOverride: process.env.MINIMIND_DB_PATH || null
    };
  });

  ipcMain.handle("history:choose-db-folder", async () => {
    if (!mainWindow) {
      return { canceled: true };
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      title: "选择数据库文件所在文件夹",
      message: `数据库文件名固定为 ${HISTORY_DB_FILENAME}`,
      properties: ["openDirectory", "createDirectory"]
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return { canceled: true };
    }
    const dbPath = path.join(result.filePaths[0], HISTORY_DB_FILENAME);
    saveHistoryDbPath(dbPath);
    return { canceled: false, path: dbPath };
  });

  ipcMain.handle("history:set-db-path", (_event, dbPath) => {
    return saveHistoryDbPath(dbPath);
  });

  ipcMain.handle("history:clear-db-path", () => {
    return saveHistoryDbPath(null);
  });

  ipcMain.handle("history:open-db-folder", async () => {
    const folder = path.dirname(getHistoryDbPath());
    try {
      if (!fs.existsSync(folder)) {
        fs.mkdirSync(folder, { recursive: true });
      }
    } catch (error) {
      return { ok: false, error: String(error) };
    }
    const error = await shell.openPath(folder);
    return error ? { ok: false, error } : { ok: true, path: folder };
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

let quitAfterCleanup = false;

app.on("before-quit", (event) => {
  if (quitAfterCleanup) {
    return;
  }
  event.preventDefault();
  quitAfterCleanup = true;
  stopVoiceServices()
    .catch((error) => sendServiceEvent("stop.error", { message: error.message }))
    .finally(() => app.exit(0));
});
