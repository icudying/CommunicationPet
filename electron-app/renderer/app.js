const els = {
  minBtn: document.getElementById("minBtn"),
  maxBtn: document.getElementById("maxBtn"),
  closeBtn: document.getElementById("closeBtn"),
  pingBtn: document.getElementById("pingBtn"),
  connectBtn: document.getElementById("connectBtn"),
  microphoneBtn: document.getElementById("microphoneBtn"),
  disconnectBtn: document.getElementById("disconnectBtn"),
  clearContextBtn: document.getElementById("clearContextBtn"),
  avatarBtn: document.getElementById("avatarBtn"),
  framingBtn: document.getElementById("framingBtn"),
  avatarModelSelect: document.getElementById("avatarModelSelect"),
  backendState: document.getElementById("backendState"),
  orb: document.getElementById("orb"),
  orbCaption: document.getElementById("orbCaption"),
  flowListening: document.getElementById("flowListening"),
  flowSpeechEnd: document.getElementById("flowSpeechEnd"),
  flowAsr: document.getElementById("flowAsr"),
  flowLlm: document.getElementById("flowLlm"),
  userStage: document.getElementById("userStage"),
  assistantStage: document.getElementById("assistantStage"),
  conversationScroll: document.getElementById("conversationScroll"),
  httpBase: document.getElementById("httpBase"),
  wsUrl: document.getElementById("wsUrl"),
  appVersion: document.getElementById("appVersion"),
  detectResourcesBtn: document.getElementById("detectResourcesBtn"),
  resourceLlamaState: document.getElementById("resourceLlamaState"),
  resourceGgufState: document.getElementById("resourceGgufState"),
  resourceAsrState: document.getElementById("resourceAsrState"),
  resourceTtsState: document.getElementById("resourceTtsState"),
  resourceHint: document.getElementById("resourceHint"),
  eventLog: document.getElementById("eventLog"),
  logCaptureToggle: document.getElementById("logCaptureToggle"),
  logCaptureHint: document.getElementById("logCaptureHint"),
  openServiceLogsBtn: document.getElementById("openServiceLogsBtn"),
  llmModeLocalBtn: document.getElementById("llmModeLocalBtn"),
  llmModeApiBtn: document.getElementById("llmModeApiBtn"),
  llmLocalPanel: document.getElementById("llmLocalPanel"),
  llmApiPanel: document.getElementById("llmApiPanel"),
  llmModelPath: document.getElementById("llmModelPath"),
  browseModelBtn: document.getElementById("browseModelBtn"),
  llmModelInfo: document.getElementById("llmModelInfo"),
  llmModelName: document.getElementById("llmModelName"),
  llmModelSize: document.getElementById("llmModelSize"),
  llmApiBaseUrl: document.getElementById("llmApiBaseUrl"),
  llmApiKey: document.getElementById("llmApiKey"),
  llmApiModel: document.getElementById("llmApiModel"),
  applyLlmBtn: document.getElementById("applyLlmBtn"),
  llmStatus: document.getElementById("llmStatus"),
  talkativenessRange: document.getElementById("talkativenessRange"),
  talkativenessLabel: document.getElementById("talkativenessLabel"),
  talkativenessHint: document.getElementById("talkativenessHint"),
  talkativenessStops: [...document.querySelectorAll(".talkativeness__stop")],
  talkativenessStatus: document.getElementById("talkativenessStatus"),
  ttsModelVersion: document.getElementById("ttsModelVersion"),
  ttsLoraSelect: document.getElementById("ttsLoraSelect"),
  refreshLoraBtn: document.getElementById("refreshLoraBtn"),
  promptAudioPath: document.getElementById("promptAudioPath"),
  browseAudioBtn: document.getElementById("browseAudioBtn"),
  promptText: document.getElementById("promptText"),
  ttsCfgValue: document.getElementById("ttsCfgValue"),
  ttsInferenceTimesteps: document.getElementById("ttsInferenceTimesteps"),
  ttsSeed: document.getElementById("ttsSeed"),
  applyTtsBtn: document.getElementById("applyTtsBtn"),
  ttsStatus: document.getElementById("ttsStatus"),
  assetsRootPath: document.getElementById("assetsRootPath"),
  browseAssetsRootBtn: document.getElementById("browseAssetsRootBtn"),
  resetAssetsRootBtn: document.getElementById("resetAssetsRootBtn"),
  assetsRootStatus: document.getElementById("assetsRootStatus"),
  toolPanel: document.getElementById("toolPanel"),
  toolTabs: [...document.querySelectorAll("[data-panel-target]")],
  toolViews: [...document.querySelectorAll("[data-panel-view]")],
  historyRefreshBtn: document.getElementById("historyRefreshBtn"),
  historySyncBtn: document.getElementById("historySyncBtn"),
  historyOpenVaultBtn: document.getElementById("historyOpenVaultBtn"),
  historyVaultPath: document.getElementById("historyVaultPath"),
  browseHistoryVaultBtn: document.getElementById("browseHistoryVaultBtn"),
  historyVaultStatus: document.getElementById("historyVaultStatus"),
  historySessionsTab: document.getElementById("historySessionsTab"),
  historyMemoriesTab: document.getElementById("historyMemoriesTab"),
  historyDatabaseTab: document.getElementById("historyDatabaseTab"),
  historyVaultBlock: document.getElementById("historyVaultBlock"),
  historyShowDeletedRow: document.getElementById("historyShowDeletedRow"),
  historyShowDeleted: document.getElementById("historyShowDeleted"),
  historyStatus: document.getElementById("historyStatus"),
  historyBody: document.getElementById("historyBody"),
  textComposer: document.getElementById("textComposer"),
  textPromptInput: document.getElementById("textPromptInput"),
  sendTextBtn: document.getElementById("sendTextBtn")
};

const DEFAULT_TEXT_PLACEHOLDER = "Type a message...";
const OFFLINE_TEXT_PLACEHOLDER = "请先启动服务";
const TEXT_STARTUP_PLACEHOLDER_FRAMES = ["正在启动", "正在启动.", "正在启动..", "正在启动..."];

// 话痨程度档位：从左到右回复越来越长，默认居中「普通」。
// 与后端 TALKATIVENESS_PROFILES (llm_client.py) 对齐；index+1 即滑块 position。
const TALKATIVENESS_LEVELS = [
  { key: "minimal", label: "极简", hint: "只说核心，一两句话" },
  { key: "concise", label: "简洁", hint: "简明扼要，直奔主题" },
  { key: "normal", label: "普通", hint: "默认回复长度" },
  { key: "talkative", label: "健谈", hint: "适当展开，补充细节" },
  { key: "chatty", label: "话痨", hint: "尽情展开，丰富详实" }
];

const state = {
  ws: null,
  micStream: null,
  micContext: null,
  micSource: null,
  micProcessor: null,
  microphoneEnabled: true,
  playbackContext: null,
  playbackAnalyser: null,
  playbackNextTime: 0,
  playbackSources: new Set(),
  // 当前轮次。动作意图与主回复并行解析，上一轮的意图可能在本轮开始后才返回，
  // 用它把这种迟到的动作丢掉，避免角色做出驴唇不对马嘴的手势。
  currentTurnId: null,
  // 化身层：服务是否在线（决定按钮可用性），要加载的化身页面地址，
  // 以及取景模式 "full"(全身) | "bust"(上半身)。
  avatarAvailable: false,
  avatarUrl: "",
  avatarModel: "red",
  avatarModels: [],
  avatarSwitching: false,
  // 对话驱动的自动形态切换：手动选择后 90 秒内不被自动覆盖，自动切换本身有冷却。
  avatarAutoSwitch: true,
  lastAvatarSwitchAt: 0,
  manualAvatarOverrideUntil: 0,
  framing: "full",
  phase: "idle",
  speechActive: false,
  serviceBusy: false,
  sessionMode: "idle",
  sessionReady: false,
  textInputEnabled: false,
  textInputLoading: false,
  textPlaceholderTimer: null,
  textPlaceholderFrame: 0,
  activeToolPanel: null,
  personaId: null,
  conversationMessages: [],
  nextMessageId: 1,
  currentUserMessageId: null,
  currentAssistantMessageId: null,
  pendingTextRequest: null,
  pendingTextRetryTimer: null,
  textInterruptPending: false,
  textInputComposing: false,
  backend: {
    httpBase: "",
    wsUrl: ""
  },
  llm: {
    mode: "local",
    modelPath: null,
    modelName: null,
    modelSize: null,
    talkativeness: "normal",
    api: {
      baseUrl: "",
      apiKey: "",
      model: ""
    }
  },
  resourceStatus: null,
  logCaptureEnabled: false,
  serviceLogsDir: "",
  turnCaptureDir: "",
  ttsOptions: {
    model_version: "1.5",
    lora_selection: null,
    prompt_audio_path: null,
    prompt_text: null,
    cfg_value: 2.0,
    inference_timesteps: 10,
    seed: -1
  }
};

function setActiveToolPanel(panelName) {
  const nextPanel = state.activeToolPanel === panelName ? null : panelName;
  state.activeToolPanel = nextPanel;
  els.toolPanel.classList.toggle("tool-panel--hidden", !nextPanel);

  for (const tab of els.toolTabs) {
    const isActive = Boolean(nextPanel && tab.dataset.panelTarget === nextPanel);
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-pressed", String(isActive));
  }

  for (const view of els.toolViews) {
    view.hidden = !nextPanel || view.dataset.panelView !== nextPanel;
  }

  if (nextPanel === "history") {
    void openHistoryPanel().catch((error) => appendLog("history.open_error", String(error)));
  }
}

function isRealtimeSocketOpen() {
  return Boolean(state.ws && state.ws.readyState === WebSocket.OPEN);
}

function isVoiceSessionActive() {
  return state.sessionMode === "voice";
}

function hasActiveTurn() {
  return (
    state.speechActive ||
    state.playbackSources.size > 0 ||
    state.phase === "speech-ended" ||
    state.phase === "asr" ||
    state.phase === "llm"
  );
}

function getReadyTurnPhase() {
  return isVoiceSessionActive() ? "listening" : "idle";
}

function getReadyOrbMode() {
  return isVoiceSessionActive() ? "listening" : "live";
}

function getReadyOrbMessage() {
  return isVoiceSessionActive()
    ? "Realtime channel connected. Microphone is streaming."
    : "Realtime channel connected. Text input is ready.";
}

function clearPendingTextRetry() {
  if (state.pendingTextRetryTimer) {
    clearTimeout(state.pendingTextRetryTimer);
    state.pendingTextRetryTimer = null;
  }
}

function renderTextComposerPlaceholder() {
  if (!els.textPromptInput) {
    return;
  }
  if (state.textInputLoading) {
    const frameIndex = state.textPlaceholderFrame % TEXT_STARTUP_PLACEHOLDER_FRAMES.length;
    els.textPromptInput.placeholder = TEXT_STARTUP_PLACEHOLDER_FRAMES[frameIndex];
    return;
  }
  els.textPromptInput.placeholder = state.sessionReady ? DEFAULT_TEXT_PLACEHOLDER : OFFLINE_TEXT_PLACEHOLDER;
}

function clearTextComposerLoadingTimer() {
  if (state.textPlaceholderTimer) {
    clearInterval(state.textPlaceholderTimer);
    state.textPlaceholderTimer = null;
  }
}

function setTextComposerLoading(loading) {
  const nextLoading = Boolean(loading);
  if (state.textInputLoading === nextLoading) {
    renderTextComposerPlaceholder();
    return;
  }

  state.textInputLoading = nextLoading;
  clearTextComposerLoadingTimer();
  state.textPlaceholderFrame = 0;
  renderTextComposerPlaceholder();

  if (nextLoading) {
    state.textPlaceholderTimer = window.setInterval(() => {
      state.textPlaceholderFrame = (state.textPlaceholderFrame + 1) % TEXT_STARTUP_PLACEHOLDER_FRAMES.length;
      renderTextComposerPlaceholder();
    }, 420);
  }
}

function updateTextComposerState() {
  const hasDraft = Boolean(String(els.textPromptInput.value || "").trim());
  const isLocked = !state.textInputEnabled;
  const isBusy = state.serviceBusy || Boolean(state.pendingTextRequest) || state.textInterruptPending;
  els.textComposer.classList.toggle("text-composer--disabled", isLocked);
  els.textComposer.classList.toggle("text-composer--pending", isBusy);
  els.textComposer.classList.toggle("text-composer--live", state.sessionReady);
  els.textComposer.setAttribute("aria-disabled", String(isLocked));
  els.textComposer.setAttribute("aria-busy", String(isBusy));
  els.textPromptInput.disabled = isLocked;
  els.sendTextBtn.disabled = isLocked || !hasDraft || state.textInputComposing;
  els.sendTextBtn.setAttribute(
    "aria-label",
    isLocked
      ? "Text input is unavailable until the voice session is ready"
      : state.textInterruptPending
        ? "Queue text message while interrupting the current turn"
        : state.pendingTextRequest
          ? "Queue text message"
          : "Send text message"
  );
  renderTextComposerPlaceholder();
}

function clearPendingTextRequest() {
  clearPendingTextRetry();
  state.pendingTextRequest = null;
  state.textInterruptPending = false;
  updateTextComposerState();
}

function restorePendingTextDraft() {
  const pending = state.pendingTextRequest;
  if (pending && !pending.dispatched && !String(els.textPromptInput.value || "").trim()) {
    els.textPromptInput.value = pending.text;
  }
  if (pending && !pending.dispatched) {
    removeConversationMessage(pending.userMessageId);
  }
  clearPendingTextRequest();
}

function queueTextRequest(text) {
  const value = String(text || "").trim();
  if (!value) {
    return null;
  }

  const existing = state.pendingTextRequest;
  let userMessageId = existing && !existing.dispatched ? existing.userMessageId : null;

  if (userMessageId) {
    updateConversationMessage(userMessageId, { text: value });
  } else {
    const message = beginConversationMessage("user", value, "Queued", "active");
    userMessageId = message.id;
  }
  state.pendingTextRequest = {
    text: value,
    userMessageId,
    dispatched: false,
    attempts: 0
  };
  setConversationStage("user", "Queued", "active");
  updateTextComposerState();
  return state.pendingTextRequest;
}

function sendRealtimeJson(payload) {
  if (!isRealtimeSocketOpen()) {
    return false;
  }
  state.ws.send(JSON.stringify(payload));
  return true;
}

function flushPendingTextRequest(reason = "direct") {
  const pending = state.pendingTextRequest;
  if (!pending || pending.dispatched) {
    updateTextComposerState();
    return false;
  }
  if (!state.sessionReady || !isRealtimeSocketOpen() || state.serviceBusy || state.textInterruptPending || hasActiveTurn()) {
    return false;
  }

  state.currentUserMessageId = pending.userMessageId;
  state.currentAssistantMessageId = null;
  pending.dispatched = true;
  pending.attempts += 1;

  const sent = sendRealtimeJson({
    type: "text.input",
    text: pending.text,
    tts_options: state.ttsOptions
  });

  if (!sent) {
    pending.dispatched = false;
    return false;
  }

  setConversationStage("user", "Done", "done");
  setOrbMode("live", "Text input sent. Waiting for the assistant.");
  appendLog("text.input.sent", {
    reason,
    attempt: pending.attempts,
    length: pending.text.length
  });
  updateTextComposerState();
  return true;
}

function schedulePendingTextFlush(reason, delayMs = 0) {
  clearPendingTextRetry();
  if (!state.pendingTextRequest) {
    updateTextComposerState();
    return;
  }
  state.pendingTextRetryTimer = window.setTimeout(() => {
    state.pendingTextRetryTimer = null;
    const sent = flushPendingTextRequest(reason);
    if (
      !sent &&
      state.pendingTextRequest &&
      !state.pendingTextRequest.dispatched &&
      !state.textInterruptPending &&
      state.sessionReady &&
      isRealtimeSocketOpen() &&
      !state.serviceBusy &&
      hasActiveTurn()
    ) {
      schedulePendingTextFlush(reason, 160);
    }
  }, delayMs);
  updateTextComposerState();
}

function retryPendingTextAfterBusyError(message) {
  const pending = state.pendingTextRequest;
  const detail = String(message?.message || "");
  if (!pending || !pending.dispatched || !/assistant request is still running/i.test(detail)) {
    return false;
  }
  pending.dispatched = false;
  setOrbMode("live", "The current turn is finishing. Queued text will send next.");
  schedulePendingTextFlush("backend.busy", 180);
  return true;
}

function shouldSuppressBusyBackendError(message) {
  const detail = String(message?.message || "");
  if (!/assistant request is still running/i.test(detail)) {
    return false;
  }
  return state.speechActive || state.phase === "speech-ended" || state.phase === "asr";
}

function handleTextComposerSubmit(event) {
  event.preventDefault();
  if (!state.textInputEnabled) {
    return;
  }
  if (state.textInputComposing) {
    return;
  }
  const text = String(els.textPromptInput.value || "").trim();
  if (!text) {
    els.textPromptInput.focus();
    return;
  }
  queueTextRequest(text);
  els.textPromptInput.value = "";
  updateTextComposerState();

  if (!state.ws) {
    void connectRealtime({ withMicrophone: false });
    return;
  }

  if (hasActiveTurn()) {
    state.textInterruptPending = true;
    updateTextComposerState();
    stopPlaybackQueue();
    if (sendRealtimeJson({ type: "interrupt", request_id: `interrupt-${Date.now()}` })) {
      setOrbMode("live", "Interrupting the current turn before sending the queued text.");
      appendLog("text.input.interrupt_requested", {
        length: text.length
      });
    } else {
      state.textInterruptPending = false;
      schedulePendingTextFlush("interrupt.send_failed", 0);
    }
    return;
  }

  schedulePendingTextFlush("submit", 0);
}

function getConversationMessage(messageId) {
  return state.conversationMessages.find((message) => message.id === messageId) || null;
}

function removeConversationMessage(messageId) {
  const index = state.conversationMessages.findIndex((message) => message.id === messageId);
  if (index < 0) {
    return false;
  }
  state.conversationMessages.splice(index, 1);
  if (state.currentUserMessageId === messageId) {
    state.currentUserMessageId = null;
  }
  if (state.currentAssistantMessageId === messageId) {
    state.currentAssistantMessageId = null;
  }
  renderConversation();
  return true;
}

function trimConversationMessages() {
  while (state.conversationMessages.length > 6) {
    const removed = state.conversationMessages.shift();
    if (removed.id === state.currentUserMessageId) {
      state.currentUserMessageId = null;
    }
    if (removed.id === state.currentAssistantMessageId) {
      state.currentAssistantMessageId = null;
    }
  }
}

function renderConversation() {
  els.conversationScroll.replaceChildren();

  for (const message of state.conversationMessages) {
    const article = document.createElement("article");
    article.className = `message message--${message.role}`;

    const bubble = document.createElement("div");
    bubble.className = `message__bubble transcript transcript--${message.role}`;
    bubble.textContent = message.text || "";

    article.appendChild(bubble);
    els.conversationScroll.appendChild(article);
  }

  els.conversationScroll.scrollTop = els.conversationScroll.scrollHeight;
}

function clearConversation() {
  clearPendingTextRequest();
  state.conversationMessages = [];
  state.currentUserMessageId = null;
  state.currentAssistantMessageId = null;
  state.nextMessageId = 1;
  renderConversation();
}

function clearContext() {
  if (state.serviceBusy) {
    return;
  }

  // If a turn is mid-flight (TTS playback or LLM reply), stop local playback
  // and interrupt the backend before wiping the context.
  if (hasActiveTurn()) {
    stopPlaybackQueue();
    sendRealtimeJson({ type: "interrupt", request_id: `interrupt-${Date.now()}` });
  }

  // Tell the backend to drop the short-term conversation context. Long-term
  // memory is intentionally left untouched.
  const contextCleared = sendRealtimeJson({
    type: "context.clear",
    request_id: `clear-${Date.now()}`
  });

  clearConversation();

  if (isVoiceSessionActive()) {
    setTurnPhase("listening");
    setOrbMode("listening", "Conversation cleared. Ready for a new turn.");
  } else {
    setTurnPhase("idle");
    setOrbMode(getReadyOrbMode(), "Conversation cleared. Ready for a new turn.");
  }

  appendLog("context.clear_requested", { sent_to_backend: contextCleared });
}

function beginConversationMessage(role, text = "", stageText = "Idle", stageTone = null) {
  const message = {
    id: state.nextMessageId,
    role,
    text,
    stageText,
    stageTone
  };
  state.nextMessageId += 1;
  state.conversationMessages.push(message);
  if (role === "user") {
    state.currentUserMessageId = message.id;
  } else {
    state.currentAssistantMessageId = message.id;
  }
  trimConversationMessages();
  renderConversation();
  return message;
}

function updateConversationMessage(messageId, updates) {
  const message = getConversationMessage(messageId);
  if (!message) {
    return null;
  }
  Object.assign(message, updates);
  renderConversation();
  return message;
}

function getCurrentConversationMessage(role) {
  const messageId = role === "user" ? state.currentUserMessageId : state.currentAssistantMessageId;
  return getConversationMessage(messageId);
}

function setConversationStage(role, text, tone) {
  const stageEl = role === "user" ? els.userStage : els.assistantStage;
  const label = role === "user" ? "User" : "AI";
  stageEl.textContent = `${label} ${text}`;
  stageEl.className = "stage-pill header-stage";
  if (tone) {
    stageEl.classList.add(`stage-pill--${tone}`);
  }
}

function setConversationText(role, text, stageText, stageTone) {
  const message = getCurrentConversationMessage(role);
  if (!message) {
    beginConversationMessage(role, text || "", stageText, stageTone);
    if (stageText) {
      setConversationStage(role, stageText, stageTone);
    }
    return;
  }
  updateConversationMessage(message.id, {
    text: text || ""
  });
  if (stageText) {
    setConversationStage(role, stageText, stageTone);
  }
}

function appendConversationText(role, text) {
  const value = text || "";
  if (!value) {
    return;
  }
  const message = getCurrentConversationMessage(role) || beginConversationMessage(role, "", "Streaming", "active");
  updateConversationMessage(message.id, {
    text: `${message.text || ""}${value}`
  });
}

function updateLlmModelDisplay() {
  const { modelPath, modelName, modelSize } = state.llm;
  els.llmModelPath.value = modelPath || "";

  if (modelName) {
    els.llmModelName.textContent = modelName;
    els.llmModelSize.textContent = modelSize || "";
    els.llmModelInfo.classList.add("llm-model-info--loaded");
  } else {
    els.llmModelName.textContent = "未选择模型";
    els.llmModelSize.textContent = "";
    els.llmModelInfo.classList.remove("llm-model-info--loaded");
  }

  if (state.llm.mode !== "api") {
    els.llmStatus.textContent = modelName
      ? `已选择: ${modelName}${modelSize ? ` (${modelSize})` : ""}`
      : "请选择一个 GGUF 格式的大语言模型文件。";
  }
}

function updateLlmApiDisplay() {
  els.llmApiBaseUrl.value = state.llm.api.baseUrl || "";
  els.llmApiKey.value = state.llm.api.apiKey || "";
  els.llmApiModel.value = state.llm.api.model || "";
}

function setLlmMode(mode, { persist = false } = {}) {
  const nextMode = mode === "api" ? "api" : "local";
  state.llm.mode = nextMode;
  const isApi = nextMode === "api";

  els.llmModeLocalBtn.classList.toggle("is-active", !isApi);
  els.llmModeApiBtn.classList.toggle("is-active", isApi);
  els.llmModeLocalBtn.setAttribute("aria-selected", String(!isApi));
  els.llmModeApiBtn.setAttribute("aria-selected", String(isApi));
  els.llmLocalPanel.hidden = isApi;
  els.llmApiPanel.hidden = !isApi;

  if (isApi) {
    const { model } = state.llm.api;
    els.llmStatus.textContent = model
      ? `API 模式: ${model}`
      : "请填写请求地址、API Key 和模型名。";
  } else {
    updateLlmModelDisplay();
  }

  if (persist) {
    void window.desktopApp.setLlmConfig({ mode: nextMode }).then(() => {
      renderResourceStatus();
    });
  } else {
    renderResourceStatus();
  }
}

function setResourceBadge(el, mode, text) {
  el.textContent = text;
  el.className = "resource-badge";
  if (mode) {
    el.classList.add(`resource-badge--${mode}`);
  }
}

function renderResourceStatus() {
  const status = state.resourceStatus;
  if (!status) {
    setResourceBadge(els.resourceLlamaState, "", "Checking");
    setResourceBadge(els.resourceGgufState, "", "Checking");
    setResourceBadge(els.resourceAsrState, "", "Checking");
    setResourceBadge(els.resourceTtsState, "", "Checking");
    els.resourceHint.textContent = "Checking installation assets...";
    return;
  }

  setResourceBadge(
    els.resourceLlamaState,
    state.llm.mode === "api"
      ? "ok"
      : status.llama.exists
        ? "ok"
        : "warn",
    state.llm.mode === "api" ? "API Mode" : status.llama.exists ? "Installed" : "Missing"
  );
  setResourceBadge(
    els.resourceAsrState,
    status.asr.exists ? "ok" : "warn",
    status.asr.exists ? "Installed" : "Missing"
  );
  setResourceBadge(
    els.resourceTtsState,
    status.tts.exists ? "ok" : "warn",
    status.tts.exists ? "Installed" : "Missing"
  );

  if (state.llm.mode === "api") {
    const apiReady = Boolean(String(state.llm.api.baseUrl || "").trim() && String(state.llm.api.model || "").trim());
    setResourceBadge(els.resourceGgufState, apiReady ? "ok" : "warn", apiReady ? "API Model" : "Configure");
  } else if (status.llm.exists) {
    setResourceBadge(els.resourceGgufState, "ok", "Ready");
  } else if (status.llm.source === "assets.multiple") {
    setResourceBadge(els.resourceGgufState, "warn", "Select Model");
  } else {
    setResourceBadge(els.resourceGgufState, "warn", "Missing");
  }

  const missing = [];
  if (state.llm.mode === "api") {
    if (!String(state.llm.api.baseUrl || "").trim()) missing.push("API base URL");
    if (!String(state.llm.api.model || "").trim()) missing.push("API model");
  } else {
    if (!status.llama.exists) missing.push("llama runtime");
    if (!status.llm.exists) missing.push(status.llm.source === "assets.multiple" ? "GGUF selection" : "GGUF");
  }
  if (!status.asr.exists) missing.push("ASR");
  if (!status.tts.exists) missing.push("TTS");

  if (!missing.length) {
    if (state.llm.mode === "api") {
      els.resourceHint.textContent = `Core assets detected. Using remote API model ${state.llm.api.model || ""}.`;
    } else {
      const ggufSource = status.llm.source === "assets.auto" ? "auto-selected GGUF" : "configured GGUF";
      els.resourceHint.textContent = `Core assets detected. Using ${ggufSource} from assets.`;
    }
    return;
  }

  els.resourceHint.textContent = `Missing: ${missing.join(", ")}. Expected under ${status.assetsRoot}.`;
}

async function detectResources() {
  if (!window.desktopApp.getResourceStatus) {
    return;
  }

  els.detectResourcesBtn.disabled = true;
  els.resourceHint.textContent = "Checking installation assets...";

  try {
    const status = await window.desktopApp.getResourceStatus();
    state.resourceStatus = status;
    renderResourceStatus();
    appendLog("assets.status", status);
  } catch (error) {
    appendLog("assets.status_error", String(error));
    els.resourceHint.textContent = "Resource detection failed.";
  } finally {
    els.detectResourcesBtn.disabled = false;
  }
}

async function loadLlmConfig() {
  try {
    const config = await window.desktopApp.getLlmConfig();
    state.llm.mode = config.mode === "api" ? "api" : "local";
    state.llm.api.baseUrl = config.api?.baseUrl || "";
    state.llm.api.apiKey = config.api?.apiKey || "";
    state.llm.api.model = config.api?.model || "";
    state.llm.talkativeness = TALKATIVENESS_LEVELS.some((l) => l.key === config.talkativeness)
      ? config.talkativeness
      : "normal";

    const modelPath = config.modelPath || null;
    if (modelPath) {
      const info = await window.desktopApp.getLlmModelInfo(modelPath);
      state.llm.modelPath = modelPath;
      state.llm.modelName = info.name || null;
      state.llm.modelSize = info.size || null;
    } else {
      state.llm.modelPath = null;
      state.llm.modelName = null;
      state.llm.modelSize = null;
    }

    updateLlmModelDisplay();
    updateLlmApiDisplay();
    setLlmMode(state.llm.mode);
    updateTalkativenessDisplay();
  } catch (error) {
    appendLog("llm.config.load_error", String(error));
  }
}

function getTalkativenessPosition() {
  const idx = TALKATIVENESS_LEVELS.findIndex((l) => l.key === state.llm.talkativeness);
  return idx >= 0 ? idx + 1 : 3;
}

// 同步滑块 UI（轨道填充、当前档位名、提示语、档位标签高亮）到 state.llm.talkativeness。
function updateTalkativenessDisplay() {
  if (!els.talkativenessRange) {
    return;
  }
  const pos = getTalkativenessPosition();
  const level = TALKATIVENESS_LEVELS[pos - 1] || TALKATIVENESS_LEVELS[2];
  els.talkativenessRange.value = String(pos);
  els.talkativenessRange.style.setProperty(
    "--talkativeness-fill",
    `${((pos - 1) / (TALKATIVENESS_LEVELS.length - 1)) * 100}%`
  );
  els.talkativenessLabel.textContent = level.label;
  els.talkativenessHint.textContent = level.hint;
  for (const stop of els.talkativenessStops) {
    stop.classList.toggle("is-active", Number(stop.dataset.level) === pos);
  }
}

// 话痨程度变更：写 state、刷新 UI、持久化到本地配置、经 WebSocket 推送到后端
// （下一轮对话立即生效，无需重启会话）。
function applyTalkativeness({ sendToBackend = true } = {}) {
  if (!els.talkativenessRange) {
    return;
  }
  const pos = Number(els.talkativenessRange.value);
  const level = TALKATIVENESS_LEVELS[pos - 1] || TALKATIVENESS_LEVELS[2];
  state.llm.talkativeness = level.key;
  updateTalkativenessDisplay();
  // 用专用 IPC 只写 talkativeness 字段，避免触碰 LLM mode 合并逻辑。
  window.desktopApp
    .setLlmTalkativeness(level.key)
    .catch((error) => appendLog("llm.talkativeness.save_error", String(error)));
  const sent = sendToBackend && isRealtimeSocketOpen();
  if (sent) {
    sendRealtimeJson({ type: "llm.configure", talkativeness: level.key });
  }
  appendLog("llm.talkativeness.set", { key: level.key, label: level.label, sent });
  els.talkativenessStatus.textContent = sent
    ? `当前话痨程度：${level.label}。已生效，下一轮对话按此长度回复。`
    : `当前话痨程度：${level.label}。已保存，连接服务后下一轮生效。`;
}

async function browseForModel() {
  try {
    const result = await window.desktopApp.showModelFileDialog();
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return;
    }
    const filePath = result.filePaths[0];
    const info = await window.desktopApp.getLlmModelInfo(filePath);
    if (!info.exists) {
      appendLog("llm.model.not_found", { path: filePath });
      return;
    }
    state.llm.modelPath = filePath;
    state.llm.modelName = info.name;
    state.llm.modelSize = info.size;
    updateLlmModelDisplay();
    appendLog("llm.model.selected", { path: filePath, name: info.name, size: info.size });
  } catch (error) {
    appendLog("llm.browse.error", String(error));
  }
}

async function applyLlmConfig() {
  if (state.llm.mode === "api") {
    return applyLlmApiConfig();
  }
  const { modelPath } = state.llm;
  if (!modelPath) {
    appendLog("llm.config.no_model", "Please select a model file first.");
    els.llmStatus.textContent = "请先选择一个 GGUF 模型文件。";
    return;
  }
  try {
    const saved = await window.desktopApp.setLlmConfig({ mode: "local", modelPath });
    if (saved) {
      appendLog("llm.config.saved", { mode: "local", modelPath });
      void detectResources();
      els.llmStatus.textContent = `配置已保存: ${state.llm.modelName}${state.llm.modelSize ? ` (${state.llm.modelSize})` : ""}。重新启动会话后生效。`;
    } else {
      appendLog("llm.config.save_failed", "Failed to save LLM configuration.");
    }
  } catch (error) {
    appendLog("llm.config.error", String(error));
  }
}

async function applyLlmApiConfig() {
  const baseUrl = els.llmApiBaseUrl.value.trim();
  const apiKey = els.llmApiKey.value.trim();
  const model = els.llmApiModel.value.trim();

  if (!baseUrl) {
    appendLog("llm.config.no_base_url", "Please enter an API base URL.");
    els.llmStatus.textContent = "请填写请求地址。";
    return;
  }
  if (!model) {
    appendLog("llm.config.no_api_model", "Please enter a model name.");
    els.llmStatus.textContent = "请填写模型名。";
    return;
  }

  state.llm.api.baseUrl = baseUrl;
  state.llm.api.apiKey = apiKey;
  state.llm.api.model = model;

  try {
    const saved = await window.desktopApp.setLlmConfig({
      mode: "api",
      api: { baseUrl, apiKey, model }
    });
    if (saved) {
      // Do not log the api key.
      appendLog("llm.config.saved", { mode: "api", baseUrl, model, apiKey: apiKey ? "***" : "" });
      renderResourceStatus();
      els.llmStatus.textContent = `API 配置已保存: ${model} @ ${baseUrl}。重新启动会话后生效。`;
    } else {
      appendLog("llm.config.save_failed", "Failed to save LLM configuration.");
    }
  } catch (error) {
    appendLog("llm.config.error", String(error));
  }
}

async function browseForAudio() {
  try {
    const result = await window.desktopApp.showAudioFileDialog();
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return;
    }
    const filePath = result.filePaths[0];
    els.promptAudioPath.value = filePath;

    // Extract filename without extension and fill Reference Text
    const fileNameWithExt = filePath.split(/[\\\/]/).pop();
    const lastDotIndex = fileNameWithExt.lastIndexOf(".");
    const fileName = lastDotIndex !== -1 ? fileNameWithExt.substring(0, lastDotIndex) : fileNameWithExt;
    els.promptText.value = fileName;

    applyTtsOptions(false);
    appendLog("tts.audio.selected", { path: filePath });
  } catch (error) {
    appendLog("tts.audio.browse_error", String(error));
  }
}

function describeAssetsRootSource(source) {
  switch (source) {
    case "env":
      return "环境变量 MINIMIND_ASSETS_ROOT";
    case "config":
      return "已选目录";
    case "default":
      return "默认目录";
    default:
      return source || "未知来源";
  }
}

async function refreshAssetsRoot() {
  if (!window.desktopApp.getAssetsRoot) {
    return;
  }
  try {
    const info = await window.desktopApp.getAssetsRoot();
    els.assetsRootPath.value = info.resolvedPath || "";
    const sourceLabel = describeAssetsRootSource(info.source);
    if (!info.exists) {
      els.assetsRootStatus.textContent = `目录不存在：${info.resolvedPath || "(空)"}。请重新选择 assets 文件夹。`;
    } else if (info.source === "env") {
      els.assetsRootStatus.textContent = `当前使用${sourceLabel}，界面选择被忽略。`;
    } else {
      els.assetsRootStatus.textContent = `当前使用${sourceLabel}：${info.resolvedPath}`;
    }
    // Disable manual controls when an env override is in effect.
    const envLocked = info.source === "env";
    els.browseAssetsRootBtn.disabled = envLocked;
    els.resetAssetsRootBtn.disabled = envLocked;
  } catch (error) {
    els.assetsRootStatus.textContent = "读取资源目录失败。";
    appendLog("assets.root.read_error", String(error));
  }
}

async function browseForAssetsRoot() {
  if (!window.desktopApp.chooseAssetsRoot) {
    return;
  }
  try {
    const result = await window.desktopApp.chooseAssetsRoot();
    if (result.canceled) {
      return;
    }
    appendLog("assets.root.selected", { path: result.path });
    await refreshAssetsRoot();
    await detectResources();
    els.assetsRootStatus.textContent = `已选择：${result.path}。如服务正在运行，请重启服务后生效。`;
  } catch (error) {
    appendLog("assets.root.select_error", String(error));
  }
}

async function resetAssetsRoot() {
  if (!window.desktopApp.clearAssetsRoot) {
    return;
  }
  try {
    await window.desktopApp.clearAssetsRoot();
    appendLog("assets.root.reset", {});
    await refreshAssetsRoot();
    await detectResources();
    els.assetsRootStatus.textContent = "已恢复默认目录。如服务正在运行，请重启服务后生效。";
  } catch (error) {
    appendLog("assets.root.reset_error", String(error));
  }
}

function readNumberInput(el, fallback, min, max) {
  const value = Number(el.value);
  const safe = Number.isFinite(value) ? value : fallback;
  return Math.max(min ?? safe, Math.min(max ?? safe, safe));
}

function readTtsOptions() {
  const promptAudioPath = String(els.promptAudioPath.value || "").trim();
  const promptText = String(els.promptText.value || "").trim();
  const hasCompleteReference = Boolean(promptAudioPath && promptText);
  return {
    model_version: String(els.ttsModelVersion.value || "").trim() || "1.5",
    lora_selection: String(els.ttsLoraSelect.value || "").trim() || null,
    prompt_audio_path: hasCompleteReference ? promptAudioPath : null,
    prompt_text: hasCompleteReference ? promptText : null,
    cfg_value: readNumberInput(els.ttsCfgValue, 2.0, 0.1, 10),
    inference_timesteps: Math.round(readNumberInput(els.ttsInferenceTimesteps, 10, 1, 100)),
    seed: Math.round(readNumberInput(els.ttsSeed, -1))
  };
}

function applyTtsConfigToUi(config) {
  els.promptAudioPath.value = config.prompt_audio_path || "";
  els.promptText.value = config.prompt_text || "";

  if (config.model_version) {
    const options = [...els.ttsModelVersion.options];
    if (options.some(opt => opt.value === config.model_version)) {
      els.ttsModelVersion.value = config.model_version;
    }
  }

  if (config.lora_selection) {
    const options = [...els.ttsLoraSelect.options];
    if (options.some(opt => opt.value === config.lora_selection)) {
      els.ttsLoraSelect.value = config.lora_selection;
    }
  } else {
    els.ttsLoraSelect.value = "";
  }
  if (config.cfg_value !== undefined && config.cfg_value !== null) {
    els.ttsCfgValue.value = config.cfg_value;
  }
  if (config.inference_timesteps !== undefined && config.inference_timesteps !== null) {
    els.ttsInferenceTimesteps.value = config.inference_timesteps;
  }
  if (config.seed !== undefined && config.seed !== null) {
    els.ttsSeed.value = config.seed;
  }
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
}

async function loadTtsConfigFromStorage() {
  try {
    const config = await window.desktopApp.getTtsConfig();
    if (config) {
      applyTtsConfigToUi(config);
      await normalizeTtsReferencePath();
      appendLog("tts.config.loaded", config);
    }
  } catch (error) {
    appendLog("tts.config.load_error", String(error));
  }
}

async function saveTtsConfigToStorage() {
  try {
    const config = state.ttsOptions;
    const saved = await window.desktopApp.saveTtsConfig(config);
    if (saved) {
      appendLog("tts.config.saved", config);
    }
  } catch (error) {
    appendLog("tts.config.save_error", String(error));
  }
}

function updateTtsStatus() {
  const options = state.ttsOptions;
  const lora = options.lora_selection || "base voice";
  const seed = Number(options.seed) >= 0 ? options.seed : "random";
  const typedReferencePath = String(els.promptAudioPath.value || "").trim();
  const typedReferenceText = String(els.promptText.value || "").trim();
  const referenceStatus = options.prompt_audio_path
    ? "reference on"
    : typedReferencePath && !typedReferenceText
      ? "reference ignored: text required"
      : "base voice";
  els.ttsStatus.textContent = `LoRA: ${lora}. ${referenceStatus}. CFG ${options.cfg_value}, ${options.inference_timesteps} steps, seed ${seed}.`;
}

async function normalizeTtsReferencePath() {
  const promptAudioPath = String(els.promptAudioPath.value || "").trim();
  if (!promptAudioPath || !window.desktopApp.getPathInfo) {
    return false;
  }

  try {
    const pathInfo = await window.desktopApp.getPathInfo(promptAudioPath);
    if (pathInfo.exists && pathInfo.type === "file") {
      return false;
    }
  } catch (error) {
    appendLog("tts.reference.check_error", String(error));
    return false;
  }

  els.promptAudioPath.value = "";
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
  await saveTtsConfigToStorage();
  appendLog("tts.reference.cleared_missing", { path: promptAudioPath });
  els.ttsStatus.textContent = "Reference audio was cleared because the file no longer exists.";
  return true;
}

function applyTtsOptions(sendToBackend = true) {
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
  saveTtsConfigToStorage();
  if (sendToBackend && state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "tts.configure", tts_options: state.ttsOptions }));
  }
  appendLog("tts.config.local", state.ttsOptions);
}

async function refreshLoraCatalog() {
  const current = els.ttsLoraSelect.value;
  let models = [];
  let source = "local";

  try {
    const localResult = await window.desktopApp.scanLoraLocal();
    models = localResult.models || [];
    appendLog("tts.lora.local_scan", { count: models.length });
  } catch (error) {
    appendLog("tts.lora.local_scan_error", String(error));
  }

  if (models.length === 0 && state.backend.httpBase) {
    try {
      const response = await fetch(`${state.backend.httpBase}/tts/lora/catalog`);
      if (response.ok) {
        const payload = await response.json();
        models = payload.models || [];
        source = "backend";
        appendLog("tts.lora.backend_scan", { count: models.length });
      }
    } catch (error) {
      appendLog("tts.lora.backend_scan_error", String(error));
    }
  }

  els.ttsLoraSelect.innerHTML = '<option value="">Base voice</option>';
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.path;
    option.textContent = model.base_model ? `${model.label} [${model.base_model}]` : model.label;
    els.ttsLoraSelect.appendChild(option);
  }
  if ([...els.ttsLoraSelect.options].some((option) => option.value === current)) {
    els.ttsLoraSelect.value = current;
  }
  state.ttsOptions = readTtsOptions();
  updateTtsStatus();
  appendLog("tts.lora.catalog", { count: models.length, source });
}

function setBackendState(text, tone) {
  els.backendState.textContent = text;
  els.backendState.className = "status-pill";
  if (tone) {
    els.backendState.classList.add(`status-pill--${tone}`);
  }
}

function setOrbMode(mode, caption) {
  if (!els.orb || !els.orbCaption) {
    return;
  }

  els.orb.className = "orb";
  if (mode) {
    els.orb.classList.add(`orb--${mode}`);
  }
  els.orbCaption.textContent = caption;
}

function setFlowStep(el, stateName) {
  if (!el) {
    return;
  }

  el.className = "flow-step";
  if (stateName) {
    el.classList.add(`flow-step--${stateName}`);
  }
}

function setTurnPhase(phase) {
  state.phase = phase;
  const done = "done";
  const active = "active";

  setFlowStep(els.flowListening, "");
  setFlowStep(els.flowSpeechEnd, "");
  setFlowStep(els.flowAsr, "");
  setFlowStep(els.flowLlm, "");

  if (phase === "idle") {
    setConversationStage("user", "Idle");
    setConversationStage("assistant", "Idle");
    return;
  }

  if (phase === "listening") {
    setFlowStep(els.flowListening, active);
    setConversationStage("user", "Listening", "active");
    setConversationStage("assistant", "Idle");
    return;
  }

  if (phase === "speech-ended") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, active);
    setConversationStage("user", "Speech End", "active");
    setConversationStage("assistant", "Idle");
    return;
  }

  if (phase === "asr") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, done);
    setFlowStep(els.flowAsr, active);
    setConversationStage("user", "ASR", "active");
    setConversationStage("assistant", "Waiting");
    return;
  }

  if (phase === "llm") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, done);
    setFlowStep(els.flowAsr, done);
    setFlowStep(els.flowLlm, active);
    setConversationStage("user", "Done", "done");
    setConversationStage("assistant", "LLM", "active");
    return;
  }

  if (phase === "completed") {
    setFlowStep(els.flowListening, done);
    setFlowStep(els.flowSpeechEnd, done);
    setFlowStep(els.flowAsr, done);
    setFlowStep(els.flowLlm, done);
    setConversationStage("user", "Done", "done");
    setConversationStage("assistant", "Done", "done");
  }
}

const LOG_TIME_FORMATTER = new Intl.DateTimeFormat("zh-CN", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit"
});

function formatLogTimestamp(date = new Date()) {
  return LOG_TIME_FORMATTER.format(date);
}

function getLogTone(type) {
  const normalized = String(type || "").toLowerCase();

  if (/(error|failed|timeout|not_found)/.test(normalized)) {
    return "error";
  }

  if (/(missing|cleared|close|stop)/.test(normalized)) {
    return "warn";
  }

  if (/(ready|ok|saved|selected|configured)/.test(normalized)) {
    return "success";
  }

  return "info";
}

function appendLog(type, payload) {
  const entry = document.createElement("div");
  entry.className = `log-entry log-entry--${getLogTone(type)}`;
  let displayPayload = payload;
  if (payload && payload.type === "assistant.audio.chunk") {
    displayPayload = {
      ...payload,
      data: `<base64 pcm_f32le ${payload.num_samples || 0} samples>`
    };
  }
  const safe = typeof displayPayload === "string" ? displayPayload : JSON.stringify(displayPayload, null, 2);

  const meta = document.createElement("div");
  meta.className = "log-entry__meta";

  const typeBadge = document.createElement("span");
  typeBadge.className = "log-entry__type";
  typeBadge.textContent = type;

  const timestamp = document.createElement("time");
  timestamp.className = "log-entry__time";
  timestamp.textContent = formatLogTimestamp();

  const content = document.createElement("pre");
  content.textContent = safe;

  meta.append(typeBadge, timestamp);
  entry.append(meta, content);
  els.eventLog.prepend(entry);
}

function renderLogCaptureState() {
  if (els.logCaptureToggle) {
    els.logCaptureToggle.checked = state.logCaptureEnabled;
  }
  if (!els.logCaptureHint) {
    return;
  }
  const captureLocation = state.turnCaptureDir || state.serviceLogsDir || "";
  els.logCaptureHint.textContent = state.logCaptureEnabled
    ? "Capture is on. Service logs and per-turn user audio/transcripts are being written for this runtime."
    : "Capture is off. No service logs or per-turn user audio/transcripts are being written.";
  els.logCaptureHint.title = captureLocation || "";
}

function applyVoiceLoggingState(loggingState) {
  if (!loggingState || typeof loggingState !== "object") {
    renderLogCaptureState();
    return;
  }
  state.logCaptureEnabled = Boolean(loggingState.enabled);
  state.serviceLogsDir = String(loggingState.logsDir || state.serviceLogsDir || "");
  state.turnCaptureDir = String(loggingState.turnCaptureDir || state.turnCaptureDir || "");
  renderLogCaptureState();
}

async function setVoiceLoggingEnabled(enabled) {
  const nextEnabled = Boolean(enabled);
  try {
    if (window.desktopApp.setVoiceLoggingEnabled) {
      const loggingState = await window.desktopApp.setVoiceLoggingEnabled(nextEnabled);
      applyVoiceLoggingState(loggingState);
    } else {
      state.logCaptureEnabled = nextEnabled;
      renderLogCaptureState();
    }
    if (isRealtimeSocketOpen()) {
      sendRealtimeJson({
        type: "logging.configure",
        enabled: state.logCaptureEnabled
      });
    }
  } catch (error) {
    appendLog("logging.capture_error", String(error));
    renderLogCaptureState();
  }
}

function renderMicrophoneState() {
  const enabled = state.microphoneEnabled;
  els.microphoneBtn.classList.toggle("is-active", enabled);
  els.microphoneBtn.setAttribute("aria-pressed", String(enabled));
  els.microphoneBtn.setAttribute("aria-label", enabled ? "关闭麦克风" : "开启麦克风");
  els.microphoneBtn.title = enabled ? "麦克风已开启" : "麦克风已关闭";
}

function setConnected(connected) {
  const voiceConnected = connected && state.sessionMode === "voice";
  els.connectBtn.disabled = voiceConnected || (connected && !state.microphoneEnabled) || state.serviceBusy;
  els.microphoneBtn.disabled = state.serviceBusy;
  els.disconnectBtn.disabled = (!connected && !state.serviceBusy) || state.serviceBusy;
  els.clearContextBtn.disabled = !connected || state.serviceBusy;
  updateTextComposerState();
  void refreshAvatarButton();
}

function applyBackendConfig(backendConfig) {
  state.backend = backendConfig;
  els.httpBase.textContent = backendConfig.httpBase;
  els.wsUrl.textContent = backendConfig.wsUrl;
}

function setServiceBusy(busy) {
  state.serviceBusy = busy;
  setConnected(Boolean(state.ws));
}

async function enableVoiceInput() {
  if (isVoiceSessionActive() && state.micStream) {
    return true;
  }

  setServiceBusy(true);
  setBackendState("Preparing Audio", "warn");
  setOrbMode("", "Requesting microphone access...");

  try {
    await ensureMicrophone();
    state.sessionMode = "voice";
    setConnected(Boolean(state.ws));
    setBackendState("WebSocket Live", "live");
    if (!hasActiveTurn()) {
      setTurnPhase("listening");
      setOrbMode("live", "Microphone is ready. Waiting for speech.");
    } else {
      setOrbMode("live", "Microphone is ready. Voice interruption is available.");
    }
    return true;
  } catch (error) {
    appendLog("mic.error", String(error));
    setBackendState("Microphone Error", "warn");
    setOrbMode("", "Microphone access failed.");
    return false;
  } finally {
    setServiceBusy(false);
  }
}

async function ensurePlaybackContext() {
  if (!state.playbackContext || state.playbackContext.state === "closed") {
    state.playbackContext = new AudioContext();
    const analyser = state.playbackContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.82;
    analyser.connect(state.playbackContext.destination);
    state.playbackAnalyser = analyser;
    window.__ttsAudioAnalyser = analyser;
    state.playbackNextTime = 0;
  }
  if (state.playbackContext.state === "suspended") {
    await state.playbackContext.resume();
  }
  return state.playbackContext;
}

function decodeBase64PcmF32(data) {
  const binary = atob(data || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

async function playPcmF32Chunk(data, sampleRate) {
  const audioContext = await ensurePlaybackContext();
  const pcm = decodeBase64PcmF32(data);
  if (!pcm.length) {
    return;
  }

  const buffer = audioContext.createBuffer(1, pcm.length, sampleRate || 44100);
  buffer.copyToChannel(pcm, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(state.playbackAnalyser || audioContext.destination);
  state.playbackSources.add(source);
  source.onended = () => {
    state.playbackSources.delete(source);
  };

  const startAt = Math.max(audioContext.currentTime + 0.02, state.playbackNextTime || 0);
  source.start(startAt);
  state.playbackNextTime = startAt + buffer.duration;
}

// ---- 化身口型 ---------------------------------------------------------------
// GPT-SoVITS 只回传裸 PCM，没有音素对齐，所以口型走「能量定开合 + 频谱重心定圆展」
// 的近似：张嘴(aa) / 展唇(ih, ee) / 圆唇(ou, oh)。通道名与 化身服务
// face/mapping.mjs 的 FACE_CHANNELS 一致，经 overrides 直接 pin 到面部解算结果上。
const VISEME_CLOSED = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
const VISEME_INTERVAL_MS = 33; // 约 30Hz，足够口型跟手又不会淹没 IPC

let visemeTimeBuffer = null;
let visemeFreqBuffer = null;
let visemeTimer = null;
let visemeQuietFrames = 0;

function clamp01(value) {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

function computeVisemes(analyser) {
  const bins = analyser.frequencyBinCount;
  if (!visemeTimeBuffer || visemeTimeBuffer.length !== bins * 2) {
    visemeTimeBuffer = new Uint8Array(bins * 2);
    visemeFreqBuffer = new Uint8Array(bins);
  }
  analyser.getByteTimeDomainData(visemeTimeBuffer);
  analyser.getByteFrequencyData(visemeFreqBuffer);

  let sumSquares = 0;
  for (let i = 0; i < visemeTimeBuffer.length; i += 1) {
    const sample = (visemeTimeBuffer[i] - 128) / 128;
    sumSquares += sample * sample;
  }
  const rms = Math.sqrt(sumSquares / visemeTimeBuffer.length);
  // 静音底噪以下完全闭嘴，越过门限后线性张开。
  const openness = clamp01((rms - 0.012) / 0.13);
  if (openness <= 0) {
    return VISEME_CLOSED;
  }

  // 低频占比高 → 圆唇；中高频占比高 → 展唇。按各自 bin 数取均值再比，避免 bin 数不等造成偏置。
  const lowEnd = Math.max(1, Math.round(bins * 0.08));
  const highStart = Math.round(bins * 0.3);
  let low = 0;
  let high = 0;
  for (let i = 1; i < lowEnd; i += 1) low += visemeFreqBuffer[i];
  for (let i = highStart; i < bins; i += 1) high += visemeFreqBuffer[i];
  const lowAvg = low / Math.max(1, lowEnd - 1);
  const highAvg = high / Math.max(1, bins - highStart);
  const roundness = clamp01((lowAvg / (lowAvg + highAvg + 1e-6) - 0.4) / 0.25);

  return {
    aa: openness * (1 - 0.55 * roundness),
    ou: openness * 0.65 * roundness,
    oh: openness * 0.45 * roundness,
    ih: openness * 0.3 * (1 - roundness),
    ee: openness * 0.22 * (1 - roundness)
  };
}

function pushVisemes(values) {
  const bridge = window.desktopApp;
  if (bridge && typeof bridge.sendVisemes === "function") {
    bridge.sendVisemes(values);
  }
}

// 化身是可选能力：主进程未提供桥接时全部静默跳过，不影响语音对话。
function avatarBridge() {
  return window.desktopApp || null;
}

function sendMotionPlan(plan) {
  const bridge = avatarBridge();
  if (bridge && typeof bridge.sendMotionPlan === "function" && plan) {
    bridge.sendMotionPlan(plan, "");
  }
}


function sendMotionSpeaking(value) {
  const bridge = avatarBridge();
  if (bridge && typeof bridge.sendMotionSpeaking === "function") {
    bridge.sendMotionSpeaking(value);
  }
}

// ---- 化身层（窗口内，取代原来的着色器球体背景）-------------------------------
// 化身页面跑在 <webview> 里，它与 .scene-backdrop 同为全窗口背景层，互斥显示。
//
// 关于尺寸：<webview> 不接受百分比尺寸，也不会跟着父元素自动重排 —— 用 % 或
// 默认尺寸时 guest 的视口会永远卡在 150px（HTML 替换元素的默认高度），角色被压成
// 一条。所以这里一律按窗口像素显式设定，并在窗口缩放时同步。
let avatarView = null;
let avatarResizeBound = false;

function sizeAvatarView() {
  if (!avatarView) {
    return;
  }
  avatarView.style.width = `${window.innerWidth}px`;
  avatarView.style.height = `${window.innerHeight}px`;
}

function ensureAvatarView(url) {
  if (avatarView) {
    return avatarView;
  }
  const stage = document.getElementById("avatarStage");
  if (!stage) {
    return null;
  }
  avatarView = document.createElement("webview");
  avatarView.id = "avatarView";
  avatarView.setAttribute("src", url);
  avatarView.style.display = "flex";
  avatarView.style.border = "0";
  stage.insertBefore(avatarView, document.getElementById("avatarStageHint"));
  sizeAvatarView();

  avatarView.addEventListener("did-finish-load", () => appendLog("avatar.loaded", { url }));
  avatarView.addEventListener("did-fail-load", (event) => {
    appendLog("avatar.load_error", String(event.errorDescription || event.errorCode));
  });
  // 页面就绪由主进程转发的 avatar.ready 事件告知（见 hast-preload 的 notifyReady）。

  if (!avatarResizeBound) {
    window.addEventListener("resize", sizeAvatarView);
    avatarResizeBound = true;
  }
  appendLog("avatar.created", { url });
  return avatarView;
}

// 切入化身模式时才创建页面：化身服务 要加载 VRM 与策略（十几秒），没必要在
// 用户没看角色时一直占着。创建后常驻，来回切换是瞬时的。
function setAvatarMode(on, url) {
  if (on) {
    document.body.classList.add("avatar-mode");
    if (!avatarView) {
      // 只在首次创建时撤销就绪标记 —— avatar.ready 只会来一次, 每次切换都撤销的话
      // "正在加载"提示会在第二次切换后一直挂着。
      document.body.classList.remove("avatar-ready");
      ensureAvatarView(url);
    }
    sizeAvatarView();
    // 重新进入时把记着的取景推过去，让页面与新按钮状态一致（首次创建时页面还没
    // 就绪，这次推送会被丢弃，由 avatar.ready 那次补上）。
    const bridge = avatarBridge();
    if (avatarView && bridge && typeof bridge.sendFraming === "function") {
      bridge.sendFraming(state.framing);
    }
  } else {
    document.body.classList.remove("avatar-mode");
  }
  updateAvatarButton();
}

// 收回化身层：退出化身模式并销毁内嵌页面。化身服务停掉后页面就没用了，
// 下次进入化身模式时 ensureAvatarView 会重新创建。
function destroyAvatarView() {
  document.body.classList.remove("avatar-mode");
  document.body.classList.remove("avatar-ready");
  if (avatarView) {
    avatarView.remove();
    avatarView = null;
  }
  // 页面被销毁，下次是全新加载，取景回到默认的全身，状态要跟着走。
  state.framing = "full";
  updateAvatarButton();
}

function updateAvatarButton() {
  const on = document.body.classList.contains("avatar-mode");
  els.avatarBtn.disabled = !state.avatarAvailable || state.serviceBusy;
  els.avatarBtn.setAttribute("aria-pressed", String(on));
  els.avatarBtn.setAttribute("aria-label", on ? "隐藏角色" : "显示角色");
  els.avatarBtn.title = on ? "隐藏角色，恢复能量球" : "显示角色，替换能量球";

  // 取景按钮只在角色显示时才有意义。
  const bust = state.framing === "bust";
  els.framingBtn.disabled = !on;
  els.framingBtn.setAttribute("aria-pressed", String(bust));
  const label = bust ? "切换为全身取景" : "切换为上半身取景";
  els.framingBtn.setAttribute("aria-label", label);
  els.framingBtn.title = label;

  if (els.avatarModelSelect) {
    els.avatarModelSelect.disabled = state.avatarSwitching || state.avatarModels.length === 0;
  }
}

function renderAvatarModelPicker() {
  const select = els.avatarModelSelect;
  if (!select) {
    return;
  }
  const models = Array.isArray(state.avatarModels) ? state.avatarModels : [];
  if (!models.length) {
    select.disabled = true;
    return;
  }
  const signature = models.map((model) => `${model.key}:${model.label || ""}`).join("|");
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...models.map((model) => {
      const option = document.createElement("option");
      option.value = model.key;
      option.textContent = model.label || model.key;
      return option;
    }));
    select.dataset.signature = signature;
  }
  if (models.some((model) => model.key === state.avatarModel)) {
    select.value = state.avatarModel;
  }
  select.disabled = state.avatarSwitching;
}

function sendAvatarForm(form) {
  const ws = state.ws;
  if (ws && ws.readyState === WebSocket.OPEN && typeof form === "string") {
    ws.send(JSON.stringify({ type: "avatar.form", form }));
  }
}

async function switchAvatarModel(key, { manual = false } = {}) {
  const bridge = avatarBridge();
  if (!bridge || typeof bridge.setAvatarModel !== "function" || !key || key === state.avatarModel) {
    return;
  }
  if (state.avatarSwitching) {
    return;
  }
  if (manual) {
    // 用户手动选过形态后，自动切换在 90 秒内不覆盖他的选择。
    state.manualAvatarOverrideUntil = Date.now() + 90000;
  }
  state.avatarSwitching = true;
  renderAvatarModelPicker();
  updateAvatarButton();
  appendLog("avatar.model_switch", { model: key, manual });
  try {
    const status = await bridge.setAvatarModel(key);
    if (status) {
      state.avatarModel = status.model || key;
      if (Array.isArray(status.models) && status.models.length) {
        state.avatarModels = status.models;
      }
      if (status.url) {
        state.avatarUrl = status.url;
      }
    }
    state.lastAvatarSwitchAt = Date.now();
    sendAvatarForm(state.avatarModel);
  } catch (error) {
    appendLog("avatar.model_switch_error", String(error));
  } finally {
    state.avatarSwitching = false;
    renderAvatarModelPicker();
    updateAvatarButton();
  }
}

// 后端根据对话给出的形态建议 -> 带冷却的自动切换。
// red=布景之形（日常/温柔/伪装），blue=幻灭之形（虚无/负面/真实）。
function maybeAutoSwitchAvatarForm(plan) {
  if (!state.avatarAutoSwitch) {
    return;
  }
  const presentation = plan && plan.presentation;
  const form = presentation && presentation.form;
  if (!form || form === state.avatarModel) {
    return;
  }
  if (!state.avatarModels.some((model) => model.key === form)) {
    return;
  }
  if (!document.body.classList.contains("avatar-mode") || !state.avatarAvailable) {
    return;
  }
  if (state.avatarSwitching) {
    return;
  }
  const immediate = Boolean(presentation.form_immediate);
  const now = Date.now();
  if (!immediate) {
    if (now < state.manualAvatarOverrideUntil) {
      return;
    }
    // 3 秒左右的模型加载 + 自动切换不应过于频繁；重话题由 form_immediate 例外。
    if (now - state.lastAvatarSwitchAt < 25000) {
      return;
    }
  }
  state.lastAvatarSwitchAt = now;
  appendLog("avatar.auto_form", {
    form,
    expression: presentation.expression,
    body: presentation.body,
    immediate
  });
  void switchAvatarModel(form, { manual: false });
}

function toggleFraming() {
  state.framing = state.framing === "bust" ? "full" : "bust";
  const bridge = avatarBridge();
  if (bridge && typeof bridge.sendFraming === "function") {
    bridge.sendFraming(state.framing);
  }
  appendLog("avatar.framing", { framing: state.framing });
  updateAvatarButton();
}

async function refreshAvatarButton() {
  const bridge = avatarBridge();
  if (!bridge || typeof bridge.getAvatarStatus !== "function") {
    els.avatarBtn.disabled = true;
    return;
  }
  try {
    const status = await bridge.getAvatarStatus();
    state.avatarAvailable = Boolean(status && status.enabled && status.available);
    state.avatarUrl = (status && status.url) || state.avatarUrl;
    if (status && Array.isArray(status.models) && status.models.length) {
      state.avatarModels = status.models;
    }
    if (status && status.model) {
      state.avatarModel = status.model;
    }
  } catch (error) {
    appendLog("avatar.status_error", String(error));
    state.avatarAvailable = false;
  }
  renderAvatarModelPicker();
  updateAvatarButton();
}

function toggleAvatar() {
  if (!state.avatarAvailable) {
    return;
  }
  setAvatarMode(!document.body.classList.contains("avatar-mode"), state.avatarUrl);
}

// 后端每个音频帧都会推一次 vad（约 15 次/秒），只在状态真正翻转时才转发。
let lastListeningSent = null;

function sendMotionListening(value) {
  const next = Boolean(value);
  if (next === lastListeningSent) {
    return;
  }
  lastListeningSent = next;
  const bridge = avatarBridge();
  if (bridge && typeof bridge.sendMotionListening === "function") {
    bridge.sendMotionListening(next);
  }
}

function visemeTick() {
  const analyser = state.playbackAnalyser;
  if (!analyser) {
    stopVisemeLoop();
    return;
  }
  const values = computeVisemes(analyser);
  const level = values.aa + values.ih + values.ou + values.ee + values.oh;

  // 音频播完后多推几帧闭合值，确保嘴巴确实合上再停表，避免一直张着嘴。
  if (state.playbackSources.size === 0 && level <= 0.02) {
    visemeQuietFrames += 1;
    if (visemeQuietFrames > 4) {
      pushVisemes(VISEME_CLOSED);
      stopVisemeLoop();
      return;
    }
  } else {
    visemeQuietFrames = 0;
  }
  pushVisemes(values);
}

function startVisemeLoop() {
  if (visemeTimer !== null || !state.playbackAnalyser) {
    return;
  }
  visemeQuietFrames = 0;
  visemeTimer = setInterval(visemeTick, VISEME_INTERVAL_MS);
}

function stopVisemeLoop() {
  if (visemeTimer !== null) {
    clearInterval(visemeTimer);
    visemeTimer = null;
  }
  pushVisemes(VISEME_CLOSED);
}

function stopPlaybackQueue() {
  for (const source of state.playbackSources) {
    try {
      source.stop();
    } catch (_) {
      // Already stopped or not yet started.
    }
  }
  state.playbackSources.clear();
  if (state.playbackContext && state.playbackContext.state !== "closed") {
    state.playbackNextTime = state.playbackContext.currentTime;
  } else {
    state.playbackNextTime = 0;
  }
}

async function releasePlayback() {
  stopVisemeLoop();
  stopPlaybackQueue();
  state.playbackNextTime = 0;
  if (state.playbackAnalyser) {
    try {
      state.playbackAnalyser.disconnect();
    } catch (_) {
      // Ignore analyser disconnect failures during shutdown.
    }
    state.playbackAnalyser = null;
  }
  window.__ttsAudioAnalyser = null;
  if (state.playbackContext) {
    try {
      await state.playbackContext.close();
    } catch (_) {
      // Ignore close failures during shutdown.
    }
    state.playbackContext = null;
  }
}

async function ensureMicrophone() {
  if (state.micStream && state.micContext && state.micProcessor && state.micSource) {
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });

  const audioContext = new AudioContext({ sampleRate: 16000 });
  await audioContext.resume();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (event) => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      pcm[i] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    state.ws.send(pcm.buffer);
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  state.micStream = stream;
  state.micContext = audioContext;
  state.micSource = source;
  state.micProcessor = processor;
}

async function releaseMicrophone() {
  if (state.micProcessor) {
    state.micProcessor.disconnect();
    state.micProcessor.onaudioprocess = null;
    state.micProcessor = null;
  }
  if (state.micSource) {
    state.micSource.disconnect();
    state.micSource = null;
  }
  if (state.micContext) {
    try {
      await state.micContext.close();
    } catch (_) {
      // Ignore close failures during shutdown.
    }
    state.micContext = null;
  }
  if (state.micStream) {
    for (const track of state.micStream.getTracks()) {
      track.stop();
    }
    state.micStream = null;
  }
}

async function toggleMicrophone() {
  if (state.serviceBusy) {
    return;
  }

  if (state.microphoneEnabled) {
    state.microphoneEnabled = false;
    renderMicrophoneState();
    if (state.micStream) {
      setServiceBusy(true);
      await releaseMicrophone();
      state.sessionMode = state.ws ? "text" : "idle";
      setServiceBusy(false);
      setTurnPhase(hasActiveTurn() ? state.phase : "idle");
      setOrbMode(state.ws ? "live" : "", state.ws ? "麦克风已关闭，文本对话仍可使用。" : "麦克风已关闭，启动时不会申请权限。");
    }
    return;
  }

  state.microphoneEnabled = true;
  renderMicrophoneState();
  if (!state.ws) {
    setOrbMode("", "麦克风已开启，启动时将申请权限。");
    return;
  }

  const enabled = await enableVoiceInput();
  if (!enabled) {
    state.microphoneEnabled = false;
    renderMicrophoneState();
    setConnected(Boolean(state.ws));
  }
}

// Persona switcher: populates the header dropdown from the personas/ dir and,
// on change, persists the choice and (if the realtime socket is live) pushes
// a persona.configure so the backend swaps both the LLM system prompt and the
// GPT-SoVITS reference audio in one atomic step.
async function initPersonaSwitcher() {
  const select = document.getElementById("personaSelect");
  if (!select) {
    return;
  }
  let personas = [];
  try {
    personas = await window.desktopApp.getPersonas();
  } catch (error) {
    appendLog("persona.list_error", String(error));
  }
  select.innerHTML = "";
  if (!personas.length) {
    select.innerHTML = `<option value="">（无可用人格）</option>`;
    select.disabled = true;
    return;
  }
  for (const persona of personas) {
    const label = persona.source ? `${persona.name}（${persona.source}）` : persona.name;
    const option = document.createElement("option");
    option.value = persona.id;
    option.textContent = label;
    select.appendChild(option);
  }
  let current = { id: "" };
  try {
    current = await window.desktopApp.getPersonaConfig();
  } catch (error) {
    appendLog("persona.get_config_error", String(error));
  }
  const targetId = current.id && personas.some((p) => p.id === current.id) ? current.id : personas[0].id;
  select.value = targetId;
  state.personaId = targetId;
  select.disabled = false;
  select.addEventListener("change", () => {
    const personaId = select.value;
    state.personaId = personaId;
    window.desktopApp.setPersonaConfig({ id: personaId }).catch((error) => {
      appendLog("persona.set_config_error", String(error));
    });
    const sent = sendRealtimeJson({ type: "persona.configure", persona: personaId });
    appendLog("persona.switch", { persona: personaId, delivered: sent });
    if (!sent) {
      setOrbMode("", `人格「${select.selectedOptions[0]?.textContent || personaId}」已保存，将在下次启动服务时生效。`);
    }
  });
}

async function bootstrap() {
  const [appInfo, backendConfig] = await Promise.all([
    window.desktopApp.getInfo(),
    window.desktopApp.getBackendConfig()
  ]);

  applyBackendConfig(backendConfig);
  if (window.desktopApp.setVoiceLoggingEnabled) {
    try {
      // Keep runtime service-log capture enabled on startup so backend stdout/
      // stderr (including startup tracebacks) land in service-logs/*.log.
      // The renderer's logCaptureEnabled flag only governs ASR turn capture,
      // not this process-log file writing.
      const loggingState = await window.desktopApp.setVoiceLoggingEnabled(true);
      applyVoiceLoggingState(loggingState);
    } catch (error) {
      appendLog("logging.capture_init_error", String(error));
      renderLogCaptureState();
    }
  } else {
    renderLogCaptureState();
  }
  els.appVersion.textContent = `${appInfo.name} ${appInfo.version}`;
  appendLog("app.ready", { appInfo, backendConfig });
  void refreshAvatarButton();
  if (window.desktopApp.onVoiceServicesEvent) {
    window.desktopApp.onVoiceServicesEvent((event) => {
      appendLog(`services.${event.type}`, event);
      if (event.type === "logging.capture_changed") {
        applyVoiceLoggingState(event);
      }
      // 化身服务就绪/加载完成/出错时同步按钮与提示状态。
      if (event.type === "avatar.ready") {
        document.body.classList.add("avatar-ready");
      }
      // 服务被停掉（挂断、启动失败回滚、退出）时化身页面必然失效，直接收回。
      if (event.type === "stop.completed") {
        destroyAvatarView();
      }
      // 页面刚就绪时把当前取景推过去 —— 首次创建时页面还没加载完，那时的推送会被丢掉。
      if (event.type === "avatar.ready") {
        const bridge = avatarBridge();
        if (bridge && typeof bridge.sendFraming === "function") {
          bridge.sendFraming(state.framing);
        }
      }
      if (event.service === "avatar" || event.type.startsWith("avatar.")
        || event.type === "start.completed" || event.type === "stop.completed") {
        void refreshAvatarButton();
      }
      if (event.type === "process.output" && state.serviceBusy) {
        setBackendState(`${event.service} Starting`, "warn");
      } else if (event.type === "warmup.begin" && state.serviceBusy) {
        setBackendState("Warming Models", "warn");
        setOrbMode("", "ASR and TTS warm-up are running before the first turn.");
      } else if (event.type === "warmup.completed" && state.serviceBusy) {
        const totalMs = Number(event.total_ms);
        const asrMs = Number(event.asr_inference_ms);
        const ttsMs = Number(event.tts_inference_ms);
        const totalSummary = Number.isFinite(totalMs) ? `${Math.round(totalMs)} ms` : "completed";
        const detailSummary = [
          Number.isFinite(asrMs) ? `ASR ${Math.round(asrMs)} ms` : "",
          Number.isFinite(ttsMs) ? `TTS ${Math.round(ttsMs)} ms` : "",
        ].filter(Boolean).join(", ");
        setBackendState("Models Warmed", "ok");
        setOrbMode("live", `${detailSummary ? `${detailSummary}. ` : ""}Warm-up ${totalSummary}. Opening microphone next.`);
      } else if (event.type === "warmup.error" && state.serviceBusy) {
        setBackendState("Warmup Failed", "warn");
        setOrbMode("", event.message || "ASR/TTS warm-up failed.");
      }
    });
  }
  await loadLlmConfig();
  await refreshAssetsRoot();
  await refreshHistoryVaultRoot();
  await detectResources();
  await refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
  await loadTtsConfigFromStorage();
  await initPersonaSwitcher();
  setTextComposerLoading(false);
  updateTextComposerState();
  setConnected(false);
}

async function pingBackend() {
  if (!state.backend.httpBase) {
    setBackendState("Backend Not Configured", "warn");
    return;
  }

  setBackendState("Checking Backend", "warn");
  setOrbMode("", "Pinging Python backend...");

  try {
    const res = await fetch(`${state.backend.httpBase}/health`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    appendLog("health.ok", data);
    setBackendState("Backend Reachable", "ok");
    setOrbMode("live", "Backend is reachable. Realtime WebSocket can be opened next.");
    void refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
  } catch (error) {
    appendLog("health.error", String(error));
    setBackendState("Backend Offline", "warn");
    setOrbMode("", "Health check failed. Start the Python backend first.");
  }
}

function handleRealtimeMessage(message) {
  appendLog(message.type || "message", message);

  switch (message.type) {
    case "session.ready":
      state.sessionReady = true;
      state.textInputEnabled = true;
      setTextComposerLoading(false);
      setBackendState("WebSocket Live", "live");
      setOrbMode("live", getReadyOrbMessage());
      setTurnPhase(getReadyTurnPhase());
      schedulePendingTextFlush("session.ready", 0);
      break;
    case "server.ready":
      setBackendState("WebSocket Live", "live");
      setOrbMode("live", "Realtime transport connected.");
      break;
    case "vad":
      sendMotionListening(Boolean(message.speaking));
      if (message.speaking) {
        state.speechActive = true;
        if (state.phase === "llm" || state.playbackSources.size > 0) {
          stopPlaybackQueue();
          setTurnPhase("listening");
          setConversationStage("assistant", "Interrupted", "warn");
          setOrbMode("listening", "User speech interrupted the assistant.");
        }
        if (state.phase === "idle" || state.phase === "completed" || state.phase === "listening") {
          setTurnPhase("listening");
        }
        setOrbMode("listening", "User speech detected.");
      } else if (state.speechActive && state.phase === "listening") {
        state.speechActive = false;
        setTurnPhase("speech-ended");
        setOrbMode("live", "Speech ended. Waiting for ASR.");
      } else if (state.phase === "idle" || state.phase === "completed" || state.phase === "listening") {
        setOrbMode("live", "Waiting for the next speech segment.");
      }
      break;
    case "asr.started":
      state.speechActive = false;
      state.currentAssistantMessageId = null;
      beginConversationMessage("user", "Transcribing...", "ASR", "active");
      setTurnPhase("asr");
      setOrbMode("listening", "Speech segment captured. ASR is transcribing.");
      break;
    case "asr.completed":
    case "user_text":
      setConversationText("user", message.text || "", "Done", "done");
      break;
    case "assistant.started":
    case "generating":
      state.currentTurnId = message.turn_id || state.currentTurnId;
      if (state.pendingTextRequest?.dispatched) {
        clearPendingTextRequest();
      }
      if (!getCurrentConversationMessage("assistant")) {
        beginConversationMessage("assistant", "", "LLM", "active");
      }
      setTurnPhase("llm");
      setOrbMode("live", "LLM is thinking...");
      break;
    case "assistant.delta":
      appendConversationText("assistant", message.text);
      break;
    case "text":
      appendConversationText("assistant", message.content);
      break;
    case "assistant.motion":
      // 动作意图与主回复并行解析，偶尔会比本轮更晚回来。若已经有更新的一轮开始，
      // 这条就是上一轮的遗留，丢掉它，否则角色会做出与当前对话无关的手势。
      if (message.turn_id && state.currentTurnId && message.turn_id !== state.currentTurnId) {
        appendLog("assistant.motion.stale", { turn_id: message.turn_id, current_turn_id: state.currentTurnId });
        break;
      }
      // 后端已把本轮用户话语解析成动作/表情计划，转给化身窗口直接驱动角色。
      appendLog("assistant.motion", message.plan);
      maybeAutoSwitchAvatarForm(message.plan);
      sendMotionPlan(message.plan);
      break;
    case "assistant.audio.start":
      void ensurePlaybackContext().then(() => startVisemeLoop());
      sendMotionSpeaking(true);
      setConversationStage("assistant", "Speaking", "active");
      setOrbMode("speaking", "TTS streaming has started.");
      break;
    case "assistant.audio.chunk":
      void playPcmF32Chunk(message.data, message.sample_rate).catch((error) => {
        appendLog("audio.play_error", String(error));
      });
      break;
    case "assistant.audio.completed":
      if (message.interrupted) {
        stopPlaybackQueue();
      }
      sendMotionSpeaking(false);
      setConversationStage("assistant", message.interrupted ? "Interrupted" : "Spoken", message.interrupted ? "warn" : "done");
      break;
    case "assistant.audio.error":
      setConversationStage("assistant", "Audio Error", "warn");
      setOrbMode("", "TTS failed while generating audio.");
      break;
    case "tts.configured":
      appendLog("tts.configured", message);
      break;
    case "persona.configured":
      appendLog("persona.configured", message);
      if (message.persona && message.persona.name) {
        setOrbMode("live", `人格已切换：${message.persona.name}`);
      }
      break;
    case "logging.configured":
      applyVoiceLoggingState({
        enabled: message.enabled,
        turnCaptureDir: message.capture_dir
      });
      break;
    case "llm.configured":
      // 后端确认话痨档位已更新；若与本地不一致则以本地为准（用户可能在途中又调了）。
      appendLog("llm.configured", message);
      if (message.talkativeness && message.talkativeness !== state.llm.talkativeness) {
        state.llm.talkativeness = message.talkativeness;
        updateTalkativenessDisplay();
      }
      break;
    case "assistant.completed":
    case "done":
      if (message.interrupted) {
        stopPlaybackQueue();
        setTurnPhase(getReadyTurnPhase());
        setConversationStage("assistant", "Interrupted", "warn");
        setOrbMode(getReadyOrbMode(), "Assistant interrupted. Ready for the queued turn.");
        schedulePendingTextFlush("assistant.completed.interrupted", 80);
      } else {
        const assistantMessage = getCurrentConversationMessage("assistant");
        if (message.text && (!assistantMessage || !(assistantMessage.text || "").trim())) {
          setConversationText("assistant", message.text, "Done", "done");
        } else if (assistantMessage && !(assistantMessage.text || "").trim()) {
          setConversationText("assistant", "LLM returned no speakable text.", "Error", "warn");
          setBackendState("LLM Empty", "warn");
          setOrbMode("", "LLM completed without text. Check whether thinking mode is disabled.");
          break;
        }
        setTurnPhase("completed");
        setOrbMode("live", "Turn completed. Ready for the next user utterance.");
        schedulePendingTextFlush("assistant.completed", 0);
      }
      // The backend persists the turn just after emitting this event, so delay
      // the refresh slightly to pick it up when the history panel is visible.
      if (state.activeToolPanel === "history" && historyState.view === "sessions" && !historyState.activeSessionId) {
        setTimeout(() => {
          void refreshHistorySessions();
        }, 800);
      }
      break;
    case "assistant.error":
      setTurnPhase(getReadyTurnPhase());
      setConversationText("assistant", `LLM error: ${message.message || "unknown error"}`, "Error", "warn");
      setBackendState("LLM Error", "warn");
      setOrbMode(
        "",
        state.llm.mode === "api"
          ? "LLM request failed. Check the remote API base URL, key, and model name."
          : "LLM request failed. Check the local LLM server."
      );
      schedulePendingTextFlush("assistant.error", 120);
      break;
    case "asr.error":
      setTurnPhase(getReadyTurnPhase());
      setConversationText("user", `ASR error: ${message.message || "unknown error"}`, "Error", "warn");
      setBackendState("ASR Error", "warn");
      setOrbMode("", "ASR failed while processing the speech segment.");
      schedulePendingTextFlush("asr.error", 120);
      break;
    case "interrupt.ack":
      state.textInterruptPending = false;
      if (message.accepted) {
        stopPlaybackQueue();
        setTurnPhase(getReadyTurnPhase());
        setConversationStage("assistant", "Interrupted", "warn");
        setOrbMode(getReadyOrbMode(), "Interrupt acknowledged. Preparing the queued turn.");
        schedulePendingTextFlush("interrupt.ack", 180);
      } else {
        setOrbMode("live", "Interrupt acknowledged.");
        schedulePendingTextFlush("interrupt.ack.noop", 0);
      }
      updateTextComposerState();
      break;
    case "error":
      if (retryPendingTextAfterBusyError(message)) {
        break;
      }
      if (shouldSuppressBusyBackendError(message)) {
        appendLog("backend.busy_ignored", {
          message: message.message || "",
          phase: state.phase,
          speechActive: state.speechActive
        });
        setOrbMode("listening", "The interrupted turn is still winding down. Waiting for the new voice turn.");
        break;
      }
      setTurnPhase(getReadyTurnPhase());
      setConversationText("assistant", `Backend error: ${message.message || "unknown error"}`, "Error", "warn");
      setBackendState("Backend Error", "warn");
      setOrbMode("", "The backend reported an error.");
      break;
    default:
      break;
  }
}

async function connectRealtime(options = {}) {
  const withMicrophone = options.withMicrophone !== false;
  if (!state.backend.wsUrl || state.serviceBusy) {
    return false;
  }
  if (state.ws) {
    if (withMicrophone) {
      return enableVoiceInput();
    }
    return true;
  }

  state.sessionMode = withMicrophone ? "voice" : "text";
  state.sessionReady = false;
  state.textInputEnabled = false;
  setTextComposerLoading(true);
  updateTextComposerState();

  try {
    await ensurePlaybackContext();
    appendLog("audio.playback_ready", { state: state.playbackContext?.state || "unknown" });
  } catch (error) {
    appendLog("audio.context_error", String(error));
    setBackendState("Audio Output Error", "warn");
    setOrbMode("", "Failed to initialize speaker playback.");
    setTextComposerLoading(false);
    restorePendingTextDraft();
    return false;
  }

  setServiceBusy(true);
  setBackendState("Starting Services", "warn");
  setOrbMode(
    "",
    state.llm.mode === "api"
      ? "Starting realtime backend with remote LLM API..."
      : "Starting realtime backend..."
  );

  try {
    if (window.desktopApp.startVoiceServices) {
      const services = await window.desktopApp.startVoiceServices(state.ttsOptions);
      appendLog("services.ready", services);
      if (services.backend) {
        applyBackendConfig(services.backend);
      }
    }
    await refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
  } catch (error) {
    appendLog("services.start_error", String(error));
    setBackendState("Service Start Failed", "warn");
    setOrbMode("", "Failed to start local services. Check the service log.");
    setServiceBusy(false);
    setTextComposerLoading(false);
    restorePendingTextDraft();
    return false;
  }

  if (withMicrophone) {
    setBackendState("Preparing Audio", "warn");
    setOrbMode("", "Services are ready. Requesting microphone access...");

    try {
      await ensureMicrophone();
    } catch (error) {
      appendLog("mic.error", String(error));
      setBackendState("Microphone Error", "warn");
      setOrbMode("", "Microphone access failed.");
      if (window.desktopApp.stopVoiceServices) {
        await window.desktopApp.stopVoiceServices().catch((stopError) => appendLog("services.stop_error", String(stopError)));
      }
      setServiceBusy(false);
      setTextComposerLoading(false);
      restorePendingTextDraft();
      return false;
    }
  }

  setBackendState("Opening WebSocket", "warn");
  setOrbMode("", withMicrophone ? "Microphone is ready. Opening realtime connection..." : "Services are ready. Opening realtime text session...");

  const ws = new WebSocket(state.backend.wsUrl);
  state.ws = ws;
  ws.binaryType = "arraybuffer";
  setTurnPhase("idle");

  ws.onopen = () => {
    console.log(`[VOICE-FE] ws.onopen url=${state.backend.wsUrl}`);
    setServiceBusy(false);
    setConnected(true);
    appendLog("ws.open", { url: state.backend.wsUrl });
    setTurnPhase(getReadyTurnPhase());
    setOrbMode("live", withMicrophone ? "Realtime session opened. Waiting for speech." : "Realtime text session opened.");
    applyTtsOptions(false);
    ws.send(JSON.stringify({
      type: "session.start",
      client: "electron-shell",
      tts_options: state.ttsOptions,
      logging_enabled: state.logCaptureEnabled
    }));
    // 重新连接后把持久化的话痨档位推给后端，确保下一轮立即按当前档位生成。
    sendRealtimeJson({ type: "llm.configure", talkativeness: state.llm.talkativeness });
    // 让后端的动作意图模型知道当前形态，避免手动选过模型后被默认值拉回。
    sendAvatarForm(state.avatarModel);
  };

  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      // Surface VAD/speech/asr events in the terminal so we can see whether the
      // backend is detecting speech at all. Only log a compact prefix to avoid
      // flooding the terminal with every vad speaking=true tick.
      if (message && (message.type === "session.ready" || message.type === "vad" || message.type === "assistant.started" || message.type === "assistant.delta" || message.type === "asr.started" || message.type === "asr.result" || message.type === "llm.configured" || message.type === "error")) {
        console.log(`[VOICE-FE] recv ${message.type}: ${JSON.stringify(message).slice(0, 200)}`);
      }
      handleRealtimeMessage(message);
    } catch (error) {
      appendLog("ws.parse_error", String(error));
    }
  };

  ws.onerror = (ev) => {
    console.log(`[VOICE-FE] ws.onerror: ${JSON.stringify(ev?.message || ev || "")}`);
    setServiceBusy(false);
    setTextComposerLoading(false);
    setBackendState("WebSocket Error", "warn");
    setOrbMode("", "WebSocket failed to connect.");
  };

  ws.onclose = async (ev) => {
    console.log(`[VOICE-FE] ws.onclose: code=${ev?.code} reason=${JSON.stringify(ev?.reason || "")} wasClean=${ev?.wasClean}`);
    appendLog("ws.close", { url: state.backend.wsUrl });
    state.ws = null;
    state.speechActive = false;
    state.sessionReady = false;
    state.sessionMode = "idle";
    state.textInputEnabled = false;
    state.textInterruptPending = false;
    setTextComposerLoading(false);
    restorePendingTextDraft();
    setTurnPhase("idle");
    setConnected(false);
    setBackendState("Backend Offline", "warn");
    setOrbMode("", "Realtime connection closed.");
    await releaseMicrophone();
    await releasePlayback();
  };
  return true;
}

async function closeRealtimeSocket() {
  if (state.ws) {
    const ws = state.ws;
    const closed = await new Promise((resolve) => {
      let settled = false;
      const finish = (value) => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(value);
        }
      };
      const timer = setTimeout(() => finish(false), 3000);
      const originalOnClose = ws.onclose;
      ws.onclose = async (event) => {
        if (originalOnClose) {
          await originalOnClose.call(ws, event);
        }
        finish(true);
      };
      try {
        ws.send(JSON.stringify({ type: "session.stop" }));
      } catch (_) {
        // Ignore close-path send failures.
      }
      ws.close();
    });
    if (!closed && state.ws === ws) {
      appendLog("ws.close_timeout", { url: state.backend.wsUrl });
      state.ws = null;
      state.speechActive = false;
      state.sessionReady = false;
      state.sessionMode = "idle";
      state.textInputEnabled = false;
      state.textInterruptPending = false;
      setTextComposerLoading(false);
      restorePendingTextDraft();
      setTurnPhase("idle");
      setConnected(false);
      await releaseMicrophone();
      await releasePlayback();
    }
    return;
  }
  await releaseMicrophone();
  await releasePlayback();
}

async function disconnectRealtime() {
  if (state.serviceBusy) {
    return;
  }

  setServiceBusy(true);
  clearPendingTextRequest();
  state.sessionReady = false;
  state.sessionMode = "idle";
  state.textInputEnabled = false;
  setTextComposerLoading(false);
  setBackendState("Stopping Session", "warn");
  setOrbMode("", "Stopping realtime session and local services...");
  clearConversation();

  try {
    await closeRealtimeSocket();
    if (window.desktopApp.stopVoiceServices) {
      const services = await window.desktopApp.stopVoiceServices();
      appendLog("services.stopped", services);
    }
    setBackendState("Services Stopped", "warn");
    setOrbMode("", "Voice session stopped. Managed local services were cleaned up.");
  } catch (error) {
    appendLog("services.stop_error", String(error));
    setBackendState("Stop Failed", "warn");
    setOrbMode("", "Failed to stop local services cleanly.");
  } finally {
    setServiceBusy(false);
    setConnected(false);
    // 挂断会连带停掉化身服务，内嵌页面随之失效，必须收回化身层：
    // 否则角色会僵在画面上、能量球也不再出现。下次开始语音后会重新创建。
    destroyAvatarView();
    void refreshAvatarButton();
  }
}

els.minBtn.addEventListener("click", () => window.desktopWindow.minimize());
els.maxBtn.addEventListener("click", () => window.desktopWindow.toggleMaximize());
els.closeBtn.addEventListener("click", () => window.desktopWindow.close());
// els.pingBtn.addEventListener("click", () => void pingBackend());
els.detectResourcesBtn.addEventListener("click", () => void detectResources());
els.connectBtn.addEventListener("click", () => void connectRealtime({ withMicrophone: state.microphoneEnabled }));
els.microphoneBtn.addEventListener("click", () => void toggleMicrophone());
els.disconnectBtn.addEventListener("click", () => void disconnectRealtime());
els.clearContextBtn.addEventListener("click", clearContext);
els.avatarBtn.addEventListener("click", () => toggleAvatar());
els.framingBtn.addEventListener("click", () => toggleFraming());
els.avatarModelSelect?.addEventListener("change", (event) => {
  void switchAvatarModel(event.target.value, { manual: true });
});
els.logCaptureToggle?.addEventListener("change", (event) => {
  void setVoiceLoggingEnabled(event.target.checked);
});
els.openServiceLogsBtn.addEventListener("click", () => {
  window.desktopApp.openVoiceServicesLogFolder().catch((error) => appendLog("services.open_log_folder_error", String(error)));
});
els.browseModelBtn.addEventListener("click", () => void browseForModel());
els.applyLlmBtn.addEventListener("click", () => void applyLlmConfig());
els.llmModeLocalBtn.addEventListener("click", () => setLlmMode("local", { persist: true }));
els.llmModeApiBtn.addEventListener("click", () => setLlmMode("api", { persist: true }));
// 话痨程度滑块：拖动释放或点击档位标签即生效。
els.talkativenessRange?.addEventListener("input", () => updateTalkativenessDisplay());
els.talkativenessRange?.addEventListener("change", () => applyTalkativeness({ sendToBackend: true }));
for (const stop of els.talkativenessStops) {
  stop.addEventListener("click", () => {
    els.talkativenessRange.value = stop.dataset.level;
    applyTalkativeness({ sendToBackend: true });
  });
}
els.browseAudioBtn.addEventListener("click", () => void browseForAudio());
els.browseAssetsRootBtn.addEventListener("click", () => void browseForAssetsRoot());
els.resetAssetsRootBtn.addEventListener("click", () => void resetAssetsRoot());
els.refreshLoraBtn.addEventListener("click", () => {
  refreshLoraCatalog().catch((error) => appendLog("tts.lora.catalog_error", String(error)));
});
els.applyTtsBtn.addEventListener("click", () => applyTtsOptions(true));
els.ttsModelVersion.addEventListener("change", () => applyTtsOptions(false));
els.ttsLoraSelect.addEventListener("change", () => applyTtsOptions(false));
els.promptAudioPath.addEventListener("input", () => applyTtsOptions(false));
els.promptAudioPath.addEventListener("change", () => applyTtsOptions(false));
els.promptText.addEventListener("change", () => applyTtsOptions(false));
els.ttsCfgValue.addEventListener("change", () => applyTtsOptions(false));
els.ttsInferenceTimesteps.addEventListener("change", () => applyTtsOptions(false));
els.ttsSeed.addEventListener("change", () => applyTtsOptions(false));
for (const tab of els.toolTabs) {
  tab.addEventListener("click", () => setActiveToolPanel(tab.dataset.panelTarget));
}
els.textPromptInput.addEventListener("input", updateTextComposerState);
els.textPromptInput.addEventListener("compositionstart", () => {
  state.textInputComposing = true;
  updateTextComposerState();
});
els.textPromptInput.addEventListener("compositionend", () => {
  state.textInputComposing = false;
  updateTextComposerState();
});
els.textComposer.addEventListener("submit", handleTextComposerSubmit);

// ---------------------------------------------------------------------------
// History panel: browse/edit/delete persisted conversations and long-term
// memory. The Markdown vault on disk is the human-editable view of the same
// data; "同步文件改动" reconciles edits/deletions made outside the app.
// ---------------------------------------------------------------------------

const historyState = {
  view: "sessions",
  includeDeleted: false,
  sessions: [],
  memories: [],
  activeSessionId: null,
  detail: null,
  database: null,
  databasePath: "",
  databasePathSource: "default",
  databaseFileName: "whispera.db"
};

function historyBase() {
  const base = state.backend.httpBase;
  return base ? base.replace(/\/+$/, "") : "";
}

async function historyRequest(path, options = {}) {
  const base = historyBase();
  if (!base) {
    throw new Error("后端地址未知，请先启动服务");
  }
  const response = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const raw = await response.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch (_) {
      data = null;
    }
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : response.statusText;
    throw new Error(`${response.status} ${detail}`);
  }
  return data;
}

function setHistoryStatus(text) {
  els.historyStatus.textContent = text || "";
}

function formatHistoryTime(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "-";
  }
  const date = new Date(seconds * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function historyEmpty(text) {
  const div = document.createElement("div");
  div.className = "history-empty";
  div.textContent = text;
  return div;
}

function historyActionButton(label, handler, { danger = false } = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `ghost-btn compact-btn history-btn${danger ? " history-btn--danger" : ""}`;
  button.textContent = label;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    Promise.resolve(handler()).catch((error) => {
      setHistoryStatus(`操作失败：${error.message}`);
      appendLog("history.action_error", String(error));
    });
  });
  return button;
}

function historyRow({ text, meta, deleted = false, onEdit = null, actions = [] }) {
  const item = document.createElement("div");
  item.className = `history-item${deleted ? " history-item--deleted" : ""}`;

  const body = document.createElement("div");
  body.className = "history-item__body";

  const textEl = document.createElement("div");
  textEl.className = "history-item__text";
  textEl.textContent = deleted ? `（已删除）${text}` : text;
  body.appendChild(textEl);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "history-item__meta";
    metaEl.textContent = meta;
    body.appendChild(metaEl);
  }
  item.appendChild(body);

  const bar = document.createElement("div");
  bar.className = "history-item__actions";
  if (onEdit) {
    bar.appendChild(historyActionButton("编辑", () => beginEdit()));
  }
  for (const action of actions) {
    bar.appendChild(action);
  }
  item.appendChild(bar);

  function beginEdit() {
    const editor = document.createElement("textarea");
    editor.className = "history-editor";
    editor.value = text;
    const editorBar = document.createElement("div");
    editorBar.className = "history-item__actions";
    editorBar.appendChild(historyActionButton("保存", async () => {
      const value = editor.value.trim();
      if (!value) {
        return;
      }
      await onEdit(value);
    }));
    editorBar.appendChild(historyActionButton("取消", () => renderHistory()));
    body.replaceChild(editor, textEl);
    bar.replaceChildren(editorBar);
    editor.focus();
  }

  return item;
}

async function openHistoryPanel() {
  await refreshHistoryVaultRoot();
  await refreshHistory();
}

async function refreshHistory() {
  if (historyState.view === "sessions") {
    await refreshHistorySessions();
  } else if (historyState.view === "memories") {
    await refreshHistoryMemories();
  } else {
    await refreshHistoryDatabase();
  }
}

function historyQuery() {
  return historyState.includeDeleted ? "?include_deleted=true" : "";
}

async function refreshHistorySessions() {
  if (!historyBase()) {
    setHistoryStatus("后端未连接。启动服务后可查看历史。");
    renderHistory();
    return;
  }
  try {
    const data = await historyRequest(`/history/sessions${historyQuery()}`);
    historyState.sessions = data.sessions || [];
    setHistoryStatus(`共 ${historyState.sessions.length} 个会话。`);
    renderHistory();
  } catch (error) {
    setHistoryStatus(`加载失败：${error.message}`);
    appendLog("history.sessions.error", String(error));
  }
}

async function refreshHistoryMemories() {
  if (!historyBase()) {
    setHistoryStatus("后端未连接。启动服务后可查看记忆。");
    renderHistory();
    return;
  }
  try {
    const data = await historyRequest(`/history/memories${historyQuery()}`);
    historyState.memories = data.memories || [];
    setHistoryStatus(`共 ${historyState.memories.length} 条长期记忆。`);
    renderHistory();
  } catch (error) {
    setHistoryStatus(`加载失败：${error.message}`);
    appendLog("history.memories.error", String(error));
  }
}

// --- Database tab ----------------------------------------------------------

function formatHistoryBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size >= 100 || unit === 0 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

async function refreshHistoryDatabase() {
  await refreshHistoryDbPath();
  if (!historyBase()) {
    historyState.database = null;
    setHistoryStatus("后端未连接。启动服务后可查看数据库。");
    renderHistory();
    return;
  }
  try {
    const info = await historyRequest("/history/info");
    historyState.database = info.database || null;
    setHistoryStatus(
      historyState.database && historyState.database.connected ? "数据库连接正常。" : "数据库不可用。"
    );
    renderHistory();
  } catch (error) {
    historyState.database = null;
    setHistoryStatus(`加载失败：${error.message}`);
    appendLog("history.db.error", String(error));
    renderHistory();
  }
}

async function refreshHistoryDbPath() {
  if (!window.desktopApp?.getHistoryDbConfig) {
    return;
  }
  try {
    const info = await window.desktopApp.getHistoryDbConfig();
    historyState.databasePath = info.resolvedPath || "";
    historyState.databaseFileName = info.fileName || "whispera.db";
    if (info.envOverride) {
      historyState.databasePathSource = "env";
    } else if (info.configuredPath) {
      historyState.databasePathSource = "custom";
    } else {
      historyState.databasePathSource = "default";
    }
  } catch (error) {
    appendLog("history.db.path_error", String(error));
  }
}

function describeHistoryDbPathSource() {
  if (historyState.databasePathSource === "env") {
    return "由环境变量 MINIMIND_DB_PATH 指定，界面选择被忽略。";
  }
  if (historyState.databasePathSource === "custom") {
    return "使用自定义位置，重启服务后生效。";
  }
  return "使用默认位置（runtime/data），重启服务后生效。";
}

async function browseHistoryDbFolder() {
  if (!window.desktopApp?.chooseHistoryDbFolder) {
    return;
  }
  try {
    const result = await window.desktopApp.chooseHistoryDbFolder();
    if (result.canceled) {
      return;
    }
    appendLog("history.db.selected", { path: result.path });
    await refreshHistoryDbPath();
    renderHistory();
    setHistoryStatus(`已选择：${result.path}。重启服务后生效。`);
  } catch (error) {
    appendLog("history.db.select_error", String(error));
  }
}

async function resetHistoryDbPath() {
  if (!window.desktopApp?.clearHistoryDbPath) {
    return;
  }
  try {
    await window.desktopApp.clearHistoryDbPath();
    appendLog("history.db.reset", {});
    await refreshHistoryDbPath();
    renderHistory();
    setHistoryStatus("已恢复默认数据库位置。重启服务后生效。");
  } catch (error) {
    appendLog("history.db.reset_error", String(error));
  }
}

async function openHistoryDbFolder() {
  if (!window.desktopApp?.openHistoryDbFolder) {
    return;
  }
  try {
    const result = await window.desktopApp.openHistoryDbFolder();
    if (result && result.ok === false) {
      setHistoryStatus(`打开目录失败：${result.error}`);
    }
  } catch (error) {
    setHistoryStatus(`打开目录失败：${error}`);
  }
}

async function openHistorySession(sessionId) {
  try {
    const data = await historyRequest(
      `/history/sessions/${encodeURIComponent(sessionId)}${historyQuery()}`
    );
    historyState.activeSessionId = sessionId;
    historyState.detail = data;
    renderHistory();
    setHistoryStatus(`${data.session.message_count} 条消息。`);
  } catch (error) {
    setHistoryStatus(`加载失败：${error.message}`);
  }
}

async function historyDeleteSession(sessionId) {
  await historyRequest(`/history/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  if (historyState.activeSessionId === sessionId) {
    historyState.activeSessionId = null;
    historyState.detail = null;
  }
  appendLog("history.session.deleted", { id: sessionId });
  await refreshHistorySessions();
}

async function historyRestoreSession(sessionId) {
  await historyRequest(`/history/sessions/${encodeURIComponent(sessionId)}/restore`, { method: "POST" });
  await refreshHistorySessions();
}

async function historyDeleteMessage(messageId) {
  await historyRequest(`/history/messages/${encodeURIComponent(messageId)}`, { method: "DELETE" });
  await openHistorySession(historyState.activeSessionId);
}

async function historyRestoreMessage(messageId) {
  await historyRequest(`/history/messages/${encodeURIComponent(messageId)}/restore`, { method: "POST" });
  await openHistorySession(historyState.activeSessionId);
}

async function historySaveMessage(messageId, content) {
  await historyRequest(`/history/messages/${encodeURIComponent(messageId)}`, {
    method: "PATCH",
    body: JSON.stringify({ content })
  });
  await openHistorySession(historyState.activeSessionId);
}

async function historyDeleteMemory(memoryId) {
  await historyRequest(`/history/memories/${encodeURIComponent(memoryId)}`, { method: "DELETE" });
  await refreshHistoryMemories();
}

async function historyRestoreMemory(memoryId) {
  await historyRequest(`/history/memories/${encodeURIComponent(memoryId)}/restore`, { method: "POST" });
  await refreshHistoryMemories();
}

async function historySaveMemory(memoryId, content) {
  await historyRequest(`/history/memories/${encodeURIComponent(memoryId)}`, {
    method: "PATCH",
    body: JSON.stringify({ content })
  });
  await refreshHistoryMemories();
}

async function historyAddMemory(content) {
  await historyRequest("/history/memories", {
    method: "POST",
    body: JSON.stringify({ content })
  });
  await refreshHistoryMemories();
}

async function historySyncVault() {
  if (!historyBase()) {
    setHistoryStatus("后端未连接，无法同步。");
    return;
  }
  setHistoryStatus("同步中…");
  try {
    const data = await historyRequest("/history/sync", { method: "POST" });
    const report = data.report || {};
    const parts = [];
    if (report.deleted_session_ids && report.deleted_session_ids.length) {
      parts.push(`删除会话 ${report.deleted_session_ids.length}`);
    }
    if (report.deleted_messages) {
      parts.push(`删除消息 ${report.deleted_messages}`);
    }
    if (report.updated_messages) {
      parts.push(`更新消息 ${report.updated_messages}`);
    }
    if (report.deleted_memory_ids && report.deleted_memory_ids.length) {
      parts.push(`删除记忆 ${report.deleted_memory_ids.length}`);
    }
    if (report.updated_memory_ids && report.updated_memory_ids.length) {
      parts.push(`更新记忆 ${report.updated_memory_ids.length}`);
    }
    if (report.created_memory_ids && report.created_memory_ids.length) {
      parts.push(`新增记忆 ${report.created_memory_ids.length}`);
    }
    appendLog("history.sync", report);
    await refreshHistory();
    setHistoryStatus(parts.length ? `同步完成：${parts.join("，")}。` : "同步完成：未检测到文件改动。");
  } catch (error) {
    setHistoryStatus(`同步失败：${error.message}`);
  }
}

function renderHistory() {
  const container = els.historyBody;
  container.replaceChildren();
  if (historyState.view === "sessions") {
    renderHistorySessions(container);
  } else if (historyState.view === "memories") {
    renderHistoryMemories(container);
  } else {
    renderHistoryDatabase(container);
  }
}

function renderHistorySessions(container) {
  if (historyState.activeSessionId && historyState.detail) {
    renderHistoryDetail(container);
    return;
  }
  if (!historyState.sessions.length) {
    container.appendChild(historyEmpty("暂无会话记录。"));
    return;
  }
  for (const session of historyState.sessions) {
    const actions = [];
    if (session.deleted) {
      actions.push(historyActionButton("恢复", () => historyRestoreSession(session.id)));
    } else {
      actions.push(historyActionButton("打开", () => openHistorySession(session.id)));
      actions.push(historyActionButton("删除", () => historyDeleteSession(session.id), { danger: true }));
    }
    const item = historyRow({
      text: session.title || session.id,
      meta: `${formatHistoryTime(session.updated_at)} · ${session.message_count} 条 · ${session.persona_id || "-"}`,
      deleted: session.deleted,
      actions
    });
    if (!session.deleted) {
      item.classList.add("history-item--clickable");
      item.addEventListener("click", (event) => {
        if (event.target.closest("button")) {
          return;
        }
        openHistorySession(session.id).catch((error) => setHistoryStatus(`加载失败：${error.message}`));
      });
    }
    container.appendChild(item);
  }
}

function renderHistoryDetail(container) {
  const detail = historyState.detail;
  const header = document.createElement("div");
  header.className = "history-detail__header";
  header.appendChild(historyActionButton("← 返回", () => {
    historyState.activeSessionId = null;
    historyState.detail = null;
    renderHistory();
    setHistoryStatus(`共 ${historyState.sessions.length} 个会话。`);
  }));
  const title = document.createElement("div");
  title.className = "history-detail__title";
  title.textContent = detail.session.title || detail.session.id;
  header.appendChild(title);
  container.appendChild(header);

  if (!detail.messages.length) {
    container.appendChild(historyEmpty("该会话暂无消息。"));
    return;
  }
  for (const message of detail.messages) {
    const actions = [];
    if (message.deleted) {
      actions.push(historyActionButton("恢复", () => historyRestoreMessage(message.id)));
    } else {
      actions.push(historyActionButton("删除", () => historyDeleteMessage(message.id), { danger: true }));
    }
    container.appendChild(historyRow({
      text: message.content,
      meta: `${message.role === "user" ? "我" : "AI"} · ${formatHistoryTime(message.created_at)}` +
        `${message.edited ? " · 已编辑" : ""}`,
      deleted: message.deleted,
      onEdit: message.deleted ? null : (value) => historySaveMessage(message.id, value),
      actions
    }));
  }
}

function renderHistoryMemories(container) {
  const addRow = document.createElement("div");
  addRow.className = "history-add";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "新增一条长期记忆，回车保存";
  addRow.appendChild(input);
  addRow.appendChild(historyActionButton("新增", () => submitNewMemory()));
  container.appendChild(addRow);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submitNewMemory();
    }
  });
  async function submitNewMemory() {
    const value = input.value.trim();
    if (!value) {
      return;
    }
    try {
      await historyAddMemory(value);
      input.value = "";
    } catch (error) {
      setHistoryStatus(`新增失败：${error.message}`);
    }
  }

  if (!historyState.memories.length) {
    container.appendChild(historyEmpty("暂无长期记忆。"));
    return;
  }
  for (const memory of historyState.memories) {
    const actions = [];
    if (memory.deleted) {
      actions.push(historyActionButton("恢复", () => historyRestoreMemory(memory.id)));
    } else {
      actions.push(historyActionButton("删除", () => historyDeleteMemory(memory.id), { danger: true }));
    }
    container.appendChild(historyRow({
      text: memory.content,
      meta: `${formatHistoryTime(memory.created_at)}${memory.edited ? " · 已编辑" : ""}`,
      deleted: memory.deleted,
      onEdit: memory.deleted ? null : (value) => historySaveMemory(memory.id, value),
      actions
    }));
  }
}

function historyStatCell(label, value, sub) {
  const cell = document.createElement("div");
  cell.className = "history-db-stat";
  const labelEl = document.createElement("span");
  labelEl.className = "history-db-stat__label";
  labelEl.textContent = label;
  const valueEl = document.createElement("span");
  valueEl.className = "history-db-stat__value";
  valueEl.textContent = String(value);
  cell.append(labelEl, valueEl);
  if (sub) {
    const subEl = document.createElement("span");
    subEl.className = "history-item__meta";
    subEl.textContent = sub;
    cell.appendChild(subEl);
  }
  return cell;
}

function renderHistoryDatabase(container) {
  const db = historyState.database;

  // Connection state
  const stateRow = document.createElement("div");
  stateRow.className = "history-db-state";
  const dot = document.createElement("span");
  dot.className = `history-db-dot ${db && db.connected ? "is-ok" : "is-bad"}`;
  const stateLabel = document.createElement("strong");
  stateLabel.textContent = !db ? "未知" : db.connected ? "已连接" : "未连接";
  const stateMeta = document.createElement("span");
  stateMeta.className = "history-item__meta";
  if (!db) {
    stateMeta.textContent = "启动服务后可读取数据库状态";
  } else if (db.error) {
    stateMeta.textContent = db.error;
  } else if (db.db_exists) {
    stateMeta.textContent = `SQLite · WAL · ${formatHistoryBytes(db.db_size_bytes)}`;
  } else {
    stateMeta.textContent = "数据库文件尚未创建";
  }
  stateRow.append(dot, stateLabel, stateMeta);
  container.appendChild(stateRow);

  // Record counts
  if (db) {
    const stats = document.createElement("div");
    stats.className = "history-db-stats";
    stats.append(
      historyStatCell("会话", db.sessions, db.sessions_deleted ? `已删除 ${db.sessions_deleted}` : ""),
      historyStatCell("消息", db.messages, db.messages_deleted ? `已删除 ${db.messages_deleted}` : ""),
      historyStatCell(
        "长期记忆",
        db.memories,
        db.memories_deleted ? `已删除 ${db.memories_deleted}` : ""
      ),
      historyStatCell("文件大小", formatHistoryBytes(db.db_size_bytes), db.db_exists ? "" : "未创建")
    );
    container.appendChild(stats);
  }

  // File location
  const pathLabel = document.createElement("label");
  pathLabel.className = "field";
  const pathTitle = document.createElement("span");
  pathTitle.textContent = `数据库文件（文件名固定为 ${historyState.databaseFileName}）`;
  const pathRow = document.createElement("div");
  pathRow.className = "file-picker-row";
  const pathInput = document.createElement("input");
  pathInput.type = "text";
  pathInput.readOnly = true;
  pathInput.value = historyState.databasePath || "（读取中…）";
  const browseButton = historyActionButton("选择...", () => browseHistoryDbFolder());
  browseButton.disabled = historyState.databasePathSource === "env";
  pathRow.append(pathInput, browseButton);
  pathLabel.append(pathTitle, pathRow);
  container.appendChild(pathLabel);

  const pathStatus = document.createElement("div");
  pathStatus.className = "voice-status";
  pathStatus.textContent = describeHistoryDbPathSource();
  container.appendChild(pathStatus);

  // Actions
  const actions = document.createElement("div");
  actions.className = "history-db-actions";
  const resetButton = historyActionButton("恢复默认位置", () => resetHistoryDbPath());
  resetButton.disabled = historyState.databasePathSource !== "custom";
  actions.append(
    historyActionButton("刷新状态", () => refreshHistoryDatabase()),
    historyActionButton("打开所在目录", () => openHistoryDbFolder()),
    resetButton
  );
  container.appendChild(actions);
}

// The Markdown-mirror controls only apply to the two record tabs; the database
// tab carries its own path picker and actions, so the unrelated chrome is hidden
// rather than showing a directory picker that would not affect the database.
function applyHistoryViewChrome(view) {
  const isDatabase = view === "database";
  els.historyVaultBlock.hidden = isDatabase;
  els.historyShowDeletedRow.hidden = isDatabase;
  els.historySyncBtn.hidden = isDatabase;
  els.historyOpenVaultBtn.hidden = isDatabase;
}

const HISTORY_VIEWS = [
  ["sessions", "historySessionsTab"],
  ["memories", "historyMemoriesTab"],
  ["database", "historyDatabaseTab"]
];

function setHistoryView(view) {
  historyState.view = view;
  historyState.activeSessionId = null;
  historyState.detail = null;
  for (const [name, elementKey] of HISTORY_VIEWS) {
    const tab = els[elementKey];
    const isActive = name === view;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  }
  applyHistoryViewChrome(view);
  void refreshHistory();
}

async function refreshHistoryVaultRoot() {
  if (!window.desktopApp.getHistoryConfig) {
    return;
  }
  try {
    const info = await window.desktopApp.getHistoryConfig();
    els.historyVaultPath.value = info.resolvedPath || "";
    const envLocked = Boolean(info.envOverride);
    els.browseHistoryVaultBtn.disabled = envLocked;
    if (envLocked) {
      els.historyVaultStatus.textContent = `由环境变量指定：${info.resolvedPath}`;
    } else if (info.configuredPath) {
      els.historyVaultStatus.textContent = `自定义目录：${info.resolvedPath}（重启服务后生效）`;
    } else {
      els.historyVaultStatus.textContent = `默认目录：${info.resolvedPath}`;
    }
  } catch (error) {
    els.historyVaultStatus.textContent = "读取目录失败。";
    appendLog("history.vault.read_error", String(error));
  }
}

async function browseHistoryVaultRoot() {
  if (!window.desktopApp.chooseHistoryVaultRoot) {
    return;
  }
  try {
    const result = await window.desktopApp.chooseHistoryVaultRoot();
    if (result.canceled) {
      return;
    }
    appendLog("history.vault.selected", { path: result.path });
    await refreshHistoryVaultRoot();
    els.historyVaultStatus.textContent = `已选择：${result.path}。重启服务后生效。`;
  } catch (error) {
    appendLog("history.vault.select_error", String(error));
  }
}

async function openHistoryVaultRoot() {
  if (!window.desktopApp.openHistoryVaultRoot) {
    return;
  }
  try {
    const result = await window.desktopApp.openHistoryVaultRoot();
    if (result && result.ok === false) {
      setHistoryStatus(`打开目录失败：${result.error}`);
    }
  } catch (error) {
    setHistoryStatus(`打开目录失败：${error}`);
  }
}

els.historyRefreshBtn.addEventListener("click", () => {
  void refreshHistory();
});
els.historySyncBtn.addEventListener("click", () => {
  void historySyncVault();
});
els.historyOpenVaultBtn.addEventListener("click", () => {
  void openHistoryVaultRoot();
});
els.historySessionsTab.addEventListener("click", () => setHistoryView("sessions"));
els.historyMemoriesTab.addEventListener("click", () => setHistoryView("memories"));
els.historyDatabaseTab.addEventListener("click", () => setHistoryView("database"));
els.historyShowDeleted.addEventListener("change", () => {
  historyState.includeDeleted = els.historyShowDeleted.checked;
  void refreshHistory();
});
els.browseHistoryVaultBtn.addEventListener("click", () => {
  void browseHistoryVaultRoot();
});

bootstrap().catch((error) => {
  appendLog("bootstrap.error", String(error));
});
