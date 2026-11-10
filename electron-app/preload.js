const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopWindow", {
  minimize: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximize: () => ipcRenderer.invoke("window:toggle-maximize"),
  close: () => ipcRenderer.invoke("window:close"),
  isMaximized: () => ipcRenderer.invoke("window:is-maximized")
});

contextBridge.exposeInMainWorld("desktopApp", {
  getInfo: () => ipcRenderer.invoke("app:get-info"),
  getBackendConfig: () => ipcRenderer.invoke("backend:get-config"),
  getResourceStatus: () => ipcRenderer.invoke("assets:get-resource-status"),
  getPathInfo: (targetPath) => ipcRenderer.invoke("path:get-info", targetPath),
  startVoiceServices: (ttsOptions) => ipcRenderer.invoke("voice-services:start", ttsOptions),
  stopVoiceServices: () => ipcRenderer.invoke("voice-services:stop"),
  getVoiceServicesStatus: () => ipcRenderer.invoke("voice-services:get-status"),
  getVoiceLoggingState: () => ipcRenderer.invoke("voice-logging:get-state"),
  setVoiceLoggingEnabled: (enabled) => ipcRenderer.invoke("voice-logging:set-enabled", enabled),
  openVoiceServicesLogFolder: () => ipcRenderer.invoke("voice-services:open-log-folder"),
  onVoiceServicesEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("voice-services:event", listener);
    return () => ipcRenderer.removeListener("voice-services:event", listener);
  },
  showModelFileDialog: () => ipcRenderer.invoke("llm:show-model-file-dialog"),
  getLlmModelPath: () => ipcRenderer.invoke("llm:get-model-path"),
  setLlmModelPath: (modelPath) => ipcRenderer.invoke("llm:set-model-path", modelPath),
  getLlmConfig: () => ipcRenderer.invoke("llm:get-config"),
  setLlmConfig: (config) => ipcRenderer.invoke("llm:set-config", config),
  setLlmTalkativeness: (level) => ipcRenderer.invoke("llm:set-talkativeness", level),
  getLlmModelInfo: (modelPath) => ipcRenderer.invoke("llm:get-model-info", modelPath),
  scanLoraLocal: () => ipcRenderer.invoke("lora:scan-local"),
  showAudioFileDialog: () => ipcRenderer.invoke("tts:show-audio-file-dialog"),
  getTtsConfig: () => ipcRenderer.invoke("tts:get-config"),
  saveTtsConfig: (config) => ipcRenderer.invoke("tts:save-config", config),
  getPersonas: () => ipcRenderer.invoke("persona:list"),
  getPersonaConfig: () => ipcRenderer.invoke("persona:get-config"),
  setPersonaConfig: (config) => ipcRenderer.invoke("persona:set-config", config),
  getAssetsRoot: () => ipcRenderer.invoke("assets:get-root"),
  chooseAssetsRoot: () => ipcRenderer.invoke("assets:choose-root"),
  setAssetsRoot: (assetsRoot) => ipcRenderer.invoke("assets:set-root", assetsRoot),
  clearAssetsRoot: () => ipcRenderer.invoke("assets:clear-root"),
  getHistoryConfig: () => ipcRenderer.invoke("history:get-config"),
  chooseHistoryVaultRoot: () => ipcRenderer.invoke("history:choose-vault-root"),
  setHistoryVaultRoot: (vaultRoot) => ipcRenderer.invoke("history:set-vault-root", vaultRoot),
  clearHistoryVaultRoot: () => ipcRenderer.invoke("history:clear-vault-root"),
  openHistoryVaultRoot: () => ipcRenderer.invoke("history:open-vault-root"),
  getHistoryDbConfig: () => ipcRenderer.invoke("history:get-db-config"),
  chooseHistoryDbFolder: () => ipcRenderer.invoke("history:choose-db-folder"),
  setHistoryDbPath: (dbPath) => ipcRenderer.invoke("history:set-db-path", dbPath),
  clearHistoryDbPath: () => ipcRenderer.invoke("history:clear-db-path"),
  openHistoryDbFolder: () => ipcRenderer.invoke("history:open-db-folder"),
  // 化身：动作计划、口型权重与说话状态都经主进程转发给内嵌的 化身服务 页面。
  // 页面的创建与显隐由渲染层自己掌握，这里只提供状态查询。
  getAvatarStatus: () => ipcRenderer.invoke("avatar:get-status"),
  // 切换角色模型: red | blue。化身页面已加载时就地重建模型。
  setAvatarModel: (key) => ipcRenderer.invoke("avatar:set-model", key),
  // 高频单向推送，用 send 而非 invoke，避免每个音频帧都产生一次往返。
  sendMotionPlan: (plan, reply) => ipcRenderer.send("motion:plan", { plan, reply }),
  sendVisemes: (values) => ipcRenderer.send("motion:visemes", values),
  // 取景模式 "full"(全身) | "bust"(上半身)
  sendFraming: (mode) => ipcRenderer.send("motion:framing", mode),
  sendMotionSpeaking: (speaking) => ipcRenderer.send("motion:speaking", speaking),
  sendMotionListening: (listening) => ipcRenderer.send("motion:listening", listening),
  sendMotionReset: () => ipcRenderer.send("motion:reset")
});
