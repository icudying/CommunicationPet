// 化身窗口的 preload。运行在 化身服务 页面的隔离世界里，把主进程收到的
// 动作计划 / 口型权重 / 说话状态转发给页面的 ES 模块。
//
// 页面侧在 embed 模式下读取 window.hastBridge（见 化身服务/web/v13_lab.mjs
// 的 bindHost），从而绕开它自带的中文模型，直接由 CommunicationPet 驱动角色。
const { contextBridge, ipcRenderer } = require("electron");

function subscribe(channel) {
  return (callback) => {
    if (typeof callback !== "function") {
      return () => {};
    }
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on(channel, listener);
    return () => ipcRenderer.removeListener(channel, listener);
  };
}

contextBridge.exposeInMainWorld("hastBridge", {
  // 动作计划（含表情与手势）。payload: { plan, reply }
  onPlan: subscribe("avatar:plan"),
  // 口型权重 {aa, ih, ou, ee, oh}，取值 0~1，由 TTS 音频实时分析得出。
  onVisemes: subscribe("avatar:visemes"),
  // 取景模式 "full"(全身) | "bust"(上半身)
  onFraming: subscribe("avatar:framing"),
  onSpeaking: subscribe("avatar:speaking"),
  onListening: subscribe("avatar:listening"),
  onReset: subscribe("avatar:reset"),
  // 切换角色模型（red | blue），由页面就地重建 VRM，不重载整个页面。
  onModel: subscribe("avatar:model"),
  // 告诉主进程页面已初始化完毕，可以开始接收消息。
  notifyReady: () => ipcRenderer.send("avatar:ready"),
  log: (message) => ipcRenderer.send("avatar:log", { message: String(message) }),
});
