# 化身 DLC（可选）

主仓库只包含语音对话，不包含 VRM 模型、骨骼动作与口型资源。化身能力通过一个独立的本机 HTTP 服务接入，默认关闭，不影响语音链路。

## 启用

1. 把兼容的化身服务放到 `vendor/avatar/`，或设置环境变量：

   ```powershell
   $env:MINIMIND_HAST_ROOT = "D:\avatar-service"
   $env:MINIMIND_HAST_ENABLED = "1"
   $env:MINIMIND_MOTION_ENABLED = "1"
   ```

2. 服务需要提供以下接口（与主进程 `electron-app/main.js` 对应）：

   - `GET /v13-api/status`：服务状态
   - `GET /configs/avatars.json`：模型列表，至少包含 `red` / `blue` 两个 key
   - 页面 `/language-lab?embed=1&model=<key>`：VRM 渲染页
   - 主进程通过 `<webview>` 注入 `hast-preload.js` 与页面通信

3. 启动应用后，顶栏的“模型”下拉与右下角“显示角色”按钮会变为可用。

## 许可

化身服务是独立的第三方项目，可能使用**非商业学习/研究许可**。本仓库不包含其代码或模型，也不对其可用性、许可或安全性作任何保证。将本项目用于商业用途前，请单独核对该服务的许可。

角色 VRM 模型同样属于第三方素材，相关权利不随本项目许可证转让。
