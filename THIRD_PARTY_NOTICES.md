# Third-Party Notices

CommunicationPet 的代码与资源包含或依赖以下第三方作品。各组件遵循其原始许可；本文件不替代原始许可文本。

## 代码

| 组件 | 位置 | 许可 | 说明 |
|---|---|---|---|
| Whispera | `electron-app/`、`realtime/`、`llm-module/` 等 | Apache-2.0 | 本项目的实时语音框架源自 Whispera。上游代码的版权归其作者所有。 |
| mem0 | `mem0/` | Apache-2.0 | Vendored 副本，上游仓库 <https://github.com/mem0ai/mem0>；本地修改见 `mem0/LOCAL_CHANGES.md`。 |
| Electron | `electron-app/node_modules/`（安装时获取） | MIT | 桌面运行时。 |
| three.js | `electron-app/node_modules/`（安装时获取） | MIT | 3D 渲染库。 |

## 外部服务与模型（不随本仓库分发）

| 组件 | 用途 | 许可 | 说明 |
|---|---|---|---|
| GPT-SoVITS | 语音合成（api_v2） | MIT | <https://github.com/RVC-Boss/GPT-SoVITS> |
| FunASR / paraformer | 中文 ASR | MIT / 模型自带许可 | 使用 GPT-SoVITS 自带的 paraformer 快照；使用前请核对模型页面许可。 |
| Silero VAD | 语音活动检测 | MIT | <https://github.com/snakers4/silero-vad> |
| HAST-SEC（可选化身服务） | VRM 动作/口型 | 非商业学习与研究许可 0.1 | 不包含在本仓库中，仅作为可选 DLC 由用户自行安装。其许可可能限制商业使用。 |

## 人格文本与语音资产

`personas/` 下的人格文本、参考音频与 GPT-SoVITS 微调权重用于演示“语音对话伙伴”的玩法：

- 这些角色名、设定与声音素材来自或改编自第三方作品，相关权利归原权利人所有；
- 本项目代码的 Apache-2.0 许可**不覆盖**这些角色设定、声音样本与微调权重；
- 如果你要将本项目用于公开发布或商业用途，请自行确认你对相应角色与声音素材拥有合法授权。

## 商标

文中出现的产品、游戏、角色与公司名称仅用于说明来源，商标归各自所有者所有。
