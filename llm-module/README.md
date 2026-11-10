# llm-module

纯标准库的 OpenAI 兼容 LLM 客户端。API 文档见 [LLM_API.md](LLM_API.md)。

## 内容

- `llm_module.LlamaServerLLM`：调用任意 OpenAI 兼容 `/v1/chat/completions` 接口（DeepSeek、OpenAI、本地 llama.cpp 等）。
- `llm_module.service`：可选 FastAPI 代理服务，提供 `/chat`、`/chat/stream` 和 `/ws/llm`。
- `scripts/llm_demo.py` / `scripts/llm_client.py`：命令行验证示例。

类名 `LlamaServerLLM` 是历史名称，实际不依赖 llama.cpp。

## 安装

```bash
pip install -e ./llm-module
```

## 直接调用

```python
from llm_module import LLMConfig, LlamaServerLLM

llm = LlamaServerLLM(LLMConfig(
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    api_key="YOUR_KEY",
))
print(llm.generate_text("你好"))
```
