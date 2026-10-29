"""Legacy Ollama conversational client; inspection tools live in agent.inspection.

Self-contained module.  Uses Ollama (llama.cpp backend) for local LLM
inference, sherpa-onnx for STT, Piper TTS for speech output.

This conversational client does not yet execute the tools described in its
prompt; do not rely on it for evidence-backed monitoring answers.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


# --- Prompt template -----------------------------------------------------

SYSTEM_PROMPT = """你是一个运行在 Jetson Orin NX 边缘设备上的智能助手。

你的名字是"哨兵"（Sentinel）。你负责：
1. 回答关于监控事件的查询（通过 query_events 工具）
2. 报告系统状态（通过 get_system_status 工具）
3. 回答一般性问题

## 规则
- 用中文回答，简洁明了，2-3句话为宜
- 当用户询问"有没有人经过"、"检测到什么"等历史事件时，使用 query_events
- 当用户询问"系统状态"、"温度"、"内存"时，使用 get_system_status
- 当你不确定时，直接告诉用户你需要更多信息
- 不要编造事件内容，只基于工具返回的结果回答

## 当前时间
{current_time}
"""

TOOL_DESCRIPTIONS = """
可用工具:
1. query_events(query: str) -> list — 语义搜索历史事件。例如: query_events("有人经过")
2. get_system_status() -> dict — 获取当前 CPU/GPU/内存/温度状态
3. get_time() -> str — 获取当前时间

当需要查询事件或系统状态时，先调用工具，然后基于工具返回的结果回答问题。
不要编造工具返回的内容。
"""


class EdgeAgent:
    """ReAct agent backed by a local LLM via Ollama.

    Parameters
    ----------
    model : str
        Ollama model name (e.g. ``"qwen2.5:7b-q4_K_M"``).
    base_url : str
        Ollama API base URL.
    tools : dict or None
        Tool name → callable mapping.  If None, uses default tools.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        tools: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._tools = tools or _default_tools()
        self._history: list[dict] = []  # Conversation history

    # -- Public API -------------------------------------------------------
    def chat(self, message: str) -> str:
        """Process a text message and return a text response.

        The agent may call tools internally before responding.
        """
        prompt = self._build_prompt(message)

        # Try tool-calling via Ollama's native API
        response = self._call_ollama(prompt)
        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    # -- Internal ---------------------------------------------------------
    def _build_prompt(self, message: str) -> str:
        """Build the full prompt with system message, tools, and history."""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        system = SYSTEM_PROMPT.format(current_time=current_time) + TOOL_DESCRIPTIONS

        parts = [f"系统: {system}"]

        # Add recent history (last 3 exchanges)
        for h in self._history[-6:]:
            role = "用户" if h["role"] == "user" else "助手"
            parts.append(f"{role}: {h['content']}")

        parts.append(f"用户: {message}")
        return "\n\n".join(parts)

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama chat API. Falls back to mock if Ollama is unavailable."""
        import requests

        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 256,
            },
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["message"]["content"]

            # Update history
            self._history.append({"role": "user", "content": prompt[:200]})
            self._history.append({"role": "assistant", "content": reply})

            return reply
        except Exception as e:
            logger.warning(f"Ollama call failed ({e}), using mock agent")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Never invent monitoring events or device measurements offline."""
        return "本地语言模型不可用，无法回答监控事件或设备状态；请通过巡检查询查看已记录的证据。"


def _default_tools() -> dict[str, Any]:
    """Return the default tool set."""
    from memory.query import format_query_results, query_events

    def _query_events(query: str) -> str:
        results = query_events(query, top_k=5)
        return format_query_results(results)

    def _get_status() -> str:
        try:
            from ops.monitor import get_system_status
            status = get_system_status()
            return json.dumps(status, ensure_ascii=False, indent=2)
        except Exception:
            return "系统状态暂时不可用"

    return {
        "query_events": _query_events,
        "get_system_status": _get_status,
    }
