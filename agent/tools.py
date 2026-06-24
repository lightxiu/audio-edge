"""Tool definitions for the Edge Agent.

Each tool is a simple Python function that the agent can call.
Tools follow the pattern: name(args) → str result.
"""

from __future__ import annotations

import json
import time
from typing import Any


def query_events(query: str, top_k: int = 5) -> str:
    """Search historical events semantically.

    Args:
        query: Natural language query like "有人经过吗".
        top_k: Max results.

    Returns:
        Formatted event list string.
    """
    from memory.query import format_query_results, query_events as _query

    results = _query(query, top_k=top_k)
    return format_query_results(results)


def query_recent_events(hours: float = 1, module: str = "") -> str:
    """Get recent events by time.

    Args:
        hours: Time window.
        module: Filter by module ("vision", "audio", "system").

    Returns:
        Formatted event list string.
    """
    from memory.query import format_query_results, query_recent

    results = query_recent(hours=hours, module=module or None)
    return format_query_results(results)


def get_system_status() -> str:
    """Get current system health status.

    Returns:
        JSON string with CPU/GPU/memory/disk/temperature stats.
    """
    try:
        from ops.monitor import get_system_status

        status = get_system_status()
        return json.dumps(status, ensure_ascii=False, indent=2)
    except ImportError:
        return json.dumps({
            "status": "unknown",
            "error": "ops module not available",
        }, ensure_ascii=False)


def get_time() -> str:
    """Get current time.

    Returns:
        Formatted time string.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# Tool registry — maps tool name to (function, description)
TOOL_REGISTRY: dict[str, tuple[Any, str]] = {
    "query_events": (query_events, "语义搜索历史事件。输入: 自然语言查询"),
    "query_recent_events": (query_recent_events, "按时间获取最近事件。输入: 小时数"),
    "get_system_status": (get_system_status, "获取系统CPU/GPU/内存/温度状态"),
    "get_time": (get_time, "获取当前时间"),
}
