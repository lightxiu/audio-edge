# audio-edge Dashboard

实时 WebSocket 仪表板 — 通过浏览器查看音频事件流。

## 快速开始

```bash
# 安装依赖
pip install fastapi "uvicorn[standard]"

# 启动 pipeline（仪表板自动在后台运行）
python -m src.cli run --mock

# 浏览器打开
# http://localhost:8765
```

## 架构

```
Pipeline 线程                     FastAPI (asyncio)
     │                                  │
     │ emit(InferenceResult)            │
     ▼                                  │
DashboardOutput ──push_event()──▶ deque buffer ──poll──▶ WebSocket ──▶ 浏览器
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `server.py` | FastAPI 应用 + WebSocket 端点 + 事件缓冲区（通用，可复用） |
| `output.py` | 对接 audio-edge pipeline 的输出后端 |
| `static/index.html` | 单文件前端（纯 HTML/CSS/JS） |
| `__init__.py` | 包标记 |

## 复用到其他项目

`server.py` 是**完全通用**的 — 不依赖 audio-edge 的任何数据结构。

1. 复制 `dashboard/` 到你的项目
2. 写一个类似 `output.py` 的适配器（只需要实现 `emit(data)` 方法）
3. 修改前端的卡片布局（按需）

YOLO 项目示例：
```python
from dashboard.server import push_event

class YOLODashboardOutput:
    def emit(self, detection_result):
        push_event({
            "task": "yolo",
            "label": f"{detection_result.class_name}",
            "confidence": detection_result.confidence,
            "timestamp": time.time(),
            "latency_ms": detection_result.latency_ms,
            "metadata": {
                "boxes": len(detection_result.boxes),
            },
        })
```

## 配置

在 `configs/default.yaml` 中：

```yaml
output:
  dashboard: true        # 启用仪表板
  dashboard_port: 8765   # 监听端口
```

## 端口选择

默认 8765。如果端口冲突，在配置中修改 `dashboard_port`。
