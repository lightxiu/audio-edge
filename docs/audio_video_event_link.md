# 音视频事件关联试验（第一阶段）

当前只关联事件元数据，不解码视频、不生成视频片段，也不判断音视频是否来自同一现场。保留三种模式：

- 纯音频：照常运行 `python -m src.cli run --config configs/windows_sound_events.yaml --duration 30 --no-report`；没有视频事件时音频候选照常显示、回听和复核。
- 纯视频：运行 `python -m dashboard`；此命令只启动本地 HTTP 服务和数据库，不访问麦克风。浏览器打开 `http://127.0.0.1:8765`，通过 HTTP 上传视频事件后可看到独立列表。
- 联合：运行音频命令并向同一服务上传视频事件；`GET /api/associations?window_sec=5` 返回音频候选和前后 5 秒的邻近视频事件。关联状态 `temporal_only`，不能当作同源或因果证明。两个模式不能同时占用 8765 端口；联合模式只运行音频命令。

视频侧接口是 `POST http://127.0.0.1:8765/api/video-events?source_id=camera-1`。body 与 C++ 项目 `HttpEventSink::__toJson` 一致：`type`、`timestamp`（Unix 毫秒）、`count`、`objects`（`label`、`name`、`probability`、`box: [x,y,w,h]`）、`message`。`source_id` 由调用方标识，原 C++ payload 中没有设备字段。重复提交相同 payload 和 source_id 不产生重复记录；返回生成的事件 ID。服务仅绑定本机，外部设备需要通过受控转发或先保存后导入，**不要为测试直接把无鉴权接口开放到公网**。

在 IDE 终端测试（服务运行期间另开一个终端）：

```powershell
$nowMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$body = @{ type = 'abnormality'; timestamp = $nowMs; count = 1; objects = @(@{ label = 2; name = 'fish'; probability = 0.8; box = @(1,2,30,40) }); message = 'test event' } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/video-events?source_id=demo' -Method Post -ContentType 'application/json' -Body $body
```

应收到带 `id`、`timestamp_ms`、`source_id` 的响应，页面“视频事件”出现一条记录。没有同步产生的真实音频候选时，关联列表应为空，这是正确结果。视频事件存于 `reviews/candidates.sqlite3`，与音频证据一样被 Git 忽略；运行实测前请确认现场采集权限。

下一阶段需确认摄像头流有无音轨、统一媒体 PTS 与时钟、断流处理及音视频证据片段；在这些完成前，简历只能称“事件时间窗关联原型”，不能称“实时音视频同步/多模态融合”。
