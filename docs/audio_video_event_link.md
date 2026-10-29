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

## 第二阶段：离线带音轨文件导入

现在可用同一个本地媒体文件生成音频候选和视频片段，不需要打开麦克风，也不需要连接实时摄像头。需先在 IDE 终端执行 `ffmpeg -version` 和 `ffprobe -version`，均输出版本号才可继续；若找不到命令，请安装 FFmpeg，并让两个命令位于 PATH 中。当前开发环境没有这两个命令，因此真实文件联调需要你运行。输入文件最长 10 分钟，需有音轨；没有画面的 WAV 也可导入，但只生成音频证据。

录制时记下**文件时间轴 0 秒对应的本地时刻和时区**，例如 `2026-10-29T14:00:00+08:00`。如果不知道绝对时刻，可任意选择一个带时区的实验基准，但它仅用于同文件回看，不能和 C++ 实时事件做绝对时间匹配。不要用文件修改时间冒充录制时间。先用 `ffprobe -v error -show_entries stream=index,codec_type,start_time -show_entries format=duration,start_time -of json 你的文件.mp4` 检查音轨与视频轨；无音轨会明确报错，纯视频仍按上面的 HTTP 模式处理。

在 IDE 的终端运行（替换为自己的文件和真实录制起始时间）：

```powershell
python -m src.pipeline.media_import "C:\path\to\recording.mp4" --recorded-at "2026-10-29T14:00:00+08:00" --source-id camera-1
```

应输出 `duration_sec`、`has_video` 和 `candidates` 列表；如果没有识别到目标，列表为空是正常结果。候选音频 WAV 和最多 5 秒的静音视频 MP4 放在本机 `evidence/`，复核记录在 `reviews/candidates.sqlite3`，不会提交到 GitHub。运行 `python -m dashboard` 并打开 `http://127.0.0.1:8765`，在候选行并排回听音频和视频；这种关联标记为 `same_media_file`。视频流的音频起始 PTS 与容器起始 PTS 的差值用于补齐前导静音，候选时间取 1 秒推理窗口的中心；这比到达时间关联稳定，但不同设备文件与实时 C++ 事件仍需实测时钟偏差。重复导入同一文件目前会产生新的候选，不要用重复导入的记录统计检出次数。

请提供一段有使用权限的 30–60 秒 MP4，包含几次可定位时间的目标声（敲门或门铃）和一段只有讲话/键盘等非目标声的对照；记下每次发生的秒数、录制起始时间与设备。运行后核对页面片段是否覆盖该秒数，并记录漏检、误检。当前版本未做实时 RTSP 音轨拆分、视频模型推理或跨设备校时，不能称作实时多模态融合。
