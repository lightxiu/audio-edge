# Windows 真实声音事件试验

目标是从“有人说话”转向**具体声音事件**：先试门铃、敲门、一般警报、闹铃、警笛。使用已有 YAMNet ONNX 权重与 AudioSet 的 521 类标签，运行时只挑这五类。输出是模型候选事件，不等于验证过的安全告警或识别准确率。

## 1. 准备

在 IDE 的终端进入 `C:\MMM\cc`。先完成 [麦克风指南](windows_real_mic_report_guide.md) 的前五步，确认真实设备能读到声音。当前配置 `configs/windows_sound_events.yaml` 的 `audio.device` 为你上次使用的索引 21；若当前设备索引变动，按 `python -X utf8 -m src.cli list-devices` 的输出修改并保存。

确认 `models/sed/yamnet.onnx` 和 `models/sed/yamnet_class_map.csv` 都存在；本配置 `strict_models: true`，模型丢失会直接报错，不会假装以 Mock 输出完成检测。

## 2. 分两轮测试

第一轮不制造目标声音，保持环境安静 30 秒，记录是否有误报：

```text
python -X utf8 -m src.cli run --config configs/windows_sound_events.yaml --duration 30 --no-report
```

预期启动日志有 `VAD: enabled`、`SED: enabled`、`YAMNet SED loaded: 521 classes` 和 `Audio capture started`，不应出现 `using mock`。安静环境下最好没有 `SED:` 事件；如有，记录标签与置信度，这是需要复核的疑似误报，不能忽略。

第二轮运行 60 秒，分别制造几次**真实可重复的目标声音**：轻敲门/桌面三次，每次间隔 5 秒以上；如有门铃，可按 2-3 次，间隔 5 秒以上。没有合适的门铃设备可以跳过，不要把口述“门铃”当成门铃声音。警笛和警报不建议在公共场所或高音量下试验；可以暂缓或在许可条件下以适中音量播放已获授权的录音，记录设备、音量、距离与背景噪声。声音经扬声器播放是功能联调，并不等于现场效果验证。

```text
python -X utf8 -m src.cli run --config configs/windows_sound_events.yaml --duration 60 --report
```

观察终端是否出现 `SED: Knock`、`SED: Doorbell` 等具体标签。浏览器打开 `http://127.0.0.1:8765`，在“音事件候选复核”中回听 1 秒证据，按“确认”或“驳回”；记录跨重启保留在 `reviews/candidates.sqlite3`，音频在 `evidence/`，二者均不上传 GitHub。只有目标类在 521 类中得分最高、且连续两个窗口相同，才会进入候选；这仍不是准确率保证。请在安静环境做门铃/敲门和说话、敲键盘等非目标对照，记录误报与漏报，再决定阈值和目标类别。真实麦克风收音仅在获得现场同意后开启，并定期自行清理本地证据文件。

最终日志若出现 `Inspection report written to ... (source=ollama)`，查看该路径所指向的新报告；纯 VAD 而无 SED 时会写事实模板，标为 `source=template`，这是预期行为。报告是模型生成的摘要，不等于人工复核结论。JSONL 记录在 `logs/windows_sound_events.jsonl`，它是追加写入，请按本次时间分辨旧记录。报告的 `monitoring` 时长应接近 60 秒；首末事件间隔可能更短，不能把它当作监测时长。

## 3. 如何判断

- 记录每个目标声音的实际播放/发生次数、是否检测到相应标签、错误标签及安静段误报次数。未检测到的目标声音也要记录，不能只选成功样本。
- YAMNet 是预训练通用模型，类别过滤与 `threshold: 0.25` 只是试验起点；阈值调高可能减少误报，也可能增加漏报，必须基于同一组有标注的音频比较。
- 同类事件默认有 3 秒冷却，因此 JSONL 行数不等于原始模型推理次数；VAD 状态也经过连续 300/600 ms 确认。不要从这一次试验推断准确率、召回率或可用于正式告警。
- 报告中的描述仍需人工核对。即便 `source=ollama`，也不能把模型生成的“风险可控”“没有误判”等当作证据；若出现这类文字，保留日志和报告反馈以继续收紧输出。

当前已具备候选留痕与人工复核，但还没有联动通知，也没有自动清理证据。严格要求目标类为全局第一名可以降低强行选目标的误报，也可能漏掉混杂声音；连续两个重叠窗口不是两次独立检出。后续需要建立有标注的目标/非目标小样本集，按逐类检出与误报校准阈值、合并相邻事件，再决定告警级别和通知策略。
