"""Video pipeline for real-time object detection on edge devices.

Self-contained module — can be copied to any project that needs
camera capture + YOLO detection.

Usage::

    from vision.output import VisionOutput

    vo = VisionOutput(camera_id=0, model_path="models/vision/yolov8n.onnx")
    vo.start()
    result = vo.get_latest()  # {"objects": [...], "counts": {...}, "timestamp": ...}
    vo.stop()
"""
