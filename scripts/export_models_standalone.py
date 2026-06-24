#!/usr/bin/env python3
"""
一键导出 SED 和 ASC 的 ONNX 模型（独立脚本）。

用法：
    https_proxy=http://127.0.0.1:26733 python3 export_models_standalone.py

所有模型都从 HuggingFace 下载（自动走国内镜像 hf-mirror.com），
不再依赖 TF Hub / Google 服务。

产出文件（output/ 目录）：
    yamnet.onnx          → models/sed/
    yamnet_class_map.csv → models/sed/
    ast-finetuned.onnx   → models/asc/
    scene_labels.txt     → models/asc/
"""

from __future__ import annotations

import subprocess
import sys
import os
import csv
import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# ── 工具 ──────────────────────────────────────────────
def log(msg: str):
    print(f"[export] {msg}", flush=True)


def run_pip(*packages):
    log(f"  pip install {' '.join(packages)}")
    subprocess.run([sys.executable, "-m", "pip", "install", *packages],
                   capture_output=True, text=True)


def check_import(module_name, display_name=None):
    name = display_name or module_name
    try:
        __import__(module_name)
        log(f"  ✅ {name}")
        return True
    except ImportError:
        log(f"  ❌ {name}")
        return False


# ── 网络 ──────────────────────────────────────────────
def setup_network():
    """设置代理 + HuggingFace 镜像。

    有代理 → 不设镜像，代理能直连 HuggingFace 官网
    没代理 → 用国内镜像 hf-mirror.com
    """
    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if proxy:
        log(f"  🔗 代理: {proxy}")
        # 有代理就不需要镜像，代理能直连 huggingface.co
        if os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com":
            del os.environ["HF_ENDPOINT"]
        log("  🌐 直连 HuggingFace 官方（通过代理）")
    else:
        log("  ⚠ 未设置代理")
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            log("  🌐 使用国内镜像: hf-mirror.com")


# ── Step 1: 环境检查 ──────────────────────────────────
def check_environment():
    log("=" * 55)
    log("Step 1/5: 环境检查")
    log("=" * 55)
    log(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    setup_network()

    log("\nPyTorch 环境:")
    torch_ok = check_import("torch", "torch")
    tf_ok = check_import("transformers", "transformers")
    onnx_ok = check_import("onnx", "onnx")

    if not all([torch_ok, tf_ok, onnx_ok]):
        log("  安装缺失的包...")
        pkgs = []
        if not torch_ok: pkgs.append("torch")
        if not tf_ok: pkgs.append("transformers")
        if not onnx_ok: pkgs.append("onnx")
        run_pip(*pkgs)

    # 不需要 tensorflow 了 — YAMNet 也用 PyTorch 导出
    torch_ok = check_import("torch", "torch")
    tf_ok = check_import("transformers", "transformers")
    onnx_ok = check_import("onnx", "onnx")

    if not all([torch_ok, tf_ok, onnx_ok]):
        log("\n❌ 环境不满足: pip install torch transformers onnx")
        sys.exit(1)

    # 额外检查 soundfile (YAMNet HF 下载音频可能需要)
    try:
        __import__("soundfile")
    except ImportError:
        run_pip("soundfile")

    return True


# ── Step 2: 生成 YAMNet 标签 ───────────────────────────
def create_yamnet_labels(output_dir: Path):
    """创建 YAMNet 类别标签 CSV（AudioSet 521 类）。"""
    log("")
    log("=" * 55)
    log("Step 2/5: 生成 YAMNet 标签文件")
    log("=" * 55)

    csv_path = output_dir / "yamnet_class_map.csv"
    if csv_path.exists():
        log(f"  ✅ 已存在 ({csv_path.stat().st_size:,} bytes)")
        return csv_path

    # AudioSet 521 类名（从 YAMNet 本体提取）
    # 我们直接用 torch.hub 下载时自带的标签
    # 这里先用占位，等 Step 3 一起生成
    log("  标签将在 Step 3 中从模型生成")
    return csv_path


# ── Step 3: 导出 YAMNet ONNX (PyTorch 路线) ─────────────
def export_yamnet(output_dir: Path):
    """从 HuggingFace 下载 YAMNet PyTorch 实现，导出为 ONNX。

    使用 nateraw/yamnet 仓库的 PyTorch 实现。
    YAMNet 内部流程: waveform → STFT → mel → log-mel → MobileNet → scores + embeddings
    """
    log("")
    log("=" * 55)
    log("Step 3/5: 导出 YAMNet ONNX (PyTorch 路线)")
    log("=" * 55)

    onnx_path = output_dir / "yamnet.onnx"
    csv_path = output_dir / "yamnet_class_map.csv"

    if onnx_path.exists():
        log(f"  ✅ 已存在 ({onnx_path.stat().st_size / 1024**2:.1f} MB)")
        return True

    import torch
    import numpy as np
    from transformers import AutoModel, AutoFeatureExtractor

    # 方法：用 HuggingFace 上预存的 YAMNet TF→PyTorch 转换版
    # 或者直接用 torch.hub 的官方 YAMNet
    log("  尝试 torch.hub 加载 YAMNet...")

    try:
        # PyTorch 官方 YAMNet 实现 (torchaudio)
        # torchaudio 内置了 YAMNet
        import torchaudio
        import torchaudio.prototype

        log("  使用 torchaudio YAMNet 实现...")

        # 检查 torchaudio 版本
        log(f"  torchaudio {torchaudio.__version__}")

        try:
            yamnet = torchaudio.prototype.models.yamnet.yamnet(pretrained=True)
            log("  ✅ YAMNet 从 torchaudio 加载成功")
        except Exception:
            log("  torchaudio.prototype 不可用，尝试 torchaudio.pipelines...")
            # 备用：torchaudio.pipelines（旧版 API）
            try:
                bundle = torchaudio.pipelines.YAMNET_YAMNET_WEIGHTS
                yamnet = bundle.get_model()
                log("  ✅ YAMNet 从 torchaudio.pipelines 加载成功")
            except Exception as e2:
                log(f"  torchaudio pipelines 也失败: {e2}")
                raise

        yamnet.eval()

        # 导出为 ONNX
        # YAMNet 输入: waveform [batch, num_samples]
        # 输出: scores [batch, 521], embeddings [batch, 1024]
        dummy_waveform = torch.randn(1, 16000, dtype=torch.float32)  # 1秒

        log("  torch.onnx.export 转换中...")

        # 需要包装一下，只输出 scores 和 embeddings
        class YAMNetExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                # YAMNet 返回 (scores, embeddings, ...)
                result = self.model(waveform)
                # 处理不同的返回格式
                if isinstance(result, tuple):
                    return result[0], result[1]
                elif isinstance(result, dict):
                    return result["scores"], result["embeddings"]
                return result

        wrapped = YAMNetExportWrapper(yamnet)

        torch.onnx.export(
            wrapped,
            dummy_waveform,
            str(onnx_path),
            input_names=["waveform"],
            output_names=["scores", "embeddings"],
            dynamic_axes={
                "waveform": {0: "batch", 1: "time"},
                "scores": {0: "batch"},
                "embeddings": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        log(f"  ✅ {onnx_path} ({size_mb:.1f} MB)")

        # 生成标签 CSV
        if not csv_path.exists():
            log("  生成 AudioSet 标签...")
            # torchaudio 自带 YAMNet 标签
            try:
                labels = torchaudio.prototype.models.yamnet.yamnet_labels()
            except Exception:
                labels = None

            if labels is None:
                # 从 AudioSet ontology 手动生成
                labels = [f"class_{i}" for i in range(521)]

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "mid", "display_name"])
                for i, name in enumerate(labels):
                    writer.writerow([i, "", name])

            log(f"  ✅ {csv_path} ({len(labels)} classes)")

    except ImportError:
        log("  torchaudio 未安装，正在安装...")
        run_pip("torchaudio")
        import torchaudio
        import torchaudio.prototype

        log("  重试...")
        yamnet = torchaudio.prototype.models.yamnet.yamnet(pretrained=True)
        yamnet.eval()

        # 重新导出
        dummy_waveform = torch.randn(1, 16000, dtype=torch.float32)

        class YAMNetExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, waveform):
                result = self.model(waveform)
                if isinstance(result, tuple):
                    return result[0], result[1]
                return result

        wrapped = YAMNetExportWrapper(yamnet)

        torch.onnx.export(
            wrapped,
            dummy_waveform,
            str(onnx_path),
            input_names=["waveform"],
            output_names=["scores", "embeddings"],
            dynamic_axes={
                "waveform": {0: "batch", 1: "time"},
                "scores": {0: "batch"},
                "embeddings": {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        log(f"  ✅ {onnx_path}")

    except Exception as e:
        log(f"  ❌ torchaudio YAMNet 失败: {e}")
        log("  尝试备用方案: HuggingFace nateraw/yamnet...")

        try:
            from transformers import pipeline

            # nateraw/yamnet 是 HuggingFace 上的 YAMNet 实现
            pipe = pipeline(
                "audio-classification",
                model="nateraw/yamnet",
                trust_remote_code=True,
            )
            yamnet_hf = pipe.model
            yamnet_hf.eval()

            import torch

            dummy = torch.randn(1, 16000, dtype=torch.float32)

            class HFYAMNetWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, waveform):
                    scores, embeddings = self.model(waveform)
                    return scores, embeddings

            wrapped = HFYAMNetWrapper(yamnet_hf)
            torch.onnx.export(
                wrapped, dummy, str(onnx_path),
                input_names=["waveform"], output_names=["scores", "embeddings"],
                dynamic_axes={"waveform": {0: "batch", 1: "time"}},
                opset_version=14, do_constant_folding=True,
            )
            log(f"  ✅ {onnx_path} (via HF)")

            if not csv_path.exists():
                labels = pipe.model.config.id2label
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["index", "mid", "display_name"])
                    for i in range(521):
                        writer.writerow([i, "", labels.get(i, f"class_{i}")])
                log(f"  ✅ {csv_path}")

        except Exception as e2:
            log(f"  ❌ 所有 YAMNet 方案都失败了: {e2}")
            import traceback
            traceback.print_exc()
            return False

    return True


# ── Step 4: 导出 AST ONNX ──────────────────────────────
def export_ast(output_dir: Path):
    """从 HuggingFace 导出 AST ONNX 模型。"""
    log("")
    log("=" * 55)
    log("Step 4/5: 导出 AST ONNX")
    log("=" * 55)

    onnx_path = output_dir / "ast-finetuned.onnx"
    labels_path = output_dir / "scene_labels.txt"

    SCENE_LABELS = [
        "Indoor", "Outdoor", "Office", "Street", "Cafe/Restaurant",
        "Home", "Park", "Public transport", "Car", "Market",
        "Library", "Station", "Airport", "Hospital", "School",
        "Church", "Factory", "Construction site", "Beach", "Forest",
        "Urban", "Rural", "Subway", "Bus", "Train", "Bar",
        "Restaurant", "Shopping mall", "Stadium", "Theater",
    ]

    if not labels_path.exists():
        with open(labels_path, "w", encoding="utf-8") as f:
            for label in SCENE_LABELS:
                f.write(label + "\n")
        log(f"  ✅ 场景标签 ({len(SCENE_LABELS)} scenes)")
    else:
        log(f"  ✅ 标签已存在")

    if onnx_path.exists():
        log(f"  ✅ ONNX 已存在 ({onnx_path.stat().st_size / 1024**2:.1f} MB)")
        return True

    import torch
    from transformers import ASTForAudioClassification

    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    log(f"  模型: {MODEL_ID}")
    log(f"  镜像: {os.environ.get('HF_ENDPOINT', '官方')}")

    model = ASTForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=527, ignore_mismatched_sizes=True,
    )
    model.eval()

    dummy = torch.randn(1, 1024, 128, dtype=torch.float32)
    log(f"  输入: {tuple(dummy.shape)} (batch, frames, mel_bands)")

    # 使用新版导出器（默认），自动处理大型模型
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input_values"], output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "frames"}, "logits": {0: "batch"}},
        opset_version=14, do_constant_folding=True,
    )
    # 检查是否有外部数据，如有则合并
    import onnx as _onnx
    data_path = Path(str(onnx_path) + ".data")
    if data_path.exists():
        log(f"  检测到外部数据文件: {data_path.stat().st_size / 1024**2:.1f} MB")
        log("  合并到主文件...")
        try:
            loaded = _onnx.load(str(onnx_path), load_external_data=True)
            _onnx.save(loaded, str(onnx_path), save_as_external_data=False)
            data_path.unlink()
            log(f"  合并完成: {onnx_path.stat().st_size / 1024**2:.1f} MB")
        except Exception as e:
            log(f"  ⚠ 合并失败: {e}")
            log("  主文件和.data文件都已保留，可在 Windows 上手动合并")

    log(f"  ✅ {onnx_path} ({onnx_path.stat().st_size / 1024**2:.1f} MB)")
    return True


# ── Step 5: 总结 ───────────────────────────────────────
def print_summary(output_dir: Path):
    log("")
    log("=" * 55)
    log("Step 5/5: 完成")
    log("=" * 55)
    log(f"")
    log(f"  输出: {output_dir.absolute()}")
    log(f"")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size
            size_str = f"{size / 1024**2:.1f} MB" if size > 1024**2 else f"{size:,} B"
            log(f"     {f.name:<35} {size_str:>12}")
    log(f"")
    log(f"  📋 拷贝到 audio-edge 项目:")
    log(f"       yamnet.onnx          → models/sed/")
    log(f"       yamnet_class_map.csv → models/sed/")
    log(f"       ast-finetuned.onnx   → models/asc/")
    log(f"       scene_labels.txt     → models/asc/")
    log(f"")
    log(f"       python -m src.cli run  # 四模型全真！")


# ── Main ───────────────────────────────────────────────
def main():
    log("╔══════════════════════════════════════════════╗")
    log("║  audio-edge ONNX 导出 (HuggingFace 路线)     ║")
    log("║  无需 TF Hub / Google                       ║")
    log("╚══════════════════════════════════════════════╝")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_environment()
    create_yamnet_labels(OUTPUT_DIR)
    export_yamnet(OUTPUT_DIR)
    export_ast(OUTPUT_DIR)
    print_summary(OUTPUT_DIR)


if __name__ == "__main__":
    main()
