"""Main pipeline orchestrator.

Wires together audio capture, VAD, inference models, scheduling,
event aggregation, and output backends into a single runnable pipeline.

Thread model:
  - Thread 1 (HIGH): PortAudio capture callback → RingBuffer
  - Thread 2 (MED):  VAD + feature extraction loop
  - Thread 3 (LOW):  Inference worker pool (KWS / SED / ASC)
  - Main thread:     Event loop, keyboard interrupt handling
"""

import signal
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.capture.ring_buffer import RingBuffer
from src.capture.stream import AudioCapture, MockAudioCapture
from src.models.base import InferenceResult
from src.models.kws import MockKWS, SherpaKWS
from src.models.vad import SileroVAD
from src.output.console import ConsoleOutput
from src.output.jsonl import JSONLOutput
from src.pipeline.aggregator import AggregatorConfig, EventAggregator
from src.pipeline.scheduler import Scheduler
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Central pipeline controller.

    Usage:
        orch = Orchestrator(config)
        orch.start()
        try:
            orch.run()  # Blocks until Ctrl+C
        finally:
            orch.stop()
    """

    def __init__(self, config: dict):
        """Initialize orchestrator from configuration.

        Args:
            config: Full configuration dictionary.
        """
        self._cfg = config
        self._running = threading.Event()
        self._threads: list[threading.Thread] = []

        # --- Audio Capture ---
        audio_cfg = config["audio"]
        use_mock = config.get("_mock_audio", False)  # Hidden flag for testing

        capture_cls = MockAudioCapture if use_mock else AudioCapture
        self._capture = capture_cls(
            sample_rate=audio_cfg["sample_rate"],
            device=audio_cfg.get("device"),
            channels=audio_cfg["channels"],
            block_size=audio_cfg["block_size"],
            buffer_duration_sec=audio_cfg["buffer_duration_sec"],
        )
        self._sample_rate = audio_cfg["sample_rate"]

        # --- VAD Model ---
        vad_cfg = config["vad"]
        if vad_cfg["enabled"]:
            self._vad = SileroVAD(
                model_path=vad_cfg["model_path"],
                threshold=vad_cfg["threshold"],
            )
        else:
            self._vad = None

        # --- KWS Model ---
        kws_cfg = config["kws"]
        if kws_cfg["enabled"]:
            # Try sherpa-onnx first, fall back to mock
            kws_model_path = Path(kws_cfg["model_path"])
            if kws_model_path.exists() and kws_model_path.is_dir():
                self._kws = SherpaKWS(
                    model_dir=kws_cfg["model_path"],
                    backend=config["inference"]["backend"],
                    use_gpu=config["inference"]["use_gpu"],
                )
            else:
                logger.info("sherpa-onnx KWS model not found, using mock KWS")
                self._kws = MockKWS()
        else:
            self._kws = None

        # --- Scheduler ---
        self._scheduler = Scheduler()
        if kws_cfg["enabled"]:
            self._scheduler.add_task(
                "kws",
                interval_sec=kws_cfg["stride_sec"],
                requires_speech=True,
            )
        if config["sed"]["enabled"]:
            self._scheduler.add_task(
                "sed",
                interval_sec=config["sed"]["interval_sec"],
                requires_speech=False,
            )
        if config["asc"]["enabled"]:
            self._scheduler.add_task(
                "asc",
                interval_sec=config["asc"]["interval_sec"],
                requires_speech=False,
            )

        # --- Event Aggregator ---
        self._aggregator = EventAggregator(
            AggregatorConfig(
                kws_cooldown_sec=kws_cfg.get("cooldown_sec", 1.5),
            )
        )

        # --- Outputs ---
        self._outputs: list = []
        out_cfg = config["output"]

        if out_cfg["console"]:
            self._outputs.append(ConsoleOutput(color=True))

        if out_cfg.get("jsonl_path"):
            jsonl_out = JSONLOutput(out_cfg["jsonl_path"])
            jsonl_out.open()
            self._outputs.append(jsonl_out)

        # --- State ---
        self._is_speech = False
        self._vad_buf = np.zeros((0, 1), dtype=np.float32)  # Accumulates for VAD window

    def start(self) -> None:
        """Start all sub-systems: audio capture, load models, start threads."""
        logger.info("=" * 50)
        logger.info("audio-edge pipeline starting")
        logger.info(f"  Audio: {self._capture.device_name} @ {self._sample_rate}Hz")
        logger.info(f"  VAD: {'enabled' if self._vad else 'disabled'}")
        logger.info(f"  KWS: {'enabled' if self._kws else 'disabled'}")
        logger.info(f"  SED: {'enabled' if 'sed' in self._scheduler.tasks else 'disabled'}")
        logger.info(f"  ASC: {'enabled' if 'asc' in self._scheduler.tasks else 'disabled'}")
        logger.info("=" * 50)

        # Load models
        if self._vad:
            self._vad.load()
        if self._kws:
            self._kws.load()

        # Start audio capture
        self._capture.start()
        self._running.set()

        # Start VAD processing thread
        if self._vad:
            vad_thread = threading.Thread(
                target=self._vad_loop,
                name="vad-processor",
                daemon=True,
            )
            vad_thread.start()
            self._threads.append(vad_thread)

    def stop(self) -> None:
        """Stop all sub-systems gracefully."""
        logger.info("Shutting down pipeline...")
        self._running.clear()

        # Stop threads
        for t in self._threads:
            t.join(timeout=2.0)

        # Stop capture
        self._capture.stop()

        # Unload models
        if self._vad:
            self._vad.unload()
        if self._kws:
            self._kws.unload()

        # Close outputs
        for out in self._outputs:
            if hasattr(out, "close"):
                out.close()

        logger.info("Pipeline stopped.")

    def run(self, duration_sec: float = 0) -> None:
        """Run the pipeline (main thread blocks).

        Args:
            duration_sec: Run for N seconds (0 = until Ctrl+C).
        """
        if not self._running.is_set():
            self.start()

        start_time = time.time()

        try:
            while self._running.is_set():
                # Check duration
                if duration_sec > 0 and (time.time() - start_time) >= duration_sec:
                    break

                # Main thread sleeps — real work happens in background threads.
                # We wake periodically to check running state and duration.
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _vad_loop(self) -> None:
        """VAD processing thread.

        Reads audio from capture, runs VAD, dispatches to inference.
        """
        window_size = 512  # Silero VAD window (32ms @ 16kHz)

        while self._running.is_set():
            # Read audio chunks from capture buffer
            available = self._capture.available
            if available < window_size:
                time.sleep(0.005)  # 5ms polling to avoid busy-wait
                continue

            # Read one window at a time
            audio_chunk = self._capture.read(window_size).squeeze()

            if len(audio_chunk) < window_size:
                continue

            # --- VAD Inference ---
            speech_changed = False
            if self._vad:
                try:
                    result = self._vad.infer(audio_chunk)
                    was_speech = self._is_speech
                    self._is_speech = (result.label == "speech")

                    if was_speech != self._is_speech:
                        speech_changed = True
                        # Emit VAD state change
                        self._emit(result)
                except Exception as e:
                    logger.error(f"VAD error: {e}")
                    continue

            # --- Inference Scheduling ---
            if self._is_speech or speech_changed:
                self._run_inference_speech(audio_chunk)
            else:
                self._run_inference_background()

    def _run_inference_speech(self, audio_chunk: np.ndarray) -> None:
        """Run speech-triggered inference tasks (KWS)."""
        # KWS: run on every speech frame
        if self._kws and self._scheduler.is_due("kws", speech_active=True):
            try:
                result = self._kws.infer(audio_chunk)
                if result.label != "no_keyword":
                    self._emit(result)
            except Exception as e:
                logger.error(f"KWS error: {e}")
            self._scheduler.mark_run("kws")

    def _run_inference_background(self) -> None:
        """Run background inference tasks (SED, ASC) regardless of speech."""
        # These will be implemented in Phase 4
        pass

    def _emit(self, result: InferenceResult) -> None:
        """Filter through aggregator and send to all outputs."""
        if self._aggregator.should_emit(result):
            for output in self._outputs:
                try:
                    output.emit(result)
                except Exception as e:
                    logger.error(f"Output error [{type(output).__name__}]: {e}")
