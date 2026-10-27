"""Main pipeline orchestrator.

Wires together audio capture, VAD, inference models, scheduling,
event aggregation, and output backends into a single runnable pipeline.

Thread model:
  - Thread 1 (HIGH): PortAudio capture callback → RingBuffer
  - Thread 2 (MED):  VAD + feature extraction loop
  - Thread 3 (LOW):  Inference worker pool (KWS / SED / ASC)
  - Main thread:     Event loop, keyboard interrupt handling
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from src.capture.stream import AudioCapture, MockAudioCapture
from src.models.asc import ASTSceneClassifier, MockASC
from src.models.base import InferenceResult
from src.models.kws import MockKWS, SherpaKWS
from src.models.sed import MockSED, YAMNetSED
from src.models.vad import SileroVAD
from src.output.console import ConsoleOutput
from src.output.jsonl import JSONLOutput
from src.output.review import ReviewStore
from src.pipeline.aggregator import AggregatorConfig, EventAggregator
from src.pipeline.scheduler import Scheduler
from src.pipeline.sed_confirmation import SEDConfirmation
from src.pipeline.vad_state import VADState
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
        self._report_events: list[dict] = []
        self._report_generated = False
        self._run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self._started_at: float | None = None
        self._ended_at: float | None = None

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
            self._vad_state = VADState(
                sample_rate=audio_cfg["sample_rate"],
                threshold=vad_cfg["threshold"],
                release_threshold=vad_cfg.get("release_threshold", vad_cfg["threshold"]),
                min_speech_duration_ms=vad_cfg["min_speech_duration_ms"],
                min_silence_duration_ms=vad_cfg["min_silence_duration_ms"],
            )
        else:
            self._vad = None
            self._vad_state = None

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

        # --- SED Model ---
        sed_cfg = config["sed"]
        if sed_cfg["enabled"]:
            sed_model_path = Path(sed_cfg["model_path"])
            if sed_model_path.exists():
                self._sed = YAMNetSED(
                    model_path=sed_cfg["model_path"],
                    labels_path=sed_cfg.get("labels_path", ""),
                    threshold=sed_cfg["threshold"],
                    target_classes=sed_cfg.get("target_classes", []),
                    backend=config["inference"]["backend"],
                    use_gpu=config["inference"]["use_gpu"],
                )
            else:
                if config["inference"].get("strict_models", False):
                    raise FileNotFoundError(f"Required YAMNet SED model not found: {sed_model_path}")
                logger.info("YAMNet SED model not found, using mock SED")
                self._sed = MockSED()
        else:
            self._sed = None
        review_cfg = config.get("review", {})
        self._review = (
            ReviewStore(review_cfg["db_path"], review_cfg["evidence_dir"])
            if review_cfg.get("enabled", False) else None
        )
        self._sed_confirmation = SEDConfirmation(sed_cfg.get("confirm_windows", 1))

        # --- ASC Model ---
        asc_cfg = config["asc"]
        if asc_cfg["enabled"]:
            asc_model_path = Path(asc_cfg["model_path"])
            if asc_model_path.exists():
                self._asc = ASTSceneClassifier(
                    model_path=asc_cfg["model_path"],
                    labels_path=asc_cfg.get("labels_path", ""),
                    threshold=0.5,
                    min_duration_sec=asc_cfg.get("min_duration_sec", 2.0),
                    backend=config["inference"]["backend"],
                    use_gpu=config["inference"]["use_gpu"],
                )
            else:
                logger.info("AST model not found, using mock ASC")
                self._asc = MockASC()
        else:
            self._asc = None

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

        if out_cfg.get("dashboard"):
            from dashboard.output import DashboardOutput

            dash_out = DashboardOutput(
                host=out_cfg.get("dashboard_host", "0.0.0.0"),
                port=out_cfg.get("dashboard_port", 8765),
                review_store=self._review,
            )
            self._outputs.append(dash_out)

        # --- State ---
        self._background_audio: Queue[np.ndarray] = Queue(maxsize=128)
        self._last_audio_drop_warning = 0.0

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
        if self._sed:
            self._sed.load()
        if self._asc:
            self._asc.load()

        if self._review:
            for output in self._outputs:
                if hasattr(output, "start"):
                    output.start()

        # Start audio capture
        self._capture.start()
        self._started_at = time.time()
        self._running.set()

        # Start VAD processing thread
        if self._vad or self._sed or self._asc:
            capture_thread = threading.Thread(
                target=self._capture_loop,
                name="audio-processor",
                daemon=True,
            )
            capture_thread.start()
            self._threads.append(capture_thread)

        # Start background inference thread (SED + ASC)
        if self._sed or self._asc:
            bg_thread = threading.Thread(
                target=self._background_loop,
                name="background-inference",
                daemon=True,
            )
            bg_thread.start()
            self._threads.append(bg_thread)

    def stop(self) -> None:
        """Stop all sub-systems gracefully."""
        logger.info("Shutting down pipeline...")
        self._running.clear()
        self._ended_at = time.time()

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
        if self._sed:
            self._sed.unload()
        if self._asc:
            self._asc.unload()

        self._write_report()

        # Close outputs
        for out in self._outputs:
            if hasattr(out, "close"):
                out.close()

        logger.info("Pipeline stopped.")

    def _write_report(self) -> None:
        if self._report_generated or not self._cfg.get("report", {}).get("enabled", False):
            return
        self._report_generated = True
        report_cfg = self._cfg["report"]
        try:
            from agent.report import EventReportGenerator

            EventReportGenerator(
                model=report_cfg["model"],
                base_url=report_cfg["base_url"],
                output_path=report_cfg["output_path"],
                timeout_sec=report_cfg["timeout_sec"],
                run_id=self._run_id,
            ).generate(self._report_events, started_at=self._started_at, ended_at=self._ended_at)
        except Exception as exc:
            logger.error("Report generation failed: {}", exc)

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
                if duration_sec > 0 and (time.time() - start_time) >= duration_sec:
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _capture_loop(self) -> None:
        """Single capture consumer; fan out each chunk to background inference."""
        window_size = 512  # Silero VAD window (32ms @ 16kHz)

        while self._running.is_set():
            available = self._capture.available
            if available < window_size:
                time.sleep(0.005)
                continue

            audio_chunk = self._capture.read(window_size).squeeze()
            if len(audio_chunk) < window_size:
                continue

            if self._sed or self._asc:
                try:
                    self._background_audio.put_nowait(audio_chunk)
                except Full:
                    self._background_audio.get_nowait()
                    self._background_audio.put_nowait(audio_chunk)
                    if time.monotonic() - self._last_audio_drop_warning > 5:
                        logger.warning("Background inference lagging; dropped old audio chunk")
                        self._last_audio_drop_warning = time.monotonic()

            if self._vad:
                try:
                    result = self._vad.infer(audio_chunk)
                    transition = self._vad_state.update(result, len(audio_chunk))
                    if transition:
                        self._emit(transition)
                except Exception as e:
                    logger.error(f"VAD error: {e}")
                    continue

            if self._vad_state and self._vad_state.is_speech:
                self._run_inference_speech(audio_chunk)

    def _run_inference_speech(self, audio_chunk: np.ndarray) -> None:
        if self._kws and self._scheduler.is_due("kws", speech_active=True):
            try:
                result = self._kws.infer(audio_chunk)
                if result.label != "no_keyword":
                    self._emit(result)
            except Exception as e:
                logger.error(f"KWS error: {e}")
            self._scheduler.mark_run("kws")

    def _background_loop(self) -> None:
        """Background inference thread for SED and ASC tasks.

        Accumulates audio into longer windows and runs inference
        at configured intervals, independent of speech activity.
        """
        sed_buf = np.zeros((0,), dtype=np.float32)
        asc_buf = np.zeros((0,), dtype=np.float32)

        sed_window = int(self._sample_rate * 1.0)  # 1 second for SED
        asc_window = int(self._sample_rate * 3.0)  # 3 seconds for ASC

        while self._running.is_set():
            try:
                chunk = self._background_audio.get(timeout=0.05)
                if self._sed:
                    sed_buf = np.concatenate([sed_buf, chunk])
                if self._asc:
                    asc_buf = np.concatenate([asc_buf, chunk])
            except Empty:
                pass

            # --- SED Inference (every ~1s) ---
            if self._sed and len(sed_buf) >= sed_window and self._scheduler.is_due("sed"):
                try:
                    result = self._sed.infer(sed_buf[:sed_window])
                    if self._sed_confirmation.update(result):
                        self._emit(result, evidence_audio=sed_buf[:sed_window])
                except Exception as e:
                    logger.error(f"SED error: {e}")
                self._scheduler.mark_run("sed")
                sed_buf = sed_buf[sed_window // 2 :]  # 50% overlap

            # --- ASC Inference (every ~3s) ---
            if self._asc and len(asc_buf) >= asc_window and self._scheduler.is_due("asc"):
                try:
                    result = self._asc.infer(asc_buf[:asc_window])
                    self._emit(result)
                except Exception as e:
                    logger.error(f"ASC error: {e}")
                self._scheduler.mark_run("asc")
                asc_buf = asc_buf[asc_window // 2 :]  # 50% overlap

            # Limit buffer growth
            max_buf = max(sed_window, asc_window) * 2
            if len(sed_buf) > max_buf:
                sed_buf = sed_buf[-max_buf:]
            if len(asc_buf) > max_buf:
                asc_buf = asc_buf[-max_buf:]


    def _emit(self, result: InferenceResult, evidence_audio: np.ndarray | None = None) -> None:
        """Filter through aggregator and send to all outputs."""
        if self._aggregator.should_emit(result):
            if self._review and result.task == "sed" and evidence_audio is not None:
                try:
                    result.metadata["candidate_id"] = self._review.add(result, evidence_audio, self._sample_rate)
                except Exception as exc:
                    logger.error("Evidence storage failed: {}", exc)
            from agent.report import event_record

            self._report_events.append(event_record(result))
            for output in self._outputs:
                try:
                    output.emit(result)
                except Exception as e:
                    logger.error(f"Output error [{type(output).__name__}]: {e}")
