"""ONNX embedding model for event text.

Uses all-MiniLM-L6-v2 (22M params, 384-dim, ~90 MB ONNX) —
the lightest battle-tested embedding model available.

Falls back gracefully if the ONNX model is not downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_PATH = Path("models/embedding/all-MiniLM-L6-v2.onnx")
EMBEDDING_DIM = 384


class Embedder:
    """ONNX text embedder.

    Parameters
    ----------
    model_path : str or Path
        Path to ONNX embedding model.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self._model_path = Path(model_path)
        self._model: object | None = None  # ort.InferenceSession
        self._tokenizer: object | None = None
        self._is_loaded = False

    # -- Public API -------------------------------------------------------
    def load(self) -> None:
        """Load ONNX model. Falls back to random embeddings if missing."""
        if self._is_loaded:
            return

        if not self._model_path.exists():
            self._is_loaded = True
            return

        import onnxruntime as ort

        self._model = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        self._is_loaded = True

    def embed(self, text: str) -> np.ndarray:
        """Convert text to a 384-dim embedding vector.

        If the real model is unavailable, returns a simple hash-based
        pseudo-embedding (deterministic, usable for testing).
        """
        if self._model is None:
            return self._fallback_embed(text)

        tokens = self._tokenize(text)
        input_name = self._model.get_inputs()[0].name
        result = self._model.run(None, {input_name: tokens})[0]
        # Mean pool across token dimension
        embedding = result.mean(axis=1).squeeze()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (len(texts), 384)."""
        return np.stack([self.embed(t) for t in texts])

    # -- Helpers ----------------------------------------------------------
    def _tokenize(self, text: str) -> np.ndarray:
        """Simple BERT-style tokenization.

        For production, use the HuggingFace tokenizer or the ONNX model's
        built-in tokenizer.  This is a minimal fallback.
        """
        # Placeholder: real tokenization needs the model's vocab
        # For now, return a dummy input that matches the model shape
        # Actual usage requires exporting the model with tokenizer or
        # using the transformers library.
        max_len = 128
        ids = np.zeros((1, max_len), dtype=np.int64)
        mask = np.zeros((1, max_len), dtype=np.int64)
        # Combined input: [input_ids, attention_mask, token_type_ids]
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros((1, max_len), dtype=np.int64),
        }

    @staticmethod
    def _fallback_embed(text: str) -> np.ndarray:
        """Deterministic pseudo-embedding from text hash."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # Expand 32 bytes → 384 floats
        vec = np.array([b / 255.0 for b in h * 12], dtype=np.float32)[:EMBEDDING_DIM]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
