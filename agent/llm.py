"""Ollama API client for local LLM inference.

Thin wrapper around Ollama's OpenAI-compatible HTTP API.
Supports chat, streaming, and embedding endpoints.

Ollama must be running: ``ollama serve``
"""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """Minimal Ollama API client.

    Parameters
    ----------
    model : str
        Model name as known to Ollama (e.g. ``"qwen2.5:7b-q4_K_M"``).
    base_url : str
        Ollama server URL.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    # -- Chat -------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        """Non-streaming chat completion.

        Parameters
        ----------
        messages : list[dict]
            List of ``{"role": "user"|"assistant"|"system", "content": str}``.
        temperature : float
        max_tokens : int

        Returns
        -------
        str
            Model response text.
        """
        import requests

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    def chat_stream(self, messages: list[dict], temperature: float = 0.7):
        """Streaming chat completion — yields text chunks."""
        import requests

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"].get("content", "")
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            raise

    # -- Embedding --------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        """Get embedding vector for text (if model supports it)."""
        import requests

        try:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            logger.warning(f"Ollama embed failed: {e}")
            raise

    # -- Model management -------------------------------------------------
    def list_models(self) -> list[str]:
        """List available models."""
        import requests

        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url
