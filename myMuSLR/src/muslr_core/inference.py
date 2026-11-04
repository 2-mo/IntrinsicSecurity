"""Standalone JSON inference over OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from openai import OpenAI


@dataclass(frozen=True)
class ModelConfig:
    endpoints: tuple[str, ...]
    model: str
    api_key: str = "EMPTY"
    timeout: float = 180.0
    image_url_mode: str = "data"
    pool_concurrency: int = 2

    def __post_init__(self) -> None:
        if not self.endpoints or any(
            urlsplit(endpoint).scheme not in {"http", "https"}
            or not urlsplit(endpoint).hostname
            for endpoint in self.endpoints
        ):
            raise ValueError("Use explicit --endpoints HTTP(S) URLs ending in /v1; 'auto' is unsupported")
        if self.pool_concurrency < 1 or self.timeout <= 0:
            raise ValueError("pool-concurrency and timeout must be positive")
        if self.image_url_mode not in {"data", "file"}:
            raise ValueError("image-url-mode must be data or file")

    @property
    def max_workers(self) -> int:
        return len(self.endpoints) * self.pool_concurrency


@dataclass(frozen=True)
class CompletionMetadata:
    output: str
    latency_sec: float
    usage: dict[str, Any] | None
    endpoint: str
    model: str


def image_block(path: Path) -> dict[str, Any]:
    """Keep local paths until the HTTP client chooses data URLs or shared files."""
    return {"type": "image_url", "image_url": {"url": path.resolve().as_uri()}}


def prepare_messages(messages: list[dict[str, Any]], image_url_mode: str) -> list[dict[str, Any]]:
    prepared = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "image_url":
                    image = dict(block["image_url"])
                    url = image["url"]
                    if url.startswith("file://") and image_url_mode == "data":
                        parsed = urlsplit(url)
                        if parsed.netloc not in {"", "localhost"}:
                            raise ValueError("Only local file:// image paths are supported")
                        path = Path(unquote(parsed.path))
                        mime = mimetypes.guess_type(path.name)[0] or "image/png"
                        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                        image["url"] = f"data:{mime};base64,{encoded}"
                    block = {**block, "image_url": image}
                blocks.append(block)
            content = blocks
        prepared.append({**message, "content": content})
    return prepared


class JSONClient:
    """Bound requests per endpoint, disable SDK retries, and own client cleanup."""

    def __init__(self, config: ModelConfig, *, max_tokens: int, temperature: float = 0.0) -> None:
        self.config = config
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._slots: Queue[tuple[str, OpenAI]] = Queue()
        self._clients: list[OpenAI] = []
        try:
            for endpoint in config.endpoints:
                local = urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}
                client = OpenAI(
                    base_url=endpoint,
                    api_key=config.api_key or "EMPTY",
                    timeout=config.timeout,
                    max_retries=0,
                    http_client=httpx.Client(timeout=config.timeout, trust_env=not local),
                )
                self._clients.append(client)
                for _ in range(config.pool_concurrency):
                    self._slots.put((endpoint, client))
        except Exception:
            self.close()
            raise

    def complete_with_metadata(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionMetadata:
        prepared = prepare_messages(messages, self.config.image_url_mode)
        endpoint, client = self._slots.get()
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=prepared,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                temperature=self.temperature if temperature is None else temperature,
                response_format=response_format,
            )
            if not response.choices:
                raise RuntimeError(f"Model returned no choices from {endpoint}")
            return CompletionMetadata(
                output=response.choices[0].message.content or "",
                latency_sec=time.monotonic() - started,
                usage=response.usage.model_dump() if response.usage else None,
                endpoint=endpoint,
                model=response.model,
            )
        finally:
            self._slots.put((endpoint, client))

    def close(self) -> None:
        for client in self._clients:
            client.close()
        self._clients.clear()

    def __enter__(self) -> JSONClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
