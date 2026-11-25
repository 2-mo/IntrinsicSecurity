import base64
import mimetypes
import time
from functools import lru_cache
from queue import Queue

import httpx


@lru_cache(maxsize=64)
def image_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def messages(sample, instruction, query):
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url(sample.image)}},
                {"type": "text", "text": f"Context:\n{sample.context}\n\n{query}"},
            ],
        },
    ]


class Backend:
    def __init__(self, endpoints, model, workers):
        self.model = model
        self.slots = Queue()
        self.client = httpx.Client(
            timeout=420, trust_env=False, limits=httpx.Limits(max_connections=workers)
        )
        try:
            for endpoint in endpoints:
                endpoint = endpoint.rstrip("/")
                models = self.client.get(endpoint + "/models")
                models.raise_for_status()
                if model not in {r["id"] for r in models.json()["data"]}:
                    raise ValueError(f"{model} is unavailable at {endpoint}")
                for _ in range((workers + len(endpoints) - 1) // len(endpoints)):
                    self.slots.put(endpoint)
        except Exception:
            self.client.close()
            raise

    def request(self, prompt, **options):
        endpoint = self.slots.get()
        try:
            for attempt in range(3):
                try:
                    response = self.client.post(
                        endpoint + "/chat/completions",
                        json={"model": self.model, "messages": prompt, **options},
                    )
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code < 500 or attempt == 2:
                        raise RuntimeError(
                            f"{endpoint}: {exc.response.text[:700]}"
                        ) from exc
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                time.sleep(2**attempt)
        finally:
            self.slots.put(endpoint)

    def complete(self, prompt, **options):
        return self.request(prompt, **options)["choices"][0]
