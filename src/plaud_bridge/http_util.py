"""
Minimal HTTP client built on urllib.

Deliberately no requests/httpx dependency. This tool should still install on a
machine you have not touched in years. What we need is small: JSON POST,
multipart POST, exponential backoff with jitter, and honest error messages.
"""

from __future__ import annotations

import json
import mimetypes
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .logging_setup import get

log = get("http")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body[:2000]

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in RETRYABLE_STATUS


def _sleep_backoff(attempt: int, base: float = 1.5, cap: float = 45.0) -> None:
    delay = min(cap, base ** attempt) * (0.6 + 0.8 * random.random())
    log.debug("backing off %.1fs (attempt %d)", delay, attempt)
    time.sleep(delay)


def _request(req: urllib.request.Request, timeout: int) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        body = b""
        try:
            body = exc.read()
        except Exception:
            pass
        raise HttpError(
            f"HTTP {exc.code} from {req.full_url}", exc.code, body.decode("utf-8", "replace")
        ) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"network error contacting {req.full_url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpError(f"timeout contacting {req.full_url}") from exc


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str],
              timeout: int = 180, max_retries: int = 4) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    last: HttpError | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        try:
            _, body = _request(req, timeout)
            return json.loads(body.decode("utf-8"))
        except HttpError as exc:
            last = exc
            if not exc.retryable or attempt == max_retries:
                break
            _sleep_backoff(attempt)
        except json.JSONDecodeError as exc:
            raise HttpError(f"non-JSON response from {url}: {exc}") from exc
    if last is None:
        # Only reachable if max_retries made the loop body never run.
        raise HttpError(f"no request attempted for {url}; check max_retries in config")
    raise last


def post_multipart(url: str, fields: dict[str, str], file_path: Path,
                   file_field: str, headers: dict[str, str],
                   timeout: int = 300, max_retries: int = 4) -> dict[str, Any]:
    """Multipart upload. Used for audio transcription endpoints."""
    boundary = f"----plaudbridge{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{file_path.name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    hdrs = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        **headers,
    }

    last: HttpError | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
        try:
            _, raw = _request(req, timeout)
            return json.loads(raw.decode("utf-8"))
        except HttpError as exc:
            last = exc
            if not exc.retryable or attempt == max_retries:
                break
            _sleep_backoff(attempt)
        except json.JSONDecodeError as exc:
            raise HttpError(f"non-JSON response from {url}: {exc}") from exc
    if last is None:
        # Only reachable if max_retries made the loop body never run.
        raise HttpError(f"no request attempted for {url}; check max_retries in config")
    raise last
