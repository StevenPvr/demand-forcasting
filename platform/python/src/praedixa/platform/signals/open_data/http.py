from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)


class HttpJsonReader(Protocol):
    def __call__(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> object: ...


class HttpTextReader(Protocol):
    def __call__(
        self,
        url: str,
        *,
        timeout_seconds: int,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> str: ...


def _http_request(url: str) -> Request:
    return Request(url, headers={"User-Agent": "praedixa-open-exogenous/1.0"})


def _read_response_text(request: Request, *, timeout_seconds: int) -> str:
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = cast(bytes, response.read())
    return payload.decode("utf-8")


def _read_http_attempt(
    *,
    request: Request,
    url: str,
    timeout_seconds: int,
    attempt: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str | None:
    try:
        return _read_response_text(request, timeout_seconds=timeout_seconds)
    except HTTPError as exc:
        if _retry_http_error(
            exc=exc,
            url=url,
            attempt=attempt,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        ):
            return None
        raise
    except URLError:
        if _retry_network_error(
            url=url,
            attempt=attempt,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        ):
            return None
        raise


def _sleep_before_retry(
    *,
    url: str,
    sleep_seconds: float,
    attempt: int,
    max_retries: int,
    message: str,
) -> None:
    LOGGER.warning(
        message,
        url,
        sleep_seconds,
        attempt + 1,
        max_retries,
    )
    time.sleep(sleep_seconds)


def _retry_sleep_seconds(retry_backoff_seconds: float, attempt: int) -> float:
    return float(retry_backoff_seconds * (2**attempt))


def _retry_http_error(
    *,
    exc: HTTPError,
    url: str,
    attempt: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> bool:
    if exc.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
        return False
    _sleep_before_retry(
        url=url,
        sleep_seconds=_retry_sleep_seconds(retry_backoff_seconds, attempt),
        attempt=attempt,
        max_retries=max_retries,
        message=f"HTTP {exc.code} on %s, retrying in %.1fs (attempt %s/%s)",
    )
    return True


def _retry_network_error(
    *,
    url: str,
    attempt: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> bool:
    if attempt >= max_retries:
        return False
    _sleep_before_retry(
        url=url,
        sleep_seconds=_retry_sleep_seconds(retry_backoff_seconds, attempt),
        attempt=attempt,
        max_retries=max_retries,
        message="Network error on %s, retrying in %.1fs (attempt %s/%s)",
    )
    return True


def http_read(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    request = _http_request(url)
    for attempt in range(max_retries + 1):
        response_text = _read_http_attempt(
            request=request,
            url=url,
            timeout_seconds=timeout_seconds,
            attempt=attempt,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        if response_text is not None:
            return response_text
    raise RuntimeError(f"Failed to download {url}")


def http_json(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> object:
    return json.loads(
        http_read(
            url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    )


def http_text(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    return http_read(
        url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def curl_text(
    url: str,
    *,
    timeout_seconds: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    command = ["curl", "-fsSL", "--max-time", str(timeout_seconds), url]
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as exc:
            if attempt < max_retries:
                sleep_seconds = retry_backoff_seconds * (2**attempt)
                LOGGER.warning(
                    "curl failed on %s, retrying in %.1fs (attempt %s/%s)",
                    url,
                    sleep_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_seconds)
                continue
            raise RuntimeError(f"Failed to download {url} with curl") from exc
    raise RuntimeError(f"Failed to download {url} with curl")
