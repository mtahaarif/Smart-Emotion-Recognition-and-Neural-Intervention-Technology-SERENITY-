import os
import json
import re
import time
from typing import Any, Iterator, List

import requests

try:
    import wordninja
except ImportError:
    wordninja = None


class CloudLLMError(RuntimeError):
    pass


class CloudLLMClient:
    """HTTP client for EC2-hosted LLM service.

    Expected API contract:
    POST {base_url}/chat
    JSON body: {"text": "..."}
    """

    def __init__(self) -> None:
        primary_api_url = os.getenv("SERENITY_CLOUD_LLM_URL", "http://16.171.3.197:8000/chat").strip()
        if not primary_api_url:
            raise CloudLLMError("Missing SERENITY_CLOUD_LLM_URL for cloud LLM API endpoint")

        fallback_urls_raw = os.getenv(
            "SERENITY_CLOUD_LLM_FALLBACK_URLS",
            "http://127.0.0.1:8000/chat",
        ).strip()
        fallback_urls = [item.strip() for item in fallback_urls_raw.split(",") if item.strip()]

        self.api_urls: List[str] = []
        for candidate in [primary_api_url, *fallback_urls]:
            if candidate and candidate not in self.api_urls:
                self.api_urls.append(candidate)

        self.active_api_url_index = 0
        self.api_url = self.api_urls[self.active_api_url_index]

        timeout_raw = os.getenv("SERENITY_CLOUD_LLM_TIMEOUT_SECONDS", "12").strip()
        try:
            self.timeout_seconds = max(1, int(timeout_raw))
        except ValueError as exc:
            raise CloudLLMError(f"Invalid SERENITY_CLOUD_LLM_TIMEOUT_SECONDS: {timeout_raw}") from exc

        connect_timeout_raw = os.getenv("SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS", "5").strip()
        try:
            self.connect_timeout_seconds = max(1, int(connect_timeout_raw))
        except ValueError as exc:
            raise CloudLLMError(
                f"Invalid SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS: {connect_timeout_raw}"
            ) from exc

        failure_threshold_raw = os.getenv("SERENITY_CLOUD_LLM_FAILURE_THRESHOLD", "2").strip()
        try:
            self.failure_threshold = max(1, int(failure_threshold_raw))
        except ValueError as exc:
            raise CloudLLMError(f"Invalid SERENITY_CLOUD_LLM_FAILURE_THRESHOLD: {failure_threshold_raw}") from exc

        cooldown_raw = os.getenv("SERENITY_CLOUD_LLM_COOLDOWN_SECONDS", "30").strip()
        try:
            self.cooldown_seconds = max(1, int(cooldown_raw))
        except ValueError as exc:
            raise CloudLLMError(f"Invalid SERENITY_CLOUD_LLM_COOLDOWN_SECONDS: {cooldown_raw}") from exc

        self.prefer_stream_accept = (
            os.getenv("SERENITY_CLOUD_LLM_PREFER_STREAM_ACCEPT", "true").strip().lower() == "true"
        )
        self.expect_sse_stream = (
            os.getenv("SERENITY_CLOUD_LLM_EXPECT_SSE", "true").strip().lower() == "true"
        )
        self.trust_polished_response = (
            os.getenv("SERENITY_TRUST_CLOUD_POLISHED_RESPONSE", "true").strip().lower() == "true"
        )

        self._consecutive_failures = 0
        self._unavailable_until_epoch = 0.0

        pool_connections = max(1, int(os.getenv("SERENITY_CLOUD_LLM_POOL_CONNECTIONS", "4")))
        pool_maxsize = max(1, int(os.getenv("SERENITY_CLOUD_LLM_POOL_MAXSIZE", "8")))
        pool_block = os.getenv("SERENITY_CLOUD_LLM_POOL_BLOCK", "false").strip().lower() == "true"

        # Reuse TCP connections across requests to reduce handshake overhead.
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=0,
            pool_block=pool_block,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Precompiled cleaners for low-overhead token normalization.
        self._camel_boundary_re = re.compile(r"([a-z])([A-Z])")
        self._whitespace_re = re.compile(r"\s+")
        self._glued_alpha_run_re = re.compile(r"[A-Za-z]{12,}")
        self._horizontal_space_re = re.compile(r"[ \t]+")
        self._space_before_punct_re = re.compile(r"\s+([,.;:!?])")
        self._junk_patterns = [
            re.compile(r"\*(?:Reflects\s*feelings|Reflectsfeelings)\*", flags=re.IGNORECASE),
            re.compile(r"\*(?:Asks\s*follow-?up\s*question|Asksfollow-?upquestion)\*", flags=re.IGNORECASE),
            re.compile(r"Reflecting feelings, then asking ONE follow-up question\.?", flags=re.IGNORECASE),
            re.compile(r"\b(?:User|Assistant):", flags=re.IGNORECASE),
        ]

    def _check_cooldown(self) -> None:
        now = time.time()
        if now >= self._unavailable_until_epoch:
            return

        remaining = max(1, int(self._unavailable_until_epoch - now))
        raise CloudLLMError(
            f"Cloud LLM API temporarily unavailable for {remaining}s after repeated failures"
        )

    def _mark_success(self) -> None:
        self._consecutive_failures = 0
        self._unavailable_until_epoch = 0.0

    def _mark_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._unavailable_until_epoch = time.time() + self.cooldown_seconds
            self._consecutive_failures = 0

    def _ordered_api_urls(self) -> List[str]:
        if not self.api_urls:
            return []

        index = max(0, min(self.active_api_url_index, len(self.api_urls) - 1))
        return self.api_urls[index:] + self.api_urls[:index]

    def _set_active_api_url(self, url: str) -> None:
        if not url:
            return

        try:
            index = self.api_urls.index(url)
        except ValueError:
            return

        self.active_api_url_index = index
        self.api_url = self.api_urls[index]

    @staticmethod
    def _truncate_after_first_asterisk(text: str, reached_first_asterisk: bool) -> tuple[str, bool]:
        if reached_first_asterisk:
            return "", True

        source = str(text or "")
        if not source:
            return "", False

        star_index = source.find("*")
        if star_index == -1:
            return source, False

        return source[:star_index], True

    def _strip_starred_segments(self, text: str, preserve_edges: bool = False) -> str:
        source = str(text or "")
        if not source:
            return ""

        star_index = source.find("*")
        cleaned = source if star_index == -1 else source[:star_index]
        cleaned = self._horizontal_space_re.sub(" ", cleaned)
        cleaned = self._space_before_punct_re.sub(r"\1", cleaned)
        return cleaned if preserve_edges else cleaned.strip()

    def _normalize_chunk(self, text: str, preserve_edges: bool = False) -> str:
        if self.trust_polished_response:
            normalized = self._strip_starred_segments(str(text or ""), preserve_edges=True).replace("\r", "")
            normalized = self._horizontal_space_re.sub(" ", normalized)
            return normalized if preserve_edges else normalized.strip()

        return self.hard_clean(text, preserve_edges=preserve_edges)

    def hard_clean(self, text: str, preserve_edges: bool = False) -> str:
        cleaned = str(text or "")
        if not cleaned:
            return ""

        for pattern in self._junk_patterns:
            cleaned = pattern.sub("", cleaned)

        if wordninja is not None:
            alpha_chars = sum(ch.isalpha() for ch in cleaned)
            space_chars = cleaned.count(" ")
            if alpha_chars >= 20 and space_chars <= max(1, alpha_chars // 24):
                def _split_glued_run(match: re.Match) -> str:
                    token = match.group(0)
                    parts = wordninja.split(token)
                    if len(parts) <= 1:
                        return token
                    return " ".join(parts)

                cleaned = self._glued_alpha_run_re.sub(_split_glued_run, cleaned)

        # Fix glued words and missing spacing around token boundaries.
        cleaned = self._camel_boundary_re.sub(r"\1 \2", cleaned)
        cleaned = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = self._whitespace_re.sub(" ", cleaned)
        return cleaned if preserve_edges else cleaned.strip()

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()

        if isinstance(payload, dict):
            for key in ("response", "reply", "answer", "text", "llm_response", "message", "delta", "token", "content"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value

            # Common OpenAI-style streaming shape.
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    delta = first.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            return content
                    message = first.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            return content

        return ""

    @classmethod
    def _extract_text_from_json_blob(cls, blob: str) -> str:
        text = str(blob or "").strip()
        if not text:
            return ""

        try:
            payload = json.loads(text)
        except ValueError:
            return ""

        return cls._extract_text(payload)

    @staticmethod
    def _decode_json_string_fragment(fragment: str) -> str:
        return (
            str(fragment or "")
            .replace(r"\n", "\n")
            .replace(r"\r", "\r")
            .replace(r"\t", "\t")
            .replace(r'\"', '"')
            .replace(r"\\", "\\")
        )

    @classmethod
    def _extract_partial_response_value(cls, blob: str) -> str:
        match = re.search(
            r'"response"\s*:\s*"(?P<body>(?:\\.|[^"\\])*)',
            str(blob or ""),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return cls._decode_json_string_fragment(match.group("body"))

    def _post(self, user_text: str, stream: bool) -> requests.Response:
        self._check_cooldown()

        headers = None
        if stream and self.prefer_stream_accept:
            headers = {
                "Accept": "text/event-stream, application/x-ndjson, application/json, text/plain",
            }

        ordered_urls = self._ordered_api_urls()
        last_exception: requests.RequestException | None = None

        for endpoint in ordered_urls:
            try:
                response = self.session.post(
                    endpoint,
                    json={"text": user_text},
                    timeout=(self.connect_timeout_seconds, self.timeout_seconds),
                    stream=stream,
                    headers=headers,
                )
                response.raise_for_status()
                self._mark_success()
                self._set_active_api_url(endpoint)
                return response
            except requests.RequestException as exc:
                last_exception = exc
                continue

        self._mark_failure()
        if last_exception is None:
            raise CloudLLMError("Cloud LLM API request failed: no available endpoint")

        attempted = ", ".join(ordered_urls)
        raise CloudLLMError(
            f"Cloud LLM API request failed across endpoints [{attempted}]: {last_exception}"
        ) from last_exception

    def ask_serenity(self, user_text: str) -> str:
        text = str(user_text or "").strip()
        if not text:
            raise CloudLLMError("Cloud LLM request text cannot be empty")

        response = self._post(text, stream=False)

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise CloudLLMError("Cloud LLM API returned non-JSON response") from exc

        answer = self._extract_text(payload)
        if answer:
            answer = self._normalize_chunk(answer)
            answer, _ = self._truncate_after_first_asterisk(answer, False)
            answer = answer.strip()
            if answer:
                return answer

        raise CloudLLMError("Cloud LLM API returned no usable response text")

    def stream_serenity(self, user_text: str) -> Iterator[str]:
        """Yield incremental response chunks when upstream supports streaming.

        If upstream responds as a standard non-streaming JSON payload, this method
        still yields a single chunk containing the response text.
        """
        text = str(user_text or "").strip()
        if not text:
            raise CloudLLMError("Cloud LLM request text cannot be empty")

        response = self._post(text, stream=True)
        emitted = False
        reached_first_asterisk = False

        with response:
            content_type = (response.headers.get("content-type") or "").lower()
            saw_sse_frames = False

            # SSE / NDJSON / line-stream responses.
            if (
                "text/event-stream" in content_type
                or "application/x-ndjson" in content_type
                or "application/jsonl" in content_type
                or "application/jsonlines" in content_type
                or self.expect_sse_stream
            ):
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    line = str(raw_line).strip()
                    if line.lower().startswith("data:"):
                        saw_sse_frames = True
                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            break

                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue

                        if bool(obj.get("done", False)):
                            break

                        token = obj.get("token")
                        if not isinstance(token, str) or not token.strip():
                            # Ignore non-token SSE frames to avoid replaying full
                            # response blobs as duplicate stream chunks.
                            continue

                        chunk = self._normalize_chunk(token, preserve_edges=True)
                        chunk, reached_first_asterisk = self._truncate_after_first_asterisk(
                            chunk,
                            reached_first_asterisk,
                        )
                        if chunk:
                            emitted = True
                            yield chunk
                        if reached_first_asterisk:
                            break
                        continue

                    # Fallback line parsing for non-SSE JSONL responses.
                    if saw_sse_frames:
                        continue

                    parsed = self._extract_text_from_json_blob(line)
                    if parsed:
                        chunk = self._normalize_chunk(parsed, preserve_edges=True)
                        chunk, reached_first_asterisk = self._truncate_after_first_asterisk(
                            chunk,
                            reached_first_asterisk,
                        )
                        if chunk:
                            emitted = True
                            yield chunk
                        if reached_first_asterisk:
                            break
                        continue

            # Standard JSON responses (non-streaming body).
            elif "application/json" in content_type:
                body_parts = []
                streamed_response_text = ""

                for raw_chunk in response.iter_content(chunk_size=64, decode_unicode=True):
                    piece = str(raw_chunk or "")
                    if not piece:
                        continue

                    body_parts.append(piece)
                    partial_blob = "".join(body_parts)
                    partial_response = self._extract_partial_response_value(partial_blob)

                    if partial_response and len(partial_response) > len(streamed_response_text):
                        delta = self._normalize_chunk(
                            partial_response[len(streamed_response_text):],
                            preserve_edges=True,
                        )
                        delta, reached_first_asterisk = self._truncate_after_first_asterisk(
                            delta,
                            reached_first_asterisk,
                        )
                        streamed_response_text = partial_response
                        if delta:
                            emitted = True
                            yield delta
                        if reached_first_asterisk:
                            break

                if reached_first_asterisk:
                    return

                body = "".join(body_parts)
                chunk = self._extract_text_from_json_blob(body)
                if not chunk:
                    chunk = self._extract_text(body)

                if chunk:
                    remaining = chunk
                    if streamed_response_text and chunk.startswith(streamed_response_text):
                        remaining = chunk[len(streamed_response_text):]

                    remaining = self._normalize_chunk(remaining, preserve_edges=True)
                    remaining, reached_first_asterisk = self._truncate_after_first_asterisk(
                        remaining,
                        reached_first_asterisk,
                    )
                    if remaining:
                        emitted = True
                        yield remaining
                    if reached_first_asterisk:
                        return

            # Plain-text chunk streams.
            elif "text/plain" in content_type:
                for raw_chunk in response.iter_content(chunk_size=128, decode_unicode=True):
                    chunk = self._normalize_chunk(str(raw_chunk or ""), preserve_edges=True)
                    chunk, reached_first_asterisk = self._truncate_after_first_asterisk(
                        chunk,
                        reached_first_asterisk,
                    )
                    if not chunk:
                        if reached_first_asterisk:
                            break
                        continue
                    emitted = True
                    yield chunk
                    if reached_first_asterisk:
                        break

            # Unknown payload type: parse as JSON blob first, then fall back to raw text.
            else:
                parts = []
                for raw_chunk in response.iter_content(chunk_size=128, decode_unicode=True):
                    chunk = str(raw_chunk or "")
                    if not chunk:
                        continue
                    parts.append(chunk)

                joined = "".join(parts).strip()
                if joined:
                    parsed = self._extract_text_from_json_blob(joined)
                    if parsed:
                        parsed = self._normalize_chunk(parsed)
                        parsed, reached_first_asterisk = self._truncate_after_first_asterisk(
                            parsed,
                            reached_first_asterisk,
                        )
                        emitted = True
                        if parsed:
                            yield parsed
                    else:
                        cleaned_joined = self._normalize_chunk(joined)
                        cleaned_joined, reached_first_asterisk = self._truncate_after_first_asterisk(
                            cleaned_joined,
                            reached_first_asterisk,
                        )
                        if cleaned_joined:
                            emitted = True
                            yield cleaned_joined

        if emitted:
            return

        # Last-resort fallback if upstream did not produce readable chunks.
        fallback = self.ask_serenity(text)
        if fallback:
            yield fallback

    def warmup(self, text: str = "Hello") -> float:
        """Prime upstream model/runtime and return warmup latency in seconds."""
        started = time.perf_counter()
        _ = self.ask_serenity(text)
        return time.perf_counter() - started

    def close(self) -> None:
        self.session.close()
