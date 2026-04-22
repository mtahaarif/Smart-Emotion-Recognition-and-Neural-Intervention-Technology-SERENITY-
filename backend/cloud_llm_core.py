import contextlib
import json
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List, Tuple
import httpx

class CloudLLMError(RuntimeError): pass

BLOCK_RE = re.compile(r"[\[\]{}<>~`Ãâð]|\([a-zA-Z\s]*\)|[\U00010000-\U0010FFFF]", re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"\s{2,}")
PROTOCOL_DELIMITER = "|||"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default)).strip()))
    except (TypeError, ValueError, AttributeError):
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default)).strip()))
    except (TypeError, ValueError, AttributeError):
        return max(minimum, default)


def _format_exception_detail(exc: Exception | None) -> str:
    if not exc:
        return "Unknown error"
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _parse_urls() -> List[str]:
    primary = os.getenv("SERENITY_CLOUD_LLM_URL", "http://51.21.162.77:8000/chat").strip()
    fallback_raw = os.getenv("SERENITY_CLOUD_LLM_FALLBACK_URLS", "")
    urls = [primary] + [u.strip() for u in str(fallback_raw).split(",") if u.strip()]
    deduped = []
    for url in urls:
        if url and url not in deduped:
            deduped.append(url)
    return deduped or ["http://51.21.162.77:8000/chat"]

class CloudLLMClient:
    """Edge-focused async client with fail-fast controls and SSE parsing."""
    __slots__ = (
        "api_urls",
        "active_url_idx",
        "timeout_seconds",
        "connect_timeout_seconds",
        "client",
        "kill_phrases",
        "tail_keep",
        "failure_count",
        "failure_threshold",
        "cooldown_seconds",
        "cooldown_until",
    )

    def __init__(self) -> None:
        self.api_urls = _parse_urls()
        self.active_url_idx = 0
        self.timeout_seconds = _env_float("SERENITY_CLOUD_LLM_TIMEOUT_SECONDS", 60.0, minimum=1.0)
        self.connect_timeout_seconds = _env_float("SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS", 3.0, minimum=0.1)

        pool_connections = _env_int("SERENITY_CLOUD_LLM_POOL_CONNECTIONS", 4, minimum=1)
        pool_maxsize = max(pool_connections, _env_int("SERENITY_CLOUD_LLM_POOL_MAXSIZE", 8, minimum=1))
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=pool_maxsize,
                max_keepalive_connections=pool_connections,
                keepalive_expiry=45.0,
            ),
            timeout=httpx.Timeout(
                connect=self.connect_timeout_seconds,
                read=self.timeout_seconds,
                write=self.connect_timeout_seconds,
                pool=self.connect_timeout_seconds,
            ),
            trust_env=os.getenv("SERENITY_CLOUD_LLM_TRUST_ENV", "false").lower() == "true",
            http2=os.getenv("SERENITY_CLOUD_LLM_HTTP2", "false").lower() == "true",
        )

        raw_kill_phrases = os.getenv("SERENITY_CLOUD_LLM_KILL_PHRASES", "user:,assistant:,reflecting,follow-up")
        self.kill_phrases = tuple(
            phrase.strip().lower()
            for phrase in str(raw_kill_phrases).split(",")
            if phrase.strip()
        )
        self.tail_keep = max(0, max((len(phrase) for phrase in self.kill_phrases), default=0) - 1)

        self.failure_count = 0
        self.failure_threshold = _env_int("SERENITY_CLOUD_LLM_FAILURE_THRESHOLD", 3, minimum=1)
        self.cooldown_seconds = _env_float("SERENITY_CLOUD_LLM_COOLDOWN_SECONDS", 20.0, minimum=1.0)
        self.cooldown_until = 0.0

    def _is_cooling_down(self) -> bool:
        return time.time() < self.cooldown_until

    def _record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.cooldown_until = time.time() + self.cooldown_seconds

    def _record_success(self) -> None:
        self.failure_count = 0
        self.cooldown_until = 0.0

    def _iter_url_indexes(self) -> List[int]:
        active = max(0, min(self.active_url_idx, len(self.api_urls) - 1))
        return [active] + [idx for idx in range(len(self.api_urls)) if idx != active]

    def _first_cutoff_index(self, text: str) -> int:
        asterisk = text.find("*")
        hash_index = text.find("#")
        if asterisk != -1 and hash_index != -1:
            return min(asterisk, hash_index)
        if asterisk != -1:
            return asterisk
        if hash_index != -1:
            return hash_index
        if match := BLOCK_RE.search(text):
            return match.start()
        return -1

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        if "\r" in text:
            text = text.replace("\r", "")
        if "\n" in text or "\t" in text:
            text = text.replace("\n", " ").replace("\t", " ")
        if "  " in text:
            text = MULTI_SPACE_RE.sub(" ", text)
        return text

    def _trim_kill_phrase(self, text: str) -> Tuple[str, bool]:
        lowered = text.lower()
        best_index = -1
        for phrase in self.kill_phrases:
            candidate = lowered.find(phrase)
            if candidate >= 0:
                if best_index == -1 or candidate < best_index:
                    best_index = candidate
        if best_index >= 0:
            return text[:best_index], True
        return text, False

    def _augment_prompt(self, text: str, require_protocol_control: bool = False) -> str:
        base_text = str(text or "").strip()
        if not require_protocol_control:
            return base_text

        if PROTOCOL_DELIMITER in base_text:
            return base_text

        return (
            f"{base_text}\n\n"
            "SYSTEM CONTROL DIRECTIVE: Keep your conversational response natural and clinically supportive. "
            "At the very end, append exactly one hidden control payload in this exact format and nothing else after it: "
            "|||{\"advance_phase\": true/false, \"detected_distortion\": \"string\"}."
        )

    def _normalize_protocol_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "advance_phase": bool(payload.get("advance_phase", False)),
            "detected_distortion": str(payload.get("detected_distortion") or "").strip(),
        }

    def _extract_protocol_payload(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        text = str(raw_text or "")
        if PROTOCOL_DELIMITER not in text:
            return text, {}

        visible_text, tail = text.split(PROTOCOL_DELIMITER, 1)
        control_candidate = str(tail or "").strip()
        if not control_candidate:
            return visible_text, {}

        payload_obj: Dict[str, Any] = {}
        try:
            parsed = json.loads(control_candidate)
            if isinstance(parsed, dict):
                payload_obj = parsed
        except json.JSONDecodeError:
            if match := re.search(r"(\{(?:.|\n)*\})", control_candidate):
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict):
                        payload_obj = parsed

        if not payload_obj:
            return visible_text, {}

        return visible_text, self._normalize_protocol_payload(payload_obj)

    async def _stream_once_events(
        self,
        url: str,
        text: str,
        require_protocol_control: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        tail = ""
        protocol_mode = False
        protocol_buffer = ""
        delimiter_tail = ""
        prompt_text = self._augment_prompt(text, require_protocol_control=require_protocol_control)

        async with self.client.stream("POST", url, json={"text": prompt_text}, headers={"Accept": "text/event-stream"}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except ValueError:
                    continue

                if payload.get("done"):
                    break

                token = payload.get("token")
                if not isinstance(token, str) or not token:
                    continue

                if require_protocol_control:
                    if protocol_mode:
                        protocol_buffer += token
                        continue

                    combined = f"{delimiter_tail}{token}"
                    delimiter_idx = combined.find(PROTOCOL_DELIMITER)
                    if delimiter_idx != -1:
                        token = combined[:delimiter_idx]
                        protocol_mode = True
                        delimiter_tail = ""
                        protocol_buffer += combined[delimiter_idx + len(PROTOCOL_DELIMITER):]
                    else:
                        if combined.endswith("||"):
                            delimiter_tail = "||"
                            token = combined[:-2]
                        elif combined.endswith("|"):
                            delimiter_tail = "|"
                            token = combined[:-1]
                        else:
                            delimiter_tail = ""
                            token = combined

                    if not token:
                        continue

                cutoff_index = self._first_cutoff_index(token)
                if cutoff_index != -1:
                    safe = self._clean(token[:cutoff_index])
                    if safe:
                        yield {"type": "assistant_delta", "delta": safe}
                    yield {"type": "cutoff"}
                    return

                combined = (tail + token).lower()
                if any(phrase in combined for phrase in self.kill_phrases):
                    yield {"type": "cutoff"}
                    return

                cleaned = self._clean(token)
                if cleaned:
                    yield {"type": "assistant_delta", "delta": cleaned}

                if self.tail_keep:
                    tail = (tail + token.lower())[-self.tail_keep:]

            if require_protocol_control and not protocol_mode and delimiter_tail:
                cleaned_tail = self._clean(delimiter_tail)
                if cleaned_tail:
                    yield {"type": "assistant_delta", "delta": cleaned_tail}

            if require_protocol_control and protocol_buffer.strip():
                _, protocol_payload = self._extract_protocol_payload(f"{PROTOCOL_DELIMITER}{protocol_buffer}")
                if protocol_payload:
                    yield {"type": "protocol_control", "payload": protocol_payload}

    async def stream_serenity_events(
        self,
        user_text: str,
        require_protocol_control: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not (text := str(user_text or "").strip()):
            return
        if self._is_cooling_down() and len(self.api_urls) == 1:
            raise CloudLLMError("Cloud LLM temporarily cooling down after failures")

        last_error: Exception | None = None
        url_indexes = self._iter_url_indexes()
        for idx in url_indexes:
            emitted_any = False
            try:
                async for event in self._stream_once_events(
                    self.api_urls[idx],
                    text,
                    require_protocol_control=require_protocol_control,
                ):
                    emitted_any = True
                    yield event
                self.active_url_idx = idx
                self._record_success()
                return
            except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
                last_error = exc
                self._record_failure()
                if emitted_any or idx == url_indexes[-1] or self._is_cooling_down():
                    break

        raise CloudLLMError(f"Stream failed: {_format_exception_detail(last_error)}")

    async def stream_serenity(self, user_text: str) -> AsyncIterator[str]:
        async for event in self.stream_serenity_events(user_text, require_protocol_control=False):
            event_type = str(event.get("type") or "")
            if event_type == "assistant_delta":
                delta = str(event.get("delta") or "")
                if delta:
                    yield delta
            elif event_type == "cutoff":
                yield "<CUTOFF>"

    async def _ask_once(
        self,
        url: str,
        text: str,
        timeout_seconds: float,
        require_protocol_control: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        prompt_text = self._augment_prompt(text, require_protocol_control=require_protocol_control)
        response = await self.client.post(
            url,
            json={"text": prompt_text},
            timeout=httpx.Timeout(
                connect=self.connect_timeout_seconds,
                read=timeout_seconds,
                write=self.connect_timeout_seconds,
                pool=self.connect_timeout_seconds,
            ),
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        payload = response.json() if "application/json" in content_type else {}
        answer = payload.get("response") or payload.get("text") or payload.get("message", {}).get("content", response.text)
        if not isinstance(answer, str):
            return "", {}

        protocol_payload: Dict[str, Any] = {}
        if require_protocol_control:
            answer, protocol_payload = self._extract_protocol_payload(answer)

        cutoff_hit = False
        cutoff_index = self._first_cutoff_index(answer)
        if cutoff_index != -1:
            answer = answer[:cutoff_index]
            cutoff_hit = True

        answer, phrase_cut = self._trim_kill_phrase(answer)
        cutoff_hit = cutoff_hit or phrase_cut
        answer = self._clean(answer).strip()

        if cutoff_hit and answer:
            punctuation_matches = list(re.finditer(r"[.!?]", answer))
            if punctuation_matches:
                answer = answer[: punctuation_matches[-1].end()].strip()
        return answer, protocol_payload

    async def _ask_with_options(
        self,
        user_text: str,
        timeout: float | None = None,
        require_protocol_control: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        if not (text := str(user_text or "").strip()):
            return "", {}
        if self._is_cooling_down() and len(self.api_urls) == 1:
            raise CloudLLMError("Cloud LLM temporarily cooling down after failures")

        effective_timeout = max(1.0, float(timeout or self.timeout_seconds))
        last_error: Exception | None = None

        url_indexes = self._iter_url_indexes()
        for idx in url_indexes:
            try:
                answer, protocol_payload = await self._ask_once(
                    self.api_urls[idx],
                    text,
                    effective_timeout,
                    require_protocol_control=require_protocol_control,
                )
                self.active_url_idx = idx
                self._record_success()
                return answer, protocol_payload
            except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
                last_error = exc
                self._record_failure()
                if self._is_cooling_down() and idx == url_indexes[0]:
                    break

        raise CloudLLMError(f"Request failed: {_format_exception_detail(last_error)}")

    async def ask_serenity(self, user_text: str, timeout: float | None = None) -> str:
        answer, _ = await self._ask_with_options(
            user_text,
            timeout=timeout,
            require_protocol_control=False,
        )
        return answer

    async def ask_serenity_with_protocol(
        self,
        user_text: str,
        timeout: float | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        return await self._ask_with_options(
            user_text,
            timeout=timeout,
            require_protocol_control=True,
        )

    async def close(self) -> None:
        await self.client.aclose()