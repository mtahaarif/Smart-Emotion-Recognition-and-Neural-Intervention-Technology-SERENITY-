import contextlib
import json
import os
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

PROTOCOL_DELIMITER = "|||"

# Hard-kill regex — fires the microsecond ANY of these chars arrive in a token.
# The stream is severed before that char ever reaches the frontend or Edge TTS.
_BLOCK_RE    = re.compile(r"[<\[\]{}>~`]|\([a-zA-Z\s]+\)", re.IGNORECASE)
_MULTI_SP_RE = re.compile(r" {2,}")
_KILL_DEFAULT = "user:,assistant:,reflecting,follow-up"


class CloudLLMError(RuntimeError):
    pass


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    try:    return max(minimum, float(os.getenv(name, str(default)).strip()))
    except: return max(minimum, default)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:    return max(minimum, int(os.getenv(name, str(default)).strip()))
    except: return max(minimum, default)


def _parse_urls() -> List[str]:
    primary  = os.getenv("SERENITY_CLOUD_LLM_URL", "http://51.21.162.77:8000/chat").strip()
    fallback = [u.strip() for u in os.getenv("SERENITY_CLOUD_LLM_FALLBACK_URLS", "").split(",") if u.strip()]
    seen, out = set(), []
    for url in [primary] + fallback:
        if url and url not in seen: seen.add(url); out.append(url)
    return out or [primary]


class CloudLLMClient:
    __slots__ = (
        "api_urls", "_active_idx", "timeout", "connect_timeout",
        "client", "kill_phrases", "_tail_keep",
        "_failures", "_failure_threshold", "_cooldown_seconds", "_cooldown_until",
    )

    def __init__(self) -> None:
        self.api_urls        = _parse_urls()
        self._active_idx     = 0
        self.timeout         = _env_float("SERENITY_CLOUD_LLM_TIMEOUT_SECONDS", 60.0)
        self.connect_timeout = _env_float("SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS", 3.0, 0.1)

        pool = _env_int("SERENITY_CLOUD_LLM_POOL_MAXSIZE", 8)
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=pool,
                max_keepalive_connections=max(2, pool // 2),
                keepalive_expiry=45.0,
            ),
            timeout=httpx.Timeout(
                connect=self.connect_timeout, read=self.timeout,
                write=self.connect_timeout, pool=self.connect_timeout,
            ),
            http2=os.getenv("SERENITY_CLOUD_LLM_HTTP2", "false").lower() == "true",
        )

        raw_kill          = os.getenv("SERENITY_CLOUD_LLM_KILL_PHRASES", _KILL_DEFAULT)
        self.kill_phrases = tuple(p.strip().lower() for p in raw_kill.split(",") if p.strip())
        self._tail_keep   = max(0, max((len(p) for p in self.kill_phrases), default=0) - 1)

        self._failures          = 0
        self._failure_threshold = _env_int("SERENITY_CLOUD_LLM_FAILURE_THRESHOLD", 3)
        self._cooldown_seconds  = _env_float("SERENITY_CLOUD_LLM_COOLDOWN_SECONDS", 20.0)
        self._cooldown_until    = 0.0

    # --- Circuit-breaker ---
    def _cooling(self) -> bool: return time.time() < self._cooldown_until

    def _record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._cooldown_until = time.time() + self._cooldown_seconds

    def _record_success(self) -> None:
        self._failures       = 0
        self._cooldown_until = 0.0

    # --- Text helpers ---
    @staticmethod
    def _clean(text: str) -> str:
        if not text: return ""
        t = text.replace("\r", "").replace("\n", " ").replace("\t", " ")
        return _MULTI_SP_RE.sub(" ", t) if "  " in t else t

    def _cutoff_index(self, text: str) -> int:
        m = _BLOCK_RE.search(text)
        return m.start() if m else -1

    def _kill_trim(self, text: str) -> Tuple[str, bool]:
        lo  = text.lower()
        idx = min((lo.find(p) for p in self.kill_phrases if lo.find(p) >= 0), default=-1)
        return (text[:idx], True) if idx >= 0 else (text, False)

    def _augment(self, text: str, proto: bool) -> str:
        base = str(text or "").strip()
        if not proto or PROTOCOL_DELIMITER in base: return base
        return (
            f"{base}\n\nSYSTEM CONTROL DIRECTIVE: At the very end append exactly: "
            f'{PROTOCOL_DELIMITER}{{"advance_phase": true/false, "detected_distortion": "string"}}'
        )

    def _parse_protocol(self, tail: str) -> Dict[str, Any]:
        candidate = str(tail or "").strip()
        if not candidate: return {}
        try:
            d = json.loads(candidate)
            if isinstance(d, dict):
                return {"advance_phase": bool(d.get("advance_phase", False)),
                        "detected_distortion": str(d.get("detected_distortion") or "").strip()}
        except json.JSONDecodeError:
            pass
        if m := re.search(r"(\{[^}]*\})", candidate):
            with contextlib.suppress(json.JSONDecodeError):
                d = json.loads(m.group(1))
                if isinstance(d, dict):
                    return {"advance_phase": bool(d.get("advance_phase", False)),
                            "detected_distortion": str(d.get("detected_distortion") or "").strip()}
        return {}

    # --- Core SSE stream (single URL) ---
    async def _stream_once(
        self, url: str, text: str, proto: bool,
    ) -> AsyncIterator[Dict[str, Any]]:
        prompt     = self._augment(text, proto)
        tail       = ""
        proto_mode = False
        proto_buf  = ""
        delim_tail = ""

        async with self.client.stream(
            "POST", url, json={"text": prompt},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if not data or data == "[DONE]": break

                try:    payload = json.loads(data)
                except ValueError: continue

                if payload.get("done"): break

                # Extract token — handle multiple server payload shapes
                token = payload.get("token")
                if isinstance(token, dict): token = token.get("text", "")
                if not token: token = payload.get("response", "")
                if not token:
                    choices = payload.get("choices", [])
                    if choices: token = choices[0].get("delta", {}).get("content", "")
                if not isinstance(token, str) or not token: continue

                # Protocol-control extraction
                if proto:
                    if proto_mode: proto_buf += token; continue
                    combined = delim_tail + token
                    d_idx    = combined.find(PROTOCOL_DELIMITER)
                    if d_idx != -1:
                        token      = combined[:d_idx]
                        proto_mode = True
                        delim_tail = ""
                        proto_buf += combined[d_idx + len(PROTOCOL_DELIMITER):]
                    elif combined.endswith("||"): delim_tail, token = "||", combined[:-2]
                    elif combined.endswith("|"):  delim_tail, token = "|",  combined[:-1]
                    else:                          delim_tail, token = "",   combined
                    if not token: continue

                # ── HARD-KILL: fires instantly on < [ { > } ~ ` ──────────────
                ci = self._cutoff_index(token)
                if ci != -1:
                    if safe := self._clean(token[:ci]):
                        yield {"type": "assistant_delta", "delta": safe}
                    yield {"type": "cutoff"}
                    return          # connection severed — no further tokens emitted

                # Kill-phrase rolling tail check
                combined_lo = (tail + token).lower()
                if any(p in combined_lo for p in self.kill_phrases):
                    yield {"type": "cutoff"}
                    return

                # Emit token to frontend INSTANTLY (zero buffering)
                if cleaned := self._clean(token):
                    yield {"type": "assistant_delta", "delta": cleaned}

                if self._tail_keep:
                    tail = (tail + token.lower())[-self._tail_keep:]

        # Flush partial delimiter suffix after normal stream end
        if proto and not proto_mode and delim_tail:
            if cleaned := self._clean(delim_tail):
                yield {"type": "assistant_delta", "delta": cleaned}

        # Emit protocol-control block (arrives after all visible tokens)
        if proto and proto_buf.strip():
            if pc := self._parse_protocol(proto_buf):
                yield {"type": "protocol_control", "payload": pc}

    # --- Public streaming API ---
    async def stream_serenity_events(
        self,
        user_text: str,
        require_protocol_control: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        text = str(user_text or "").strip()
        if not text: return
        if self._cooling() and len(self.api_urls) == 1:
            raise CloudLLMError("LLM in cooldown")

        idxs = [self._active_idx] + [i for i in range(len(self.api_urls)) if i != self._active_idx]
        last_error: Optional[Exception] = None

        for idx in idxs:
            emitted = False
            try:
                async for ev in self._stream_once(self.api_urls[idx], text, require_protocol_control):
                    emitted = True
                    yield ev
                self._active_idx = idx
                self._record_success()
                return
            except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
                last_error = exc
                self._record_failure()
                if emitted or idx == idxs[-1] or self._cooling(): break

        raise CloudLLMError(f"Stream failed: {type(last_error).__name__}: {last_error}")

    # --- Non-streaming (single request) ---
    async def _ask_once(
        self, url: str, text: str, timeout: float, proto: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        resp = await self.client.post(
            url, json={"text": self._augment(text, proto)},
            timeout=httpx.Timeout(
                connect=self.connect_timeout, read=timeout,
                write=self.connect_timeout, pool=self.connect_timeout,
            ),
        )
        resp.raise_for_status()
        ct      = resp.headers.get("content-type", "")
        payload = resp.json() if "application/json" in ct else {}
        answer  = payload.get("response") or payload.get("text") or resp.text or ""
        if not isinstance(answer, str): return "", {}

        protocol: Dict[str, Any] = {}
        if proto and PROTOCOL_DELIMITER in answer:
            answer, tail = answer.split(PROTOCOL_DELIMITER, 1)
            protocol     = self._parse_protocol(tail)

        ci = self._cutoff_index(answer)
        if ci != -1: answer = answer[:ci]
        answer, _ = self._kill_trim(answer)
        answer    = self._clean(answer).strip()
        if answer:
            if m := list(re.finditer(r"[.!?]", answer)):
                answer = answer[:m[-1].end()].strip()
        return answer, protocol

    async def _ask_with_fallback(
        self, user_text: str, timeout: Optional[float], proto: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        text = str(user_text or "").strip()
        if not text: return "", {}
        eff  = max(1.0, float(timeout or self.timeout))
        idxs = [self._active_idx] + [i for i in range(len(self.api_urls)) if i != self._active_idx]
        last: Optional[Exception] = None
        for idx in idxs:
            try:
                ans, pc = await self._ask_once(self.api_urls[idx], text, eff, proto)
                self._active_idx = idx
                self._record_success()
                return ans, pc
            except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
                last = exc
                self._record_failure()
                if self._cooling(): break
        raise CloudLLMError(f"Request failed: {type(last).__name__}: {last}")

    async def ask_serenity(self, user_text: str, timeout: Optional[float] = None) -> str:
        ans, _ = await self._ask_with_fallback(user_text, timeout, False)
        return ans

    async def ask_serenity_with_protocol(
        self, user_text: str, timeout: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        return await self._ask_with_fallback(user_text, timeout, True)

    async def close(self) -> None:
        await self.client.aclose()