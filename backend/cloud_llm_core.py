import json
import os
import re
import time
from typing import Dict, Iterator, Optional

import requests


class CloudLLMError(RuntimeError):
    pass


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _first_special_cutoff(text: str) -> int:
    star_idx = text.find("*")
    hash_idx = text.find("#")
    if star_idx == -1:
        return hash_idx
    if hash_idx == -1:
        return star_idx
    return min(star_idx, hash_idx)


class CloudLLMClient:
    """Edge-optimized HTTP client with low-overhead streaming cutoffs."""

    __slots__ = (
        "api_url",
        "timeout",
        "connect_timeout",
        "session",
        "_space_re",
        "_camel_re",
        "kill_phrases",
        "_tail_keep",
    )

    def __init__(self) -> None:
        self.api_url = os.getenv("SERENITY_CLOUD_LLM_URL", "http://16.171.3.197:8000/chat").strip()
        self.timeout = _env_int("SERENITY_CLOUD_LLM_TIMEOUT_SECONDS", 60)
        self.connect_timeout = _env_int("SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS", 5)

        pool_connections = _env_int("SERENITY_CLOUD_LLM_POOL_CONNECTIONS", 4)
        pool_maxsize = max(pool_connections, _env_int("SERENITY_CLOUD_LLM_POOL_MAXSIZE", 8))

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=0,
            pool_block=False,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._space_re = re.compile(r"\s+")
        self._camel_re = re.compile(r"([a-z])([A-Z])")

        raw_phrases = os.getenv(
            "SERENITY_CLOUD_LLM_KILL_PHRASES",
            "user:,assistant:,reflecting,follow-up",
        )
        self.kill_phrases = tuple(
            phrase.strip().lower()
            for phrase in str(raw_phrases).split(",")
            if phrase.strip()
        )
        max_phrase_len = max((len(phrase) for phrase in self.kill_phrases), default=0)
        self._tail_keep = max(0, max_phrase_len - 1)

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        if "\r" in text:
            text = text.replace("\r", "")
        if self._camel_re.search(text):
            text = self._camel_re.sub(r"\1 \2", text)
        if "\n" in text or "\t" in text or "  " in text:
            text = self._space_re.sub(" ", text)
        return text

    def _find_kill_start(self, token_lower: str, tail_lower: str) -> Optional[int]:
        if not self.kill_phrases:
            return None
        joined = tail_lower + token_lower
        earliest: Optional[int] = None
        for phrase in self.kill_phrases:
            idx = joined.find(phrase)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
        if earliest is None:
            return None
        return earliest - len(tail_lower)

    def stream_serenity(self, user_text: str) -> Iterator[str]:
        text = str(user_text or "").strip()
        if not text:
            return

        tail_lower = ""

        try:
            with self.session.post(
                self.api_url,
                json={"text": text},
                timeout=(self.connect_timeout, self.timeout),
                stream=True,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue

                    if obj.get("done"):
                        break

                    token = obj.get("token")
                    if not isinstance(token, str) or not token:
                        continue

                    cutoff = _first_special_cutoff(token)
                    if cutoff != -1:
                        safe = self._clean_text(token[:cutoff])
                        if safe:
                            yield safe
                        return

                    token_lower = token.lower()
                    kill_start = self._find_kill_start(token_lower, tail_lower)
                    if kill_start is not None:
                        if kill_start > 0:
                            safe = self._clean_text(token[:kill_start])
                            if safe:
                                yield safe
                        return

                    cleaned = self._clean_text(token)
                    if cleaned:
                        yield cleaned

                    if self._tail_keep:
                        tail_lower = (tail_lower + token_lower)[-self._tail_keep :]

        except requests.RequestException as exc:
            raise CloudLLMError(f"Stream failed: {exc}") from exc

    def _extract_answer_text(self, response: requests.Response) -> str:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "application/json" not in content_type:
            return str(response.text or "")

        try:
            payload = response.json()
        except ValueError:
            return str(response.text or "")

        if not isinstance(payload, dict):
            return ""

        answer = payload.get("response") or payload.get("text")
        if isinstance(answer, str):
            return answer

        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        return ""

    def ask_serenity(self, user_text: str) -> str:
        text = str(user_text or "").strip()
        if not text:
            return ""

        try:
            response = self.session.post(
                self.api_url,
                json={"text": text},
                timeout=(self.connect_timeout, self.timeout),
            )
            response.raise_for_status()

            answer = self._extract_answer_text(response)
            cutoff = _first_special_cutoff(answer)
            if cutoff != -1:
                answer = answer[:cutoff]

            return self._clean_text(answer).strip()

        except requests.RequestException as exc:
            raise CloudLLMError(f"Request failed: {exc}") from exc

    def warmup(self, text: str = "Hello") -> float:
        started = time.perf_counter()
        try:
            self.ask_serenity(text)
        except Exception:
            pass
        return time.perf_counter() - started

    def close(self) -> None:
        self.session.close()