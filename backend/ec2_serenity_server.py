import asyncio
import json
import logging
import os
import re
import threading
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


LOGGER = logging.getLogger("serenity-ec2")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class ChatRequest(BaseModel):
    text: str
    stream: Optional[bool] = None
    username: Optional[str] = None


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except Exception:
        return default


ROLE_PREFIX_RE = re.compile(r"\b(?:assistant|user|system)\s*:\s*", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")

# Prompt-leakage / chain-of-thought / role-play traces that should never reach UI/TTS.
LEAK_PATTERNS = [
    re.compile(r"Reflect(?:ing)?\s+feelings?.*$", flags=re.IGNORECASE),
    re.compile(r"then\s+ask(?:s|ing)?(?:\s+one)?\s+follow-?up\s+question.*$", flags=re.IGNORECASE),
    re.compile(r"\*[^*\n]{1,200}\*", flags=re.IGNORECASE),
    re.compile(r"\*[^*\n]{1,200}$", flags=re.IGNORECASE),
]


def _strip_starred_segments(text: str) -> str:
    source = str(text or "")
    if not source:
        return ""

    output_chars: List[str] = []
    in_starred_segment = False
    index = 0
    length = len(source)

    while index < length:
        char = source[index]
        if char == "*":
            while index < length and source[index] == "*":
                index += 1
            in_starred_segment = not in_starred_segment
            continue

        if not in_starred_segment:
            output_chars.append(char)
        index += 1

    cleaned = "".join(output_chars)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _sentence_key(sentence: str) -> str:
    return re.sub(r"\W+", "", sentence.lower())


def _collapse_repeated_ngrams(text: str) -> str:
    cleaned = str(text or "")
    for n in (12, 10, 8, 6, 5, 4):
        previous = None
        while cleaned != previous:
            previous = cleaned
            cleaned = re.sub(
                rf"(?i)\b((?:[\w']+\W+){{{n - 1}}}[\w']+)(?:\W+\1\b)+",
                r"\1",
                cleaned,
            )
    return cleaned


def _dedupe_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text

    unique: List[str] = []
    seen = set()

    for sentence in sentences:
        key = _sentence_key(sentence)
        if key and key in seen:
            continue

        if unique:
            prev = unique[-1]
            similarity = SequenceMatcher(None, prev.lower(), sentence.lower()).ratio()
            prev_key = _sentence_key(prev)
            if similarity > 0.92:
                continue
            if key and prev_key and (key in prev_key or prev_key in key):
                continue

        if key:
            seen.add(key)
        unique.append(sentence)

    return " ".join(unique).strip()


def clean_assistant_text(text: str, max_sentences: int = 4, max_words: int = 90) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return "I am here with you. Let's take one small step together."

    cleaned = SPECIAL_TOKEN_RE.sub("", cleaned)
    cleaned = _strip_starred_segments(cleaned)

    # Keep only the first assistant segment if the model leaks a transcript.
    if ROLE_PREFIX_RE.search(cleaned):
        pieces = [part.strip() for part in ROLE_PREFIX_RE.split(cleaned) if part.strip()]
        if pieces:
            cleaned = pieces[0]

    for pattern in LEAK_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    cleaned = _collapse_repeated_ngrams(cleaned)
    cleaned = _dedupe_sentences(cleaned)

    # Remove accidental immediate word duplication.
    cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    sentences = _split_sentences(cleaned)
    if sentences:
        cleaned = " ".join(sentences[:max_sentences]).strip()

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).strip().rstrip(",;:") + "."

    if not cleaned:
        cleaned = "I am here with you. Let's take one small step together."

    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."

    return cleaned


def _stream_word_chunks(text: str, words_per_chunk: int) -> Iterable[str]:
    words = text.split()
    if not words:
        return

    step = max(1, words_per_chunk)
    total = len(words)
    for start in range(0, total, step):
        chunk_words = words[start : start + step]
        chunk = " ".join(chunk_words)
        if start + step < total:
            chunk += " "
        yield chunk


class SerenityRuntime:
    def __init__(self) -> None:
        self.model_id = os.getenv("SERENITY_EC2_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0").strip()
        self.adapter_path = os.getenv("SERENITY_ADAPTER_PATH", "").strip()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_input_tokens = _env_int("SERENITY_MAX_INPUT_TOKENS", 2048)
        self.max_new_tokens = _env_int("SERENITY_MAX_NEW_TOKENS", 120)
        self.temperature = _env_float("SERENITY_TEMPERATURE", 0.6)
        self.top_p = _env_float("SERENITY_TOP_P", 0.9)
        self.repetition_penalty = _env_float("SERENITY_REPETITION_PENALTY", 1.14)
        self.no_repeat_ngram_size = _env_int("SERENITY_NO_REPEAT_NGRAM", 4)
        self.do_sample = _env_bool("SERENITY_DO_SAMPLE", True)
        self.stream_words_per_chunk = _env_int("SERENITY_STREAM_WORDS_PER_CHUNK", 2)
        self.stream_delay_seconds = _env_float("SERENITY_STREAM_DELAY_SECONDS", 0.0)

        self.tokenizer = None
        self.model = None
        self._manual_device = True
        self._load_model()

    def _load_model(self) -> None:
        LOGGER.info("Loading tokenizer: %s", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_in_4bit = _env_bool("SERENITY_LOAD_IN_4BIT", self.device == "cuda")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_kwargs: Dict[str, object] = {"torch_dtype": torch_dtype}

        if self.device == "cuda" and load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
            self._manual_device = False
        else:
            self._manual_device = True

        LOGGER.info("Loading model on %s: %s", self.device, self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        if self.adapter_path and os.path.isdir(self.adapter_path):
            if PeftModel is None:
                LOGGER.warning(
                    "SERENITY_ADAPTER_PATH is set, but peft is not installed. Adapter will be skipped."
                )
            else:
                LOGGER.info("Applying LoRA/PEFT adapter: %s", self.adapter_path)
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        if self._manual_device:
            self.model.to(self.device)

        self.model.eval()
        LOGGER.info("Model loaded successfully.")

    def _build_messages(self, user_text: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are Serenity, a supportive mental-health companion. "
            "Give one concise, empathetic response in plain text. "
            "Do not include role labels, stage directions, bullet lists, markdown, or internal notes. "
            "Do not repeat sentences. Ask at most one follow-up question."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    def generate(self, user_text: str) -> str:
        user_text = str(user_text or "").strip()
        if not user_text:
            raise ValueError("text cannot be empty")

        messages = self._build_messages(user_text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"System: {messages[0]['content']}\nUser: {user_text}\nAssistant:"

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        if self._manual_device:
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        generation_kwargs: Dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_tokens:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return clean_assistant_text(raw_text)


app = FastAPI(title="Serenity EC2 LLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: Optional[SerenityRuntime] = None


def get_runtime() -> SerenityRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = SerenityRuntime()
    return _RUNTIME


def _sse_event(payload: Dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> Dict[str, object]:
    loaded = _RUNTIME is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_id": getattr(_RUNTIME, "model_id", None),
    }


@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    user_text = str(payload.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    accept_header = (request.headers.get("accept") or "").lower()
    stream_requested = bool(payload.stream) or "text/event-stream" in accept_header

    try:
        runtime = get_runtime()
        final_text = await run_in_threadpool(runtime.generate, user_text)
    except Exception as exc:
        LOGGER.exception("Generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {exc}") from exc

    if not stream_requested:
        return JSONResponse({"response": final_text})

    async def event_stream():
        for token_chunk in _stream_word_chunks(final_text, runtime.stream_words_per_chunk):
            yield _sse_event({"token": token_chunk, "done": False})
            if runtime.stream_delay_seconds > 0:
                await asyncio.sleep(runtime.stream_delay_seconds)

        yield _sse_event({"done": True, "response": final_text})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ec2_serenity_server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
        reload=False,
    )