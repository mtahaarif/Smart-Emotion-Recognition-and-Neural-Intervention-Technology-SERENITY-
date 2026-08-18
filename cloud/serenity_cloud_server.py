import json
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_title: str
    source_url: str
    source_type: str
    word_count: int
    chunk_index: int

    def to_dict(self) -> Dict:
        return asdict(self)


class KnowledgeScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def scrape_wikipedia_article(self, url: str) -> Optional[Dict]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            title_el = soup.find("h1", class_="firstHeading")
            title = title_el.get_text(strip=True) if title_el else "Unknown"

            content_div = soup.find("div", class_="mw-parser-output")
            if not content_div:
                return None

            paragraphs: List[str] = []
            for p in content_div.find_all("p"):
                text = p.get_text().strip()
                if len(text) > 40 and not text.startswith("["):
                    text = re.sub(r"\[\d+\]", "", text)
                    paragraphs.append(text)

            content = "\n\n".join(paragraphs).strip()
            if not content:
                return None

            return {
                "title": title,
                "url": url,
                "content": content,
                "source": "Wikipedia",
                "word_count": len(content.split()),
            }
        except Exception:
            return None

    def scrape_urls(self, urls: List[str], delay: float = 0.8) -> List[Dict]:
        articles: List[Dict] = []
        for i, url in enumerate(urls):
            article = self.scrape_wikipedia_article(url)
            if article and article["word_count"] > 50:
                articles.append(article)
            if i < len(urls) - 1:
                time.sleep(delay)
        return articles


class TextProcessor:
    def __init__(self, chunk_size: int = 180, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:\-\'\"()]", "", text)
        return text.strip()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def create_chunks(self, text: str, min_words: int = 40) -> List[str]:
        sentences = self.split_into_sentences(text)

        chunks: List[str] = []
        current: List[str] = []
        current_wc = 0

        for sentence in sentences:
            sentence_wc = len(sentence.split())
            if current and current_wc + sentence_wc > self.chunk_size:
                chunk_text = " ".join(current).strip()
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)

                overlap_sentences: List[str] = []
                overlap_wc = 0
                for sent in reversed(current):
                    sent_wc = len(sent.split())
                    if overlap_wc + sent_wc <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_wc += sent_wc
                    else:
                        break

                current = overlap_sentences
                current_wc = overlap_wc

            current.append(sentence)
            current_wc += sentence_wc

        if current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text.split()) >= min_words:
                chunks.append(chunk_text)

        return chunks

    def process_articles(self, articles: List[Dict]) -> List[Chunk]:
        output: List[Chunk] = []
        for article_idx, article in enumerate(articles):
            cleaned = self.clean_text(article["content"])
            chunk_texts = self.create_chunks(cleaned)
            for chunk_idx, text in enumerate(chunk_texts):
                output.append(
                    Chunk(
                        chunk_id=f"article_{article_idx}_chunk_{chunk_idx}",
                        text=text,
                        source_title=article["title"],
                        source_url=article["url"],
                        source_type=article["source"],
                        word_count=len(text.split()),
                        chunk_index=chunk_idx,
                    )
                )
        return output


class EmbeddingEngine:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.index = None
        self.chunks: List[Chunk] = []

    def build_index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            chunks = [
                Chunk(
                    chunk_id="fallback_0",
                    text="Empathy, active listening, and compassionate support are helpful.",
                    source_title="Fallback",
                    source_url="",
                    source_type="internal",
                    word_count=9,
                    chunk_index=0,
                )
            ]

        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.index = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            return []

        q = self.vectorizer.transform([query])
        scores = (self.index @ q.T).toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def get_context(self, query: str, top_k: int = 3, max_words: int = 300) -> Tuple[str, List[Dict]]:
        results = self.search(query, top_k=top_k)
        ctx_parts: List[str] = []
        metadata: List[Dict] = []
        total_words = 0

        for chunk, score in results:
            if total_words + chunk.word_count > max_words:
                break
            ctx_parts.append(chunk.text)
            total_words += chunk.word_count
            metadata.append({"source": chunk.source_title, "score": score})

        return "\n\n".join(ctx_parts), metadata

    def save_assets(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False)

        if self.index is not None:
            sparse.save_npz(cache_dir / "matrix.npz", self.index)

    @classmethod
    def load_assets(cls, cache_dir: Path) -> Optional["EmbeddingEngine"]:
        vec_file = cache_dir / "vectorizer.pkl"
        chunks_file = cache_dir / "chunks.json"
        mat_file = cache_dir / "matrix.npz"

        if not (vec_file.exists() and chunks_file.exists() and mat_file.exists()):
            return None

        engine = cls()
        with open(vec_file, "rb") as f:
            engine.vectorizer = pickle.load(f)

        with open(chunks_file, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        engine.chunks = [Chunk(**item) for item in raw_chunks]
        engine.index = sparse.load_npz(mat_file)
        return engine


class SerenityGenerator:
    STOP_MARKERS = (
        "\nUser:",
        "\nAssistant:",
        "User:",
        "Assistant:",
        "Reflects:",
    )

    def __init__(self, embedding_engine: EmbeddingEngine, api_url: str):
        self.embedding_engine = embedding_engine
        self.api_url = api_url
        self.history: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.history_lock = Lock()
        self.sentence_split_re = re.compile(r"(?<=[.!?])\s+")

    def get_system_prompt(self) -> str:
        return (
            "You are Serenity, a warm empathetic therapist.\n"
            "CRITICAL RULES:\n"
            "- Keep responses SHORT (2-4 sentences, max 60 words)\n"
            "- Be conversational and warm\n"
            "- Reflect feelings, then ask ONE follow-up question\n"
            "- No lists, no bullet points, no markdown markers\n"
            "- Never output role labels like User or Assistant\n"
        )

    def _history_slice(self, session_id: str) -> List[Dict[str, str]]:
        with self.history_lock:
            return list(self.history[session_id][-8:])

    def _append_history(self, session_id: str, user_text: str, assistant_text: str) -> None:
        with self.history_lock:
            self.history[session_id].append({"role": "User", "content": user_text})
            self.history[session_id].append({"role": "Assistant", "content": assistant_text})
            self.history[session_id] = self.history[session_id][-12:]

    def reset_conversation(self, session_id: str = "default") -> None:
        with self.history_lock:
            self.history[session_id] = []

    def _build_prompt(
        self,
        user_message: str,
        session_id: str,
        use_rag: bool,
        top_k_chunks: int,
    ) -> Tuple[str, List[Dict]]:
        context = ""
        rag_metadata: List[Dict] = []

        if use_rag and self.embedding_engine is not None:
            context, rag_metadata = self.embedding_engine.get_context(
                user_message,
                top_k=top_k_chunks,
                max_words=250,
            )

        prompt = self.get_system_prompt()
        if context:
            prompt += f"\nRelevant context:\n{context}"

        for turn in self._history_slice(session_id):
            role = turn.get("role", "").strip()
            content = turn.get("content", "").strip()
            if role and content:
                prompt += f"\n{role}: {content}"

        prompt += f"\nUser: {user_message}\nAssistant:"
        return prompt, rag_metadata

    @staticmethod
    def _truncate_at_stop_marker(text: str) -> Tuple[str, bool]:
        stop_positions = [text.find(marker) for marker in SerenityGenerator.STOP_MARKERS if text.find(marker) != -1]
        if not stop_positions:
            return text, False
        cut = min(stop_positions)
        return text[:cut], True

    @staticmethod
    def _flush_complete_words(buffer: str) -> Tuple[List[str], str]:
        matches = list(re.finditer(r"\S+\s+", buffer))
        if not matches:
            return [], buffer

        flush_upto = matches[-1].end()
        emit_text = buffer[:flush_upto]
        remain = buffer[flush_upto:]
        chunks = re.findall(r"\S+\s*", emit_text)
        return chunks, remain

    def _iter_llama_tokens(self, response: requests.Response) -> Iterator[str]:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = str(raw_line).strip()
            if not line:
                continue

            if line == "[DONE]":
                break

            if line.lower().startswith("data:"):
                line = line[5:].strip()

            if not line:
                continue

            if line == "[DONE]":
                break

            try:
                payload = json.loads(line)
            except Exception:
                continue

            token = payload.get("content") or payload.get("token") or payload.get("text")
            if isinstance(token, str) and token:
                yield token.replace("\r", "")

            if bool(payload.get("done", False)) or bool(payload.get("stop", False)):
                break

    def _drain_complete_sentences(self, buffer: str) -> Tuple[List[str], str]:
        parts = self.sentence_split_re.split(buffer)
        if len(parts) <= 1:
            return [], buffer
        completed = [p.strip() for p in parts[:-1] if p.strip()]
        return completed, parts[-1]

    def stream_response(
        self,
        user_message: str,
        session_id: str = "default",
        use_rag: bool = True,
        top_k_chunks: int = 2,
        max_new_tokens: int = 96,
        temperature: float = 0.55,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.12,
    ) -> Iterator[Dict]:
        prompt, rag_metadata = self._build_prompt(
            user_message=user_message,
            session_id=session_id,
            use_rag=use_rag,
            top_k_chunks=top_k_chunks,
        )

        payload = {
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": list(self.STOP_MARKERS),
            "stream": True,
        }

        pending = ""
        canonical_parts: List[str] = []
        sentence_buffer = ""
        sentence_seq = 0
        force_stop = False

        with requests.post(self.api_url, json=payload, stream=True, timeout=(10, 180)) as resp:
            resp.raise_for_status()

            for token in self._iter_llama_tokens(resp):
                pending += token

                pending, hit_stop = self._truncate_at_stop_marker(pending)
                if hit_stop:
                    force_stop = True

                word_chunks, pending = self._flush_complete_words(pending)

                for chunk in word_chunks:
                    canonical_parts.append(chunk)
                    current_text = "".join(canonical_parts)
                    yield {
                        "type": "token",
                        "token": chunk,
                        "text": current_text,
                        "done": False,
                    }

                    sentence_buffer += chunk
                    completed_sentences, sentence_buffer = self._drain_complete_sentences(sentence_buffer)
                    for sentence_text in completed_sentences:
                        sentence_seq += 1
                        yield {
                            "type": "sentence",
                            "sequence": sentence_seq,
                            "text": sentence_text,
                            "done": False,
                        }

                if force_stop:
                    break

        tail = pending.strip()
        if tail:
            canonical_parts.append(tail)
            current_text = "".join(canonical_parts)
            yield {
                "type": "token",
                "token": tail,
                "text": current_text,
                "done": False,
            }
            sentence_buffer += tail

        final_text = "".join(canonical_parts).strip()
        if not final_text:
            final_text = "I am here with you. What feels most difficult right now?"
            canonical_parts = [final_text]
            yield {
                "type": "token",
                "token": final_text,
                "text": final_text,
                "done": False,
            }

        if final_text[-1] not in ".!?":
            canonical_parts.append(".")
            final_text = "".join(canonical_parts).strip()
            yield {
                "type": "token",
                "token": ".",
                "text": final_text,
                "done": False,
            }
            sentence_buffer += "."

        if sentence_buffer.strip():
            sentence_seq += 1
            yield {
                "type": "sentence",
                "sequence": sentence_seq,
                "text": sentence_buffer.strip(),
                "done": False,
            }

        self._append_history(session_id, user_message, final_text)

        yield {
            "type": "done",
            "done": True,
            "response": final_text,
            "metadata": rag_metadata,
        }


def default_urls() -> List[str]:
    return [
        "https://en.wikipedia.org/wiki/Empathy",
        "https://en.wikipedia.org/wiki/Emotional_intelligence",
        "https://en.wikipedia.org/wiki/Active_listening",
        "https://en.wikipedia.org/wiki/Cognitive_behavioral_therapy",
        "https://en.wikipedia.org/wiki/Mindfulness",
        "https://en.wikipedia.org/wiki/Compassion",
        "https://en.wikipedia.org/wiki/Coping",
    ]


def load_urls_from_env() -> List[str]:
    raw = os.getenv("SERENITY_WIKI_URLS_JSON", "").strip()
    if not raw:
        return default_urls()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass

    return default_urls()


def build_or_load_engine(cache_dir: Path) -> EmbeddingEngine:
    loaded = EmbeddingEngine.load_assets(cache_dir)
    if loaded is not None:
        return loaded

    scraper = KnowledgeScraper()
    processor = TextProcessor(chunk_size=180, overlap=30)

    urls = load_urls_from_env()
    articles = scraper.scrape_urls(urls, delay=0.8)
    chunks = processor.process_articles(articles)

    engine = EmbeddingEngine(max_features=5000)
    engine.build_index(chunks)
    engine.save_assets(cache_dir)
    return engine


app = FastAPI(title="Serenity EC2 LLM", version="2.0.0")
serenity: Optional[SerenityGenerator] = None


class Query(BaseModel):
    text: str
    session_id: str = "default"
    use_rag: bool = True


@app.on_event("startup")
def startup() -> None:
    global serenity

    cache_dir = Path(os.getenv("SERENITY_KB_CACHE_DIR", "./kb_cache"))
    engine = build_or_load_engine(cache_dir)

    llama_url = os.getenv("SERENITY_LOCAL_LLM_URL", "http://127.0.0.1:8080/completion")
    serenity = SerenityGenerator(embedding_engine=engine, api_url=llama_url)


@app.get("/health")
def health() -> Dict:
    ready = serenity is not None and serenity.embedding_engine is not None
    return {"status": "ok", "ready": ready}


@app.post("/chat")
def chat_api(query: Query):
    if serenity is None:
        raise HTTPException(status_code=503, detail="Serenity is not initialized")

    user_text = (query.text or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    def stream():
        try:
            for event in serenity.stream_response(
                user_message=user_text,
                session_id=query.session_id,
                use_rag=query.use_rag,
            ):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        except requests.RequestException as exc:
            err = {
                "type": "done",
                "done": True,
                "response": "",
                "error": f"upstream_gguf_server_error: {str(exc)}",
            }
            yield "data: " + json.dumps(err, ensure_ascii=False) + "\n\n"
        except Exception as exc:
            err = {
                "type": "done",
                "done": True,
                "response": "",
                "error": f"internal_error: {str(exc)}",
            }
            yield "data: " + json.dumps(err, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/reset")
def reset_session(query: Query) -> Dict:
    if serenity is None:
        raise HTTPException(status_code=503, detail="Serenity is not initialized")
    serenity.reset_conversation(query.session_id)
    return {"ok": True, "session_id": query.session_id}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)