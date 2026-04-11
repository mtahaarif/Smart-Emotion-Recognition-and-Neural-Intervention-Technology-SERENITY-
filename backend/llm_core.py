import logging
import os
import pickle
import re
import hashlib
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import faiss
import requests
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_FALLBACK_MODEL_IDS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "serenity_faiss.index")
DEFAULT_CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "serenity_chunks.pkl")
SENTENCE_BOUNDARY_REGEX = re.compile(r"(?<=[.!?])\s+")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


DEFAULT_GENERATION_MAX_NEW_TOKENS = _env_int("SERENITY_LLM_MAX_NEW_TOKENS", 80)
DEFAULT_GENERATION_TEMPERATURE = _env_float("SERENITY_LLM_TEMPERATURE", 0.70)
DEFAULT_GENERATION_TOP_P = _env_float("SERENITY_LLM_TOP_P", 0.90)


def _parse_fallback_model_ids() -> List[str]:
    raw = os.getenv("SERENITY_MODEL_FALLBACK_IDS", "").strip()
    if not raw:
        return DEFAULT_FALLBACK_MODEL_IDS.copy()
    return [item.strip() for item in raw.split(",") if item.strip()]

# =============================================================================
# CLASSES (Your Original Logic)
# =============================================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_title: str
    source_url: str
    source_type: str
    word_count: int
    chunk_index: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

class KnowledgeScraper:
    def __init__(self, output_dir: str = "knowledge_base"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def scrape_wikipedia_article(self, url: str) -> Optional[Dict[str, object]]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('h1', class_='firstHeading')
            title_text = title.get_text() if title else "Unknown"
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                return None

            paragraphs = []
            for p in content_div.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 40 and not text.startswith('['):
                    text = re.sub(r'\[\d+\]', '', text)
                    paragraphs.append(text)

            content = '\n\n'.join(paragraphs)
            return {
                'title': title_text,
                'url': url,
                'content': content,
                'source': 'Wikipedia',
                'word_count': len(content.split()),
            }
        except Exception as exc:
            LOGGER.warning("Error scraping %s: %s", url, exc)
            return None

    def scrape_urls(self, urls: List[str], delay: float = 1.0) -> List[Dict[str, object]]:
        articles = []
        for idx, url in enumerate(urls):
            LOGGER.info("Scraping %s/%s: %s", idx + 1, len(urls), url)
            article = self.scrape_wikipedia_article(url)
            if article and int(article.get("word_count", 0)) > 50:
                articles.append(article)
            if idx < len(urls) - 1:
                time.sleep(delay)
        return articles

class TextProcessor:
    def __init__(self, chunk_size: int = 180, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)
        return text.strip()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str, min_words: int = 40) -> List[str]:
        sentences = self.split_into_sentences(text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)

                overlap_sentences: List[str] = []
                overlap_words = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_words + sent_words <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_words += sent_words
                    else:
                        break

                current_chunk = overlap_sentences
                current_word_count = overlap_words

            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text.split()) >= min_words:
                chunks.append(chunk_text)

        return chunks

    def process_article(self, article: Dict[str, object], article_idx: int) -> List[Chunk]:
        cleaned_content = self.clean_text(str(article.get('content', '')))
        chunk_texts = self.create_chunks(cleaned_content)
        chunks: List[Chunk] = []

        for chunk_idx, chunk_text in enumerate(chunk_texts):
            chunks.append(
                Chunk(
                    chunk_id=f"article_{article_idx}_chunk_{chunk_idx}",
                    text=chunk_text,
                    source_title=str(article.get('title', 'Unknown')),
                    source_url=str(article.get('url', '')),
                    source_type=str(article.get('source', 'Wikipedia')),
                    word_count=len(chunk_text.split()),
                    chunk_index=chunk_idx,
                )
            )

        return chunks

    def process_articles(self, articles: List[Dict[str, object]]) -> List[Chunk]:
        all_chunks = []
        for idx, article in enumerate(articles):
            chunks = self.process_article(article, idx)
            all_chunks.extend(chunks)
        return all_chunks

class EmbeddingEngine:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        LOGGER.info("Loading embedding model: %s", model_name)
        embedding_device = os.getenv("SERENITY_EMBEDDING_DEVICE", "cpu").strip().lower()
        self.model = SentenceTransformer(model_name, device=embedding_device)
        self.index = None
        self.chunks: List[Chunk] = []
    
    def build_index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(
            [c.text for c in chunks],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            return []

        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        similarities, indices = self.index.search(
            query_emb.astype('float32'),
            min(top_k, self.index.ntotal),
        )

        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], similarities[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_words: int = 500,
    ) -> Tuple[str, List[Dict[str, object]]]:
        results = self.search(query, top_k)
        context_parts: List[str] = []
        metadata: List[Dict[str, object]] = []
        total_words = 0

        for idx, (chunk, score) in enumerate(results):
            if total_words + chunk.word_count > max_words and idx > 0:
                break

            context_parts.append(f"[Source {idx + 1}]: {chunk.text}")
            total_words += chunk.word_count
            metadata.append(
                {
                    "source": chunk.source_title,
                    "url": chunk.source_url,
                    "similarity": score,
                    "chunk_id": chunk.chunk_id,
                }
            )

        return "\n\n".join(context_parts), metadata

    def save_assets(self, index_path: str, chunks_path: str) -> None:
        if self.index is None:
            raise ValueError("Cannot save assets because FAISS index is not initialized.")

        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as chunks_file:
            pickle.dump(self.chunks, chunks_file)

    def load_assets(self, index_path: str, chunks_path: str) -> None:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as chunks_file:
            loaded_chunks = pickle.load(chunks_file)

        if not isinstance(loaded_chunks, list):
            raise ValueError("Invalid chunks payload: expected a list of Chunk objects.")

        self.chunks = loaded_chunks

class SerenityGenerator:
    def __init__(self, embedding_engine: EmbeddingEngine, model_id: str = DEFAULT_MODEL_ID):
        LOGGER.info("Loading quantized Qwen model: %s", model_id)

        requested_model_id = model_id
        candidate_model_ids: List[str] = [requested_model_id]
        candidate_model_ids.extend(
            [fallback for fallback in _parse_fallback_model_ids() if fallback != requested_model_id]
        )

        self.tokenizer = None
        self.model = None
        self.active_model_id = requested_model_id
        load_errors: List[str] = []

        use_4bit = (
            os.getenv("SERENITY_LLM_USE_4BIT", "false").strip().lower() == "true"
            and torch.cuda.is_available()
        )

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        if torch.cuda.is_available():
            gpu_fraction = float(os.getenv("SERENITY_GPU_MEMORY_FRACTION", "0.80"))
            gpu_fraction = max(0.30, min(gpu_fraction, 0.95))
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            allowed_gpu_memory_mb = int((total_gpu_memory * gpu_fraction) / (1024 * 1024))
            model_kwargs["max_memory"] = {
                0: f"{allowed_gpu_memory_mb}MB",
                "cpu": os.getenv("SERENITY_CPU_MAX_MEMORY", "6GB"),
            }

        for candidate_id in candidate_model_ids:
            tokenizer_errors: List[str] = []
            local_tokenizer = None

            tokenizer_attempts = [
                {"trust_remote_code": True, "use_fast": False},
                {"trust_remote_code": True},
                {"use_fast": False},
                {},
            ]

            for attempt_kwargs in tokenizer_attempts:
                try:
                    LOGGER.info("Attempting tokenizer load for %s with args: %s", candidate_id, attempt_kwargs)
                    local_tokenizer = AutoTokenizer.from_pretrained(candidate_id, **attempt_kwargs)
                    break
                except Exception as exc:
                    tokenizer_errors.append(str(exc))

            if local_tokenizer is None:
                load_errors.append(
                    f"Tokenizer load failed for {candidate_id}: {' | '.join(tokenizer_errors)}"
                )
                continue

            if local_tokenizer.pad_token is None:
                local_tokenizer.pad_token = local_tokenizer.eos_token

            try:
                local_model = AutoModelForCausalLM.from_pretrained(candidate_id, **model_kwargs)
            except Exception as exc:
                load_errors.append(f"Model load failed for {candidate_id}: {exc}")
                continue

            self.tokenizer = local_tokenizer
            self.model = local_model
            self.active_model_id = candidate_id

            # Align generation config with notebook-style sampling defaults.
            gen_cfg = self.model.generation_config
            gen_cfg.do_sample = True
            gen_cfg.temperature = DEFAULT_GENERATION_TEMPERATURE
            gen_cfg.top_p = DEFAULT_GENERATION_TOP_P
            gen_cfg.top_k = 0

            if candidate_id != requested_model_id:
                LOGGER.warning(
                    "Primary model %s unavailable. Falling back to %s.",
                    requested_model_id,
                    candidate_id,
                )
            break

        if self.tokenizer is None or self.model is None:
            joined = " || ".join(load_errors)
            raise RuntimeError(
                "Failed to initialize any local LLM candidate. "
                f"Primary: {requested_model_id}. Errors: {joined}"
            )

        self.embedding_engine = embedding_engine
        self.conversation_history: List[Dict[str, str]] = []

    def _sample_generation_kwargs(
        self,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, object]:
        sampling_temperature = DEFAULT_GENERATION_TEMPERATURE if temperature is None else float(temperature)
        sampling_top_p = DEFAULT_GENERATION_TOP_P if top_p is None else float(top_p)
        sampling_top_p = max(0.05, min(sampling_top_p, 1.0))
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": sampling_temperature,
            "top_p": sampling_top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def _greedy_generation_kwargs(self, max_new_tokens: int) -> Dict[str, object]:
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.08,
            "renormalize_logits": True,
            "remove_invalid_values": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def _generate_with_fallback(
        self,
        inputs,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        sampling_kwargs = self._sample_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        try:
            return self.model.generate(**inputs, **sampling_kwargs)
        except RuntimeError as exc:
            error_text = str(exc).lower()
            if "probability tensor contains either" in error_text or "nan" in error_text or "inf" in error_text:
                LOGGER.warning(
                    "Numerically unstable logits detected; retrying generation with deterministic decoding."
                )
                stricter_kwargs = self._greedy_generation_kwargs(max_new_tokens=max_new_tokens)
                return self.model.generate(**inputs, **stricter_kwargs)
            raise

    @staticmethod
    def _contains_self_harm_signal(user_msg: str) -> bool:
        text = str(user_msg or "").lower()
        markers = [
            "kill myself",
            "suicide",
            "end my life",
            "don't want to live",
            "not want to live",
            "hurt myself",
            "life is worth this pain",
            "life is not worth",
        ]
        return any(marker in text for marker in markers)

    @staticmethod
    def _fallback_empathetic_response(emotion_context: str, user_msg: str = "") -> str:
        if SerenityGenerator._contains_self_harm_signal(user_msg):
            return (
                "I am really glad you told me this, and your safety matters deeply right now. "
                "If you might act on these thoughts, please call local emergency services immediately "
                "or contact a trusted person to stay with you. Are you safe right now, and who can you "
                "reach out to in the next few minutes?"
            )

        emotion = str(emotion_context or "neutral").strip().lower()
        templates = {
            "angry": [
                "I can hear how intense this feels right now. Let's loosen your shoulders and take three slow breaths together. What part of this is hitting you the hardest?",
                "That sounds deeply frustrating. Before we respond, let's pause for one steady breath so your body can settle a little. What happened right before this feeling spiked?",
                "I hear how much pressure this is creating. Let's slow down together for one breath before we decide your next step. What feels most urgent to handle first?",
                "That frustration sounds exhausting. We can channel it without hurting you. What boundary or need feels most ignored here?",
            ],
            "sad": [
                "This sounds really heavy, and I am glad you shared it. Place one hand on your chest and take a gentle breath with me. What would feel even 5% more supportive right now?",
                "I hear the weight in this. You do not have to carry it all at once. What is one small thing that usually helps you feel less alone?",
                "It makes sense that this feels draining. Let's make tonight as gentle as possible. What is one tiny action that would make the next hour easier?",
                "Thank you for being honest about how hard this is. You still deserve care in this moment. What has felt most painful today?",
            ],
            "fear": [
                "It makes sense that this feels overwhelming. Let's ground for a moment: notice five things you can see and three things you can hear. Are you safe right now?",
                "I can hear the anxiety in this. Let's slow your breathing down first so your mind has more space. What are you most afraid might happen next?",
                "Your nervous system sounds overloaded right now. Let's steady the body first so your thoughts soften. What is the scariest part of this moment?",
                "That fear is understandable. You are not weak for feeling it. What would help you feel 10% safer in the next hour?",
            ],
            "happy": [
                "I can hear some real positive energy. Let's anchor it so it lasts longer. What are you doing differently that's helping this feeling show up?",
                "This sounds like an important bright spot. I'd like to help you hold onto it. What is one small way you can protect this momentum today?",
                "I am glad this moment feels lighter. Let's reinforce what is working for you. What part of today do you want to repeat tomorrow?",
                "That warmth in your voice matters. You built this moment somehow. What helped create this shift?",
            ],
            "calm": [
                "You sound steady in this moment, and that is a strength. Let's use it intentionally. What one small intention do you want for the next hour?",
                "I hear a grounded tone right now. That gives us room to choose your next step with care. What feels most meaningful to focus on first?",
                "This calm is valuable. We can use it to make one thoughtful decision. What would future-you thank you for doing today?",
                "You sound centered, which gives us clarity. What gentle next step fits your values best right now?",
            ],
            "neutral": [
                "Thank you for opening up. Let's take one slow breath in for four, hold for two, and exhale for six. What feels most important to unpack first?",
                "I am here with you. We can take this one piece at a time. What part of this has been staying on your mind the most?",
                "I appreciate you sharing this honestly. We do not have to solve everything at once. What is the hardest piece to carry right now?",
                "You are not alone in this moment. Let's make this manageable together. What would be most helpful to talk through first?",
            ],
            "surprise": [
                "That sounds like a sudden emotional shift. Let's slow the moment down and name what changed. What felt most unexpected to you?",
                "I can hear how abrupt that felt. Taking a breath can help your mind catch up to what happened. Which part still feels hardest to process?",
                "That sounds like it hit you fast. Your reaction makes sense. What changed so quickly that your system felt thrown off?",
                "A sudden shift can feel disorienting. Let's put words to it gently. What part is still echoing in your mind?",
            ],
            "disgust": [
                "I can hear strong discomfort in this. Let's create a little distance with one slow breath. What boundary feels crossed for you here?",
                "That reaction makes sense if something felt deeply off. We can move carefully from here. What do you need most to feel safe and respected right now?",
                "Your discomfort is valid. Something likely felt deeply misaligned. What boundary would protect you best going forward?",
                "That sense of aversion can be protective. Let's listen to it carefully. What felt most unacceptable in that situation?",
            ],
        }
        choices = templates.get(emotion, templates["neutral"])
        selector_seed = f"{emotion}:{user_msg}".lower()
        digest = hashlib.sha256(selector_seed.encode("utf-8", errors="ignore")).hexdigest()
        index = int(digest, 16) % len(choices)
        return choices[index]

    @staticmethod
    def _is_low_quality_output(text: str) -> bool:
        if not text:
            return True

        normalized = " ".join(str(text).split())
        if len(normalized) < 12:
            return True

        if not any(ch.isalpha() for ch in normalized):
            return True

        words = [word for word in re.split(r"\s+", normalized.lower()) if word]
        if len(words) >= 8 and len(set(words)) <= 2:
            return True

        alphabetic_chars = [ch for ch in normalized if ch.isalpha()]
        latin_chars = [ch for ch in normalized if ("a" <= ch.lower() <= "z")]
        if len(alphabetic_chars) >= 20:
            latin_ratio = len(latin_chars) / max(len(alphabetic_chars), 1)
            if latin_ratio < 0.55:
                return True

        mojibake_markers = ["Ã", "Ð", "à¸", "æ", "å", "ï¿½"]
        marker_hits = sum(normalized.count(marker) for marker in mojibake_markers)
        if marker_hits >= 6:
            return True

        # Catch outputs that are mostly repeated punctuation or repeated symbols.
        if re.search(r"(.)\1{7,}", normalized):
            return True

        total_chars = max(len(normalized), 1)
        printable_chars = sum(ch.isprintable() for ch in normalized)
        ascii_chars = sum(ch.isascii() for ch in normalized)

        if printable_chars / total_chars < 0.98:
            return True

        if (ascii_chars / total_chars) < 0.82:
            return True

        weird_symbol_hits = sum(normalized.count(symbol) for symbol in ["{", "}", "<", ">", "\\", "/", "_", "=", "%"])
        if weird_symbol_hits >= 8:
            return True

        return False

    @staticmethod
    def _normalize_for_overlap(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
        return " ".join(cleaned.split())

    def _is_repetitive_against_history(self, text: str, history: Optional[List[Dict[str, str]]]) -> bool:
        if not history:
            return False

        candidate = self._normalize_for_overlap(text)
        if len(candidate) < 24:
            return False

        candidate_tokens = set(candidate.split())
        if not candidate_tokens:
            return False

        for turn in history[-4:]:
            prior_assistant = str(turn.get("assistant_text", "")).strip()
            if not prior_assistant:
                continue

            prior = self._normalize_for_overlap(prior_assistant)
            if not prior:
                continue

            if candidate == prior:
                return True

            prior_tokens = set(prior.split())
            overlap = candidate_tokens.intersection(prior_tokens)
            union = candidate_tokens.union(prior_tokens)
            if union and (len(overlap) / len(union)) > 0.88:
                return True

        return False

    @staticmethod
    def _drain_complete_sentences(buffer: str) -> Tuple[List[str], str]:
        if not buffer:
            return [], ""

        segments = SENTENCE_BOUNDARY_REGEX.split(buffer)
        if len(segments) <= 1:
            return [], buffer

        completed = [segment.strip() for segment in segments[:-1] if segment.strip()]
        pending = segments[-1]
        return completed, pending

    @staticmethod
    def get_system_prompt() -> str:
        return (
            "You are Serenity, a warm empathetic therapist.\n\n"
            "CRITICAL RULES:\n"
            "- Keep responses SHORT (2-4 sentences, max 60 words)\n"
            "- Be conversational and warm, not formal\n"
            "- Reflect feelings, then ask ONE follow-up question\n"
            "- NO bullet points, NO lists, NO numbered steps\n"
            "- Talk like a real therapist in a gentle conversation\n"
            "- Use the knowledge context subtly, don't lecture\n\n"
            "Examples of good responses:\n"
            "\"I hear you - breakups are really painful. What's been the hardest part for you?\"\n"
            "\"That sounds overwhelming. When did you first start noticing these feelings?\"\n"
            "\"It makes sense you'd feel anxious about that. Have you been able to talk to anyone about this?\"\n\n"
            "Remember: Brief, empathetic, conversational. One thoughtful question."
        )

    @staticmethod
    def _sanitize_assistant_surface(text: str) -> str:
        cleaned = str(text or "")
        cleaned = re.sub(r"\n\d+\..*", "", cleaned)
        cleaned = re.sub(r"\n-.*", "", cleaned)
        cleaned = re.sub(r"\*\*.*?\*\*", "", cleaned)
        return " ".join(cleaned.split()).strip()

    @staticmethod
    def _truncate_to_word_limit(text: str, max_words: int = 60) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).strip().rstrip(",;:") + "."

    def _build_multimodal_messages(
        self,
        user_msg: str,
        dominant_emotion: str,
        history: Optional[List[Dict[str, str]]] = None,
        emotion_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Dict[str, str]]:
        history = history or []
        rag_context, _metadata = self.embedding_engine.get_context(
            user_msg,
            top_k=2,
            max_words=250,
        )

        system_prompt = self.get_system_prompt()
        if dominant_emotion:
            system_prompt += f"\nDetected emotion: {dominant_emotion}."
        if rag_context:
            system_prompt += f"\n\nRelevant context (use subtly):\n{rag_context}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        source_history = history[-4:] if history else self.conversation_history[-4:]
        for turn in source_history:
            if "role" in turn and "content" in turn:
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
                continue

            prior_user = str(turn.get("user_text", "")).strip()
            prior_assistant = str(turn.get("assistant_text", "")).strip()
            if prior_user:
                messages.append({"role": "user", "content": prior_user})
            if prior_assistant:
                messages.append({"role": "assistant", "content": prior_assistant})

        messages.append({"role": "user", "content": user_msg})
        return messages

    def _finalize_response(
        self,
        raw_response: str,
        emotion_context: str,
        user_msg: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        allow_fallback: bool = True,
    ) -> str:
        normalized = self._sanitize_assistant_surface(raw_response)
        if self._is_low_quality_output(normalized):
            if allow_fallback:
                LOGGER.warning("Low-quality LLM output detected; returning empathetic fallback response.")
                return self._truncate_to_word_limit(
                    self._sanitize_assistant_surface(
                        self._fallback_empathetic_response(emotion_context, user_msg)
                    )
                )
            return normalized

        if self._is_repetitive_against_history(normalized, history):
            LOGGER.warning("Response was overly similar to recent turns; returning varied fallback response.")
            return self._truncate_to_word_limit(
                self._sanitize_assistant_surface(
                    self._fallback_empathetic_response(emotion_context, user_msg)
                )
            )

        return self._truncate_to_word_limit(normalized)

    def generate_response(
        self,
        user_message: str,
        use_rag: bool = True,
        top_k_chunks: int = 2,
        max_new_tokens: int = DEFAULT_GENERATION_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_GENERATION_TEMPERATURE,
        top_p: float = DEFAULT_GENERATION_TOP_P,
    ) -> Tuple[str, List[Dict[str, object]]]:
        context = ""
        rag_metadata: List[Dict[str, object]] = []
        if use_rag and self.embedding_engine is not None:
            context, rag_metadata = self.embedding_engine.get_context(
                user_message,
                top_k=top_k_chunks,
                max_words=250,
            )

        messages: List[Dict[str, str]] = []
        system_content = self.get_system_prompt()
        if context:
            system_content += f"\n\nRelevant context (use subtly):\n{context}"
        messages.append({"role": "system", "content": system_content})

        for msg in self.conversation_history[-4:]:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        generation_kwargs = self._sample_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        response = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        finalized = self._finalize_response(response, "neutral", user_msg=user_message)

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": finalized})

        return finalized, rag_metadata

    def reset_conversation(self) -> None:
        self.conversation_history = []
    
    def generate(self, user_msg: str, emotion_context: str = "") -> str:
        messages = self._build_multimodal_messages(
            user_msg=user_msg,
            dominant_emotion=emotion_context,
            history=None,
            emotion_probabilities=None,
        )
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        with torch.inference_mode():
            output_ids = self._generate_with_fallback(
                inputs=inputs,
                max_new_tokens=DEFAULT_GENERATION_MAX_NEW_TOKENS,
                temperature=DEFAULT_GENERATION_TEMPERATURE,
                top_p=DEFAULT_GENERATION_TOP_P,
            )
        response = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        finalized = self._finalize_response(response, emotion_context, user_msg=user_msg)

        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": finalized})
        return finalized

    def generate_multimodal(
        self,
        user_msg: str,
        dominant_emotion: str,
        history: Optional[List[Dict[str, str]]] = None,
        emotion_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        history = history or []
        messages = self._build_multimodal_messages(
            user_msg=user_msg,
            dominant_emotion=dominant_emotion,
            history=history,
            emotion_probabilities=emotion_probabilities,
        )

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self._generate_with_fallback(
                inputs=inputs,
                max_new_tokens=DEFAULT_GENERATION_MAX_NEW_TOKENS,
                temperature=DEFAULT_GENERATION_TEMPERATURE,
                top_p=DEFAULT_GENERATION_TOP_P,
            )

        response = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return self._finalize_response(
            response,
            dominant_emotion,
            user_msg=user_msg,
            history=history,
        )

    def generate_multimodal_streaming(
        self,
        user_msg: str,
        dominant_emotion: str,
        history: Optional[List[Dict[str, str]]] = None,
        emotion_probabilities: Optional[Dict[str, Dict[str, float]]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        history = history or []
        messages = self._build_multimodal_messages(
            user_msg=user_msg,
            dominant_emotion=dominant_emotion,
            history=history,
            emotion_probabilities=emotion_probabilities,
        )

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = self._sample_generation_kwargs(
            max_new_tokens=DEFAULT_GENERATION_MAX_NEW_TOKENS,
            temperature=DEFAULT_GENERATION_TEMPERATURE,
            top_p=DEFAULT_GENERATION_TOP_P,
        )
        generation_kwargs.update(inputs)
        generation_kwargs["streamer"] = streamer

        generation_errors: List[Exception] = []

        def _stream_worker() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as exc:
                generation_errors.append(exc)

        worker = threading.Thread(target=_stream_worker, daemon=True)
        worker.start()

        accumulated = ""
        pending_sentence = ""

        for token_chunk in streamer:
            if not token_chunk:
                continue
            accumulated += token_chunk
            pending_sentence += token_chunk

            if on_token:
                on_token(token_chunk)

            completed, pending_sentence = self._drain_complete_sentences(pending_sentence)
            if on_sentence:
                for sentence in completed:
                    on_sentence(sentence)

        worker.join()

        if generation_errors:
            first_error = generation_errors[0]
            LOGGER.warning("Streaming generation failed; falling back to non-streaming path: %s", first_error)
            with torch.inference_mode():
                output_ids = self._generate_with_fallback(
                    inputs=inputs,
                    max_new_tokens=DEFAULT_GENERATION_MAX_NEW_TOKENS,
                    temperature=DEFAULT_GENERATION_TEMPERATURE,
                    top_p=DEFAULT_GENERATION_TOP_P,
                )
            accumulated = self.tokenizer.decode(
                output_ids[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True,
            )
            pending_sentence = accumulated

        if pending_sentence.strip() and on_sentence:
            on_sentence(pending_sentence.strip())

        return self._finalize_response(
            accumulated,
            dominant_emotion,
            user_msg=user_msg,
            history=history,
            allow_fallback=True,
        )

def build_and_persist_knowledge_base(
    urls: List[str],
    index_path: str = DEFAULT_INDEX_PATH,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """
    One-time offline build step.

    Scrapes source URLs, chunks text, embeds, and persists FAISS/chunks assets to disk.
    This should be run explicitly during preparation, not during API startup.
    """
    LOGGER.info("Building knowledge base assets.")

    scraper = KnowledgeScraper()
    articles = scraper.scrape_urls(urls)
    if not articles:
        raise RuntimeError("No articles were scraped. Cannot build knowledge base assets.")

    processor = TextProcessor()
    chunks = processor.process_articles(articles)
    if not chunks:
        raise RuntimeError("No chunks generated from scraped articles.")

    engine = EmbeddingEngine(model_name=embedding_model)
    engine.build_index(chunks)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    engine.save_assets(index_path=index_path, chunks_path=chunks_path)
    LOGGER.info("Knowledge base assets saved to %s and %s", index_path, chunks_path)


def init_rag_system(
    model_id: str = DEFAULT_MODEL_ID,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    index_path: str = DEFAULT_INDEX_PATH,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> SerenityGenerator:
    """
    Runtime initialization path.

    Loads persisted FAISS/chunks assets and initializes the quantized model.
    No dynamic scraping or index building occurs here.
    """
    LOGGER.info("Initializing SERENITY runtime components.")

    engine = EmbeddingEngine(model_name=embedding_model)
    engine.load_assets(index_path=index_path, chunks_path=chunks_path)

    serenity = SerenityGenerator(embedding_engine=engine, model_id=model_id)
    LOGGER.info("SERENITY runtime initialization complete.")
    return serenity