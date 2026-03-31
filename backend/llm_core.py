import logging
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "serenity_faiss.index")
DEFAULT_CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "serenity_chunks.pkl")

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

class KnowledgeScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    def scrape_wikipedia_article(self, url: str) -> Optional[Dict[str, str]]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200: return None
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('h1', class_='firstHeading').get_text()
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div: return None
            
            paragraphs = []
            for p in content_div.find_all('p'):
                text = p.get_text().strip()
                if len(text) > 40:
                    text = re.sub(r'\[\d+\]', '', text)
                    paragraphs.append(text)
            return {'title': title, 'url': url, 'content': '\n\n'.join(paragraphs), 'source': 'Wikipedia'}
        except Exception:
            return None

    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        articles = []
        for url in urls:
            print(f"   ...scraping {url}")
            article = self.scrape_wikipedia_article(url)
            if article: articles.append(article)
        return articles

class TextProcessor:
    def process_articles(self, articles: List[Dict[str, str]]) -> List[Chunk]:
        all_chunks = []
        for idx, article in enumerate(articles):
            # Simple chunking by approximate words
            words = article['content'].split()
            chunk_size = 180
            for i in range(0, len(words), chunk_size - 30): # Overlap
                chunk_text = " ".join(words[i:i + chunk_size])
                all_chunks.append(Chunk(
                    chunk_id=f"{idx}_{i}", text=chunk_text,
                    source_title=article['title'], source_url=article['url'],
                    source_type='Wikipedia', word_count=len(chunk_text.split()), chunk_index=i
                ))
        return all_chunks

class EmbeddingEngine:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        LOGGER.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[Chunk] = []
    
    def build_index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode([c.text for c in chunks], convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
    
    def get_context(self, query: str, top_k: int = 2) -> str:
        if self.index is None:
            return ""

        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        _, indices = self.index.search(query_emb.astype('float32'), top_k)
        return "\n\n".join(
            [
                f"[Source: {self.chunks[i].source_title}]: {self.chunks[i].text}"
                for i in indices[0]
                if i < len(self.chunks)
            ]
        )

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

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        self.embedding_engine = embedding_engine
    
    def generate(self, user_msg: str, emotion_context: str = "") -> str:
        rag_context = self.embedding_engine.get_context(user_msg)
        
        system_prompt = f"""You are Serenity, an empathetic therapist. The user is feeling {emotion_context}.
        Keep responses SHORT (max 50 words). Be warm. Ask one follow-up question.
        
        Context from psychology knowledge base:
        {rag_context}
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()

    def generate_multimodal(
        self,
        user_msg: str,
        dominant_emotion: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        history = history or []
        rag_context = self.embedding_engine.get_context(user_msg)

        serialized_history = "\n".join(
            [
                f"User: {turn.get('user_text', '')}\nAssistant: {turn.get('assistant_text', '')}"
                for turn in history
            ]
        )

        system_prompt = (
            "You are Serenity, a calm and empathetic psychologist-like AI assistant. "
            "Keep responses concise (40-80 words), validating feelings first and then offering one practical next step.\n\n"
            f"Detected emotion: {dominant_emotion}.\n\n"
            "Recent conversation history:\n"
            f"{serialized_history if serialized_history else 'No prior turns.'}\n\n"
            "Relevant psychology knowledge:\n"
            f"{rag_context if rag_context else 'No RAG context found.'}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                temperature=0.65,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()

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