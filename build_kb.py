import logging
from pathlib import Path
from typing import List

from backend.llm_core import EmbeddingEngine, KnowledgeScraper, TextProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

THERAPEUTIC_URLS: List[str] = [
    "https://en.wikipedia.org/wiki/Empathy",
    "https://en.wikipedia.org/wiki/Emotional_intelligence",
    "https://en.wikipedia.org/wiki/Active_listening",
    "https://en.wikipedia.org/wiki/Cognitive_behavioral_therapy",
    "https://en.wikipedia.org/wiki/Mindfulness",
    "https://en.wikipedia.org/wiki/Compassion",
    "https://en.wikipedia.org/wiki/Coping",
]

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "backend" / "serenity_faiss.index"
CHUNKS_PATH = BASE_DIR / "backend" / "serenity_chunks.pkl"


def main() -> None:
    LOGGER.info("Starting offline knowledge base build.")

    scraper = KnowledgeScraper()
    articles = scraper.scrape_urls(THERAPEUTIC_URLS)
    if not articles:
        raise RuntimeError("No articles were scraped. Aborting knowledge base build.")

    processor = TextProcessor()
    chunks = processor.process_articles(articles)
    if not chunks:
        raise RuntimeError("No chunks were generated. Aborting knowledge base build.")

    engine = EmbeddingEngine()
    engine.build_index(chunks)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine.save_assets(str(INDEX_PATH), str(CHUNKS_PATH))

    LOGGER.info("Knowledge base build complete.")
    LOGGER.info("FAISS index: %s", INDEX_PATH)
    LOGGER.info("Chunks file: %s", CHUNKS_PATH)
    LOGGER.info("Chunk count: %d", len(chunks))


if __name__ == "__main__":
    main()
