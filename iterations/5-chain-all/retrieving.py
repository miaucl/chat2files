"""Simple example of how to use Haystack with Qdrant as a document store and retriever."""
import logging
from pathlib import Path

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.readers import ExtractiveReader
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DIR = Path(__file__).resolve().parent

document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="5-chain-all",
    embedding_dim=384,
    similarity="cosine",  # or "dot" or "euclidean"
)
reader = ExtractiveReader(model="deepset/roberta-base-squad2")
retrieving_pipeline = Pipeline()
retrieving_pipeline.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
retrieving_pipeline.add_component(
    "retriever", QdrantEmbeddingRetriever(document_store=document_store)
)
retrieving_pipeline.add_component(instance=reader, name="reader")

retrieving_pipeline.connect("embedder.embedding", "retriever.query_embedding")
retrieving_pipeline.connect("retriever.documents", "reader.documents")

retrieving_pipeline.draw(str(DIR / "retrieving.png"))


def run(inputs: dict) -> dict:
    """Run the retrieving pipeline with the given inputs."""
    LOGGER.info("Running retrieving pipeline with inputs: %s", inputs)
    response = retrieving_pipeline.run(
        inputs, include_outputs_from=["retriever", "reader"]
    )
    LOGGER.info("Pipeline run completed.")
    return response
