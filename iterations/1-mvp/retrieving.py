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
    index="1-mvp",
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

question = "What ingredients would I need to make vegan keto eggplant lasagna, vegan persimmon flan, and vegan hemp cheese?"

LOGGER.info("Running pipeline with question: %s", question)
response = retrieving_pipeline.run(
    {
        "embedder": {"text": question},
        "retriever": {"top_k": 5},
        "reader": {"query": question, "top_k": 3},
    },
    include_outputs_from=["retriever", "reader"],
)
LOGGER.info(
    "Found %d documents with following answers", len(response["retriever"]["documents"])
)
LOGGER.info("=" * 80)
for answer in response["reader"]["answers"]:
    LOGGER.info("Answer: %s", answer.data)
    LOGGER.info("Document: %s", answer.document)
    LOGGER.info("Score: %s", answer.score)
    LOGGER.info("-" * 80)

retrieving_pipeline.draw(str(DIR / "retrieving.png"))
