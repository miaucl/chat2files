"""Indexing pipeline for recipe files using Haystack and Qdrant."""
import logging
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DIR = Path(__file__).resolve().parent
DATA = DIR.parent.parent / "data" / "recipe_files"

document_store = QdrantDocumentStore(
    url="http://localhost:6333",
    index="5-chain-all",
    embedding_dim=384,
    similarity="cosine",  # or "dot" or "euclidean"
)
file_type_router = FileTypeRouter(
    mime_types=["text/plain", "application/pdf", "text/markdown"]
)
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
document_joiner = DocumentJoiner()

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(
    split_by="word", split_length=150, split_overlap=50
)

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
document_writer = DocumentWriter(document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=file_type_router, name="file_type_router")
indexing_pipeline.add_component(
    instance=text_file_converter, name="text_file_converter"
)
indexing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
indexing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
indexing_pipeline.add_component(instance=document_joiner, name="document_joiner")
indexing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
indexing_pipeline.add_component(instance=document_splitter, name="document_splitter")
indexing_pipeline.add_component(instance=document_embedder, name="document_embedder")
indexing_pipeline.add_component(instance=document_writer, name="document_writer")

indexing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
indexing_pipeline.connect(
    "file_type_router.text/markdown", "markdown_converter.sources"
)
indexing_pipeline.connect("text_file_converter", "document_joiner")
indexing_pipeline.connect("pypdf_converter", "document_joiner")
indexing_pipeline.connect("markdown_converter", "document_joiner")
indexing_pipeline.connect("document_joiner", "document_cleaner")
indexing_pipeline.connect("document_cleaner", "document_splitter")
indexing_pipeline.connect("document_splitter", "document_embedder")
indexing_pipeline.connect("document_embedder", "document_writer")

indexing_pipeline.draw(str(DIR / "indexing.png"))

LOGGER.info("Fetching data from %s", str(DATA))
indexing_pipeline.run({"file_type_router": {"sources": list(DATA.glob("**/*"))}})

LOGGER.info("%d document(s) written to the store", document_store.count_documents())
