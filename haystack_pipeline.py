from haystack import Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Document

def create_pipeline(documents):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    query_embedder = OpenAITextEmbedder()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    pipeline = Pipeline()
    pipeline.add_component("query_embedder", query_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    return pipeline
