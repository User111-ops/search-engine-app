import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.utils import Document
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="Recherche IA dans vos documents")
st.title("📄 Recherche intelligente avec Haystack")

uploaded_files = st.file_uploader("Déposez vos fichiers ici", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    documents = []

    for file in uploaded_files:
        content = file.read().decode("utf-8", errors="ignore")
        documents.append(Document(content=content, meta={"name": file.name, "path": "upload"}))

    st.success("✅ Fichiers bien reçus !")

    question = st.text_input("Que cherchez-vous ?")

    if question:
        with st.spinner("Recherche en cours..."):
            document_store = InMemoryDocumentStore()
            document_store.write_documents(documents)

            embedder = OpenAITextEmbedder()
            retriever = InMemoryEmbeddingRetriever(document_store=document_store)

            pipeline = Pipeline()
            pipeline.add_component("query_embedder", embedder)
            pipeline.add_component("retriever", retriever)
            pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

            results = pipeline.run({"query_embedder": {"text": question}})
            docs = results["retriever"]["documents"]

            if docs:
                top_doc = docs[0]
                st.markdown("### 📌 Résultat")
                st.write(top_doc.content)
                st.caption(f"Match in `{top_doc.meta['name']}` --> `{top_doc.meta['path']}`")
            else:
                st.warning("No match found.")

