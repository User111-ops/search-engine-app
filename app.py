import streamlit as st
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import convert_files_to_docs
from pathlib import Path

st.set_page_config(page_title="Recherche simple", layout="wide")
st.title("🔍 Recherche par mot-clé")

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_pipeline():
    document_store = InMemoryDocumentStore()
    retriever = BM25Retriever(document_store=document_store)
    pipeline = DocumentSearchPipeline(retriever)
    return document_store, pipeline

document_store, pipeline = load_pipeline()

uploaded_files = st.file_uploader("Upload .txt or .docx files", type=["txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"{len(uploaded_files)} fichiers chargés.")
    docs = convert_files_to_docs(dir_path=str(UPLOAD_DIR))
    document_store.write_documents(docs)

query = st.text_input("Entrez un mot-clé ou une phrase à rechercher :")

if st.button("Rechercher") and query:
    result = pipeline.run(query=query, params={"Retriever": {"top_k": 5}})
    docs = result.get("documents", [])

    if not docs:
        st.warning("Aucun résultat.")
    else:
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}.** {doc.content[:300]}...")
            st.caption(f"From file: {doc.meta.get('name', 'inconnu')}")
