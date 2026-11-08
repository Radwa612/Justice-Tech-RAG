from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db_maroc", 
    embedding_function=embeddings
)

# Récupération de tous les métadonnées enregistrées
all_docs = vectorstore.get()
sources = set(meta.get("source") for meta in all_docs["metadatas"])

print("--- FICHIERS ACTUELLEMENT DANS CHROMADB ---")
for src in sources:
    print(f"- {src}")