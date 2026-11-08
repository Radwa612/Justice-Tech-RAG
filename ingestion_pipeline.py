import os
import pymupdf as fitz # PyMuPDF : bibliothèque d'extraction de texte PDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil



def extraction_pdf(dossier_data):
    """
    Étape 1 : Lecture et extraction du texte des fichiers PDF.
    Chaque page devient un objet Document avec ses métadonnées (nom du fichier, numéro de page).
    """
    documents_extraits = []
    print("\n--- ÉTAPE 1 : EXTRACTION DES PDF ---")
    
    # Parcours de tous les fichiers du dossier
    for nom_fichier in os.listdir(dossier_data):
        if nom_fichier.endswith(".pdf"):
            chemin_pdf = os.path.join(dossier_data, nom_fichier)
            document_pdf = fitz.open(chemin_pdf)
            
            # Lecture page par page
            for numero_page in range(document_pdf.page_count):
                page = document_pdf[numero_page]
                texte = page.get_text("text")
                
                # Ignorer les pages vides
                if texte.strip():
                    metadonnees = {
                        "source": nom_fichier,
                        "page": numero_page + 1
                    }
                    doc = Document(page_content=texte, metadata=metadonnees)
                    documents_extraits.append(doc)
            
            print(f"  ✓ {nom_fichier} extrait ({document_pdf.page_count} pages)")
            document_pdf.close()
            
    print(f" Total : {len(documents_extraits)} pages extraites au total.\n")
    return documents_extraits


def decoupage_texte(documents):
    """
    Étape 2 : Découpage sémantique (Chunking).
    Divise les textes en segments plus petits adaptés à la recherche juridique.
    """
    print("--- ÉTAPE 2 : DÉCOUPAGE DU TEXTE (CHUNKING) ---")
    
    # Découpeur adapté à la structure des textes juridiques marocains
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=800,        # Taille du segment de texte
        chunk_overlap=150,      # Chevauchement pour conserver la continuité du contexte
        separators=["Article ", "المادة ", "\n\n", "\n", " "]
    )
    
    chunks = decoupeur.split_documents(documents)
    print(f" Total : {len(chunks)} chunks générés.\n")
    return chunks


def charger_modele_embedding():
    """
    Étape 3 : Chargement du modèle de vectorisation local.
    Convertit le texte en vecteurs numériques (exécuté en RAM/CPU local).
    """
    print("--- ÉTAPE 3 : CHARGEMENT DU MODÈLE D'EMBEDDING ---")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(" Modèle BAAI/bge-m3 prêt en mémoire.\n")
    return embeddings


def creation_base_vectorielle(chunks, embeddings, dossier_base):
    # 1. Nettoyage de l'ancienne base L2 si elle existe
    if os.path.exists(dossier_base):
        print(f"🧹 Suppression de l'ancienne base L2 dans '{dossier_base}'...")
        shutil.rmtree(dossier_base)

    # 2. Création de la base avec la métrique COSINE explicite
    vectorstore = Chroma(
        persist_directory=dossier_base,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}  # <--- SEULE LIGNE AJOUTÉE
    )
    
    # Découpage par paquets de 100 chunks pour alléger le calcul
    taille_paquet = 100
    total = len(chunks)
    
    for i in range(0, total, taille_paquet):
        paquet = chunks[i:i + taille_paquet]
        vectorstore.add_documents(paquet)
        print(f"📦 Chunks {i} à {min(i + taille_paquet, total)} / {total} indexés...")
        
    print("✅ Base ChromaDB créée avec succès !")
    return vectorstore


# --- POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    DOSSIER_PDF = "./data"
    DOSSIER_BASE_VECTORIELLE = "./chroma_db_maroc"
    
    # Vérification de l'existence du dossier de données
    if not os.path.exists(DOSSIER_PDF):
        os.makedirs(DOSSIER_PDF)
        print(f"⚠️ Le dossier '{DOSSIER_PDF}' a été créé. Déposez-y vos documents PDF et relancez le script.")
    else:
        # Exécution séquentielle des 4 étapes
        pages_extraites = extraction_pdf(DOSSIER_PDF)
        chunks_texte = decoupage_texte(pages_extraites)
        moteur_embedding = charger_modele_embedding()
        base_donnees = creation_base_vectorielle(chunks_texte, moteur_embedding, DOSSIER_BASE_VECTORIELLE)
        
        print(" PIPELINE D'INGESTION TERMINÉ AVEC SUCCÈS !")
        print(f"La base ChromaDB est prête et disponible dans : {DOSSIER_BASE_VECTORIELLE}")