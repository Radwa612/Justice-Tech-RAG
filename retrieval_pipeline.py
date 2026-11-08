import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def charger_base_vectorielle():
    """
    Étape 1 : Chargement du modèle d'embedding et connexion à ChromaDB.
    """
    print("1. Chargement de la base vectorielle...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db_maroc", 
        embedding_function=embeddings
    )
    
    # On récupère les 3 chunks les plus pertinents
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def initialiser_modele_llm():
    """
    Étape 2 : Configuration du modèle LLM via l'API Groq Cloud.
    """
    print("2. Connexion au modèle de langage Groq...")
    os.environ["GROQ_API_KEY"] = "ENTRER LA CLE ICI"
    
    llm = ChatGroq(
        model_name="qwen/qwen3.8-27b",
        temperature=0
    )
    return llm


def construire_chaine_rag(retriever, llm):
    """
    Étape 3 : Création du prompt et de la chaîne de recherche avec mémoire.
    """
    print("3. Configuration du Prompt et de la mémoire...")
    
    system_prompt = """Tu es un Assistant IA Juridique expert en droit marocain.
Ton rôle est d'aider le citoyen à comprendre ses droits de manière simple, précise et fiable.

Consignes strictes :
1. Utilise EXCLUSIVEMENT le contexte juridique fourni ci-dessous.
2. Tiens compte de l'historique de la conversation pour comprendre les questions de suivi.
3. Si la réponse ne se trouve pas dans le contexte, indique "Non trouvé".

CONTEXTE JURIDIQUE :
{context}

RÉPONDS STRICTEMENT EN SUIVANT CE FORMAT :
---
1. LE DROIT : (Explication claire et accessible)
2. LA SOURCE OFFICIELLE : (Texte de loi et numéro d'article)
3. LA PROCÉDURE / COMMENT AGIR : (Étapes pratiques à suivre)
---"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # Historique des échanges
        ("human", "{question}")
    ])

    # Fonction explicite pour chercher et formater les documents de ChromaDB
    def rechercher_et_preparer_contexte(inputs):
        question_utilisateur = inputs["question"]
        historique = inputs.get("chat_history", [])
        
        # Recherche vectorielle dans ChromaDB
        documents_trouves = retriever.invoke(question_utilisateur)
        
        # Consolidation des extraits de textes
        textes = []
        for doc in documents_trouves:
            textes.append(doc.page_content)
        contexte_formate = "\n\n".join(textes)

        return {
            "context": contexte_formate,
            "question": question_utilisateur,
            "chat_history": historique
        }

    # Pipeline RAG
    base_chain = rechercher_et_preparer_contexte | prompt | llm
    
    # Gestion de la mémoire par session utilisateur
    registre_sessions = {}

    def obtenir_historique_session(session_id: str):
        if session_id not in registre_sessions:
            registre_sessions[session_id] = InMemoryChatMessageHistory()
        return registre_sessions[session_id]

    # Enrobage final avec la gestion d'historique
    chaine_avec_memoire = RunnableWithMessageHistory(
        base_chain,
        obtenir_historique_session,
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    
    return chaine_avec_memoire


# --- TEST DU SCRIPT (EXÉCUTION DIRECTE) ---
if __name__ == "__main__":
    retriever = charger_base_vectorielle()
    llm = initialiser_modele_llm()
    rag_chain = construire_chaine_rag(retriever, llm)

    # Identifiant unique de session de test
    configuration_session = {"configurable": {"session_id": "test_session_1"}}

    print("\n==================================================")
    print("--- TEST 1 : Question Principale ---")
    question_1 = "Mon employeur m'a licencié sans préavis, quels sont mes droits ?"
    print(f"Utilisateur : {question_1}\n")
    
    reponse_1 = rag_chain.invoke({"question": question_1}, config=configuration_session)
    print(f"Assistant :\n{reponse_1.content}\n")

    print("==================================================")
    print("--- TEST 2 : Question de Suivi (Utilise l'historique) ---")
    question_2 = "Et quelle est la durée minimale de ce préavis selon la loi ?"
    print(f"Utilisateur : {question_2}\n")
    
    reponse_2 = rag_chain.invoke({"question": question_2}, config=configuration_session)
    print(f"Assistant :\n{reponse_2.content}\n")
   