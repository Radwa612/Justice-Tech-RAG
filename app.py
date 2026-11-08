import os
import re
import json
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq

# Configuration de la page Streamlit
st.set_page_config(page_title="Moroccan Legal AI Assistant", layout="wide", page_icon="⚖️")
st.title("⚖️ Assistant Juridique - Droit Marocain")

# Configuration des clés API
os.environ["GROQ_API_KEY"] = "ENTRER LA CLE ICI "
client_juge = Groq()

# --- FONCTION DE CALCUL DU SCORE DE CONFIANCE VECTORIEL ---
def calculer_score_confiance(vectorstore, question, k=8):
    try:
        resultats = vectorstore.similarity_search_with_score(question, k=k)
        if not resultats:
            return 0.0
        
        scores = []
        for doc, distance in resultats:
            # En espace 'cosine', la similitude est (1 - distance) * 100
            similitude = (1.0 - distance) * 100
            scores.append(max(0.0, min(100.0, similitude)))
            
        score_moyen = sum(scores) / len(scores)
        return round(score_moyen, 1)
    except Exception:
        return 0.0

# --- FONCTION ÉVALUATION SCORE 2 : CONFORMITÉ PAR LE LLM JUGE ---
def evaluer_score_2_conformite(question, reponse_rag):
    """
    Évalue en direct la conformité juridique et l'absence d'hallucinations dans la réponse générée.
    """
    prompt = f"""
Tu es un juge juridique expert en droit marocain.
Ton objectif est de vérifier si la réponse fournie est JURIDIQUEMENT COHÉRENTE, CONFORME aux textes marocains et EXEMPTE D'HALLUCINATIONS.

QUESTION DE L'UTILISATEUR :
{question}

RÉPONSE DU RAG À ÉVALUER :
{reponse_rag}

CONSIGNES :
1. Vérifie la présence de fondements juridiques réels (articles, lois).
2. Sanctionne les affirmations vagues, contradictoires ou fausses.
3. Attribue une note de conformité de 0 à 100%.

Réponds STRICTEMENT sous ce format JSON :
{{
  "score_pourcentage": 85,
  "explication": "La réponse s'appuie correctement sur les règles du droit de la consommation sans hallucination."
}}
"""
    try:
        response = client_juge.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        resultat = json.loads(response.choices[0].message.content)
        return resultat.get("score_pourcentage", 0), resultat.get("explication", "Évaluation indisponible.")
    except Exception as e:
        return 0, f"Erreur d'évaluation par le Juge : {e}"

# --- INITIALISATION EN CACHE ---
@st.cache_resource
def init_models_and_chain():
    # Étape 1 : Chargement du modèle d'embedding local
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Étape 2 : Connexion à la base vectorielle ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db_maroc", 
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.3
        }
    )

    # Étape 3 : Configuration du LLM Groq
    llm = ChatGroq(
        model_name="qwen/qwen3.8-27b",
        temperature=0,
        max_tokens=500
    )

    # Étape 4 : Définition du Prompt avec les 6 rubriques imposées
    template_prompt = """Tu es un assistant juridique spécialisé en droit marocain. 
Ta mission est d'analyser la question de l'utilisateur à partir du CONTEXTE fourni et de structurer ta réponse STRICTEMENT selon les 6 rubriques imposées.

RÈGLE D'INTERPRÉTATION :
Les lois rédigées posent des règles générales (ex: "fournisseur", "prestataire", "affichage des prix"). Si la question décrit un cas concret (ex: restaurant, magasin, achat), applique les principes généraux correspondants de la Loi 31-08 sur la protection du consommateur ou du Code du travail. Ne réponds JAMAIS "Non trouvé" si le contexte contient un principe général applicable au cas décrit.

CONTEXTE JURIDIQUE :
{context}

QUESTION :
{question}

RÉPONDS STRICTEMENT SOUS CE FORMAT (NE CHANGE AUCUNE BALISE) :

SITUATION : Résume le problème juridique concret identifié.
SOURCE & TEXTE : Cite la loi et l'article exact trouvé dans le contexte (ex: Loi 31-08, Article X). Si aucun texte du contexte ne s'applique, écris "Non trouvé".
EXPLICATION : Explique la règle juridique applicable en faisant le lien direct entre la loi et le cas de l'utilisateur.
DROITS : Liste les droits explicites accordés à l'utilisateur selon le texte.
PROCÉDURE ÉCONOMIQUE : Indique les démarches pratiques et recours possibles (ex: négociation, signalement, association de consommateurs).
ÉVALUATION : Fournis une évaluation globale de la prise en charge de la demande selon le contexte.
"""

    prompt = ChatPromptTemplate.from_template(template_prompt)

    # Étape 5 : Chaîne RAG
    def format_docs(docs):
        textes = [doc.page_content for doc in docs]
        return "\n\n".join(textes)

    def prepare_rag_input(inputs):
        question_text = inputs["question"]
        documents_trouves = retriever.invoke(question_text)
        contexte_formate = format_docs(documents_trouves)
        return {
            "context": contexte_formate,
            "question": question_text
        }

    chain = prepare_rag_input | prompt | llm
    
    return llm, chain, vectorstore

# Chargement des ressources
llm, rag_chain, vectorstore = init_models_and_chain()

# --- GESTION DE L'HISTORIQUE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Si le message possède des métriques d'évaluation
        if "score_vectoriel" in message and "score_juge" in message:
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"🎯 **Confiance Vectorielle :** {message['score_vectoriel']}%")
            with col2:
                badge = "✅ Validé" if message['score_juge'] >= 70 else "⚠️ Incomplet / Non conforme"
                st.caption(f"⚖️ **Score 2 (Juge) :** {message['score_juge']}% ({badge})")
            st.caption(f"💬 *Avis Juge :* {message.get('avis_juge', '')}")

# Détection des salutations
GREETING_PATTERN = r"^(heey|hey|hello|hi|bonjour|bonsoir|salut|salam|coucou|سلام)\b"

# --- INTERACTION UTILISATEUR ---
if user_input := st.chat_input("Posez votre question sur le droit marocain..."):
    
    # 1. Message Utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Réponse de l'Assistant
    with st.chat_message("assistant"):
        clean_input = user_input.strip().lower()
        mots = clean_input.split()

        # ROUTE 1 : Salutation
        if re.match(GREETING_PATTERN, clean_input) or len(mots) <= 2:
            greeting_prompt = f"Tu es un assistant juridique expert en droit marocain. Réponds de manière courtoise et professionnelle à : '{user_input}'"
            response = llm.invoke(greeting_prompt)
            st.write(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})

        # ROUTE 2 : Question Juridique RAG + Évaluation Juge
        else:
            with st.spinner("Analyse des textes de loi et génération de la réponse..."):
                # Calcul du score de confiance vectoriel
                score_vectoriel = calculer_score_confiance(vectorstore, user_input, k=8)
                
                # Exécution de la chaîne RAG
                response = rag_chain.invoke({"question": user_input})
                reponse_rag = response.content
                st.write(reponse_rag)

            # Évaluation en direct par le LLM Juge (Score 2)
            with st.spinner("⚖️ Évaluation de conformité par le LLM Juge..."):
                score_juge, avis_juge = evaluer_score_2_conformite(user_input, reponse_rag)

            st.divider()
            
            # Affichage des deux scores sous la réponse
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 Confiance Contexte Vectoriel", value=f"{score_vectoriel}%")
            with col2:
                st.metric(label="⚖️ Score 2 - Conformité Juge", value=f"{score_juge}%")

            if score_juge >= 70:
                st.success(f"✅ **Réponse Conforme** — {avis_juge}")
            else:
                st.warning(f"⚠️ **Attention** — {avis_juge}")

            # Enregistrement dans l'historique
            st.session_state.messages.append({
                "role": "assistant", 
                "content": reponse_rag,
                "score_vectoriel": score_vectoriel,
                "score_juge": score_juge,
                "avis_juge": avis_juge
            })