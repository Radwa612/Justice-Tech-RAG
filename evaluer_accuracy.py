import json
import os
import time
import pandas as pd
from groq import Groq

# Importation sécurisée depuis retrieval_pipeline.py
try:
    from retrieval_pipeline import (
        charger_base_vectorielle,
        initialiser_modele_llm,
        construire_chaine_rag,
    )
except ImportError:
    print("❌ Erreur : Impossible d'importer depuis 'retrieval_pipeline.py'. Vérifiez que le fichier existe.")
    def charger_base_vectorielle(): return None
    def initialiser_modele_llm(): return None
    def construire_chaine_rag(r, l): return None

# Configuration de la clé API pour le LLM Juge
os.environ["GROQ_API_KEY"] = "ENTRER LA CLE ICI "
client_juge = Groq()


def evaluer_score_2_conformite(question, reponse_rag, reponse_reference):
    """
    Score 2 : Le LLM Juge évalue la conformité juridique et l'absence d'hallucinations.
    """
    prompt = f"""
Tu es un juge juridique expert en droit marocain.
Ton unique objectif est de vérifier si la réponse du RAG est JURIDIQUEMENT CORRECTE par rapport à la réponse de référence.

QUESTION :
{question}

RÉPONSE DE RÉFÉRENCE (Vérité Terrain) :
{reponse_reference}

RÉPONSE DU RAG À ÉVALUER :
{reponse_rag}

CONSIGNES STRICTES :
1. Ignore la forme ou la structure de la réponse.
2. Vérifie si le fond juridique est exact (articles, règles, délais).
3. Vérifie l'absence d'hallucinations.

Réponds STRICTEMENT sous ce format JSON :
{{
  "score_pourcentage": 88,
  "explication": "Réponse conforme aux textes juridiques, aucun risque d'hallucination."
}}
"""
    max_retries = 3
    for tentative in range(max_retries):
        try:
            response = client_juge.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            resultat = json.loads(response.choices[0].message.content)
            return resultat["score_pourcentage"], resultat["explication"]
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                attente = (tentative + 1) * 5
                print(f"⚠️ Rate Limit pour le Juge (429). Pause de {attente}s...")
                time.sleep(attente)
            else:
                return 0, f"Erreur d'évaluation : {str(e)}"
                
    return 0, "Échec de l'évaluation par le Juge après plusieurs tentatives."


def lancer_benchmark_score_3(nom_fichier="justicetech_test.json", seuil_validation=70):
    """
    Score 3 : Fait tourner tout le dataset de test sur le RAG réel, calcule la Global Accuracy
    et sauvegarde les résultats détaillés en JSON et CSV.
    """
    print("🔄 Initialisation du système RAG...")
    retriever = charger_base_vectorielle()
    llm = initialiser_modele_llm()
    rag_chain = construire_chaine_rag(retriever, llm)

    if rag_chain is None:
        print("❌ Impossible de lancer le benchmark : Le pipeline RAG n'a pas pu être initialisé.")
        return

    # Chemin d'accès au fichier JSON
    dossier_actuel = os.path.dirname(os.path.abspath(__file__))
    chemin_json = os.path.join(dossier_actuel, nom_fichier)

    if not os.path.exists(chemin_json):
        print(f"❌ Erreur : Le fichier '{nom_fichier}' est introuvable dans {dossier_actuel}")
        return

    with open(chemin_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_questions = len(dataset)
    reponses_validees = 0
    details_resultats = []

    print(f"\n🚀 Lancement du Benchmark sur {total_questions} questions...\n" + "=" * 50)

    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        ref = item["reponse_reference"]
        config_session = {"configurable": {"session_id": f"eval_session_{idx}"}}

        print(f"⏳ [{idx}/{total_questions}] Traitement de la question...")

        # 1. Exécution du RAG avec gestion des erreurs / Rate Limits
        reponse_rag = ""
        for tentative in range(3):
            try:
                res_rag = rag_chain.invoke({"question": question}, config=config_session)
                reponse_rag = res_rag.content if hasattr(res_rag, 'content') else str(res_rag)
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    attente = (tentative + 1) * 6
                    print(f"⚠️ Rate Limit RAG (429). Pause de {attente}s...")
                    time.sleep(attente)
                else:
                    reponse_rag = f"Erreur RAG : {e}"
                    break

        time.sleep(3)  # Pause préventive après RAG

        # 2. Évaluation par le LLM Juge
        score_2, explication = evaluer_score_2_conformite(question, reponse_rag, ref)
        
        time.sleep(3)  # Pause préventive après Juge

        # 3. Validation
        est_valide = score_2 >= seuil_validation
        if est_valide:
            reponses_validees += 1

        print(f"Question {idx}/{total_questions} : {question[:60]}...")
        print(f"📊 Score 2 : {score_2}% | Validé : {'✅' if est_valide else '❌'}")
        print(f"💬 Avis Juge : {explication}\n" + "-" * 40)

        # Ajout aux détails pour la sauvegarde
        details_resultats.append({
            "id": item.get("id", idx),
            "domaine": item.get("domaine", "Non spécifié"),
            "question": question,
            "reponse_reference": ref,
            "reponse_rag": reponse_rag,
            "score_conformite": score_2,
            "est_valide": est_valide,
            "avis_juge": explication
        })

    # Calcul global
    global_accuracy = (reponses_validees / total_questions) * 100 if total_questions > 0 else 0

    print("\n" + "=" * 50)
    print(f"🏆 SCORE 3 (GLOBAL ACCURACY) : {global_accuracy:.2f}%")
    print(f"Résultat : {reponses_validees}/{total_questions} questions validées (Seuil >= {seuil_validation}%)")
    print("=" * 50)

    # --- SAUVEGARDE EN JSON ET CSV (Phase 3) ---
    chemin_export_json = os.path.join(dossier_actuel, "resultats_benchmark_detailles.json")
    with open(chemin_export_json, "w", encoding="utf-8") as f_out:
        json.dump({
            "global_accuracy": global_accuracy,
            "total_questions": total_questions,
            "questions_validees": reponses_validees,
            "details": details_resultats
        }, f_out, ensure_ascii=False, indent=4)

    chemin_export_csv = os.path.join(dossier_actuel, "resultats_benchmark_detailles.csv")
    df = pd.DataFrame(details_resultats)
    df.to_csv(chemin_export_csv, index=False, encoding="utf-8-sig")

    print(f"💾 Sauvegarde terminée :\n 📄 {chemin_export_json}\n 📊 {chemin_export_csv}")


if __name__ == "__main__":
    lancer_benchmark_score_3()