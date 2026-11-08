⚖️ **Justice-Tech : Assistant Juridique RAG (Droit Marocain)**

Bienvenue dans la documentation technique du projet **Justice-Tech**, développé dans le cadre d'un Stage de Fin d'Année (PFA) chez **3D Smart Factory**.
Cette application est un système de **Génération Accrue par Récupération (RAG)** conçu pour démocratiser et vulgariser l'accès au droit marocain (Droit du travail, Protection du consommateur, Loi bancaire et Contrats).
---

🛠️ **Architecture & Technologies**
- **Interface Utilisateur :** Streamlit
- **Framework RAG :** LangChain
- **Base de Données Vectorielle :** ChromaDB
- **Modèle d'Embedding :** `BAAI/bge-m3`
- **Moteur d'Inférence / LLM :** `Qwen 27B` (via l'API Groq)
- **Modèle LLM Juge :** `GPT-OSS-120B` (Évaluation automatique)
--

📁 **Structure du Projet**

```text
2.3_2.4_code_source/
├── app.py                  # Application principale Streamlit (Interface & pipeline RAG)
├── ingestion.py            # Script d'ingestion, découpage des textes de loi (chunking) et vectorisation
├── evaluation.py           # Module de contrôle qualité (Score vectoriel + LLM Juge)
├── requirements.txt        # Liste des dépendances Python
├── README.md               # Le présent fichier de documentation
└── .env.example            # Fichier modèle pour les variables d'environnement
