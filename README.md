# ComplianceGuard 🛡️

**Assistant juridique intelligent pour les startups en Tunisie** - Système de Question-Réponse basé sur GraphRAG (Graph Retrieval-Augmented Generation) avec fallback web automatique.

## 📋 Description

ComplianceGuard est un système d'IA qui aide les entrepreneurs tunisiens à naviguer dans le cadre réglementaire du **Startup Act** et des lois connexes. Il combine :

- **Recherche vectorielle** (similarité sémantique via Qdrant)
- **Graphe de connaissances** (relations juridiques via Neo4j)
- **Agent Web** (recherche Google + scraping via LangChain)
- **LLM** (Azure OpenAI) pour générer des réponses contextuelles

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ComplianceGuard                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────── ask_question.py ─────────────────────────┐  │
│  │                                                               │  │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │  │
│  │   │   Qdrant    │    │   Neo4j     │    │     Ollama      │   │  │
│  │   │  (Vectors)  │    │  (Graph)    │    │ (qwen3-embedding:0.6b) │   │  │
│  │   └──────┬──────┘    └──────┬──────┘    └────────┬────────┘   │  │
│  │          │                  │                    │            │  │
│  │          └─────────┬────────┴────────────────────┘            │  │
│  │                    │                                          │  │
│  │           ┌────────▼────────┐                                 │  │
│  │           │ Hybrid Retriever│                                 │  │
│  │           │  (GraphRAG)     │                                 │  │
│  │           └────────┬────────┘                                 │  │
│  │                    │                                          │  │
│  │         Contexte suffisant?                                   │  │
│  │              │         │                                      │  │
│  │            OUI        NON ──────────┐                         │  │
│  │              │                      │                         │  │
│  │              │         ┌────────────▼─────────────┐           │  │
│  │              │         │    Web Fallback          │           │  │
│  │              │         │  (Serper + Scraping)     │           │  │
│  │              │         └────────────┬─────────────┘           │  │
│  │              │                      │                         │  │
│  │              └──────────┬───────────┘                         │  │
│  │                         │                                     │  │
│  │                ┌────────▼────────┐                            │  │
│  │                │  Azure OpenAI   │                            │  │
│  │                │  (Llama 4)      │                            │  │
│  │                └────────┬────────┘                            │  │
│  │                         │                                     │  │
│  │                    Réponse                                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────── main.py ───────────────────────────────┐  │
│  │                   Agent Web (LangChain)                       │  │
│  │                                                               │  │
│  │   ┌─────────────────┐    ┌─────────────────┐                  │  │
│  │   │  serper_search  │    │ scrape_website  │                  │  │
│  │   │  (Google API)   │    │ (WebBaseLoader) │                  │  │
│  │   └────────┬────────┘    └────────┬────────┘                  │  │
│  │            │                      │                           │  │
│  │            └──────────┬───────────┘                           │  │
│  │                       │                                       │  │
│  │              ┌────────▼────────┐                              │  │
│  │              │ LLM with Tools  │                              │  │
│  │              │ (bind_tools)    │                              │  │
│  │              └────────┬────────┘                              │  │
│  │                       │                                       │  │
│  │               Rapport (report.md)                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🤖 Agents LangChain

### 1. Agent GraphRAG (`ask_question.py`)

Recherche dans les documents juridiques locaux avec fallback web automatique.

**Outils intégrés :**
- `HybridRetriever` - Combine recherche vectorielle (Qdrant) et traversée de graphe (Neo4j)
- `WebFallback` - Recherche Serper + scraping si contexte local insuffisant

**Fonctionnalités :**
- Détection automatique des salutations/non-questions
- Fallback web transparent
- Sources juridiques tracées

### 2. Agent Web (`main.py` / `chain.py`)

Agent LangChain avec outils bindés pour recherche web active.

**Outils LangChain :**

| Outil | Description |
|-------|-------------|
| `serper_search` | Recherche Google via Serper API (10 résultats) |
| `scrape_website` | Extraction du contenu d'une URL via WebBaseLoader |

**Flux :**
```python
llm = AzureChatOpenAI(**config)
llm_with_tools = llm.bind_tools([serper_search, scrape_website])
# L'agent décide quand appeler chaque outil
```

### 3. Agent Rédacteur (`agent_redacteur.py`)

Génère des documents juridiques adaptés au projet startup.

**Documents disponibles :**

| Document | Description |
|----------|-------------|
| `statuts` | Statuts de société (SUARL, SARL, SA) conformes au Code des Sociétés |
| `cgu` | Conditions Générales d'Utilisation conformes à la loi 2004-63 |
| `contrat_investissement` | Convention d'investissement avec clauses Startup Act |
| `demande_label` | Formulaire de demande du label Startup (Décret 2018-840) |

**Utilisation :**
```powershell
# Générer un seul document
python -m complianceguard.agent_redacteur --nom "MaStartup" --activite "Description" --doc statuts

# Générer le pack complet
python -m complianceguard.agent_redacteur --nom "MaStartup" --activite "Description" \
    --fondateurs "Nom1" "Nom2" --capital 5000 --siege "Tunis" --type SARL --doc all
```

## 📁 Structure du Projet

```
AI project/
├── complianceguard/
│   ├── main.py              # Agent de recherche avec outils (web search, scraping)
│   ├── ask_question.py      # CLI interactif pour questions juridiques (GraphRAG)
│   ├── chain.py             # Chaîne LangChain avec outils bindés
│   ├── config.py            # Configuration centralisée (Pydantic Settings)
│   ├── ingest.py            # Ingestion des PDFs vers Neo4j + Qdrant
│   └── tools/
│       ├── retriever.py     # Retriever hybride (Vector + Graph)
│       ├── graph_agent.py   # Agent de traversée du graphe Neo4j
│       └── custom_tool.py   # Outils personnalisés
├── Data/                    # Documents juridiques sources (PDFs)
│   ├── Loi_2018_20_FR.pdf          # Startup Act
│   ├── Decret_2018_840_Startup.pdf # Décret d'application
│   ├── Circulaire_2019_01_FR.pdf   # Circulaire BCT (devises)
│   └── ...
├── reports/                 # Rapports générés
├── requirements.txt         # Dépendances Python
└── .env                     # Variables d'environnement (non versionné)
```

## 🚀 Installation

### Prérequis

- Python 3.12+
- [Ollama](https://ollama.com/download) (pour les embeddings locaux)
- Compte Neo4j Aura (gratuit)
- Compte Qdrant Cloud (gratuit)
- Clé API Azure OpenAI

### 1. Cloner et créer l'environnement virtuel

```powershell
cd "c:\Users\rarou\Desktop\AI project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
# Azure OpenAI
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=https://your-resource.services.ai.azure.com
AZURE_API_VERSION=2024-05-01-preview
AZURE_MODEL=Llama-4-Maverick-17B-128E-Instruct-FP8

# Neo4j Aura
NEO4J_URI=neo4j+ssc://xxxx.databases.neo4j.io
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password

# Qdrant Cloud
QDRANT_URL=https://xxxx.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=complianceguard_chunks

# Ollama embeddings
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b

# Serper (recherche web)
SERPER_API_KEY=your_serper_key
```

### 4. Télécharger le modèle d'embedding

```powershell
ollama pull qwen3-embedding:0.6b
```

Si vous migrez depuis `bge-m3` (ou un autre modèle), vous devez réindexer les vecteurs Qdrant pour éviter les incompatibilités de dimension :

```powershell
# 1) Supprimer la collection Qdrant existante (Dashboard Qdrant ou API)
# 2) Relancer l'ingestion complète pour recréer les vecteurs avec qwen
python -m complianceguard.ingest
```

## 💻 Utilisation

### Mode Questions-Réponses (GraphRAG + Web Fallback)

```powershell
# Mode interactif (avec fallback web activé par défaut)
python -m complianceguard.ask_question

# Question unique
python -m complianceguard.ask_question -q "Quels documents pour le congé startup ?"

# Sans fallback web (GraphRAG uniquement)
python -m complianceguard.ask_question --no-web
```

**Comportement intelligent :**
- Salutations ("bonjour") → Message d'accueil
- Questions juridiques → GraphRAG puis web si nécessaire
- Questions hors sujet → Réponse appropriée sans sources inutiles

### Mode Agent de Recherche Web (LangChain Tools)

```powershell
python -m complianceguard.main
```

Génère un rapport détaillé (`report.md`) avec :
- Recherche web via Serper API
- Scraping des sources pertinentes
- Validation des liens HTTP (status codes)
- Sources vérifiées et tracées

## 📚 Sources Juridiques Intégrées

| Document | Description |
|----------|-------------|
| Loi n° 2018-20 | Startup Act tunisien |
| Décret n° 2018-840 | Décret d'application du Startup Act |
| Circulaire BCT n° 2019-01 | Comptes en devises pour startups |
| Circulaire BCT n° 2019-02 | Investissements étrangers |
| Code du Travail | Droit du travail tunisien |
| Code des Sociétés | Droit des sociétés commerciales |

## 🔧 Corrections Appliquées

Durant le développement, les corrections suivantes ont été apportées :

1. **chain.py** : `llm` → `self.llm` (variable d'instance)
2. **chain.py** : `ChatOpenAI` → `AzureChatOpenAI` (compatibilité Azure)
3. **.env** : Ajout de `AZURE_MODEL` pour le nom du déploiement
4. **.env** : `neo4j+s://` → `neo4j+ssc://` (SSL Windows fix)

## 🛠️ Technologies

| Catégorie | Technologie | Usage |
|-----------|-------------|-------|
| **Framework** | LangChain | Orchestration des agents et outils |
| **LLM** | Azure OpenAI (Llama 4) | Génération de réponses |
| **Graphe** | Neo4j Aura | Relations juridiques (entités, articles) |
| **Vecteurs** | Qdrant Cloud | Recherche sémantique sur chunks |
| **Embeddings** | Ollama + qwen3-embedding:0.6b | Vectorisation multilingue locale |
| **Web Search** | Serper API | Recherche Google |
| **Scraping** | WebBaseLoader | Extraction de contenu web |
| **Config** | Pydantic Settings | Validation des variables d'environnement |

## 📄 Licence

Projet privé - Usage interne uniquement.

---

*Développé pour faciliter la conformité juridique des startups tunisiennes* 🇹🇳
