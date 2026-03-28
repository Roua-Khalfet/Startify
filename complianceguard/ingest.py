"""
ComplianceGuard — ingest.py
============================
Pipeline d'ingestion des PDFs juridiques tunisiens vers Neo4j GraphRAG.
Lit tous les PDFs dans /Data, les découpe en chunks, extrait les entités
juridiques avec un LLM, et construit le knowledge graph dans Neo4j.

Usage:
    python ingest.py

Prérequis (requirements.txt):
    langchain langchain-community langchain-openai
    langchain-experimental neo4j pypdf
    python-dotenv tiktoken
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "Data"

# Métadonnées officielles par fichier — utilisées pour enrichir les nœuds
PDF_META = {
    "Loi_2018_20_FR.pdf": {
        "type": "loi", "reference": "Loi n° 2018-20",
        "date": "17 avril 2018", "jort": "JORT n°32",
        "domaines": ["label_startup", "IS_exoneration", "conge_startup",
                     "bourse_startup", "financement", "compte_devises"],
    },
    "Decret_2018_840_Startup.pdf": {
        "type": "decret", "reference": "Décret n° 2018-840",
        "date": "11 octobre 2018", "jort": "JORT n°84",
        "domaines": ["procedure_labelisation", "conditions_label",
                     "conge_startup", "bourse_startup"],
    },
    "Circulaire_2019_01_FR.pdf": {
        "type": "circulaire_BCT", "reference": "Circulaire BCT n° 2019-01",
        "date": "30 janvier 2019", "jort": "BCT",
        "domaines": ["compte_startup_devises", "changes", "levee_fonds"],
    },
    "Circulaire_2019_02_FR.pdf": {
        "type": "circulaire_BCT", "reference": "Circulaire BCT n° 2019-02",
        "date": "30 janvier 2019", "jort": "BCT",
        "domaines": ["carte_technologique", "transferts_courants"],
    },
    "Code_Societes_Commerciales_FR.pdf": {
        "type": "code", "reference": "Code des Sociétés Commerciales",
        "date": "2000", "jort": "Loi n°2000-93",
        "domaines": ["SARL", "SA", "SAS", "capital", "statuts"],
    },
    "Code_Droits_Procedures_Fiscaux_2023.pdf": {
        "type": "code", "reference": "Code des Droits et Procédures Fiscaux",
        "date": "2023", "jort": "JORT",
        "domaines": ["IS", "TVA", "declarations", "controle_fiscal"],
    },
    "Code_Travail_FR.pdf": {
        "type": "code", "reference": "Code du Travail",
        "date": "1966", "jort": "Loi n°1966-27",
        "domaines": ["contrats_travail", "licenciement", "conges", "salaire"],
    },
    "Loi_63-2004_FR.pdf": {
        "type": "loi", "reference": "Loi n° 2004-63",
        "date": "2004", "jort": "JORT",
        "domaines": ["protection_donnees", "INPDP", "vie_privee"],
    },
    "Loi_2000-83_FR.pdf": {
        "type": "loi", "reference": "Loi n° 2000-83",
        "date": "2000", "jort": "JORT",
        "domaines": ["echanges_electroniques", "signature_electronique"],
    },
    "Loi_2016_71_FR.pdf": {
        "type": "loi", "reference": "Loi n° 2016-71",
        "date": "2016", "jort": "JORT",
        "domaines": ["investissement", "APII", "incitations", "FOPRODI"],
    },
    "Rapport_IC_Startup_Acts_FR.pdf": {
        "type": "rapport", "reference": "Rapport IC Startup Acts",
        "date": "2023", "jort": None,
        "domaines": ["analyse_comparative", "recommandations", "ecosysteme"],
    },
}

# ── CONNEXION NEO4J ───────────────────────────────────────────────────────────

def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


# ── ÉTAPE 1 : CHARGEMENT ET CHUNKING DES PDFs ────────────────────────────────

def load_pdfs() -> list[Document]:
    """
    Charge tous les PDFs du dossier /Data avec leurs métadonnées.
    Retourne une liste de Documents LangChain.
    """
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\nArt", "\n\nArticle", "\n\n", "\n", " "],
        # On coupe en priorité aux articles pour préserver le contexte légal
    )

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        fname = pdf_path.name
        meta = PDF_META.get(fname, {
            "type": "autre", "reference": fname, "date": "", "jort": "", "domaines": []
        })

        print(f"  Chargement : {fname}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Enrichir les métadonnées de chaque page
        for page in pages:
            page.metadata.update({
                "source_file": fname,
                "doc_type": meta["type"],
                "reference": meta["reference"],
                "date": meta["date"],
                "jort": meta.get("jort") or "",
                "domaines": ",".join(meta.get("domaines") or []),
            })

        chunks = splitter.split_documents(pages)
        docs.extend(chunks)
        print(f"    → {len(pages)} pages, {len(chunks)} chunks")

    return docs


# ── ÉTAPE 2 : EXTRACTION DES ENTITÉS JURIDIQUES ──────────────────────────────

LEGAL_ENTITY_PROMPT = """
Tu es un expert en droit tunisien. Extrait les entités et relations juridiques 
du texte suivant pour construire un knowledge graph.

Types de nœuds à identifier :
- Article (ex: "Article 3", "Art. 19")
- Loi (ex: "Loi n° 2018-20")
- Organisme (ex: "APII", "BCT", "CNSS", "Comité de labélisation")
- Avantage (ex: "exonération IS", "congé startup", "bourse startup")
- Obligation (ex: "dépôt dossier", "tenue comptabilité")
- Condition (ex: "moins de 100 salariés", "8 ans d'existence")
- Délai (ex: "30 jours", "6 mois")
- Montant (ex: "1000 DT", "15 millions DT")

Relations à identifier :
- PRÉVOIT (loi → avantage/obligation)
- CONDITIONNE (condition → avantage)
- CONCERNE (article → organisme)
- MODIFIE (loi/décret → loi antérieure)
- APPLIQUE (décret → loi mère)
- DÉPEND_DE (avantage → condition)
"""

def build_graph_from_docs(docs: list[Document], graph: Neo4jGraph):
    """
    Utilise LLMGraphTransformer pour extraire entités et relations
    et les insérer dans Neo4j.
    """
    # Groq API (ultra-fast inference, compatible OpenAI)
    # Using GPT-OSS 120B which supports json_schema structured outputs
    # (llama-3.3-70b-versatile does NOT support structured outputs)
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=8192,  # Increase output limit for complex entity extraction
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Article", "Loi", "Decret", "Circulaire",
            "Organisme", "Avantage", "Obligation",
            "Condition", "Delai", "Montant", "Domaine"
        ],
        allowed_relationships=[
            "PREVOIT", "PRÉVOIT",  # both ASCII and accented versions
            "CONDITIONNE", "CONCERNE",
            "MODIFIE", "APPLIQUE", "DEPEND_DE",
            "COUVRE", "IMPLIQUE", "REFERENCE"
        ],
        node_properties=["description", "valeur", "reference", "date"],
        relationship_properties=["note"],
        additional_instructions=LEGAL_ENTITY_PROMPT,
    )

    print(f"\nExtraction des entités sur {len(docs)} chunks...")
    batch_size = 10
    total_batches = (len(docs) - 1) // batch_size + 1
    max_retries = 3

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                graph_docs = transformer.convert_to_graph_documents(batch)
                graph.add_graph_documents(
                    graph_docs,
                    baseEntityLabel=True,
                    include_source=True,
                )
                print(f"  Batch {batch_num}/{total_batches} ✓")
                success = True
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait = 2 ** attempt * 3  # 6s, 12s, 24s
                    print(f"  Batch {batch_num} rate-limited (tentative {attempt}/{max_retries}), "
                          f"attente {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Batch {batch_num} erreur: {e}")
                    break

        if not success:
            print(f"  Batch {batch_num} ignorée après échecs.")

        # Pause pour Groq rate limit (30 req/min free tier)
        time.sleep(3)


# ── ÉTAPE 3 : INDEX VECTORIEL NEO4J ──────────────────────────────────────────

def build_vector_index(docs: list[Document]) -> Neo4jVector:
    """
    Crée l'index vectoriel dans Neo4j pour la recherche sémantique.
    Combine avec le graph pour le GraphRAG hybride.
    """
    # Embeddings locaux avec HuggingFace (multilingual pour le français)
    print("  Chargement du modele d'embeddings local...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    print("\nCréation de l'index vectoriel Neo4j...")
    vector_index = Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        index_name="complianceguard_chunks",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
        pre_delete_collection=True,  # Recrée l'index à chaque ingest
    )
    print("  Index vectoriel créé ✓")
    return vector_index


# ── ÉTAPE 4 : RELATIONS INTER-DOCUMENTS ──────────────────────────────────────

INTER_DOC_RELATIONS = """
MERGE (l1:Loi {reference: 'Loi n° 2018-20'})
MERGE (d1:Decret {reference: 'Décret n° 2018-840'})
MERGE (c1:Circulaire {reference: 'Circulaire BCT n° 2019-01'})
MERGE (c2:Circulaire {reference: 'Circulaire BCT n° 2019-02'})
MERGE (d1)-[:APPLIQUE {note: 'Décret pris en application Art. 3,6,7,8,9,10,13'}]->(l1)
MERGE (c1)-[:APPLIQUE {note: 'Pris en application Art. 17 Loi 2018-20'}]->(l1)
MERGE (c2)-[:APPLIQUE {note: 'Pris en application Loi 2018-20'}]->(l1)
MERGE (c1)-[:REFERENCE]->(d1)
MERGE (c2)-[:REFERENCE]->(d1)
"""

def add_inter_doc_relations(graph: Neo4jGraph):
    """Ajoute les relations manuelles entre les textes officiels."""
    print("\nAjout des relations inter-documents...")
    graph.query(INTER_DOC_RELATIONS)
    print("  Relations inter-documents ✓")


# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────

def run_ingestion():
    print("=" * 55)
    print("ComplianceGuard — Pipeline d'ingestion GraphRAG")
    print("=" * 55)

    graph = get_graph()

    # 1. Charger les PDFs
    print("\n[1/4] Chargement et chunking des PDFs...")
    docs = load_pdfs()
    print(f"  Total: {len(docs)} chunks générés\n")

    # 2. Construire le knowledge graph
    print("[2/4] Extraction des entités juridiques → Neo4j...")
    build_graph_from_docs(docs, graph)

    # 3. Index vectoriel
    print("\n[3/4] Construction de l'index vectoriel...")
    build_vector_index(docs)

    # 4. Relations inter-documents
    print("\n[4/4] Relations inter-documents...")
    add_inter_doc_relations(graph)

    print("\n" + "=" * 55)
    print("Ingestion terminée avec succès !")
    print(f"  {len(docs)} chunks indexés")
    print("  Knowledge graph Neo4j peuplé")
    print("  Index vectoriel 'complianceguard_chunks' créé")
    print("=" * 55)


if __name__ == "__main__":
    run_ingestion()