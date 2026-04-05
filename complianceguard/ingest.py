import os
import re
import sys
import time
import uuid
import logging
import warnings
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from langchain_openai import AzureChatOpenAI

load_dotenv()

def _suppress_noisy_neo4j_logs() -> None:
    # Les notifications de dépréciation APOC arrivent via les loggers Neo4j,
    # pas via le module warnings.
    for logger_name in (
        "neo4j.notifications",
        "neo4j._sync.work.result",
        "neo4j._async.work.result",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        logger.disabled = True
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())


_suppress_noisy_neo4j_logs()

# Ces warnings Pydantic sont bruyants mais non bloquants pour le pipeline.
warnings.filterwarnings(
    "ignore",
    message=r"PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Received notification from DBMS server:.*apoc\.create\.addLabels.*",
    category=Warning,
)


def _is_transient_neo4j_error(err: str) -> bool:
    """Détecte les erreurs réseau/routage Neo4j qui méritent un retry."""
    msg = (err or "").lower()
    transient_patterns = [
        "unable to retrieve routing information",
        "failed to read from defunct connection",
        "defunct connection",
        "serviceunavailable",
        "connection reset",
        "connection refused",
        "timed out",
        "oerror('no data')",
        "ssl",
    ]
    return any(p in msg for p in transient_patterns)


def _missing_required_apoc_procedures(graph: Neo4jGraph) -> list[str]:
    """Retourne les procédures APOC requises par langchain_neo4j qui sont absentes."""
    required = {
        "apoc.meta.data",
        "apoc.merge.node",
        "apoc.merge.relationship",
        "apoc.create.addLabels",
    }

    query_show = (
        "SHOW PROCEDURES YIELD name "
        "WHERE name IN $required "
        "RETURN collect(name) AS names"
    )
    query_legacy = (
        "CALL dbms.procedures() YIELD name "
        "WHERE name IN $required "
        "RETURN collect(name) AS names"
    )

    available: set[str] = set()
    for q in (query_show, query_legacy):
        try:
            rows = graph.query(q, params={"required": sorted(required)})
            names = rows[0].get("names", []) if rows else []
            available = {str(n) for n in names if n}
            break
        except Exception:
            continue

    return sorted(required - available)

# ── CONFIG ────────────────────────────────────────────────────────────────────

CHUNKS_DIR       = Path(__file__).parent.parent / "chunks"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "complianceguard_chunks")
QDRANT_USER_COLLECTION = os.getenv("QDRANT_USER_COLLECTION_NAME", "user_uploads")
ARTICLE_MAX_CHARS = 2000


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(min_value, value)
    except Exception:
        return default


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
        return max(min_value, value)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


# ── CONNEXIONS ────────────────────────────────────────────────────────────────

def get_llm():
    """
    Retourne le LLM configuré.
    Azure uniquement. Accepte les variables d'environnement :
      - AZURE_OPENAI_* (recommandé)
      - AZURE_* (compatibilité avec l'ancien config.py)
      - model/MODEL (compatibilité Azure AI Foundry)

    temperature=0 pour des extractions déterministes.
    """
    azure_endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        or os.getenv("AZURE_API_BASE", "").strip()
    )
    azure_model_or_deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_MODEL", "").strip()
        or os.getenv("MODEL", "").strip()
        or os.getenv("model", "").strip()
    )
    api_version = (
        os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        or os.getenv("AZURE_API_VERSION", "2024-02-01").strip()
    )
    api_key = (
        os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_API_KEY", "").strip()
    )

    missing: list[str] = []
    if not azure_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT (ou AZURE_API_BASE)")
    if not azure_model_or_deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT (ou AZURE_MODEL/model)")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY (ou AZURE_API_KEY)")

    if missing:
        raise RuntimeError(
            "Configuration Azure OpenAI incomplète: " + ", ".join(missing)
        )

    # Certains .env utilisent "azure/<deployment>". Azure attend seulement le nom du déploiement.
    deployment_name = azure_model_or_deployment.strip()
    if "/" in deployment_name:
        deployment_name = deployment_name.split("/", 1)[1].strip()

    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=deployment_name,
        api_version=api_version,
        api_key=api_key,
        temperature=0,
    )


def get_graph() -> Neo4jGraph:
    # Connexions plus stables avec Aura: keep-alive + rotation plus fréquente
    # des sockets pour limiter les erreurs de connexion "defunct".
    driver_config = {
        "keep_alive": True,
        "max_connection_lifetime": int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "240")),
        "connection_timeout": int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "20")),
        "max_connection_pool_size": int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")),
    }

    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        refresh_schema=False,
        driver_config=driver_config,
    )


def get_ollama_embeddings() -> OllamaEmbeddings:
    model    = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"  Embeddings Ollama : {model}")
    return OllamaEmbeddings(model=model, base_url=base_url)


def _extract_collection_vector_size(collection_info: object) -> int | None:
    """Extrait la dimension vecteur d'une collection Qdrant (single ou multi-vector)."""
    try:
        vectors = collection_info.config.params.vectors  # type: ignore[attr-defined]
    except Exception:
        return None

    if vectors is None:
        return None

    # Single-vector
    if hasattr(vectors, "size"):
        try:
            return int(vectors.size)
        except Exception:
            return None

    # Multi-vector (dict)
    if isinstance(vectors, dict):
        for vp in vectors.values():
            if hasattr(vp, "size"):
                try:
                    return int(vp.size)
                except Exception:
                    continue

    return None


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "").strip()
    if url:
        kwargs: dict = {"url": url}
        key = os.getenv("QDRANT_API_KEY", "").strip()
        if key:
            kwargs["api_key"] = key
        return QdrantClient(**kwargs)
    path = os.getenv("QDRANT_PATH", "").strip() or str(
        Path(__file__).parent.parent / ".qdrant"
    )
    print(f"  Qdrant local : {path}")
    return QdrantClient(path=path)


def ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = True
    collection_info = None
    try:
        collection_info = client.get_collection(collection_name)
    except Exception:
        exists = False

    if recreate and exists:
        print(f"  Recréation de la collection '{collection_name}'...")
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        print(f"  Création de la collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        return

    # Collection existante: vérifier la compatibilité de dimension avec le modèle courant.
    existing_size = _extract_collection_vector_size(collection_info)
    if existing_size is not None and existing_size != vector_size:
        current_model = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
        raise RuntimeError(
            "Dimension embedding incompatible avec la collection Qdrant "
            f"'{collection_name}': collection={existing_size}, modèle({current_model})={vector_size}. "
            "Relancez l'ingestion avec VECTOR_PRE_DELETE_COLLECTION=true ou supprimez/recréez la collection."
        )


def fast_ingest_file(file_path: str | Path) -> dict:
    """
    Ingestion rapide d'un document uploadé utilisateur dans Qdrant.
    N'alimente pas Neo4j et n'impacte pas le pipeline GraphRAG principal.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Import paresseux pour éviter de casser l'import global du module ingest
    # si les utilitaires de parsing ne sont pas installés dans l'environnement.
    try:
        from complianceguard.document_utils import (
            build_document_converter,
            extract_failed_pages,
            fallback_extract_pdf_pages,
            split_semantic_chunks,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Document parsing helpers are not available. Install: pip install \"unstructured[pdf,docx]\""
        ) from exc

    # 1. Parse du document
    converter = build_document_converter()
    conv_result = converter.convert(str(file_path), raises_on_error=False)
    dl_doc = conv_result.document

    conversion_errors = conv_result.errors or []
    failed_pages = extract_failed_pages(conversion_errors)
    if conversion_errors:
        print(
            f"[warn] Le parser a signalé {len(conversion_errors)} erreurs de conversion "
            f"(pages en échec détectées: {len(failed_pages)})."
        )

    # 2. Chunks sémantiques
    docs = []
    semantic_chunks = split_semantic_chunks(
        dl_doc.export_to_markdown(),
        max_chars=ARTICLE_MAX_CHARS,
        overlap=300,
    )

    chunk_index = 0
    for chunk in semantic_chunks:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        seed = f"user_{file_path.name}:{chunk_index}:{content[:80]}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "source_file": file_path.name,
                    "doc_type": "user_upload",
                    "fallback_page": None,
                    "chunk_type": str(chunk.get("chunk_type", "preambule") or "preambule"),
                    "text": content,
                },
            )
        )
        chunk_index += 1

    # 3. Fallback sur pages en échec du parser
    if failed_pages:
        fallback_index = len(docs)
        try:
            fallback_pages = fallback_extract_pdf_pages(file_path, failed_pages)
            for page_num, page_text in fallback_pages:
                for piece in split_semantic_chunks(page_text, max_chars=ARTICLE_MAX_CHARS, overlap=300):
                    content = str(piece.get("content", "")).strip()
                    if not content:
                        continue

                    seed = f"user_fallback_{file_path.name}:{page_num}:{fallback_index}:{content[:80]}"
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "id": chunk_id,
                                "chunk_id": chunk_id,
                                "chunk_index": fallback_index,
                                "source_file": file_path.name,
                                "doc_type": "user_upload",
                                "fallback_page": page_num,
                                "chunk_type": str(piece.get("chunk_type", "preambule") or "preambule"),
                                "text": content,
                            },
                        )
                    )
                    fallback_index += 1
        except Exception as exc:
            print(f"[warn] fallback pypdf impossible pour {file_path.name}: {exc}")

    if not docs:
        return {"status": "error", "message": "No text extracted."}

    # 4. Embeddings + upsert Qdrant
    embeddings = get_ollama_embeddings()
    vector_size = len(embeddings.embed_query("probe"))
    client = get_qdrant_client()

    ensure_qdrant_collection(client, QDRANT_USER_COLLECTION, vector_size, recreate=False)

    vectors = embeddings.embed_documents([d.page_content for d in docs])
    points = []
    for doc, vector in zip(docs, vectors):
        points.append(
            models.PointStruct(
                id=doc.metadata["chunk_id"],
                vector=vector,
                payload=doc.metadata,
            )
        )

    client.upsert(collection_name=QDRANT_USER_COLLECTION, points=points, wait=True)

    return {
        "status": "success",
        "file_name": file_path.name,
        "chunks_indexed": len(docs),
        "collection": QDRANT_USER_COLLECTION,
    }


# ── MÉTADONNÉES PAR FICHIER ───────────────────────────────────────────────────

PDF_META: dict[str, dict] = {
    "Loi_2018_20_FR.pdf": {
        "type": "loi",
        "reference": "Loi n° 2018-20",
        "date": "17 avril 2018",
        "jort": "JORT n°32",
        "domaines": [
            "label_startup", "IS_exoneration", "conge_startup",
            "bourse_startup", "financement", "compte_devises",
        ],
    },
    "Decret_2018_840_Startup.pdf": {
        "type": "decret",
        "reference": "Décret n° 2018-840",
        "date": "11 octobre 2018",
        "jort": "JORT n°84",
        "domaines": [
            "procedure_labelisation", "conditions_label",
            "conge_startup", "bourse_startup",
        ],
    },
    "Circulaire_2019_01_FR.pdf": {
        "type": "circulaire_BCT",
        "reference": "Circulaire BCT n° 2019-01",
        "date": "30 janvier 2019",
        "jort": "BCT",
        "domaines": ["compte_startup_devises", "changes", "levee_fonds"],
    },
    "Circulaire_2019_02_FR.pdf": {
        "type": "circulaire_BCT",
        "reference": "Circulaire BCT n° 2019-02",
        "date": "30 janvier 2019",
        "jort": "BCT",
        "domaines": ["carte_technologique", "transferts_courants"],
    },
    "Code_Societes_Commerciales_FR.pdf": {
        "type": "code",
        "reference": "Code des Sociétés Commerciales",
        "date": "2000",
        "jort": "Loi n°2000-93",
        "domaines": ["SARL", "SA", "SAS", "capital", "statuts"],
    },
    "Code_Droits_Procedures_Fiscaux_2023.pdf": {
        "type": "code",
        "reference": "Code des Droits et Procédures Fiscaux",
        "date": "2023",
        "jort": "JORT",
        "domaines": ["IS", "TVA", "declarations", "controle_fiscal"],
    },
    "Code_Travail_FR.pdf": {
        "type": "code",
        "reference": "Code du Travail",
        "date": "1966",
        "jort": "Loi n°1966-27",
        "domaines": ["contrats_travail", "licenciement", "conges", "salaire"],
    },
    "Loi_63-2004_FR.pdf": {
        "type": "loi",
        "reference": "Loi n° 2004-63",
        "date": "2004",
        "jort": "JORT",
        "domaines": ["protection_donnees", "INPDP", "vie_privee"],
    },
    "Loi_2000-83_FR.pdf": {
        "type": "loi",
        "reference": "Loi n° 2000-83",
        "date": "2000",
        "jort": "JORT",
        "domaines": ["echanges_electroniques", "signature_electronique"],
    },
    "Loi_2016_71_FR.pdf": {
        "type": "loi",
        "reference": "Loi n° 2016-71",
        "date": "2016",
        "jort": "JORT",
        "domaines": ["investissement", "APII", "incitations", "FOPRODI"],
    },
    "Rapport_IC_Startup_Acts_FR.pdf": {
        "type": "rapport",
        "reference": "Rapport IC Startup Acts",
        "date": "2023",
        "jort": None,
        "domaines": ["analyse_comparative", "recommandations", "ecosysteme"],
    },
}

# Mapping chunks pré-générés -> PDF source de référence.
# Permet de conserver les métadonnées juridiques existantes (type, référence, domaines...).
CHUNK_FILE_TO_PDF: dict[str, str] = {
    "Chunks pour RAG - Loi n° 2016-71 (Investissement).md": "Loi_2016_71_FR.pdf",
    "Chunks pour RAG - Loi n° 2018-20 (Startups).md": "Loi_2018_20_FR.pdf",
    "Chunks_startupact.md": "Rapport_IC_Startup_Acts_FR.pdf",
    "Circulaire_2019_01_RAG_chunks.md": "Circulaire_2019_01_FR.pdf",
    "Circulaire_2019_02_RAG_chunks.md": "Circulaire_2019_02_FR.pdf",
    "Code_Droits_Procedures_Fiscaux_RAG_Chunks.md": "Code_Droits_Procedures_Fiscaux_2023.pdf",
    "Code_Societes_Commerciales_RAG_Chunks.md": "Code_Societes_Commerciales_FR.pdf",
    "code_travail_chunked.md": "Code_Travail_FR.pdf",
    "Code_Travail_FR_RAG_chunks.md": "Code_Travail_FR.pdf",
    "loi_2000_83_chunks.md": "Loi_2000-83_FR.pdf",
    "loi_63-2004_rag_chunks.md": "Loi_63-2004_FR.pdf",
    "startup_decret_chunks.md": "Decret_2018_840_Startup.pdf",
}

CHUNK_HEADER_RE = re.compile(
    r"^\s*#{2,6}\s*(?:chunk(?:[_\s-]*\d+)?)\b",
    re.IGNORECASE,
)
CHUNK_INLINE_HEADER_RE = re.compile(
    r"\bchunk[_\s-]*\d+\b",
    re.IGNORECASE,
)
CHUNK_METADATA_LINE_RE = re.compile(
    r"^\*\*(?:id|page\(s\)|tokens_estim[ée]s|overlap_avec|source|document|strat[ée]gie|nombre de chunks)\s*:\*\*",
    re.IGNORECASE,
)
ARTICLE_NUM_RE = re.compile(
    r"\bArt(?:icle)?\.?\s*(\d+(?:\s*(?:bis|ter|quater|quinquies))?)\b",
    re.IGNORECASE,
)
GRAPH_OVERLAP_LINE_RE = re.compile(
    r"^\s*>?\s*\*\(overlap[^)]*\)\*.*$",
    re.IGNORECASE,
)
GRAPH_FOOTER_LINE_RE = re.compile(
    r"(?:journal officiel de la r[ée]publique tunisienne|page\s+\d+|n°\s*\d+)",
    re.IGNORECASE,
)
MULTI_SPACE_RE = re.compile(r"\s{2,}")


# ── ÉTAPE 0 : INITIALISATION NEO4J ───────────────────────────────────────────

# Contraintes d'unicité + index de lookup.
# Les contraintes garantissent que MERGE ne crée pas de doublons.
NEO4J_SETUP_STATEMENTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Loi)        REQUIRE n.reference IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Decret)     REQUIRE n.reference IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Circulaire) REQUIRE n.reference IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Article)    REQUIRE n.article_key IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (n:Article)  ON (n.id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Article)  ON (n.article_key)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Chunk)    ON (n.chunk_id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Avantage) ON (n.id)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id)",
]


def init_neo4j(graph: Neo4jGraph) -> None:
    print("Initialisation Neo4j (contraintes + index)...")
    for stmt in NEO4J_SETUP_STATEMENTS:
        try:
            graph.query(stmt)
        except Exception as e:
            # Déjà existant ou version Neo4j incompatible — non bloquant
            print(f"  [warn] {stmt[:60]}... → {e}")
    print("  Schéma Neo4j prêt ✓")


# ── ÉTAPE 1 : CHARGEMENT DES CHUNKS PRÉ-GÉNÉRÉS ─────────────────────────────

# Détecte les clauses "Vu …" pour l'extraction de références inter-documents.
_VU_RE = re.compile(
    r"Vu\s+(?:la\s+|le\s+|l['']\s*)?(?:loi|décret|code|circulaire|avis)"
    r"(?:[^;.\n]|\n(?!\n))+[;.]",
    re.IGNORECASE,
)
# Extrait un numéro de référence dans une clause "Vu …"
_REF_NUM_RE = re.compile(
    r"(?:loi|décret|circulaire|code)\s+n[°o]?\s*([\d]{2,4}[-/][\d]{2,4})",
    re.IGNORECASE,
)


def _split_chunk_sections(raw_lines: list[str]) -> list[list[str]]:
    """
    Découpe un fichier markdown de chunks en sections.
    Si aucun header de chunk n'est trouvé, tout le contenu devient un seul chunk.
    """
    sections: list[list[str]] = []
    current: list[str] | None = None

    for line in raw_lines:
        if CHUNK_HEADER_RE.match(line):
            if current is not None:
                sections.append(current)
            current = []
            continue
        if current is not None:
            current.append(line)

    if current is not None:
        sections.append(current)

    if not sections:
        return [raw_lines]
    return sections


def _clean_chunk_lines(lines: list[str]) -> list[str]:
    """Nettoie les lignes de structure markdown pour ne garder que le texte utile."""
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(">") or s.startswith("---"):
            continue
        if CHUNK_INLINE_HEADER_RE.search(s) and len(s.split()) <= 4:
            continue
        if CHUNK_METADATA_LINE_RE.match(s):
            continue
        out.append(s)
    return out


def _extract_vu_refs(full_text: str) -> list[str]:
    """
    Retourne les numéros de référence légale cités dans les clauses 'Vu …'.
    Ex : ["2018-20", "2018-840", "2016-35"]
    """
    refs: list[str] = []
    for m in _VU_RE.finditer(full_text):
        for ref_m in _REF_NUM_RE.finditer(m.group()):
            refs.append(ref_m.group(1).strip())
    return list(dict.fromkeys(refs))  # dédupliqué, ordre préservé


def _infer_chunk_type_and_article(
    content: str,
    previous_article_num: str,
) -> tuple[str, str]:
    """Déduit le type de chunk et le numéro d'article éventuel."""
    match = ARTICLE_NUM_RE.search(content)
    if match:
        article_num = re.sub(r"\s+", " ", match.group(1).strip())
        starts_with_article = bool(ARTICLE_NUM_RE.match(content.lstrip()))
        if starts_with_article:
            return "article", article_num
        return "article_fragment", article_num

    if previous_article_num:
        return "article_fragment", previous_article_num

    return "preambule", ""


def _resolve_target_chunk_files(target_files: set[str] | None) -> tuple[list[str], set[str]]:
    """Résout des cibles CLI/.env (pdf ou md) vers des fichiers markdown de chunks."""
    available = {p.name for p in CHUNKS_DIR.glob("*.md")}

    if not target_files:
        default_targets = [name for name in CHUNK_FILE_TO_PDF if name in available]
        return sorted(default_targets), set()

    resolved: set[str] = set()
    unresolved: set[str] = set()

    for requested in target_files:
        name = Path(requested).name

        if name in available:
            resolved.add(name)
            continue

        mapped = [chunk_name for chunk_name, pdf_name in CHUNK_FILE_TO_PDF.items() if pdf_name == name]
        mapped = [chunk_name for chunk_name in mapped if chunk_name in available]
        if mapped:
            resolved.update(mapped)
            continue

        unresolved.add(name)

    return sorted(resolved), unresolved


def load_chunk_files(
    target_files: set[str] | None = None,
) -> tuple[list[Document], dict[str, list[str]]]:
    """
    Charge les chunks pré-générés depuis /chunks (GraphRAG uniquement).
    Aucune extraction PDF ni chunking n'est effectué ici.
    """
    docs: list[Document] = []
    doc_vu_refs: dict[str, list[str]] = {}

    chunk_files, unresolved_targets = _resolve_target_chunk_files(target_files)

    if unresolved_targets:
        for missing in sorted(unresolved_targets):
            print(f"  [Avertissement] Cible introuvable (ni chunk .md ni mapping PDF) : {missing}")

    for chunk_file_name in chunk_files:
        chunk_path = CHUNKS_DIR / chunk_file_name
        if not chunk_path.exists():
            print(f"  [warn] Chunk introuvable : {chunk_file_name}")
            continue

        source_pdf = CHUNK_FILE_TO_PDF.get(chunk_file_name, "")
        meta = PDF_META.get(
            source_pdf,
            {
                "type": "autre",
                "reference": source_pdf or chunk_file_name,
                "date": "",
                "jort": "",
                "domaines": [],
            },
        )
        doc_ref = str(meta.get("reference") or source_pdf or chunk_file_name)

        print(f"  Chargement chunks : {chunk_file_name}")
        raw_text = chunk_path.read_text(encoding="utf-8", errors="ignore")

        vu_refs = _extract_vu_refs(raw_text)
        if vu_refs:
            doc_vu_refs[doc_ref] = vu_refs

        sections = _split_chunk_sections(raw_text.splitlines())
        chunk_defs: list[tuple[str, str, str]] = []  # (content, article_num, chunk_type)
        previous_article_num = ""

        for section in sections:
            cleaned_lines = _clean_chunk_lines(section)
            content = "\n".join(cleaned_lines).strip()
            if not content:
                continue

            chunk_type, article_num = _infer_chunk_type_and_article(content, previous_article_num)
            if article_num:
                previous_article_num = article_num

            chunk_defs.append((content, article_num, chunk_type))

        if not chunk_defs:
            print(f"    [warn] Aucun chunk exploitable détecté dans {chunk_file_name}")
            continue

        for chunk_index, (content, art_num, chunk_type) in enumerate(chunk_defs):
            seed = f"{chunk_file_name}:{art_num}:{chunk_index}:{content[:80]}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": chunk_id,  # Neo4j ID
                        "chunk_id": chunk_id,  # Qdrant ID
                        "chunk_index": chunk_index,
                        "chunk_type": chunk_type,
                        "article_num": art_num if art_num != "0" else "",
                        "fallback_page": None,
                        "source_kind": "prechunked_markdown",
                        "source_file": chunk_file_name,
                        "source_pdf": source_pdf,
                        "doc_type": meta["type"],
                        "reference": doc_ref,
                        "date": meta.get("date", ""),
                        "jort": meta.get("jort") or "",
                        "domaines": ",".join(meta.get("domaines") or []),
                    },
                )
            )

        n_articles = sum(1 for _, _, t in chunk_defs if t == "article")
        n_fragments = sum(1 for _, _, t in chunk_defs if t == "article_fragment")
        n_preambules = sum(1 for _, _, t in chunk_defs if t == "preambule")
        print(
            f"    → {n_articles} articles | {n_fragments} fragments | "
            f"{n_preambules} preambules | {len(chunk_defs)} chunks total"
        )

    if not docs:
        raise ValueError("Aucun chunk chargé. Vérifie le dossier /chunks et les cibles fournies.")

    return docs, doc_vu_refs


def _clean_chunk_text_for_graph(raw_text: str) -> str:
    """Nettoie le bruit fréquent des fichiers de chunks avant extraction LLM."""
    lines: list[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("---") or s.startswith("## CHUNK"):
            continue
        if CHUNK_METADATA_LINE_RE.match(s):
            continue
        if GRAPH_OVERLAP_LINE_RE.match(s):
            continue
        # On garde les lignes utiles mais on supprime les pieds de page/headers courts répétitifs.
        if GRAPH_FOOTER_LINE_RE.search(s) and len(s.split()) <= 12:
            continue
        lines.append(s)

    text = " ".join(lines)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _build_graph_input_docs(docs: list[Document]) -> list[Document]:
    """
    Prépare des documents enrichis pour le LLM à partir des chunks :
    - nettoyage du bruit OCR/chunking,
    - ajout d'un contexte voisin (chunk précédent/suivant),
    - injection des métadonnées explicites (référence, article, source).
    """
    operative_types = {"article", "article_fragment", "paragraphe"}
    by_file: dict[str, list[Document]] = defaultdict(list)

    for d in docs:
        by_file[str(d.metadata.get("source_file", ""))].append(d)

    graph_docs: list[Document] = []

    for file_docs in by_file.values():
        sorted_docs = sorted(file_docs, key=lambda d: int(d.metadata.get("chunk_index", 0) or 0))

        for idx, doc in enumerate(sorted_docs):
            chunk_type = str(doc.metadata.get("chunk_type", "") or "")
            if chunk_type not in operative_types:
                continue

            main_text = _clean_chunk_text_for_graph(doc.page_content)
            if not main_text:
                continue

            article_num = str(doc.metadata.get("article_num", "") or "").strip()
            if article_num:
                article_marker = re.search(
                    rf"\bArt(?:icle)?\.?\s*{re.escape(article_num)}\b",
                    main_text,
                    re.IGNORECASE,
                )
                if article_marker and article_marker.start() > 0:
                    main_text = main_text[article_marker.start():].strip()

            prev_text = ""
            next_text = ""
            if idx > 0:
                prev_text = _clean_chunk_text_for_graph(sorted_docs[idx - 1].page_content)[:450]
            if idx + 1 < len(sorted_docs):
                next_text = _clean_chunk_text_for_graph(sorted_docs[idx + 1].page_content)[:450]

            enriched_content = (
                "META\n"
                f"reference: {doc.metadata.get('reference', '')}\n"
                f"source_pdf: {doc.metadata.get('source_pdf', '')}\n"
                f"source_file: {doc.metadata.get('source_file', '')}\n"
                f"article_num: {article_num or 'inconnu'}\n"
                f"chunk_type: {chunk_type}\n"
                f"chunk_index: {doc.metadata.get('chunk_index', 0)}\n\n"
                "TEXTE_A_EXTRAIRE\n"
                f"{main_text}\n\n"
                "CONTEXTE_AVANT (utiliser uniquement pour compléter une phrase tronquée)\n"
                f"{prev_text}\n\n"
                "CONTEXTE_APRES (utiliser uniquement pour compléter une phrase tronquée)\n"
                f"{next_text}"
            )

            graph_docs.append(
                Document(
                    page_content=enriched_content,
                    metadata=dict(doc.metadata),
                )
            )

    return graph_docs


def _seed_graph_from_chunks(docs: list[Document], graph: Neo4jGraph) -> None:
    """
    Seed Neo4j avec les informations déterministes issues des chunks :
    - noeuds Document/Chunk,
    - ancrage initial Document-[:MENTIONS]->Article via article_num.
    """
    doc_rows: list[dict] = []
    mention_rows: list[dict] = []

    for doc in docs:
        chunk_id = str(doc.metadata.get("chunk_id") or doc.metadata.get("id") or "").strip()
        if not chunk_id:
            continue

        reference = str(
            doc.metadata.get("reference")
            or doc.metadata.get("source_pdf")
            or doc.metadata.get("source_file")
            or ""
        ).strip()

        doc_rows.append(
            {
                "id": chunk_id,
                "text": doc.page_content,
                "reference": reference,
                "source_file": str(doc.metadata.get("source_file", "") or ""),
                "source_pdf": str(doc.metadata.get("source_pdf", "") or ""),
                "doc_type": str(doc.metadata.get("doc_type", "") or ""),
                "chunk_type": str(doc.metadata.get("chunk_type", "") or ""),
                "article_num": str(doc.metadata.get("article_num", "") or ""),
                "chunk_index": int(doc.metadata.get("chunk_index", 0) or 0),
                "domaines": str(doc.metadata.get("domaines", "") or ""),
                "date": str(doc.metadata.get("date", "") or ""),
            }
        )

        article_num = str(doc.metadata.get("article_num", "") or "").strip()
        if article_num and reference:
            mention_rows.append(
                {
                    "chunk_id": chunk_id,
                    "article_num": article_num,
                    "article_key": f"{reference}::article:{article_num}",
                    "reference": reference,
                    "titre_court": f"Article {article_num}",
                }
            )

    if not doc_rows:
        return

    graph.query(
        """
        UNWIND $rows AS row
        MERGE (d:Document {id: row.id})
        SET d.text = row.text,
            d.reference = row.reference,
            d.source_file = row.source_file,
            d.source_pdf = row.source_pdf,
            d.doc_type = row.doc_type,
            d.chunk_type = row.chunk_type,
            d.article_num = row.article_num,
            d.chunk_index = row.chunk_index,
            d.domaines = row.domaines,
            d.date = row.date
        MERGE (c:Chunk {chunk_id: row.id})
        SET c.id = row.id,
            c.chunk_id = row.id,
            c.text = row.text,
            c.reference = row.reference,
            c.source_file = row.source_file,
            c.source_pdf = row.source_pdf,
            c.chunk_type = row.chunk_type,
            c.article_num = row.article_num,
            c.chunk_index = row.chunk_index,
            c.date = row.date
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        params={"rows": doc_rows},
    )

    if mention_rows:
        graph.query(
            """
            UNWIND $rows AS row
            MERGE (a:Article {article_key: row.article_key})
            SET a.id = coalesce(a.id, row.article_key),
                a.numero = row.article_num,
                a.reference = row.reference,
                a.source = row.reference,
                a.titre_court = coalesce(a.titre_court, row.titre_court)
            MERGE (d:Document {id: row.chunk_id})
            MERGE (d)-[:MENTIONS]->(a)
            """,
            params={"rows": mention_rows},
        )

    print(
        f"  Seed Neo4j chunks: {len(doc_rows)} Document/Chunk | "
        f"{len(mention_rows)} liens Document->Article"
    )


# ── ÉTAPE 2 : EXTRACTION DES ENTITÉS JURIDIQUES ──────────────────────────────

# Prompt resserré, orienté qualité plutôt que exhaustivité.
# Règles explicites pour éviter le bruit (boilerplate, numéros non-significatifs).
LEGAL_ENTITY_PROMPT = """
Tu es un juriste expert en droit tunisien chargé de construire un knowledge graph de conformité.
Le texte fourni est un chunk juridique enrichi avec sections META, TEXTE_A_EXTRAIRE,
CONTEXTE_AVANT, CONTEXTE_APRES.

RÈGLES STRICTES — à suivre impérativement :

1. PÉRIMÈTRE : extrais principalement depuis la section TEXTE_A_EXTRAIRE.
    Utilise CONTEXTE_AVANT et CONTEXTE_APRES uniquement pour compléter une phrase
    manifestement tronquée. Ignore le bruit de chunking ("overlap", "CHUNK", "Page",
    "Journal Officiel", champs techniques).

2. MÉTADONNÉES :
    - utilise META.reference comme source juridique par défaut,
    - si META.article_num est renseigné, utilise-le comme numero d'article prioritaire,
    - n'invente jamais de référence légale absente du texte.

3. NŒUDS UTILES SEULEMENT :
   - Un nœud "Loi" ou "Decret" n'est créé que si le texte le MODIFIE ou en DÉPEND directement.
     Une simple citation en référence ne justifie pas un nœud.
   - Un nœud "Montant" ou "Delai" n'est créé que si la valeur est une limite légale
     actionnable (ex: "100.000 DT", "30 jours"). Ignore les numéros d'articles,
     les années, les numéros de page.
   - Un nœud "Organisme" n'est créé que s'il a un rôle actif dans la disposition.

4. DÉDUPLICATION : si la même entité apparaît plusieurs fois, crée UN SEUL nœud.

5. RELATIONS DIRECTES : évite les chaînes A→B→C quand A→C suffit.

6. PROPRIÉTÉS OBLIGATOIRES selon le type :
   - Article  : numero (ex: "14"), titre_court (≤ 5 mots), source (ex: "Circulaire BCT 2019-02")
   - Montant  : valeur (nombre seul), devise ("DT" ou "EUR"), contexte (usage de ce montant)
   - Delai    : duree (nombre seul), unite ("jours", "mois", "ans")
   - Avantage : description (phrase courte résumant le bénéfice)
   - Condition : description (critère précis et quantifié si possible)

Types de nœuds autorisés :
  Article, Loi, Decret, Circulaire, Organisme, Avantage, Obligation, Condition, Delai, Montant

Types de relations autorisés (SANS ACCENTS) :
  PREVOIT    — (Article/Loi/Decret) → (Avantage/Obligation)
  CONDITIONNE — (Condition) → (Avantage)
  CONCERNE   — (Article) → (Organisme)
  MODIFIE    — (Loi/Decret/Circulaire) → (Loi/Decret/Circulaire)
  APPLIQUE   — (Decret/Circulaire) → (Loi)
  DEPEND_DE  — (Avantage) → (Condition)
  FIXE       — (Article) → (Montant/Delai)
"""


def build_graph_from_docs(docs: list[Document], graph: Neo4jGraph) -> None:
    """
    Extraction entités + relations → Neo4j via LLMGraphTransformer.

    Seuls les chunks "article" et "article_fragment" sont envoyés au LLM.
    Les préambules sont exclus : ils ne contiennent que du boilerplate.
    Batch = 5 pour maximiser la précision (1 article ≈ 1 appel LLM).
    """
    missing_apoc = _missing_required_apoc_procedures(graph)
    if missing_apoc:
        print(
            "  [warn] APOC manquant sur Neo4j. "
            "Extraction LLM vers Neo4j ignorée pour cette exécution."
        )
        print("         Procédures absentes: " + ", ".join(missing_apoc))
        print(
            "         Action: recréer le conteneur Neo4j avec APOC activé "
            "(script start-local-stack mis à jour)."
        )
        return

    llm = get_llm()

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Article", "Loi", "Decret", "Circulaire",
            "Organisme", "Avantage", "Obligation",
            "Condition", "Delai", "Montant",
        ],
        allowed_relationships=[
            "PREVOIT", "CONDITIONNE", "CONCERNE",
            "MODIFIE", "APPLIQUE", "DEPEND_DE", "FIXE",
        ],
        node_properties=[
            "description", "valeur", "reference", "date",
            "numero", "titre_court", "source",
            "duree", "unite", "devise", "contexte",
        ],
        relationship_properties=["note"],
        additional_instructions=LEGAL_ENTITY_PROMPT,
    )

    # Seed déterministe depuis les chunks pour stabiliser le graphe de base.
    _seed_graph_from_chunks(docs, graph)

    # Préparer des entrées LLM chunk-aware (texte nettoyé + contexte voisin).
    operative = _build_graph_input_docs(docs)
    excluded  = len(docs) - len(operative)
    print(
        f"\n  Extraction sur {len(operative)} chunks opérationnels "
        f"({excluded} préambules exclus)"
    )

    if not operative:
        print("  [warn] Aucun chunk opérationnel à extraire pour Neo4j.")
        return

    # Accélération pilotable par env:
    # - GRAPH_BATCH_SIZE: taille des lots LLM/Neo4j (défaut 10)
    # - GRAPH_BATCH_THROTTLE_SECONDS: pause entre lots (défaut 0.25s)
    # - GRAPH_MAX_RETRIES: tentatives max par lot (défaut 5)
    batch_size    = _env_int("GRAPH_BATCH_SIZE", 10, min_value=1)
    total_batches = max(1, (len(operative) - 1) // batch_size + 1)
    max_retries   = _env_int("GRAPH_MAX_RETRIES", 5, min_value=1)
    base_throttle = _env_float("GRAPH_BATCH_THROTTLE_SECONDS", 0.25, min_value=0.0)

    for i in range(0, len(operative), batch_size):
        batch     = operative[i: i + batch_size]
        batch_num = i // batch_size + 1
        success   = False
        attempts_used = 0

        for attempt in range(1, max_retries + 1):
            attempts_used = attempt
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
                err = str(e)
                if "DeploymentNotFound" in err:
                    raise RuntimeError(
                        "Azure LLM introuvable (DeploymentNotFound). "
                        "Vérifiez la cohérence endpoint/modele dans .env: "
                        "AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (Azure OpenAI) "
                        "ou AZURE_API_BASE + model/AZURE_MODEL (Azure AI Foundry)."
                    ) from e
                if "429" in err:
                    wait = 2 ** attempt * 5  # 10s, 20s, 40s
                    print(
                        f"  Batch {batch_num} rate-limited "
                        f"(tentative {attempt}/{max_retries}), attente {wait}s..."
                    )
                    time.sleep(wait)
                elif _is_transient_neo4j_error(err):
                    wait = min(90, 2 ** attempt * 5)
                    print(
                        f"  Batch {batch_num} erreur réseau Neo4j "
                        f"(tentative {attempt}/{max_retries}), attente {wait}s..."
                    )
                    time.sleep(wait)
                    try:
                        graph = get_graph()
                    except Exception as reconnect_err:
                        print(f"  [warn] reconnexion Neo4j échouée: {reconnect_err}")
                else:
                    print(f"  Batch {batch_num} erreur : {e}")
                    break  # erreur non-récupérable → passer au batch suivant

        if not success:
            print(f"  Batch {batch_num} abandonné après {attempts_used} tentative(s).")

        # Throttle dynamique: faible si succès direct, plus prudent après retries.
        if base_throttle > 0:
            throttle = base_throttle
            if attempts_used > 1:
                throttle = min(2.0, base_throttle * attempts_used)
            time.sleep(throttle)


def canonicalize_article_nodes(graph: Neo4jGraph) -> None:
    """
    Canonise les noeuds Article pour éviter les collisions inter-documents.

    Règle d'identité:
      article_key = <reference_document>::article:<numero>

    Puis:
      - recopie les relations Article vers le noeud canonique,
            - relabellise les anciens noeuds génériques en :ArticleLegacy,
            - supprime les noeuds legacy (par défaut) pour nettoyer le graphe.
    """
    print("\nNormalisation des noeuds Article (clé canonique par document)...")

    mapping_query = """
    MATCH (d:Document)-[:MENTIONS]->(a:Article)
    WITH d, a,
         trim(coalesce(toString(a.numero), '')) AS a_num,
         trim(coalesce(toString(d.article_num), '')) AS d_num,
         trim(coalesce(toString(a.id), '')) AS a_id,
         trim(coalesce(toString(d.reference), toString(d.source_file), 'Document')) AS doc_ref
    WITH d, a, doc_ref,
         CASE
            WHEN a_num <> '' AND toLower(a_num) <> 'null' THEN a_num
            WHEN d_num <> '' AND toLower(d_num) <> 'null' AND d_num <> 'para' AND d_num <> 'preambule' THEN d_num
            WHEN a_id =~ '(?i)^article\\s+.+$' THEN trim(replace(replace(a_id, 'Article', ''), 'article', ''))
            ELSE ''
         END AS article_num
    WHERE article_num <> ''
    WITH d, a, doc_ref, article_num, doc_ref + '::article:' + article_num AS article_key
    MERGE (ca:Article {article_key: article_key})
    SET ca.id = article_key,
        ca.numero = article_num,
        ca.reference = doc_ref,
        ca.source = CASE
            WHEN coalesce(trim(toString(a.source)), '') IN ['', 'null'] THEN doc_ref
            ELSE trim(toString(a.source))
        END,
        ca.titre_court = CASE
            WHEN coalesce(trim(toString(a.titre_court)), '') = '' THEN 'Article ' + article_num
            ELSE trim(toString(a.titre_court))
        END,
        a.canonical_article_key = article_key
    MERGE (d)-[:MENTIONS]->(ca)
    RETURN count(DISTINCT ca) AS canonical_articles, count(*) AS mapped_mentions
    """

    rel_types = [
        "PREVOIT", "CONCERNE", "FIXE", "MODIFIE",
        "APPLIQUE", "DEPEND_DE", "CONDITIONNE", "REFERENCE",
    ]

    migrated = 0
    for rel in rel_types:
        out_q = f"""
        MATCH (a:Article)-[:{rel}]->(x)
        WHERE a.canonical_article_key IS NOT NULL
        MATCH (ca:Article {{article_key: a.canonical_article_key}})
        MERGE (ca)-[:{rel}]->(x)
        RETURN count(*) AS c
        """
        in_q = f"""
        MATCH (x)-[:{rel}]->(a:Article)
        WHERE a.canonical_article_key IS NOT NULL
        MATCH (ca:Article {{article_key: a.canonical_article_key}})
        MERGE (x)-[:{rel}]->(ca)
        RETURN count(*) AS c
        """
        out_rows = graph.query(out_q)
        in_rows = graph.query(in_q)
        migrated += int((out_rows[0].get("c", 0) if out_rows else 0) or 0)
        migrated += int((in_rows[0].get("c", 0) if in_rows else 0) or 0)

    relabel_query = """
    MATCH (a:Article)
    WHERE a.canonical_article_key IS NOT NULL
      AND (a.id =~ '(?i)^Article\\s+.+$' OR coalesce(trim(toString(a.source)), '') IN ['', 'null'])
    REMOVE a:Article
    SET a:ArticleLegacy
    RETURN count(*) AS relabeled
    """

    delete_legacy_query = """
    MATCH (a:ArticleLegacy)
    DETACH DELETE a
    RETURN count(*) AS deleted
    """

    map_rows = graph.query(mapping_query)
    relabel_rows = graph.query(relabel_query)

    keep_legacy = os.getenv("KEEP_ARTICLE_LEGACY", "").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    deleted = 0
    if not keep_legacy:
        del_rows = graph.query(delete_legacy_query)
        deleted = int((del_rows[0].get("deleted", 0) if del_rows else 0) or 0)

    canonical_articles = int((map_rows[0].get("canonical_articles", 0) if map_rows else 0) or 0)
    mapped_mentions = int((map_rows[0].get("mapped_mentions", 0) if map_rows else 0) or 0)
    relabeled = int((relabel_rows[0].get("relabeled", 0) if relabel_rows else 0) or 0)

    print(
        "  Articles canonisés : "
        f"{canonical_articles} | mentions remappées : {mapped_mentions} | "
        f"relations migrées : {migrated} | anciens noeuds relabellisés : {relabeled}"
    )
    if keep_legacy:
        print("  Legacy conservé (KEEP_ARTICLE_LEGACY=true).")
    else:
        print(f"  Legacy supprimé : {deleted}")


# ── ÉTAPE 3 : LIENS NEXT ENTRE CHUNKS ────────────────────────────────────────

def add_chunk_links(docs: list[Document], graph: Neo4jGraph) -> None:
    """
    Crée des relations NEXT entre chunks consécutifs du même document.
    Permet au retriever de récupérer le contexte autour d'un résultat vectoriel
    sans ré-émettre un appel LLM.
    """
    print("\nCréation des liens NEXT entre chunks...")

    by_file: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        by_file[doc.metadata["source_file"]].append(doc)

    total_links = 0
    for fname, file_docs in by_file.items():
        sorted_docs = sorted(file_docs, key=lambda d: d.metadata["chunk_index"])
        for prev_doc, next_doc in zip(sorted_docs, sorted_docs[1:]):
            try:
                graph.query(
                    """
                    MATCH (a:Document {id: $prev_id})
                    MATCH (b:Document {id: $next_id})
                    MERGE (a)-[:NEXT]->(b)
                    """,
                    params={
                        "prev_id": prev_doc.metadata["chunk_id"],
                        "next_id": next_doc.metadata["chunk_id"],
                    },
                )
                total_links += 1
            except Exception as e:
                print(f"  [warn] lien NEXT {fname} : {e}")

    print(f"  {total_links} liens NEXT créés ✓")


# ── ÉTAPE 4 : INDEX VECTORIEL QDRANT ─────────────────────────────────────────

def build_vector_index(
    docs: list[Document],
    pre_delete_collection: bool = True,
) -> None:
    """
    Indexe tous les chunks dans Qdrant.

    Payload stocké :
      - text         : texte complet (pas tronqué)
      - chunk_id     : lien vers le nœud Document Neo4j
      - article_num  : numéro de l'article source
            - chunk_type   : preambule | article | article_fragment | paragraphe | fallback_page
      - domaines     : tags métier pour le filtrage
    """
    embeddings  = get_ollama_embeddings()
    vector_size = len(embeddings.embed_query("probe"))
    # Accélération pilotable par env:
    # - QDRANT_BATCH_SIZE: taille du lot d'upsert (défaut 128)
    # - QDRANT_UPSERT_WAIT: wait=True/False (défaut true)
    batch_size  = _env_int("QDRANT_BATCH_SIZE", 128, min_value=1)
    upsert_wait = _env_bool("QDRANT_UPSERT_WAIT", True)

    print(f"\nConnexion à Qdrant (collection : {QDRANT_COLLECTION})...")
    client = get_qdrant_client()
    ensure_qdrant_collection(client, QDRANT_COLLECTION, vector_size, pre_delete_collection)

    total_batches = max(1, (len(docs) - 1) // batch_size + 1)
    print(f"Indexation de {len(docs)} chunks...")

    for i in range(0, len(docs), batch_size):
        batch     = docs[i: i + batch_size]
        batch_num = i // batch_size + 1

        vectors = embeddings.embed_documents([d.page_content for d in batch])
        points: list[models.PointStruct] = []

        for doc, vector in zip(batch, vectors):
            chunk_id = str(doc.metadata.get("chunk_id") or uuid.uuid4())
            points.append(models.PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "chunk_id":    chunk_id,
                    "source_file": doc.metadata.get("source_file", ""),
                    "reference":   doc.metadata.get("reference", ""),
                    "doc_type":    doc.metadata.get("doc_type", ""),
                    "chunk_type":  doc.metadata.get("chunk_type", ""),
                    "article_num": doc.metadata.get("article_num", ""),
                    "fallback_page": doc.metadata.get("fallback_page"),
                    "chunk_index": int(doc.metadata.get("chunk_index", 0) or 0),
                    "domaines":    doc.metadata.get("domaines", ""),
                    "date":        doc.metadata.get("date", ""),
                    # Texte COMPLET stocké — pas de troncature à 500 chars
                    "text":        doc.page_content,
                },
            ))

        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=upsert_wait)
        print(f"  Batch Qdrant {batch_num}/{total_batches} ✓")

    print("  Index vectoriel Qdrant ✓")


# ── ÉTAPE 5 : RELATIONS INTER-DOCUMENTS ──────────────────────────────────────

# Relations connues et vérifiées entre les textes du corpus Startup Act.
# Format : (référence_source, TYPE_RELATION, référence_cible, note)
_STATIC_RELATIONS: list[tuple[str, str, str, str]] = [
    (
        "Décret n° 2018-840",
        "APPLIQUE",
        "Loi n° 2018-20",
        "Art. 3,6,7,8,9,10,13 de la Loi 2018-20",
    ),
    (
        "Circulaire BCT n° 2019-01",
        "APPLIQUE",
        "Loi n° 2018-20",
        "Art. 17 — comptes devises startup",
    ),
    (
        "Circulaire BCT n° 2019-02",
        "APPLIQUE",
        "Loi n° 2018-20",
        "Carte Technologique Internationale",
    ),
    (
        "Circulaire BCT n° 2019-01",
        "REFERENCE",
        "Décret n° 2018-840",
        "",
    ),
    (
        "Circulaire BCT n° 2019-02",
        "REFERENCE",
        "Décret n° 2018-840",
        "",
    ),
]

# Map numéro court → (label Neo4j, référence complète)
# Utilisé pour résoudre les références extraites dynamiquement des "Vu …".
_KNOWN_REFS: dict[str, tuple[str, str]] = {
    "2018-20":  ("Loi",        "Loi n° 2018-20"),
    "840-2018": ("Decret",     "Décret n° 2018-840"),
    "2018-840": ("Decret",     "Décret n° 2018-840"),
    "2019-01":  ("Circulaire", "Circulaire BCT n° 2019-01"),
    "2019-02":  ("Circulaire", "Circulaire BCT n° 2019-02"),
    "2016-35":  ("Loi",        "Loi n° 2016-35"),
}

_DOC_LABEL_BY_TYPE: dict[str, str] = {
    "loi": "Loi",
    "decret": "Decret",
    "circulaire_BCT": "Circulaire",
    "code": "Code",
    "rapport": "Rapport",
}

# Référence canonique -> label Neo4j canonique
_REFERENCE_LABELS: dict[str, str] = {
    meta["reference"]: _DOC_LABEL_BY_TYPE.get(meta.get("type", ""), "Document")
    for meta in PDF_META.values()
}

for _, (lbl, full_ref) in _KNOWN_REFS.items():
    _REFERENCE_LABELS.setdefault(full_ref, lbl)


def _label_for_reference(reference: str) -> str:
    return _REFERENCE_LABELS.get(reference, "Document")


def add_inter_doc_relations(
    graph: Neo4jGraph,
    doc_vu_refs: dict[str, list[str]],
) -> None:
    """
    Construit les relations entre documents dans Neo4j :
      1. Relations statiques vérifiées (corpus Startup Act)
      2. Relations dynamiques extraites des clauses "Vu …" de chaque document
    """
    print("\nRelations inter-documents...")

    # 1. Statiques
    for src_ref, rel_type, tgt_ref, note in _STATIC_RELATIONS:
        src_label = _label_for_reference(src_ref)
        tgt_label = _label_for_reference(tgt_ref)
        try:
            graph.query(
                f"""
                MERGE (src:{src_label} {{reference: $src}})
                MERGE (tgt:{tgt_label} {{reference: $tgt}})
                MERGE (src)-[:{rel_type} {{note: $note}}]->(tgt)
                """,
                params={"src": src_ref, "tgt": tgt_ref, "note": note},
            )
        except Exception as e:
            print(f"  [warn] relation statique {src_ref} → {tgt_ref} : {e}")

    # 2. Dynamiques (depuis "Vu …")
    dynamic_count = 0
    for citing_ref, cited_numbers in doc_vu_refs.items():
        src_label = _label_for_reference(citing_ref)
        for num in cited_numbers:
            if num not in _KNOWN_REFS:
                continue  # référence inconnue, on ignore
            tgt_label, cited_full_ref = _KNOWN_REFS[num]
            if cited_full_ref == citing_ref:
                continue  # pas d'auto-référence
            try:
                graph.query(
                    f"""
                    MERGE (src:{src_label} {{reference: $citing}})
                    MERGE (tgt:{tgt_label} {{reference: $cited}})
                    MERGE (src)-[:REFERENCE {{note: 'extrait_preambule'}}]->(tgt)
                    """,
                    params={"citing": citing_ref, "cited": cited_full_ref},
                )
                dynamic_count += 1
            except Exception as e:
                print(f"  [warn] relation dynamique {citing_ref} → {cited_full_ref} : {e}")

    print(
        f"  {len(_STATIC_RELATIONS)} statiques + {dynamic_count} dynamiques créées ✓"
    )


# ── UTILS CLI ─────────────────────────────────────────────────────────────────

def parse_target_files(cli_args: list[str]) -> set[str] | None:
    """
    Construit la liste des cibles d'ingestion depuis :
      1. Arguments CLI : python ingest.py file1.pdf file2.pdf ou chunk1.md
      2. Variable .env : INGEST_ONLY_FILES=file1.pdf,chunk1.md
    Retourne None si aucun ciblage → tous les chunks pré-mappés seront traités.
    """
    selected: set[str] = set()
    env_value = os.getenv("INGEST_ONLY_FILES", "").strip()
    if env_value:
        selected.update(p.strip() for p in env_value.split(",") if p.strip())
    if cli_args:
        selected.update(Path(a).name for a in cli_args if a.strip())
    return selected or None


# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────

def run_ingestion() -> None:
    print("=" * 60)
    print("ComplianceGuard — Pipeline d'ingestion GraphRAG v2")
    print("=" * 60)

    graph = get_graph()

    # ── 0. Schéma Neo4j ───────────────────────────────────────────
    print("\n[0/5] Initialisation Neo4j...")
    init_neo4j(graph)

    # ── Sélection des fichiers ────────────────────────────────────
    target_files = parse_target_files(sys.argv[1:])
    if target_files:
        print("\nMode ingestion ciblée :")
        for n in sorted(target_files):
            print(f"  - {n}")

    # ── 1. Chargement des chunks pré-générés ──────────────────────
    print("\n[1/5] Chargement des chunks pré-générés...")
    docs, doc_vu_refs = load_chunk_files(target_files=target_files)
    by_type: dict[str, int] = defaultdict(int)
    for d in docs:
        by_type[d.metadata["chunk_type"]] += 1
    print(f"  Total : {len(docs)} chunks")
    for ctype, count in sorted(by_type.items()):
        print(f"    {ctype:<20} : {count}")

    # ── 2. Extraction entités → Neo4j ─────────────────────────────
    print("\n[2/5] Extraction des entités juridiques → Neo4j...")
    build_graph_from_docs(docs, graph)
    canonicalize_article_nodes(graph)

    # ── 3. Liens NEXT entre chunks ────────────────────────────────
    print("\n[3/5] Liens de séquence (NEXT) dans Neo4j...")
    add_chunk_links(docs, graph)

    # ── 4. Index vectoriel Qdrant ─────────────────────────────────
    print("\n[4/5] Index vectoriel Qdrant...")
    pre_delete_env = os.getenv("VECTOR_PRE_DELETE_COLLECTION", "").strip().lower()
    if pre_delete_env:
        pre_delete = pre_delete_env in {"1", "true", "yes", "y", "on"}
    else:
        pre_delete = target_files is None  # plein rechargement = recréer, mode ciblé = conserver
    if not pre_delete:
        print("  Mode incrémental : collection Qdrant existante conservée.")
    build_vector_index(docs, pre_delete_collection=pre_delete)

    # ── 5. Relations inter-documents ──────────────────────────────
    print("\n[5/5] Relations inter-documents...")
    add_inter_doc_relations(graph, doc_vu_refs)

    # ── Résumé ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ingestion terminée avec succès !")
    print(f"  Chunks générés  : {len(docs)}")
    print(f"  Neo4j           : graphe peuplé + liens NEXT + relations inter-docs")
    print(f"  Qdrant          : collection '{QDRANT_COLLECTION}' indexée")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()