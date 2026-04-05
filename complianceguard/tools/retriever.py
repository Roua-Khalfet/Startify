"""
ComplianceGuard — retriever.py
================================
Retriever hybride GraphRAG : combine recherche vectorielle et
traversée du knowledge graph Neo4j pour des réponses juridiques précises.

Deux modes :
  - VectorRetriever     : similarité sémantique sur les chunks
  - GraphRetriever      : traversée du graphe de relations légales
  - HybridRetriever     : fusion des deux (recommandé pour ComplianceGuard)
"""

import os
from complianceguard.config import config, get_azure_llm_kwargs, get_ollama_embed_kwargs
from typing import Any, List
from pathlib import Path
import re
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

load_dotenv()


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(min_value, value)
    except Exception:
        return default


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

# ── CONNEXIONS ────────────────────────────────────────────────────────────────

def get_qdrant_client() -> QdrantClient:
    qdrant_url = config.QDRANT_URL.strip()
    qdrant_api_key = config.QDRANT_API_KEY.strip()
    if qdrant_url:
        kwargs = {"url": qdrant_url}
        if qdrant_api_key:
            kwargs["api_key"] = qdrant_api_key
        return QdrantClient(**kwargs)

    qdrant_path = config.QDRANT_PATH.strip() or str(Path(__file__).resolve().parents[2] / ".qdrant")
    return QdrantClient(path=qdrant_path)


def get_embeddings_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(**get_ollama_embed_kwargs())


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
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        refresh_schema=False,
        driver_config=driver_config,
    )


def _extract_collection_vector_size(collection_info: object) -> int | None:
    """Extrait la dimension vecteur d'une collection Qdrant (single ou multi-vector)."""
    try:
        vectors = collection_info.config.params.vectors  # type: ignore[attr-defined]
    except Exception:
        return None

    if vectors is None:
        return None

    if hasattr(vectors, "size"):
        try:
            return int(vectors.size)
        except Exception:
            return None

    if isinstance(vectors, dict):
        for vp in vectors.values():
            if hasattr(vp, "size"):
                try:
                    return int(vp.size)
                except Exception:
                    continue

    return None


class ComplianceGuardRetriever(BaseRetriever):
    """
    Retriever hybride pour ComplianceGuard.
    Combine :
      1. Recherche vectorielle (similarité sémantique sur chunks)
      2. Traversée du graphe Neo4j (relations juridiques)
      3. Re-ranking par pertinence légale
    """

    qdrant_client: QdrantClient
    embeddings: OllamaEmbeddings
    qdrant_collection: str
    graph: Neo4jGraph
    k_vector: int = 4       # Nombre de chunks vectoriels
    k_graph: int = 5        # Nombre de résultats graph
    search_mode: str = "all" # "all", "kb", "notebook"

    class Config:
        arbitrary_types_allowed = True

    def _run_graph_query_with_retry(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> list[dict[str, Any]]:
        """Exécute une requête graph avec retry/reconnexion sur erreurs transitoires."""
        params = params or {}
        delay_s = 0.8

        for attempt in range(1, retries + 1):
            try:
                return self.graph.query(cypher, params=params)
            except Exception as e:
                err = str(e)
                is_last = attempt >= retries
                if is_last or not _is_transient_neo4j_error(err):
                    raise

                print(
                    f"[GraphSearch] Neo4j transient error (attempt {attempt}/{retries}): {err}. "
                    f"Retry in {delay_s:.1f}s"
                )
                time.sleep(delay_s)
                delay_s = min(delay_s * 1.8, 4.0)

                # Recréation explicite du client graph pour éviter de réutiliser
                # une connexion/sockets déjà marquées defunct.
                try:
                    self.graph = get_graph()
                except Exception:
                    pass

        return []

    def _collection_compatible(self, collection_name: str, query_dim: int) -> bool:
        """
        Vérifie qu'une collection Qdrant existe et est compatible avec la
        dimension du modèle d'embedding actif.
        """
        try:
            info = self.qdrant_client.get_collection(collection_name)
        except Exception:
            return False

        collection_dim = _extract_collection_vector_size(info)
        if collection_dim is None:
            return True

        if collection_dim != query_dim:
            print(
                f"[QdrantSearch] Collection '{collection_name}' ignorée: "
                f"dimension={collection_dim}, embedding={query_dim} "
                f"(modèle={config.OLLAMA_EMBED_MODEL})."
            )
            return False

        return True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        docs = []

        # 1. Recherche vectorielle
        vector_docs = self._vector_search(query)
        docs.extend(vector_docs)

        # 2. Recherche dans le graphe Neo4j via Cypher
        graph_docs = self._graph_search(query)
        docs.extend(graph_docs)

        # 3. Dédoublonnage et tri par score de pertinence légale
        seen, unique = set(), []
        for d in docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(d)

        # Laisse de la place aux preuves relationnelles (paths + summary) ajoutées
        # après les résultats vectoriels pour mieux couvrir les requêtes multi-hop.
        return unique[: self.k_vector + self.k_graph + 6]

    def _qdrant_search(self, query_vector: list[float], collection_name: str) -> list[Any]:
        """Compatibilité qdrant-client: search (legacy) et query_points (new)."""
        if hasattr(self.qdrant_client, "search"):
            return self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=self.k_vector,
                with_payload=True,
            )

        result = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=self.k_vector,
            with_payload=True,
        )
        return list(getattr(result, "points", []) or [])

    def _fetch_doc_by_id(self, doc_id: str, payload: dict[str, Any], score: float) -> Document | None:
        doc_query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (d)-[:MENTIONS]->(e)
        RETURN d.text AS text,
               d.reference AS reference,
               d.source_file AS source_file,
               collect(DISTINCT coalesce(e.description, e.id))[0..5] AS entities
        LIMIT 1
        """
        chunk_query = """
        MATCH (c:Chunk)
        WHERE c.id = $doc_id OR c.chunk_id = $doc_id
        RETURN c.text AS text,
               c.reference AS reference,
               c.source_file AS source_file,
               [] AS entities
        LIMIT 1
        """

        try:
            rows = self._run_graph_query_with_retry(doc_query, params={"doc_id": doc_id})
            if not rows:
                rows = self._run_graph_query_with_retry(chunk_query, params={"doc_id": doc_id})
        except Exception as e:
            print(f"[GraphSearch] Lookup doc_id={doc_id} échoué: {e}")
            rows = []

        if rows:
            row = rows[0]
            text = row.get("text") or ""
            reference = row.get("reference") or payload.get("reference", "")
            source_file = row.get("source_file") or payload.get("source_file", "")
            entities = [e for e in (row.get("entities") or []) if e]
        else:
            text = payload.get("text", payload.get("text_snippet", ""))
            reference = payload.get("reference", "")
            source_file = payload.get("source_file", "")
            entities = []

        if not text:
            return None

        if entities:
            text = f"{text}\nEntités liées: {', '.join(entities)}"

        return Document(
            page_content=text,
            metadata={
                "retrieval_source": "vector_qdrant",
                "reference": reference,
                "source_file": source_file,
                "doc_id": doc_id,
                "score": score,
            },
        )

    def _vector_search(self, query: str) -> List[Document]:
        """Recherche vectorielle dans Qdrant (Corpus + User Uploads) puis Neo4j."""
        try:
            query_vector = self.embeddings.embed_query(query)
            query_dim = len(query_vector)

            results_corpus = []
            if self.search_mode in ["all", "kb"]:
                if self._collection_compatible(self.qdrant_collection, query_dim):
                    results_corpus = self._qdrant_search(query_vector, self.qdrant_collection)
            
            user_collection = os.getenv("QDRANT_USER_COLLECTION_NAME", "user_uploads")
            results_user = []
            if self.search_mode in ["all", "notebook"]:
                try:
                    if self._collection_compatible(user_collection, query_dim):
                        results_user = self._qdrant_search(query_vector, user_collection)
                except Exception:
                    pass
            
            results = results_corpus + results_user
            results = sorted(results, key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)[:self.k_vector]
        except Exception as e:
            print(f"[QdrantSearch] Erreur: {e}")
            return []

        docs = []
        for item in results:
            payload = getattr(item, "payload", None) or {}
            score = float(getattr(item, "score", 0.0) or 0.0)

            # User uploads live only in Qdrant — skip Neo4j lookup
            if payload.get("doc_type") == "user_upload":
                text = payload.get("text", "")
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "retrieval_source": "vector_qdrant_user",
                            "reference": payload.get("source_file", ""),
                            "source_file": payload.get("source_file", ""),
                            "doc_id": payload.get("chunk_id", str(getattr(item, "id", ""))),
                            "score": score,
                            "doc_type": "user_upload",
                        },
                    ))
                continue

            doc_id = str(payload.get("doc_id") or getattr(item, "id", ""))
            if not doc_id:
                continue

            doc = self._fetch_doc_by_id(doc_id, payload, score)
            if doc:
                docs.append(doc)

        return docs

    def _graph_search(self, query: str) -> List[Document]:
        """
        Recherche dans le graphe Neo4j.
        Extrait les nœuds liés à la question via des requêtes Cypher ciblées.
        """
        if self.search_mode == "notebook":
            return [] # Pas de recherche dans la base de graphe pour les uploads locaux
            
        # Requête générale : trouve les avantages et obligations liés à la question
        cypher = """
        CALL db.index.fulltext.queryNodes('legal_entities', $query)
        YIELD node, score
        WITH node, score
        ORDER BY score DESC
        LIMIT $limit
        OPTIONAL MATCH (node)-[r]->(related)
         RETURN coalesce(node.description, node.valeur, node.reference, node.id) AS description,
             coalesce(node.reference, node.id) AS reference,
               labels(node) AS types,
             collect({
                 type: type(r),
                 target: coalesce(related.description, related.valeur, related.reference, related.id)
             }) AS relations,
               score
        """
        try:
            results = self._run_graph_query_with_retry(
                cypher,
                params={"query": query, "limit": self.k_graph}
            )
            docs = []
            for r in results:
                if r.get("description"):
                    content = f"[{', '.join(r['types'])}] {r['reference']}: {r['description']}"
                    if r.get("relations"):
                        rels = "; ".join(
                            f"{rel['type']} → {rel['target']}"
                            for rel in r["relations"][:3]
                            if rel.get("target")
                        )
                        if rels:
                            content += f"\nRelations: {rels}"
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "retrieval_source": "graph",
                            "reference": r.get("reference", ""),
                            "score": r.get("score", 0),
                        }
                    ))

            # Fallback relationnel: injecte des chemins explicites src-[REL]->tgt
            # pour renforcer les requêtes multi-hop orientées inter-documents.
            rel_types = [
                "APPLIQUE", "REFERENCE", "MODIFIE",
                "PREVOIT", "CONDITIONNE", "CONCERNE",
                "DEPEND_DE", "FIXE",
            ]

            tokens = [
                tok for tok in re.findall(r"[a-zA-Z0-9_-]{4,}", query.lower())
                if tok not in {"avec", "dans", "pour", "entre", "quels", "quelles", "donne"}
            ]

            relation_cypher = """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN $rel_types
              AND (
                size($tokens) = 0 OR
                any(tok IN $tokens WHERE
                  toLower(coalesce(a.reference, '')) CONTAINS tok OR
                  toLower(coalesce(b.reference, '')) CONTAINS tok OR
                  toLower(coalesce(a.description, '')) CONTAINS tok OR
                  toLower(coalesce(b.description, '')) CONTAINS tok
                )
              )
            RETURN coalesce(a.reference, a.id) AS src,
                   type(r) AS rel,
                   coalesce(b.reference, b.id) AS tgt
            LIMIT $limit
            """

            rel_rows = self._run_graph_query_with_retry(
                relation_cypher,
                params={
                    "rel_types": rel_types,
                    "tokens": tokens,
                    "limit": self.k_graph,
                },
            )

            if not rel_rows:
                rel_rows = self._run_graph_query_with_retry(
                    """
                    MATCH (a)-[r]->(b)
                    WHERE type(r) IN $rel_types
                    RETURN coalesce(a.reference, a.id) AS src,
                           type(r) AS rel,
                           coalesce(b.reference, b.id) AS tgt
                    LIMIT $limit
                    """,
                    params={"rel_types": rel_types, "limit": self.k_graph},
                )

            for row in rel_rows:
                src = row.get("src") or ""
                rel = row.get("rel") or ""
                tgt = row.get("tgt") or ""
                if not (src and rel and tgt):
                    continue
                docs.append(
                    Document(
                        page_content=f"[Path] {src} -[{rel}]-> {tgt}",
                        metadata={
                            "retrieval_source": "graph_path",
                            "reference": src,
                            "score": 1.0,
                        },
                    )
                )

            summary_rows = self._run_graph_query_with_retry(
                """
                MATCH ()-[r]->()
                WHERE type(r) IN $rel_types
                RETURN type(r) AS rel, count(*) AS c
                ORDER BY c DESC
                LIMIT 12
                """,
                params={"rel_types": rel_types},
            )

            if summary_rows:
                rel_summary = ", ".join(
                    f"{row.get('rel')}({row.get('c')})"
                    for row in summary_rows
                    if row.get("rel")
                )
                if rel_summary:
                    docs.append(
                        Document(
                            page_content=f"[RelationSummary] {rel_summary}",
                            metadata={
                                "retrieval_source": "graph_summary",
                                "reference": "graph",
                                "score": 1.0,
                            },
                        )
                    )

            return docs
        except Exception as e:
            print(f"[GraphSearch] Erreur Cypher: {e}")
            return []


def get_hybrid_retriever(search_mode: str = "all") -> ComplianceGuardRetriever:
    """Factory pour obtenir le retriever hybride configuré."""
    return ComplianceGuardRetriever(
        qdrant_client=get_qdrant_client(),
        embeddings=get_embeddings_model(),
        qdrant_collection=config.QDRANT_COLLECTION_NAME,
        graph=get_graph(),
        k_vector=_env_int("RETRIEVER_K_VECTOR", 4, min_value=1),
        k_graph=_env_int("RETRIEVER_K_GRAPH", 5, min_value=1),
        search_mode=search_mode,
    )


# ── SETUP INDEX FULLTEXT NEO4J ────────────────────────────────────────────────

def setup_fulltext_index():
    """
    Crée l'index fulltext Neo4j pour la recherche dans le graphe.
    À exécuter une seule fois après l'ingestion.
    """
    graph = get_graph()
    try:
        graph.query("""
        CREATE FULLTEXT INDEX legal_entities IF NOT EXISTS
        FOR (n:Article|Loi|Decret|Circulaire|Organisme|Avantage|Obligation|Condition)
        ON EACH [n.description, n.reference, n.valeur]
        """)
        print("Index fulltext 'legal_entities' créé ✓")
    except Exception as e:
        print(f"Index fulltext: {e}")


if __name__ == "__main__":
    print("Setup des index Neo4j...")
    setup_fulltext_index()
    print("Retriever prêt.")