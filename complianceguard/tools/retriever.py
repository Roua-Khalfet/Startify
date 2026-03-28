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
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List

load_dotenv()

# ── CONNEXIONS ────────────────────────────────────────────────────────────────

def get_vector_store() -> Neo4jVector:
    return Neo4jVector.from_existing_index(
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        ),
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        index_name="complianceguard_chunks",
        text_node_property="text",
        embedding_node_property="embedding",
    )


def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )


# ── GRAPH RETRIEVER : requêtes Cypher générées par le LLM ────────────────────

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
Tu es un expert en droit tunisien et en bases de données Neo4j.
Génère une requête Cypher pour répondre à la question juridique suivante.

Schéma du graphe :
{schema}

Nœuds disponibles : Article, Loi, Decret, Circulaire, Organisme,
                    Avantage, Obligation, Condition, Delai, Montant

Relations disponibles : PREVOIT, CONDITIONNE, CONCERNE, MODIFIE,
                        APPLIQUE, DEPEND_DE, COUVRE, IMPLIQUE, REFERENCE

Règles :
- Retourne toujours les propriétés 'description', 'reference', 'valeur'
- Limite les résultats à 10 nœuds maximum
- Préfère les relations PREVOIT et CONDITIONNE pour les questions sur les avantages
- Utilise APPLIQUE pour tracer les textes d'application d'une loi

Question : {question}

Requête Cypher :
""")

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es ComplianceGuard, assistant juridique spécialisé en droit entrepreneurial tunisien.
Réponds à la question en te basant UNIQUEMENT sur les informations du graphe juridique.

Contexte du graphe :
{context}

Question : {question}

Instructions :
- Cite toujours l'article et la loi source (ex: "Selon l'Art. 3 de la Loi 2018-20...")
- Si l'information est incomplète, indique ce qui manque
- Utilise un langage clair et accessible
- Structure ta réponse : Réponse directe → Base légale → Conditions → Étapes pratiques

Réponse :
""")


def get_graph_qa_chain() -> GraphCypherQAChain:
    """Chaîne qui génère du Cypher depuis une question naturelle."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    graph = get_graph()

    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=CYPHER_QA_PROMPT,
        verbose=True,
        return_intermediate_steps=True,
        validate_cypher=True,
        top_k=10,
    )


# ── HYBRID RETRIEVER ──────────────────────────────────────────────────────────

class ComplianceGuardRetriever(BaseRetriever):
    """
    Retriever hybride pour ComplianceGuard.
    Combine :
      1. Recherche vectorielle (similarité sémantique sur chunks)
      2. Traversée du graphe Neo4j (relations juridiques)
      3. Re-ranking par pertinence légale
    """

    vector_store: Neo4jVector
    graph: Neo4jGraph
    k_vector: int = 4       # Nombre de chunks vectoriels
    k_graph: int = 3        # Nombre de résultats graph

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        docs = []

        # 1. Recherche vectorielle
        vector_docs = self.vector_store.similarity_search(
            query, k=self.k_vector
        )
        for d in vector_docs:
            d.metadata["retrieval_source"] = "vector"
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

        return unique[:self.k_vector + self.k_graph]

    def _graph_search(self, query: str) -> List[Document]:
        """
        Recherche dans le graphe Neo4j.
        Extrait les nœuds liés à la question via des requêtes Cypher ciblées.
        """
        # Requête générale : trouve les avantages et obligations liés à la question
        cypher = """
        CALL db.index.fulltext.queryNodes('legal_entities', $query)
        YIELD node, score
        WITH node, score
        ORDER BY score DESC
        LIMIT $limit
        OPTIONAL MATCH (node)-[r]->(related)
        RETURN node.description AS description,
               node.reference AS reference,
               labels(node) AS types,
               collect({type: type(r), target: related.description}) AS relations,
               score
        """
        try:
            results = self.graph.query(
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
            return docs
        except Exception as e:
            print(f"[GraphSearch] Erreur Cypher: {e}")
            return []


def get_hybrid_retriever() -> ComplianceGuardRetriever:
    """Factory pour obtenir le retriever hybride configuré."""
    return ComplianceGuardRetriever(
        vector_store=get_vector_store(),
        graph=get_graph(),
        k_vector=4,
        k_graph=3,
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