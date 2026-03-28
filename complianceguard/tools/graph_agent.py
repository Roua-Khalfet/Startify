"""
ComplianceGuard — graph_agent.py
==================================
Intègre le GraphRAG comme Tool LangChain dans ton agent existant (chain.py).
Ajoute graph_tool à ta liste de tools dans main.py — c'est tout.

Utilisation dans main.py :
    from complianceguard.tools.graph_agent import graph_tool, graph_qa_tool
    tools = [web_search_tool, graph_tool, graph_qa_tool]  # ajoute ici
"""

import os
from dotenv import load_dotenv
from langchain.tools import Tool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field

from .retriever import get_hybrid_retriever, get_graph_qa_chain

load_dotenv()

# ── LLM PRINCIPAL ────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ── PROMPT COMPLIANCEGUARD ────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es ComplianceGuard, assistant juridique spécialisé en droit 
entrepreneurial tunisien. Tu as accès à une base de connaissances juridiques 
construite à partir des textes officiels suivants :

• Loi n° 2018-20 du 17 avril 2018 (Startup Act) — JORT n°32
• Décret gouvernemental n° 2018-840 du 11 octobre 2018 — JORT n°84  
• Circulaire BCT n° 2019-01 — Comptes startup en devises
• Circulaire BCT n° 2019-02 — Carte Technologique Internationale
• Code des Sociétés Commerciales (Loi n°2000-93)
• Code des Droits et Procédures Fiscaux 2023
• Code du Travail tunisien
• Loi n° 2004-63 sur la protection des données personnelles
• Loi n° 2016-71 sur l'investissement
• Rapport IC sur les Startup Acts

Règles de réponse :
1. Cite toujours la source exacte : "Selon l'Art. X de la Loi n° YYYY-ZZ..."
2. Structure : Réponse directe → Base légale → Conditions → Étapes pratiques
3. Si tu n'es pas sûr, dis-le clairement et recommande de consulter un avocat
4. Réponds en français (ou en arabe si la question est en arabe)
5. Ne jamais inventer des articles ou des lois qui n'existent pas

Contexte juridique récupéré :
{context}
"""

# ── CHAÎNE RAG PRINCIPALE ────────────────────────────────────────────────────

def build_rag_chain():
    """
    Construit la chaîne RAG hybride : retriever → prompt → LLM.
    """
    retriever = get_hybrid_retriever()

    def format_docs(docs):
        formatted = []
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("reference", d.metadata.get("source_file", ""))
            retrieval = d.metadata.get("retrieval_source", "")
            formatted.append(
                f"[{i}] Source: {source} ({retrieval})\n{d.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── TOOL 1 : RECHERCHE DANS LA BASE JURIDIQUE ────────────────────────────────

class LegalQueryInput(BaseModel):
    question: str = Field(
        description="Question juridique sur le droit tunisien des startups, "
                    "la fiscalité, la création d'entreprise, le droit social, "
                    "ou la propriété intellectuelle en Tunisie."
    )

_rag_chain = None

def query_legal_knowledge_base(question: str) -> str:
    """
    Interroge la base de connaissances juridiques tunisienne (GraphRAG).
    Utilise à la fois la recherche vectorielle et le knowledge graph Neo4j.
    """
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = build_rag_chain()
    try:
        return _rag_chain.invoke(question)
    except Exception as e:
        return f"Erreur lors de la recherche juridique: {str(e)}"


graph_tool = StructuredTool.from_function(
    func=query_legal_knowledge_base,
    name="base_juridique_tunisienne",
    description="""
    Utilise cet outil pour toute question juridique relative à :
    - La création de startup et le label Startup Act (Loi 2018-20)
    - Les avantages fiscaux : exonération IS, déductions, TVA
    - Le congé pour création de startup et la bourse startup
    - La réglementation des changes et comptes en devises (BCT)
    - Le Code des sociétés : SARL, SA, SAS, capital, statuts
    - Le droit du travail tunisien : contrats, CNSS, licenciement
    - La protection des données personnelles (Loi 2004-63)
    - L'investissement et les incitations (Loi 2016-71)
    
    NE PAS utiliser pour des actualités récentes ou des questions
    non-juridiques (utiliser web_search à la place).
    """,
    args_schema=LegalQueryInput,
)


# ── TOOL 2 : REQUÊTE DIRECTE SUR LE GRAPHE ───────────────────────────────────

class GraphQueryInput(BaseModel):
    question: str = Field(
        description="Question spécifique sur les relations entre textes de loi, "
                    "articles, organismes ou procédures juridiques tunisiennes."
    )

_graph_qa_chain = None

def query_legal_graph(question: str) -> str:
    """
    Interroge directement le knowledge graph Neo4j via Cypher généré par LLM.
    Utile pour tracer des relations entre lois, articles, et obligations.
    """
    global _graph_qa_chain
    if _graph_qa_chain is None:
        _graph_qa_chain = get_graph_qa_chain()
    try:
        result = _graph_qa_chain.invoke({"query": question})
        answer = result.get("result", "")
        steps = result.get("intermediate_steps", [])
        if steps:
            cypher = steps[0].get("query", "")
            if cypher:
                answer += f"\n\n[Requête Cypher: {cypher[:200]}...]"
        return answer
    except Exception as e:
        return f"Erreur graph: {str(e)}"


graph_qa_tool = StructuredTool.from_function(
    func=query_legal_graph,
    name="graphe_juridique_tunisien",
    description="""
    Utilise cet outil pour des questions de type :
    - "Quels articles de la Loi 2018-20 concernent les avantages fiscaux ?"
    - "Quels organismes sont impliqués dans la procédure de labellisation ?"
    - "Quel décret applique l'article 17 de la Loi 2018-20 ?"
    - "Quelles conditions doit remplir une startup pour bénéficier de l'IS à 0% ?"
    
    Préférer base_juridique_tunisienne pour les questions générales.
    """,
    args_schema=GraphQueryInput,
)


# ── VÉRIFICATION DE CONFORMITÉ ────────────────────────────────────────────────

class ComplianceCheckInput(BaseModel):
    description_projet: str = Field(
        description="Description du projet startup à vérifier : "
                    "secteur, modèle économique, équipe, structure juridique envisagée."
    )

def check_compliance(description_projet: str) -> str:
    """
    Vérifie la conformité légale d'un projet startup tunisien.
    Identifie les obligations, avantages disponibles, et risques.
    """
    chain = build_rag_chain()

    prompt = f"""
    Analyse ce projet startup et génère un rapport de conformité légale complet :

    DESCRIPTION DU PROJET :
    {description_projet}

    Génère un rapport structuré avec :
    1. ÉLIGIBILITÉ AU LABEL STARTUP (Loi 2018-20, Art. 3)
       - Critères remplis / manquants
       
    2. FORME JURIDIQUE RECOMMANDÉE
       - SARL, SA ou SAS selon le profil
       - Capital minimum requis
       
    3. AVANTAGES DISPONIBLES
       - Fiscal (IS 0%, déductions)
       - Social (CNSS prise en charge)
       - Financement (fonds de garantie, SICAR)
       
    4. OBLIGATIONS LÉGALES
       - Comptabilité, déclarations fiscales
       - Obligations CNSS
       - Protection des données si applicable
       
    5. RISQUES ET POINTS D'ATTENTION
    
    6. ÉTAPES PRATIQUES (ordre chronologique)
    
    Cite les articles de loi pour chaque point.
    """
    try:
        return chain.invoke(prompt)
    except Exception as e:
        return f"Erreur vérification conformité: {str(e)}"


compliance_tool = StructuredTool.from_function(
    func=check_compliance,
    name="verification_conformite_startup",
    description="""
    Utilise cet outil quand un utilisateur décrit son projet startup
    et veut savoir : son éligibilité au label, les avantages disponibles,
    les obligations légales, et les étapes à suivre.
    
    Input : description complète du projet (secteur, équipe, financement, etc.)
    Output : rapport de conformité légale structuré avec citations.
    """,
    args_schema=ComplianceCheckInput,
)


# ── EXPORT TOOLS ──────────────────────────────────────────────────────────────

# À importer dans main.py :
# from complianceguard.tools.graph_agent import COMPLIANCEGUARD_TOOLS
COMPLIANCEGUARD_TOOLS = [
    graph_tool,
    graph_qa_tool,
    compliance_tool,
]