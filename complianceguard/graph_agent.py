import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

# Import existing functions
from complianceguard.crag import crag_answer
from complianceguard.ask_question import answer_question, _web_search, _build_llm, _sanitize_answer_text

logger = logging.getLogger(__name__)

# 1. State Definition
class ComplianceState(TypedDict):
    question: str
    has_pdf: bool
    
    crag_answer: str
    crag_sources: List[str]
    crag_metadata: Dict[str, Any]
    
    graphrag_answer: str
    graphrag_sources: List[str]
    
    # --- A2A Routing Addition ---
    needs_web_search: bool
    
    web_context: str
    web_sources: List[str]
    
    final_answer: str
    final_sources: List[str]
    
    # --- A2A Peer-Review Additions ---
    draft: str
    feedback: str
    revision_count: int

# 2. Nodes (Tools)
def node_crag(state: ComplianceState):
    """Exécute le pipeline CRAG si un PDF est présent."""
    if not state.get("has_pdf"):
        return {
            "crag_answer": "", 
            "crag_sources": [],
            "crag_metadata": {"action": "skipped"}
        }
    
    try:
        logger.info("[LangGraph] Running CRAG tool...")
        ans, sources, metadata = crag_answer(state["question"], enable_web_fallback=False, mode="notebook")
        return {
            "crag_answer": ans, 
            "crag_sources": sources,
            "crag_metadata": metadata
        }
    except Exception as e:
        logger.error(f"[LangGraph] CRAG error: {e}")
        return {
            "crag_answer": "", 
            "crag_sources": [],
            "crag_metadata": {"error": str(e)}
        }

def node_graphrag(state: ComplianceState):
    """Exécute le pipeline GraphRAG (Neo4j + Qdrant global)."""
    try:
        logger.info("[LangGraph] Running GraphRAG tool...")
        ans, sources = answer_question(state["question"], enable_web_fallback=False, mode="kb")
        return {
            "graphrag_answer": ans, 
            "graphrag_sources": sources
        }
    except Exception as e:
        logger.error(f"[LangGraph] GraphRAG error: {e}")
        return {
            "graphrag_answer": "", 
            "graphrag_sources": []
        }

def node_web(state: ComplianceState):
    """Exécute la recherche web (Serper)."""
    try:
        logger.info("[LangGraph] Running Web Scraping tool...")
        context, sources = _web_search(state["question"])
        return {
            "web_context": context, 
            "web_sources": sources
        }
    except Exception as e:
        logger.error(f"[LangGraph] Web Scraping error: {e}")
        return {
            "web_context": "", 
            "web_sources": []
        }

def node_evaluator(state: ComplianceState):
    """Agent Évaluateur (A2A) : Vérifie si les sources locales sont suffisantes ou s'il faut scrapper le web."""
    logger.info("[LangGraph] Agent Évaluateur : Assessing internal knowledge completeness...")
    print("\n[A2A - Agent Évaluateur] Analyse des connaissances internes de la base de données...")
    
    graph_ans = state.get("graphrag_answer", "")
    crag_ans = state.get("crag_answer", "")
    
    # Heuristique rapide : si GraphRAG ne trouve rien, on part sur le web direct
    if "Désolé" in graph_ans or (not graph_ans.strip() and not crag_ans.strip()):
        print("[A2A - Agent Évaluateur] Décision : La base de données est muette. Recours au Web obligatoire. 🌐")
        return {"needs_web_search": True}
        
    llm = _build_llm()
    system_prompt = (
        "Tu es un évaluateur juridique de haut niveau. Ta tâche est de lire une question utilisateur "
        "et le contexte extrait d'une base de données interne. "
        "Si le contexte contient TOUTES les informations nécessaires pour répondre parfaitement, réponds 'OUI'. "
        "S'il manque des informations cruciales ou que le contexte est incomplet, réponds 'NON'."
    )
    human_prompt = f"Question de l'utilisateur : {state['question']}\n\nContexte interne trouvé :\n{graph_ans}\n{crag_ans}\n\nLe contexte est-il suffisant ? (OUI ou NON) :"
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        decision = response.content.strip().upper()
        needs_web = "NON" in decision
        logger.info(f"[LangGraph] Agent Évaluateur decision: {decision} -> Web Search Required: {needs_web}")
        
        if needs_web:
            print(f"[A2A - Agent Évaluateur] Décision: {decision}. Le contexte est incomplet. Activation de l'Agent Web. 🌐")
        else:
            print(f"[A2A - Agent Évaluateur] Décision: {decision}. Le contexte interne est suffisant. Pas besoin du web. ✅")
            
        return {"needs_web_search": needs_web}
    except Exception as e:
        logger.error(f"[LangGraph] Evaluator error: {e}")
        print("[A2A - Agent Évaluateur] Erreur lors de l'évaluation, Fallback sur le web par sécurité. 🌐")
        return {"needs_web_search": True}  # Par sécurité, on scrap si erreur

def node_draft(state: ComplianceState):
    """Agent Rédacteur : Combine les résultats et génère un brouillon."""
    logger.info("[LangGraph] Agent Rédacteur : Drafting answer...")
    print("\n[A2A - Agent Rédacteur] Rédaction du document juridique en cours... 📝")
    llm = _build_llm()
    
    parts = []
    sources = []
    
    # CRAG
    if state.get("crag_answer") and "Désolé, je ne trouve pas" not in state["crag_answer"]:
        parts.append(f"--- ANALYSE DES DOCUMENTS UPLOADÉS ---\n{state['crag_answer']}")
        sources.extend(state.get("crag_sources", []))
        
    # GraphRAG
    if state.get("graphrag_answer"):
        parts.append(f"--- ANALYSE DE LA BASE DE CONNAISSANCES INTERNE ---\n{state['graphrag_answer']}")
        sources.extend(state.get("graphrag_sources", []))
        
    # Web
    if state.get("web_context"):
        parts.append(f"--- INFORMATIONS WEB ---\n{state['web_context']}")
        sources.extend(state.get("web_sources", []))
        
    context = "\n\n".join(parts)
    
    if not context.strip():
        # Deduplicate sources while preserving order
        unique_sources = list(dict.fromkeys(sources))
        print("[A2A - Agent Rédacteur] Impossible de rédiger, aucun contexte disponible.")
        return {
            "final_answer": "Désolé, je n'ai pas pu trouver d'informations pertinentes pour répondre à votre question.",
            "final_sources": unique_sources,
            "draft": "Désolé, je n'ai pas pu trouver d'informations pertinentes pour répondre à votre question."
        }
    
    system_prompt = (
        "Tu es ComplianceGuard, un juriste et consultant expert en droit tunisien (notamment le Startup Act). "
        "Tu reçois des éléments de contexte extraits de différentes sources (Documents utilisateurs, Base de graphe interne, Web). "
        "Ton objectif est de fournir une synthèse juridique irréprochable, professionnelle et exploitable.\n\n"
        "RÈGLES STRICTES :\n"
        "1. Ne révèle jamais tes mécanismes internes (ne dis pas 'Selon les documents uploadés' ni 'La base interne indique...'). "
        "Affirme les faits comme un expert sûr de lui.\n"
        "2. Formate toujours ta réponse en markdown propre, divisée exactement en 3 sections :\n"
        "   - **Réponse directe** : Une synthèse claire et sans jargon inutile qui répond immédiatement à la question.\n"
        "   - **Conditions principales** : Les prérequis, critères d'éligibilité ou exceptions légales applicables (sous forme de puces).\n"
        "   - **Étapes pratiques** : La marche à suivre concrète, numérotée, pour appliquer ce droit ou cette procédure.\n"
        "3. Cite précisément les textes de loi fournis dans le contexte (numéro, date, article) sans inventer de références."
    )
    
    # Intégration du feedback du réviseur s'il s'agit d'une révision
    feedback_section = ""
    if state.get("feedback") and state.get("feedback") != "APPROVED":
        print(f"[A2A - Agent Rédacteur] Analyse du retour de l'Avocat Sénior : On corrige les failles signalées... 🔄")
        feedback_section = f"\n\n/!\\ ATTENTION - RETOUR DE L'AVOCAT RÉVISEUR /!\\ :\n{state['feedback']}\nCorrige ton brouillon en fonction de ce retour."

    human_prompt = f"Question: {state['question']}\n\nContexte combiné:\n{context}{feedback_section}\n\nBrouillon:"
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        draft_ans = _sanitize_answer_text(response.content)
        print("[A2A - Agent Rédacteur] Brouillon terminé, envoi pour validation au Réviseur. 📨")
    except Exception as e:
        logger.error(f"[LangGraph] Drafting error: {e}")
        draft_ans = "Erreur lors de la génération du brouillon."
        print("[A2A - Agent Rédacteur] Erreur lors de la rédaction.")
    
    unique_sources = list(dict.fromkeys(sources))
            
    return {
        "draft": draft_ans,
        "final_answer": draft_ans, # En attendant la validation
        "final_sources": unique_sources
    }

def node_review(state: ComplianceState):
    """Agent Réviseur : Lit le brouillon et décide s'il est validé ou nécessite correction."""
    current_revisions = state.get("revision_count", 0)
    print(f"\n[A2A - Agent Réviseur] Lecture critique du brouillon (Itération {current_revisions + 1}) 🧐 ...")
    
    # Limiter les boucles infinies (2 révisions maximum)
    if current_revisions >= 2:
        logger.info("[LangGraph] Agent Réviseur : Max revisions reached. Approving by default.")
        print("[A2A - Agent Réviseur] Maximum de révisions atteint. Forçage de l'approbation pour éviter la boucle infinie. ⚠️")
        return {"feedback": "APPROVED"}
        
    logger.info(f"[LangGraph] Agent Réviseur : Reviewing draft (Revision {current_revisions + 1})...")
    llm = _build_llm()
    
    system_prompt = (
        "Tu es un Avocat Senior expert en droit tunisien supervisant un juriste junior. "
        "Tu vas lire sa réponse (le brouillon) à une question juridique. "
        "Ta mission : vérifier que la réponse contient bien les 3 sections obligatoires "
        "('Réponse directe', 'Conditions principales', 'Étapes pratiques') et ne dévoile pas le fonctionnement interne de l'IA.\n"
        "SI LA RÉPONSE EST PARFAITE (respecte la structure et semble robuste juridique): retourne UNIQUEMENT le mot 'APPROVED'.\n"
        "SI LA RÉPONSE EST DÉFICIENTE (il manque une section, le ton est mauvais, etc) : explique précisément ce qu'il faut corriger. Ne répond PAS toi-même à la question, donne juste tes consignes de correction."
    )
    
    human_prompt = f"Question de l'utilisateur: {state['question']}\n\nBrouillon du junior:\n{state.get('draft', '')}\n\nTa décision (APPROVED ou remarques) :"
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        feedback = response.content.strip()
        
        if feedback == "APPROVED":
            print("[A2A - Agent Réviseur] Décision : APPROVED. Le document final est prêt ! ✅")
        else:
            print("[A2A - Agent Réviseur] Décision : REJETÉ. Il y a des erreurs, renvoi au Rédacteur. ❌")
            print(f"[A2A - Agent Réviseur] -> Motif(s) : {feedback}")
            
    except Exception as e:
        logger.error(f"[LangGraph] Review error: {e}")
        feedback = "APPROVED" # En cas d'erreur de LLM, on laisse passer
        print("[A2A - Agent Réviseur] Erreur interne pendant la révision, auto-approbation par sécurité.")
        
    return {
        "feedback": feedback,
        "revision_count": current_revisions + 1
    }

def should_continue(state: ComplianceState) -> str:
    """Condition pour boucler ou terminer."""
    # Si pas de recherche fructueuse d'emblée, ou si approuvé
    if state.get("draft", "").startswith("Désolé"):
        return "end"
    if state.get("feedback") == "APPROVED":
        return "end"
    return "rewrite"

# 3. Graph Definition
def build_compliance_agent():
    workflow = StateGraph(ComplianceState)
    
    workflow.add_node("node_crag", node_crag)
    workflow.add_node("node_graphrag", node_graphrag)
    workflow.add_node("node_evaluator", node_evaluator)
    workflow.add_node("node_web", node_web)
    workflow.add_node("node_draft", node_draft)
    workflow.add_node("node_review", node_review)
    
    # Parallel execution from START to local DBs
    workflow.add_edge(START, "node_crag")
    workflow.add_edge(START, "node_graphrag")
    
    # Local DBs converge to the Evaluator
    workflow.add_edge("node_crag", "node_evaluator")
    workflow.add_edge("node_graphrag", "node_evaluator")
    
    # Routing Logic : L'évaluateur décide d'aller sur le web ou directement à la rédaction
    def route_after_eval(state: ComplianceState) -> str:
        if state.get("needs_web_search"):
            return "search_web"
        return "skip_web"
        
    workflow.add_conditional_edges(
        "node_evaluator",
        route_after_eval,
        {
            "search_web": "node_web",
            "skip_web": "node_draft"
        }
    )
    
    # Si on est allé sur le web, la prochaine étape est la rédaction
    workflow.add_edge("node_web", "node_draft")
    
    # Le brouillon part en review
    workflow.add_edge("node_draft", "node_review")
    
    # Conditional edge A2A pour valider le brouillon
    workflow.add_conditional_edges(
        "node_review", 
        should_continue, 
        {
            "rewrite": "node_draft",
            "end": END
        }
    )
    
    return workflow.compile()

# Global agent instance
compliance_agent = build_compliance_agent()
