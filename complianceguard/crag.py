import logging
import os
import re
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from complianceguard.tools.retriever import get_hybrid_retriever
from complianceguard.ask_question import _build_llm, _web_search, _build_context, _collect_sources, _is_greeting_or_non_question, _sanitize_answer_text, _report_inference_error

class ItemScore(BaseModel):
    item_index: int = Field(description="The index of the scored item (1-indexed).")
    score: float = Field(description="Relevance score in [-1, 1].")


class ItemScoreFormat(BaseModel):
    scores: List[ItemScore] = Field(description="List of item scores")


class QueryRewriteFormat(BaseModel):
    keywords: List[str] = Field(description="Up to three concise keywords for web search")


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return max(min_value, value)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def _clamp_score(value: float) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, score))


def _score_text_items(question: str, items: List[str], llm, item_label: str = "Item", preview_chars: int = 400) -> List[float]:
    """
    Returns one relevance score per item in [-1, 1].
    """
    if not items:
        return []

    items_text = ""
    for i, item in enumerate(items, start=1):
        clean = _normalize_text(item)
        preview = (clean[:preview_chars] + "...") if len(clean) > preview_chars else clean
        items_text += f"{item_label} {i}:\n{preview}\n\n"

    system_prompt = """You are a legal retrieval relevance scorer.
Given a question and candidate text items, score each item independently for relevance.

Return a score in [-1, 1] for each item:
- 1.0 means highly relevant and directly useful.
- 0.0 means unclear or weakly relevant.
- -1.0 means irrelevant.

Return JSON matching the requested schema.
"""

    human_prompt = f"Question: {question}\n\n{items_text}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    try:
        scorer_llm = llm.with_structured_output(ItemScoreFormat)
        res = scorer_llm.invoke(messages)
        score_map: Dict[int, float] = {}
        for s in res.scores:
            score_map[int(s.item_index)] = _clamp_score(s.score)
        return [score_map.get(i, 0.0) for i in range(1, len(items) + 1)]
    except Exception as e:
        _report_inference_error("crag.score_items", e, question=question)
        return [0.0] * len(items)


def _score_to_grade(score: float, upper_threshold: float, lower_threshold: float) -> str:
    if score >= upper_threshold:
        return "CORRECT"
    if score < lower_threshold:
        return "INCORRECT"
    return "AMBIGUOUS"


def _split_into_knowledge_strips(text: str, max_sentences_per_strip: int = 2, max_chars_per_strip: int = 600) -> List[str]:
    """
    Decompose text into small strips, roughly sentence groups.
    """
    clean = _normalize_text(text)
    if not clean:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if len(sentences) <= 2:
        if len(clean) > max_chars_per_strip:
            return [clean[:max_chars_per_strip] + "..."]
        return [clean]

    strips: List[str] = []
    bucket: List[str] = []
    for sentence in sentences:
        bucket.append(sentence)
        if len(bucket) >= max_sentences_per_strip:
            strip = " ".join(bucket).strip()
            if strip:
                if len(strip) > max_chars_per_strip:
                    strip = strip[:max_chars_per_strip] + "..."
                strips.append(strip)
            bucket = []

    if bucket:
        strip = " ".join(bucket).strip()
        if strip:
            if len(strip) > max_chars_per_strip:
                strip = strip[:max_chars_per_strip] + "..."
            strips.append(strip)

    if not strips:
        fallback = clean[:max_chars_per_strip] + ("..." if len(clean) > max_chars_per_strip else "")
        strips.append(fallback)
    return strips


def _refine_documents(question: str, docs: List[Document], llm, strip_keep_threshold: float = 0.0, max_strips_per_doc: int = 5) -> List[Document]:
    """
    Decompose -> score -> filter -> recompose for internal documents.
    """
    refined_docs: List[Document] = []
    for doc in docs:
        strips = _split_into_knowledge_strips(doc.page_content)
        if not strips:
            continue

        strip_scores = _score_text_items(question, strips, llm, item_label="Knowledge strip", preview_chars=450)
        selected_idx = [i for i, score in enumerate(strip_scores) if score >= strip_keep_threshold]

        if not selected_idx:
            top_idx = max(range(len(strip_scores)), key=lambda idx: strip_scores[idx])
            selected_idx = [top_idx]

        if len(selected_idx) > max_strips_per_doc:
            top_by_score = sorted(selected_idx, key=lambda idx: strip_scores[idx], reverse=True)[:max_strips_per_doc]
            selected_idx = sorted(top_by_score)

        selected_strips = [strips[idx] for idx in selected_idx]
        recomposed = " ".join(selected_strips).strip()
        if not recomposed:
            continue

        metadata = dict(doc.metadata)
        metadata["refined"] = True
        metadata["strips_selected"] = len(selected_strips)
        refined_docs.append(Document(page_content=recomposed, metadata=metadata))

    return refined_docs


def _refine_web_context(question: str, web_context: str, llm, strip_keep_threshold: float = 0.0, max_strips: int = 12) -> str:
    """
    Applies the same strip selection logic to web context.
    """
    strips = _split_into_knowledge_strips(web_context)
    if not strips:
        return ""

    strip_scores = _score_text_items(question, strips, llm, item_label="Web strip", preview_chars=450)
    selected_idx = [i for i, score in enumerate(strip_scores) if score >= strip_keep_threshold]
    if not selected_idx:
        top_idx = max(range(len(strip_scores)), key=lambda idx: strip_scores[idx])
        selected_idx = [top_idx]

    if len(selected_idx) > max_strips:
        top_by_score = sorted(selected_idx, key=lambda idx: strip_scores[idx], reverse=True)[:max_strips]
        selected_idx = sorted(top_by_score)

    selected = [strips[idx] for idx in selected_idx]
    return "\n\n".join(selected).strip()


def _rewrite_web_query(question: str, llm) -> str:
    """
    Rewrites the user question into short web-search keywords.
    """
    system_prompt = """Rewrite the user question into up to three concise web-search keywords.
Rules:
- Keep named entities and legal references.
- Keep it in French if the question is in French.
- Return JSON matching the requested schema.
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}"),
    ]

    try:
        rewrite_llm = llm.with_structured_output(QueryRewriteFormat)
        res = rewrite_llm.invoke(messages)
        keywords = [
            _normalize_text(k).strip(",; ")
            for k in (res.keywords or [])
            if _normalize_text(k).strip(",; ")
        ]
        if keywords:
            return ", ".join(keywords[:3])
    except Exception as e:
        _report_inference_error("crag.rewrite_web_query", e, question=question)

    return question


def grade_documents(question: str, docs: List[Document], llm, upper_threshold: float = 0.6, lower_threshold: float = -0.2) -> List[Tuple[Document, str, float]]:
    """
    Grades documents using score-based relevance evaluation.
    Returns list of (Document, Grade, Score).
    """
    if not docs:
        return []

    doc_texts = [_normalize_text(doc.page_content) for doc in docs]
    scores = _score_text_items(question, doc_texts, llm, item_label="Document", preview_chars=350)

    graded: List[Tuple[Document, str, float]] = []
    for doc, score in zip(docs, scores):
        grade = _score_to_grade(score, upper_threshold=upper_threshold, lower_threshold=lower_threshold)
        graded.append((doc, grade, score))

    return graded


def decide_action(graded_docs: List[Tuple[Document, str, float]], upper_threshold: float = 0.6, lower_threshold: float = -0.2) -> str:
    """
    Decides the next action based on CRAG-like score thresholds.
    Returns: 'use_docs', 'web_search', or 'combine'
    """
    if not graded_docs:
        return 'web_search'

    scores = [score for _, _, score in graded_docs]

    # Correct if at least one document is confidently relevant.
    if any(score >= upper_threshold for score in scores):
        return 'use_docs'

    # Incorrect if all documents are confidently below relevance.
    if all(score < lower_threshold for score in scores):
        return 'web_search'

    # Ambiguous otherwise.
    return 'combine'


def crag_answer(question: str, enable_web_fallback: bool = True, mode: str = "notebook") -> Tuple[str, List[str], Dict[str, Any]]:
    """
    CRAG Pipeline. Returns (Answer, Sources, Metadata).
    """
    # Notebook mode is restricted to uploaded files only.
    if mode == "notebook":
        enable_web_fallback = False

    # 1. Handle greetings
    if _is_greeting_or_non_question(question):
        return (
            "Bonjour ! Je suis l'assistant juridique ComplianceGuard, spécialisé dans le Startup Act tunisien.\n\n"
            "Posez-moi une question juridique ou sur un document que vous avez uploade, par exemple :\n"
            "- Quels documents pour obtenir le label startup ?\n"
            "- Quels sont les avantages fiscaux du Startup Act ?\n"
            "- Comment obtenir le congé startup ?",
            [],
            {"action": "greeting"}
        )

    llm = _build_llm()
    retriever = get_hybrid_retriever(search_mode=mode)

    upper_threshold = 0.6
    lower_threshold = -0.2
    strip_keep_threshold = 0.0
    enable_refinement = _env_bool("CRAG_ENABLE_REFINEMENT", True)
    max_internal_docs = _env_int("CRAG_MAX_INTERNAL_DOCS", 4, min_value=1)
    
    # 2. Retrieve Documents
    logger.info(f"[CRAG] Retrieving documents for query: {question}")
    docs = retriever.invoke(question)
    
    # 3. Grade Documents
    logger.info("[CRAG] Grading retrieved documents...")
    graded_docs = grade_documents(
        question,
        docs,
        llm,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
    
    # 4. Decide Action
    action = decide_action(
        graded_docs,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
    logger.info(f"[CRAG] Decision: {action}")
    
    internal_ranked = [(doc, score) for doc, grade, score in graded_docs if grade != "INCORRECT"]
    internal_ranked = sorted(internal_ranked, key=lambda item: item[1], reverse=True)[:max_internal_docs]
    internal_candidates = [doc for doc, _ in internal_ranked]

    refined_internal_docs: List[Document] = []
    if enable_refinement and internal_candidates:
        refined_internal_docs = _refine_documents(
            question,
            internal_candidates,
            llm,
            strip_keep_threshold=strip_keep_threshold,
        )
    selected_internal_docs = refined_internal_docs if refined_internal_docs else internal_candidates
    
    context = ""
    sources = []
    web_query = ""
    
    # 5. Build Context based on Decision
    if action == 'use_docs':
        context = _build_context(selected_internal_docs)
        sources = _collect_sources(selected_internal_docs)
        
    elif action == 'web_search':
        if enable_web_fallback:
            web_query = _rewrite_web_query(question, llm)
            web_context, web_sources = _web_search(web_query)
            context = _refine_web_context(
                question,
                web_context,
                llm,
                strip_keep_threshold=strip_keep_threshold,
            )
            sources = web_sources
        else:
            context = ""
            
    elif action == 'combine':
        context = _build_context(selected_internal_docs)
        sources = _collect_sources(selected_internal_docs)
        if enable_web_fallback:
            web_query = _rewrite_web_query(question, llm)
            web_context, web_sources = _web_search(web_query)
            refined_web_context = _refine_web_context(
                question,
                web_context,
                llm,
                strip_keep_threshold=strip_keep_threshold,
            )
            if refined_web_context:
                if context.strip():
                    context += "\n\n--- RÉSULTATS WEB ---\n\n" + refined_web_context
                else:
                    context = refined_web_context
            for ws in web_sources:
                if ws not in sources:
                    sources.append(ws)

    # 6. Generate final answer
    if not context.strip():
        ans = "Désolé, je ne trouve pas d'informations suffisantes (ni en base ni sur le web) pour répondre avec certitude."
        return ans, [], {
            "action": action,
            "grades": [],
            "grader_thresholds": {
                "upper": upper_threshold,
                "lower": lower_threshold,
                "strip_keep": strip_keep_threshold,
            },
            "web_query": web_query,
        }

    system_prompt = """Tu es ComplianceGuard, un juriste IA pour les startups.
Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
Cite tes sources quand c'est pertinent (loi, article, lien web).
Ne mens jamais et ne donne pas de conseil financier. Si la réponse n'est pas dans le contexte, dis-le clairement."""

    human_prompt = f"Contexte:\n{context}\n\nQuestion:\n{question}\n\nRéponse:"
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
    except Exception as e:
        _report_inference_error("crag.final_answer.invoke", e, question=question)
        raise
    
    answer = _sanitize_answer_text(response.content)
    
    # Pack metadata
    metadata = {
        "action": action,
        "grades": [
            {
                "doc_id": d.metadata.get("chunk_id", d.metadata.get("doc_id", "")),
                "grade": g,
                "score": round(s, 4),
            }
            for d, g, s in graded_docs
        ],
        "documents_retrieved": len(docs),
        "documents_internal_candidates": len(internal_candidates),
        "documents_refined": len(refined_internal_docs),
        "documents_used": len(selected_internal_docs),
        "latency_toggles": {
            "enable_refinement": enable_refinement,
            "max_internal_docs": max_internal_docs,
        },
        "grader_thresholds": {
            "upper": upper_threshold,
            "lower": lower_threshold,
            "strip_keep": strip_keep_threshold,
        },
        "web_query": web_query,
    }
    
    return answer, sources, metadata
