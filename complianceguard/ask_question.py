#!/usr/bin/env python
"""Simple CLI to ask one legal question and get one direct answer."""

from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path
from typing import Iterable
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

logger = logging.getLogger(__name__)

_PDF_NAME_RE = re.compile(r"\b[\w\-]+(?:_[\w\-]+)*\.pdf\b", re.IGNORECASE)

LEGAL_REF_FULL_TITLES: dict[str, str] = {
    "Loi n° 2018-20": "Loi n° 2018-20 du 17 avril 2018 (Startup Act)",
    "Décret n° 2018-840": "Décret gouvernemental n° 2018-840 du 11 octobre 2018",
    "Circulaire BCT n° 2019-01": "Circulaire BCT n° 2019-01 du 30 janvier 2019",
    "Circulaire BCT n° 2019-02": "Circulaire BCT n° 2019-02 du 30 janvier 2019",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Compat: certains environnements utilisent QDRANT_COLLECTION au lieu de
# QDRANT_COLLECTION_NAME.
if not os.getenv("QDRANT_COLLECTION_NAME", "").strip():
    legacy_collection = os.getenv("QDRANT_COLLECTION", "").strip()
    if legacy_collection:
        os.environ["QDRANT_COLLECTION_NAME"] = legacy_collection

try:
    from complianceguard.tools.retriever import get_hybrid_retriever, setup_fulltext_index
except ModuleNotFoundError:
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from tools.retriever import get_hybrid_retriever, setup_fulltext_index


def _build_llm() -> ChatOpenAI | AzureChatOpenAI:
    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = (
        os.getenv("GROQ_MODEL", "").strip()
        or "openai/gpt-oss-120b"
    )
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()

    # Prefer Groq for CRAG when explicitly requested or when a Groq key is present.
    use_groq = llm_provider == "groq" or (groq_api_key and llm_provider != "azure")
    if use_groq:
        if not groq_api_key:
            raise RuntimeError("Configuration Groq incomplete: GROQ_API_KEY")

        print(f"[LLM] Provider=groq model={groq_model}")

        return ChatOpenAI(
            model=groq_model,
            api_key=groq_api_key,
            base_url=groq_base_url,
            temperature=0,
        )

    azure_endpoint = (
        os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        or os.getenv("AZURE_API_BASE", "").strip()
    )
    model_or_deployment = (
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
    if not model_or_deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT (ou AZURE_MODEL/model)")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY (ou AZURE_API_KEY)")

    if missing:
        raise RuntimeError("Configuration Azure incomplete: " + ", ".join(missing))

    deployment_name = model_or_deployment
    if "/" in deployment_name:
        deployment_name = deployment_name.split("/", 1)[1].strip()

    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=deployment_name,
        api_version=api_version,
        api_key=api_key,
        temperature=0,
    )


def _looks_like_groq_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    return (
        provider == "groq"
        or "groq" in msg
        or "api.groq.com" in msg
        or "openaierror" in msg
        or "ratelimit" in msg
        or "429" in msg
    )


def _report_inference_error(stage: str, exc: Exception, question: str = "") -> None:
    prefix = "[GROQ][INFERENCE][ERROR]" if _looks_like_groq_error(exc) else "[LLM][INFERENCE][ERROR]"
    q_preview = " ".join((question or "").split())[:180]
    print(f"{prefix} stage={stage} message={exc}")
    if q_preview:
        print(f"{prefix} question={q_preview}")
    logger.exception("%s stage=%s question_preview=%s", prefix, stage, q_preview)


def _is_legal_ref(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if not re.search(r"\b(Loi|Decret|Décret|Circulaire|Code|Rapport)\b", text, re.IGNORECASE):
        return False
    return (
        "n°" in text
        or "no " in text.lower()
        or text.lower().startswith("code")
        or text.lower().startswith("rapport")
        or " bct " in f" {text.lower()} "
    )


def _sanitize_answer_text(text: str) -> str:
    # Supprime toute mention de nom de fichier type *.pdf dans la réponse finale.
    cleaned = _PDF_NAME_RE.sub("", text)
    # Normalise les sections en mode console (évite les titres markdown collés).
    cleaned = cleaned.replace("### Reponse directe", "Reponse directe:")
    cleaned = cleaned.replace("### Réponse directe", "Reponse directe:")
    cleaned = cleaned.replace("### Conditions principales", "Conditions principales:")
    cleaned = cleaned.replace("### Etapes pratiques", "Etapes pratiques:")
    cleaned = cleaned.replace("### Étapes pratiques", "Etapes pratiques:")
    cleaned = re.sub(r"\s+Conditions principales:\s*", "\n\nConditions principales:\n", cleaned)
    cleaned = re.sub(r"\s+Etapes pratiques:\s*", "\n\nEtapes pratiques:\n", cleaned)

    # Développe les références juridiques vers leur forme complète.
    for short_ref, full_title in LEGAL_REF_FULL_TITLES.items():
        cleaned = re.sub(rf"\b{re.escape(short_ref)}\b", full_title, cleaned)

    # Évite les citations de numéro seul du type "2018-840".
    cleaned = re.sub(
        r"(?<!n°\s)(?<!nº\s)(?<!no\s)\b2018-840\b",
        "Décret gouvernemental n° 2018-840 du 11 octobre 2018",
        cleaned,
    )

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)
    return cleaned.strip()


def _build_context(docs: Iterable, max_docs: int = 8, max_chars: int = 1200) -> str:
    def _truncate_for_context(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text

        # Keep both the beginning and end of long chunks. In legal documents,
        # operative conditions are often listed near section ends.
        head_len = max(200, (limit // 2) - 20)
        tail_len = max(200, limit - head_len - 20)
        head = text[:head_len].rstrip()
        tail = text[-tail_len:].lstrip()
        return f"{head} ... {tail}"

    chunks: list[str] = []
    for i, doc in enumerate(list(docs)[:max_docs], start=1):
        ref = str(doc.metadata.get("reference", "") or "").strip()
        text = (doc.page_content or "").strip().replace("\n", " ")
        text = " ".join(text.split())
        if not text:
            continue
        text = _truncate_for_context(text, max_chars)
        if _is_legal_ref(ref):
            chunks.append(f"[{i}] reference={ref}\n{text}")
        else:
            chunks.append(f"[{i}]\n{text}")
    return "\n\n".join(chunks)


def _collect_sources(docs: Iterable) -> list[str]:
    refs: list[str] = []
    for doc in docs:
        ref = str(doc.metadata.get("reference", "")).strip()
        if not ref or ref == "graph":
            continue
        if not _is_legal_ref(ref):
            continue
        display_ref = LEGAL_REF_FULL_TITLES.get(ref, ref)
        if display_ref not in refs:
            refs.append(display_ref)
    return refs


def _is_context_insufficient(context: str, docs: list) -> bool:
    """Vérifie si le contexte GraphRAG est insuffisant."""
    if not docs or len(docs) == 0:
        return True
    if not context or len(context.strip()) < 100:
        return True
    return False


def _is_greeting_or_non_question(text: str) -> bool:
    """Détecte les salutations et messages non-questions."""
    text_lower = text.lower().strip()
    
    greetings = [
        "bonjour", "bonsoir", "salut", "hello", "hi", "hey",
        "coucou", "salam", "bsr", "bjr", "cc", "merci", "thanks",
        "ok", "oui", "non", "d'accord", "super", "bien", "cool"
    ]
    
    # Message trop court (moins de 3 mots)
    if len(text_lower.split()) < 3:
        for greeting in greetings:
            if greeting in text_lower:
                return True
        # Vérifier si c'est juste un ou deux mots sans point d'interrogation
        if "?" not in text and len(text_lower) < 20:
            return True
    
    return False


def _web_search(query: str) -> tuple[str, list[str]]:
    """Recherche web via Serper API."""
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_key:
        return "", []
    
    print("[Web] Recherche en cours...")
    search = GoogleSerperAPIWrapper(serper_api_key=serper_key, k=5)
    results = search.results(query + " Tunisie startup réglementation")
    
    web_context = ""
    web_sources = []
    
    # Extraire les résultats organiques
    organic = results.get("organic", [])[:3]
    for item in organic:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        web_context += f"[{title}]\n{snippet}\n\n"
        if link:
            web_sources.append(link)
    
    # Scraper les 2 premières URLs pour plus de contexte
    for url in web_sources[:2]:
        try:
            print(f"[Web] Scraping {url[:50]}...")
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(url)
            docs = loader.load()
            content = "\n".join(doc.page_content for doc in docs)
            content = " ".join(content.split())[:2000]
            web_context += f"\n[Contenu de {url}]\n{content}\n"
        except Exception as e:
            print(f"[Web] Erreur scraping: {e}")
    
    return web_context, web_sources


def answer_question(question: str, max_docs: int = 8, enable_web_fallback: bool = True, mode: str = "kb") -> tuple[str, list[str]]:
    # Détecter les salutations et non-questions
    if _is_greeting_or_non_question(question):
        return (
            "Bonjour ! Je suis l'assistant juridique ComplianceGuard, spécialisé dans le Startup Act tunisien.\n\n"
            "Posez-moi une question juridique, par exemple :\n"
            "- Quels documents pour obtenir le label startup ?\n"
            "- Quels sont les avantages fiscaux du Startup Act ?\n"
            "- Comment obtenir le congé startup ?",
            []
        )
    
    setup_fulltext_index()
    retriever = get_hybrid_retriever(search_mode=mode)
    docs = retriever.invoke(question)

    context = _build_context(docs, max_docs=max_docs)
    sources = _collect_sources(docs)
    source_type = "GraphRAG"
    
    # Fallback web si contexte insuffisant
    if enable_web_fallback and _is_context_insufficient(context, docs):
        print("[GraphRAG] Contexte insuffisant, recherche web en cours...")
        web_context, web_sources = _web_search(question)
        if web_context:
            context = web_context
            sources = web_sources
            source_type = "Web"
    
    llm = _build_llm()

    system_prompt = (
        "Tu es un assistant juridique tunisien. "
        "Donne une reponse directe, claire et pratique. "
        "Ne parle pas du pipeline, du graphe, des relations techniques, ni du scoring. "
        "Ne cite jamais de noms de fichiers (ex: *.pdf, source_file). "
        "Si tu cites une source, cite uniquement un intitulé juridique complet (nature + numero + date quand connue), "
        "jamais un numéro seul comme '2018-840'. "
        "Structure strictement ta reponse en 3 blocs: Reponse directe, Conditions principales, Etapes pratiques. "
        "Si le contexte est insuffisant, dis-le explicitement."
    )

    human_prompt = (
        f"Question:\n{question}\n\n"
        f"Contexte juridique recupere ({source_type}):\n"
        f"{context}\n\n"
        "Reponds en francais, de maniere concise et concrete."
    )

    try:
        result = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
    except Exception as exc:
        _report_inference_error("answer_question.invoke", exc, question=question)
        raise
    answer = (result.content or "").strip() if hasattr(result, "content") else str(result)
    answer = _sanitize_answer_text(answer)
    
    if source_type == "Web":
        answer += f"\n\n[Source: Recherche Web]"
    
    return answer, sources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask one question to ComplianceGuard GraphRAG.")
    parser.add_argument("-q", "--question", default="", help="Question juridique a poser")
    parser.add_argument("--max-docs", type=int, default=8, help="Nombre max de documents de contexte")
    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Ne pas afficher la liste des sources detectees",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Desactiver le fallback vers la recherche web",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    enable_web = not args.no_web

    if args.question.strip():
        answer, refs = answer_question(args.question.strip(), max_docs=max(1, args.max_docs), enable_web_fallback=enable_web)
        print("\n" + "=" * 60)
        print("REPONSE")
        print("=" * 60)
        print(answer)
        if not args.hide_sources:
            print("\nSources:")
            if refs:
                for ref in refs[:8]:
                    print(f"- {ref}")
            else:
                print("- (aucune source explicite)\n")
        return 0

    print("Mode interactif. Tapez 'exit' pour quitter. (Web fallback: " + ("ON" if enable_web else "OFF") + ")")
    while True:
        q = input("\nQuestion > ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            return 0

        answer, refs = answer_question(q, max_docs=max(1, args.max_docs), enable_web_fallback=enable_web)
        print("\n" + "-" * 60)
        print(answer)
        if not args.hide_sources:
            print("\nSources:")
            if refs:
                for ref in refs[:8]:
                    print(f"- {ref}")
            else:
                print("- (aucune source explicite)")


if __name__ == "__main__":
    raise SystemExit(main())
