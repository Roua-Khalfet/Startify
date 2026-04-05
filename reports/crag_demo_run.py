import os
import sys
import json
import logging
import traceback
from pathlib import Path

# Django bootstrap
sys.path.insert(0, str(Path("backend").resolve()))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from rest_framework.test import APIRequestFactory
from django.core.files.uploadedfile import SimpleUploadedFile
from api.views import upload_document, chat

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("complianceguard.crag").setLevel(logging.INFO)
logging.getLogger("complianceguard.ask_question").setLevel(logging.INFO)


def _print_runtime_provider_context() -> None:
    provider = os.getenv("LLM_PROVIDER", "").strip() or "(auto)"
    groq_model = os.getenv("GROQ_MODEL", "").strip() or "(default)"
    groq_base = os.getenv("GROQ_BASE_URL", "").strip() or "https://api.groq.com/openai/v1"
    print("[RUNTIME] LLM_PROVIDER:", provider)
    print("[RUNTIME] GROQ_MODEL:", groq_model)
    print("[RUNTIME] GROQ_BASE_URL:", groq_base)


def _looks_like_inference_error(text: str) -> bool:
    msg = (text or "").lower()
    patterns = [
        "groq",
        "api.groq.com",
        "openaierror",
        "llm",
        "inference",
        "rate limit",
        "429",
        "timeout",
        "model",
        "agents ne sont pas disponibles",
    ]
    return any(p in msg for p in patterns)

root = Path.cwd()
pdf_path = root / "complianceguard" / "test.pdf"

print("=" * 70)
print("CRAG DEMO RUN - NOTEBOOK MODE (UPLOAD ONLY)")
print("=" * 70)
_print_runtime_provider_context()
print(f"PDF used: {pdf_path}")
print(f"PDF exists: {pdf_path.exists()}")

if not pdf_path.exists():
    raise FileNotFoundError(f"Missing test file: {pdf_path}")

factory = APIRequestFactory()

print("\n[STEP 1] Upload document")
try:
    with open(pdf_path, "rb") as f:
        uploaded = SimpleUploadedFile("test.pdf", f.read(), content_type="application/pdf")
        req_upload = factory.post("/api/upload/", {"file": uploaded}, format="multipart")
        resp_upload = upload_document(req_upload)
except Exception as exc:
    print("[PIPELINE][ERROR] Upload step failed:", exc)
    traceback.print_exc()
    raise

print(f"Upload HTTP status: {resp_upload.status_code}")
try:
    upload_data = dict(resp_upload.data)
except Exception:
    upload_data = {"raw": str(resp_upload.data)}
print("Upload response JSON:")
print(json.dumps(upload_data, indent=2, ensure_ascii=False))

print("\n[STEP 2] Ask in notebook mode (CRAG)")
question = "Les conditions d’octroi du label startup "
chat_payload = {
    "message": question,
    "mode": "notebook",
    "project_context": ""
}
try:
    req_chat = factory.post("/api/chat/", chat_payload, format="json")
    resp_chat = chat(req_chat)
except Exception as exc:
    print("[PIPELINE][ERROR] Chat step crashed:", exc)
    traceback.print_exc()
    raise

print(f"Chat HTTP status: {resp_chat.status_code}")
try:
    chat_data = dict(resp_chat.data)
except Exception:
    chat_data = {"raw": str(resp_chat.data)}
print("Chat response JSON:")
print(json.dumps(chat_data, indent=2, ensure_ascii=False))

source_type = chat_data.get("source_type", "") if isinstance(chat_data, dict) else ""
response_text = chat_data.get("response", "") if isinstance(chat_data, dict) else ""
if source_type == "error" or _looks_like_inference_error(response_text):
    print("\n[INFERENCE][ALERT] Une erreur potentielle LLM/Groq a été détectée dans la réponse API.")
    print("[INFERENCE][ALERT] source_type:", source_type or "(vide)")
    preview = " ".join(str(response_text).split())[:300]
    print("[INFERENCE][ALERT] response_preview:", preview)
    print("[INFERENCE][ALERT] Vérifie les logs backend pour les lignes [GROQ][INFERENCE][ERROR].")

metadata = chat_data.get("metadata", {}) if isinstance(chat_data, dict) else {}
print("\n[STEP 3] CRAG metadata summary")
print("action:", metadata.get("action"))
print("documents_retrieved:", metadata.get("documents_retrieved"))
print("documents_internal_candidates:", metadata.get("documents_internal_candidates"))
print("documents_refined:", metadata.get("documents_refined"))
print("documents_used:", metadata.get("documents_used"))
print("web_query:", repr(metadata.get("web_query")))

print("\n[STEP 4] Key interpretation")
print("- If action is use_docs/combine and web_query is empty, CRAG stayed on internal uploaded docs.")
print("- Notebook mode in backend forces enable_web_fallback=False.")
print("=" * 70)
