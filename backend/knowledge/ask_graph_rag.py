import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_plain_csv(stdout: str) -> list[dict[str, str]]:
    import csv

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.reader(lines)
    rows = list(reader)
    if not rows:
        return []
    headers = [header.strip() for header in rows[0]]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        if not row:
            continue
        padded = row + [""] * (len(headers) - len(row))
        record: dict[str, str] = {}
        for index, header in enumerate(headers):
            value = padded[index].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] == '"':
                value = value[1:-1]
            record[header] = value
        records.append(record)
    return records


def http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
            if not raw.strip():
                return response.status, {}
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="ignore") if error.fp else ""
        try:
            parsed: dict[str, Any] | list[Any] | str = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return error.code, parsed
    except Exception as error:  # noqa: BLE001
        return 599, str(error)


def cypher_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


class Neo4jRunner:
    def __init__(self, container: str, username: str, password: str, database: str) -> None:
        self.container = container
        self.username = username
        self.password = password
        self.database = database

    def run(self, query: str) -> list[dict[str, str]]:
        cmd = [
            "docker",
            "exec",
            self.container,
            "cypher-shell",
            "-u",
            self.username,
            "-p",
            self.password,
            "-d",
            self.database,
            "--format",
            "plain",
            query,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Cypher query failed: "
                f"{' '.join(cmd)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return parse_plain_csv(completed.stdout)


class AzureChat:
    def __init__(self, timeout_seconds: int) -> None:
        self.api_key = os.getenv("AZURE_API_KEY", "").strip()
        self.api_base = os.getenv("AZURE_API_BASE", "").strip()
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-05-01-preview").strip()
        configured_model = os.getenv("AZURE_MODEL", os.getenv("model", "Kimi-K2.5")).strip()
        self.model = self._normalize_model_id(configured_model)
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _normalize_model_id(model_name: str) -> str:
        normalized = model_name.strip()
        for prefix in ("azure/", "openai/", "model/"):
            if normalized.lower().startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        return normalized.strip()

    def is_enabled(self) -> bool:
        return bool(self.api_key and self.api_base and self.model)

    def _candidate_endpoints(self) -> list[str]:
        explicit = os.getenv("AZURE_CHAT_COMPLETIONS_URL", "").strip()
        if explicit:
            return [explicit]

        base = self.api_base.rstrip("/") + "/openai/v1/chat/completions"
        candidates: list[str] = []
        if self.api_version:
            candidates.append(base + "?" + urllib.parse.urlencode({"api-version": self.api_version}))

        for version in ["2024-10-21", "2024-08-01-preview", "2024-06-01", "2024-05-01-preview"]:
            url = base + "?" + urllib.parse.urlencode({"api-version": version})
            if url not in candidates:
                candidates.append(url)

        if base not in candidates:
            candidates.append(base)

        return candidates

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_enabled():
            raise RuntimeError("Azure LLM is not configured (AZURE_API_KEY / AZURE_API_BASE / model).")

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error = "unknown"
        for endpoint in self._candidate_endpoints():
            status, data = http_json(
                "POST",
                endpoint,
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                payload=payload,
                timeout=self.timeout_seconds,
            )
            if status < 300 and isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                last_error = f"invalid response payload: {data}"
                continue

            lowered = str(data).lower()
            if status == 404 or "api version not supported" in lowered:
                last_error = f"{status} at {endpoint}: {data}"
                continue

            raise RuntimeError(f"Azure chat error {status}: {data}")

        raise RuntimeError(f"Azure chat failed across endpoints: {last_error}")


class GroqChat:
    def __init__(self, timeout_seconds: int) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
        self.model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").strip()
        self.timeout_seconds = timeout_seconds

    def _sdk_base_url(self) -> str:
        """Groq SDK expects host base (e.g., https://api.groq.com), not /openai/v1 suffix."""
        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/openai/v1"):
            normalized = normalized[: -len("/openai/v1")]
        return normalized

    def is_enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_enabled():
            raise RuntimeError("Groq is not configured (GROQ_API_KEY / GROQ_BASE_URL / GROQ_MODEL).")

        try:
            from groq import Groq  # type: ignore
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"Groq SDK unavailable: {error}") from error

        client = Groq(api_key=self.api_key, base_url=self._sdk_base_url())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_tokens = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "1024"))

        # Preferred path: stream chunks exactly as Groq SDK docs.
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=True,
            stop=None,
        )

        parts: list[str] = []
        for chunk in completion:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:  # noqa: BLE001
                delta = ""
            if delta:
                parts.append(delta)

        content = "".join(parts).strip()
        if content:
            return content

        # Fallback to non-streaming if streaming yields no text.
        non_stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
        fallback_content = ""
        if getattr(non_stream, "choices", None):
            choice = non_stream.choices[0]
            message = getattr(choice, "message", None)
            fallback_content = getattr(message, "content", "") or ""

        if not fallback_content.strip():
            raise RuntimeError("Groq empty content from both stream and non-stream responses.")

        return fallback_content.strip()


def resolve_llm_provider(provider_arg: str) -> str:
    lowered = provider_arg.strip().lower()
    if lowered in {"azure", "groq"}:
        return lowered

    env_pref = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_pref in {"azure", "groq"}:
        return env_pref

    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq"
    return "azure"


def embed_query(ollama_url: str, model: str, question: str, timeout: int) -> list[float]:
    endpoint = ollama_url.rstrip("/") + "/api/embed"
    payload = {
        "model": model,
        "input": [question],
        "truncate": True,
    }
    status, data = http_json("POST", endpoint, payload=payload, timeout=timeout)
    if status >= 300:
        raise RuntimeError(f"Ollama embed failed status={status} response={data}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Ollama response: {data}")

    vectors = data.get("embeddings")
    if isinstance(vectors, list) and vectors and isinstance(vectors[0], list):
        return vectors[0]

    single = data.get("embedding")
    if isinstance(single, list):
        return single

    raise RuntimeError(f"Ollama response missing embedding field: {data}")


def qdrant_search(
    qdrant_url: str,
    api_key: str,
    collection: str,
    vector: list[float],
    top_k: int,
    timeout: int,
) -> list[dict[str, Any]]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["api-key"] = api_key.strip()

    base = qdrant_url.rstrip("/")
    candidates = [
        (
            f"{base}/collections/{urllib.parse.quote(collection)}/points/search",
            {"vector": vector, "limit": top_k, "with_payload": True, "with_vector": False},
        ),
        (
            f"{base}/collections/{urllib.parse.quote(collection)}/points/query",
            {"query": vector, "limit": top_k, "with_payload": True, "with_vector": False},
        ),
    ]

    last_error = ""
    for url, payload in candidates:
        status, data = http_json("POST", url, headers=headers, payload=payload, timeout=timeout)
        if status >= 300:
            last_error = f"status={status} response={data}"
            continue

        if not isinstance(data, dict):
            last_error = f"invalid payload from qdrant: {data}"
            continue

        result = data.get("result")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            points = result.get("points")
            if isinstance(points, list):
                return points

        last_error = f"qdrant payload shape not recognized: {data}"

    raise RuntimeError(f"Qdrant search failed: {last_error}")


def looks_like_broad_fiscal_question(question: str, response_mode: str) -> bool:
    lowered = question.lower()
    markers = [
        "obligation",
        "declaration",
        "fisc",
        "impot",
        "tva",
        "retenue",
        "cnss",
        "startup",
        "tunisie",
    ]
    hit_count = sum(1 for marker in markers if marker in lowered)
    return response_mode in {"risk", "obligations", "general"} and hit_count >= 3


def build_retrieval_queries(question: str, response_mode: str) -> list[str]:
    queries = [question]

    if looks_like_broad_fiscal_question(question, response_mode):
        queries.extend(
            [
                "Tunisie startup obligations fiscales IS impot sur les societes declaration annuelle",
                "Tunisie startup obligations TVA taxe sur la valeur ajoutee declaration et paiement",
                "Tunisie obligations retenue a la source declaration fiscale sanctions",
                "Tunisie startup CNSS obligations sociales employeur",
                "Tunisie decret 2018 840 startup obligations fiscales impot sur les societes",
                "Tunisie declaration prix de transfert et declaration pays par pays article 17 ter",
            ]
        )

    if response_mode == "comparison":
        queries.append(question + " differences conditions delais sanctions")

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique


def _theme_hit_count(text: str) -> int:
    lowered = text.lower()
    themes = [
        ["impot sur les societ", "is", "irpp"],
        ["taxe sur la valeur ajout", "tva"],
        ["retenue a la source", "retenues a la source"],
        ["cnss", "securite sociale"],
        ["declaration", "declarer"],
        ["startup", "label startup"],
    ]
    return sum(1 for terms in themes if any(term in lowered for term in terms))


def _lexical_overlap_bonus(question: str, text: str) -> float:
    question_terms = {
        term
        for term in re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", question.lower())
        if term not in {"pour", "avec", "dans", "tunisie", "startup", "savoir", "veux", "obligations"}
    }
    if not question_terms:
        return 0.0

    lowered_text = text.lower()
    overlap = sum(1 for term in question_terms if term in lowered_text)
    return min(0.03, overlap * 0.004)


def merge_hits_across_queries(
    query_hits: list[tuple[str, list[dict[str, Any]]]],
    question: str,
    desired_k: int,
) -> list[dict[str, Any]]:
    by_chunk: dict[str, dict[str, Any]] = {}

    for _, hits in query_hits:
        for rank, hit in enumerate(hits):
            payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
            if not isinstance(payload, dict):
                continue
            chunk_id = str(payload.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            try:
                score = float(hit.get("score", 0.0)) if isinstance(hit, dict) else 0.0
            except Exception:  # noqa: BLE001
                score = 0.0

            rank_bonus = 1.0 / (rank + 1)
            existing = by_chunk.get(chunk_id)
            if existing is None:
                by_chunk[chunk_id] = {
                    "hit": dict(hit),
                    "max_score": score,
                    "sum_score": score,
                    "appearances": 1,
                    "rank_bonus": rank_bonus,
                }
            else:
                existing["sum_score"] += score
                existing["appearances"] += 1
                existing["rank_bonus"] += rank_bonus
                if score > existing["max_score"]:
                    existing["max_score"] = score
                    existing["hit"] = dict(hit)

    rescored: list[dict[str, Any]] = []
    for item in by_chunk.values():
        hit = item["hit"]
        payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
        if not isinstance(payload, dict):
            payload = {}

        text = str(payload.get("text", ""))
        doc_id = str(payload.get("doc_id", ""))
        combined_text = f"{doc_id} {text}"
        theme_hits = _theme_hit_count(combined_text)

        legal_doc_bonus = 0.0
        preferred_docs = {
            "code_droits_procedures_fiscaux_2023",
            "loi_2018_56_fr",
            "decret_2018_840_startup",
            "loi_2000_82",
        }
        if doc_id in preferred_docs:
            legal_doc_bonus += 0.05
        if doc_id.startswith("rapport_"):
            legal_doc_bonus -= 0.08

        final_score = (
            float(item["max_score"])
            + 0.07 * max(0, int(item["appearances"]) - 1)
            + 0.015 * float(item["rank_bonus"])
            + 0.02 * theme_hits
            + _lexical_overlap_bonus(question, combined_text)
            + legal_doc_bonus
        )

        hit["score"] = final_score
        rescored.append(hit)

    rescored.sort(key=lambda value: float(value.get("score", 0.0)), reverse=True)

    # Keep source diversity so one dominant topic/doc does not crowd out all other fiscal categories.
    selected: list[dict[str, Any]] = []
    per_doc_count: dict[str, int] = {}
    max_per_doc = max(2, desired_k // 3)
    for hit in rescored:
        payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
        doc_id = str(payload.get("doc_id", "__unknown__")) if isinstance(payload, dict) else "__unknown__"
        if per_doc_count.get(doc_id, 0) >= max_per_doc:
            continue
        selected.append(hit)
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
        if len(selected) >= desired_k:
            break

    if len(selected) < desired_k:
        selected_ids = {
            str((hit.get("payload", {}) if isinstance(hit, dict) else {}).get("chunk_id", ""))
            for hit in selected
        }
        for hit in rescored:
            payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
            chunk_id = str(payload.get("chunk_id", "")) if isinstance(payload, dict) else ""
            if not chunk_id or chunk_id in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(chunk_id)
            if len(selected) >= desired_k:
                break

    return selected[:desired_k]


def _hit_chunk_id(hit: dict[str, Any]) -> str:
    payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("chunk_id", ""))


def _hit_text_blob(hit: dict[str, Any]) -> str:
    payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
    if not isinstance(payload, dict):
        return ""
    doc_id = str(payload.get("doc_id", ""))
    text = str(payload.get("text", ""))
    return f"{doc_id} {text}".lower()


def infer_required_themes(question: str) -> list[str]:
    lowered = question.lower()
    required: list[str] = []

    if "tva" in lowered or "valeur ajout" in lowered:
        required.append("tva")
    if "impot sur les societ" in lowered or re.search(r"\bis\b", lowered):
        required.append("is")
    if "retenue" in lowered:
        required.append("retenue")
    if "cnss" in lowered or "securite sociale" in lowered or "sécurité sociale" in lowered:
        required.append("cnss")

    if looks_like_broad_fiscal_question(question, "risk") and not required:
        required = ["is", "tva", "retenue"]

    # De-duplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for theme in required:
        if theme in seen:
            continue
        seen.add(theme)
        unique.append(theme)
    return unique


def hit_matches_theme(hit: dict[str, Any], theme: str) -> bool:
    blob = _hit_text_blob(hit)
    if theme == "tva":
        return bool(re.search(r"\btva\b", blob)) or "taxe sur la valeur ajout" in blob
    if theme == "is":
        return (
            "impôt sur les sociétés" in blob
            or "impot sur les societes" in blob
            or bool(re.search(r"\birpp\b", blob))
            or bool(re.search(r"\bis\b", blob))
        )
    if theme == "retenue":
        return "retenue à la source" in blob or "retenue a la source" in blob
    if theme == "cnss":
        return "cnss" in blob or "securite sociale" in blob or "sécurité sociale" in blob
    return False


def enforce_theme_coverage(
    selected_hits: list[dict[str, Any]],
    candidate_hits: list[dict[str, Any]],
    question: str,
    desired_k: int,
) -> list[dict[str, Any]]:
    required_themes = infer_required_themes(question)
    if not required_themes:
        return selected_hits[:desired_k]

    unique_candidates: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for hit in sorted(candidate_hits, key=lambda value: float(value.get("score", 0.0)), reverse=True):
        chunk_id = _hit_chunk_id(hit)
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        unique_candidates.append(hit)

    working: list[dict[str, Any]] = []
    working_ids: set[str] = set()
    for hit in selected_hits:
        chunk_id = _hit_chunk_id(hit)
        if not chunk_id or chunk_id in working_ids:
            continue
        working.append(hit)
        working_ids.add(chunk_id)

    for theme in required_themes:
        if any(hit_matches_theme(hit, theme) for hit in working):
            continue
        for candidate in unique_candidates:
            chunk_id = _hit_chunk_id(candidate)
            if chunk_id in working_ids:
                continue
            if hit_matches_theme(candidate, theme):
                working.append(candidate)
                working_ids.add(chunk_id)
                break

    # Boost required-theme carriers so they survive later truncation (max_sources).
    for hit in working:
        theme_count = sum(1 for theme in required_themes if hit_matches_theme(hit, theme))
        if theme_count > 0:
            try:
                hit["score"] = float(hit.get("score", 0.0)) + (0.22 * theme_count)
            except Exception:  # noqa: BLE001
                pass

    working.sort(key=lambda value: float(value.get("score", 0.0)), reverse=True)

    while len(working) > desired_k:
        removed = False
        for index in range(len(working) - 1, -1, -1):
            candidate = working[index]
            removable = True
            for theme in required_themes:
                if hit_matches_theme(candidate, theme):
                    others = any(
                        i != index and hit_matches_theme(working[i], theme)
                        for i in range(len(working))
                    )
                    if not others:
                        removable = False
                        break
            if removable:
                del working[index]
                removed = True
                break

        if not removed:
            working = working[:desired_k]
            break

    return working[:desired_k]


def build_chunk_context(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for hit in hits:
        payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        items.append(
            {
                "chunk_id": str(payload.get("chunk_id", "")),
                "doc_id": str(payload.get("doc_id", "")),
                "text": str(payload.get("text", "")),
                "section_title": str(payload.get("section_title", "")),
                "source_page_start": str(payload.get("source_page_start", "")),
                "source_page_end": str(payload.get("source_page_end", "")),
                "score": float(hit.get("score", 0.0)) if isinstance(hit, dict) else 0.0,
            }
        )
    return [item for item in items if item["chunk_id"]]


def detect_language(question: str) -> str:
    if any("\u0600" <= char <= "\u06FF" for char in question):
        return "ar"

    lowered = question.lower()
    english_markers = ["what", "how", "when", "should", "must", "can i", "risk", "deadline"]
    french_markers = ["quoi", "comment", "quand", "dois", "obligation", "risque", "delai"]

    english_hits = sum(1 for marker in english_markers if marker in lowered)
    french_hits = sum(1 for marker in french_markers if marker in lowered)

    if english_hits > french_hits:
        return "en"
    return "fr"


def detect_response_mode(question: str, forced_mode: str) -> str:
    if forced_mode != "auto":
        return forced_mode

    lowered = question.lower()

    if re.search(r"\b(est-ce que|dois-je|puis-je|faut-il|oui ou non|yes or no)\b", lowered):
        return "yes_no"
    if re.search(r"\b(vs|versus|difference|comparer|compare|meilleur|choisir entre)\b", lowered):
        return "comparison"
    if re.search(r"\b(comment|etapes|demarche|procedure|processus|marche a suivre)\b", lowered):
        return "procedure"
    if re.search(r"\b(risque|risques|sanction|penalite|amende|danger|exposition)\b", lowered):
        return "risk"
    if re.search(r"\b(c[’']?est quoi|qu[’']?est ce que|definition|definir|explique)\b", lowered):
        return "definition"
    if re.search(r"\b(quand|delai|date limite|echeance|deadline)\b", lowered):
        return "timeline"
    if re.search(r"\b(obligation|declar|conformite|compliance|doit|must)\b", lowered):
        return "obligations"

    return "general"


def detect_verbosity(question: str, forced_style: str) -> str:
    if forced_style != "auto":
        return forced_style

    lowered = question.lower()
    if re.search(r"\b(bref|court|resume|tl;dr|en 3 points|succinct)\b", lowered):
        return "concise"
    if re.search(r"\b(detail|detaille|approfondi|complet|exhaustif)\b", lowered):
        return "detailed"
    return "standard"


def language_name(code: str) -> str:
    return {
        "fr": "French",
        "en": "English",
        "ar": "Arabic",
    }.get(code, "French")


def mode_sections(mode: str) -> list[str]:
    sections = {
        "yes_no": [
            "Reponse courte (Oui/Non/Incertain)",
            "Justification basee sur les sources",
            "Conditions et exceptions",
            "Actions recommandees",
            "Sources utilisees",
        ],
        "comparison": [
            "Resume de comparaison",
            "Comparatif structure (tableau compact)",
            "Quand choisir option A ou B",
            "Risques et limites",
            "Sources utilisees",
        ],
        "procedure": [
            "Resume court",
            "Etapes pratiques (ordre chronologique)",
            "Documents / informations necessaires",
            "Erreurs frequentes a eviter",
            "Sources utilisees",
        ],
        "risk": [
            "Resume court",
            "Matrice des risques (risque, impact, priorite)",
            "Mesures de mitigation",
            "Points a verifier",
            "Sources utilisees",
        ],
        "definition": [
            "Definition simple",
            "Implication pratique pour une startup",
            "Exemple concret",
            "Limites / conditions",
            "Sources utilisees",
        ],
        "timeline": [
            "Resume court",
            "Echeances et delais",
            "Priorites immediates",
            "Points non determines",
            "Sources utilisees",
        ],
        "obligations": [
            "Resume court",
            "Obligations confirmees",
            "Risques et sanctions",
            "Actions immediates",
            "Limites et points a verifier",
            "Sources utilisees",
        ],
        "general": [
            "Resume court",
            "Points cles",
            "Actions recommandees",
            "Limites et points a verifier",
            "Sources utilisees",
        ],
    }
    return sections.get(mode, sections["general"])


def style_instructions(style: str) -> str:
    if style == "concise":
        return "Keep answer concise: 5-10 lines max, short bullets only."
    if style == "detailed":
        return "Provide rich detail with practical nuance, while staying grounded in evidence."
    return "Use balanced detail: concise but actionable."


def build_source_catalog(
    chunks: list[dict[str, Any]],
    by_chunk_articles: dict[str, list[dict[str, str]]],
    by_chunk_docs: dict[str, list[dict[str, str]]],
    article_support: dict[str, list[dict[str, str]]],
    max_sources: int,
) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        excerpt = chunk.get("text", "").strip().replace("\n", " ")
        if len(excerpt) > 520:
            excerpt = excerpt[:520] + "..."

        refs_articles = by_chunk_articles.get(chunk_id, [])
        refs_docs = by_chunk_docs.get(chunk_id, [])

        articles_short: list[str] = []
        for ref in refs_articles[:3]:
            art_label = f"{ref.get('target_doc', '')}::article::{ref.get('article_no', '')}".strip(":")
            if art_label:
                articles_short.append(art_label)

        docs_short = [ref.get("target_doc", "") for ref in refs_docs[:3] if ref.get("target_doc", "")]

        cards.append(
            {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk_id,
                "pages": f"{chunk.get('source_page_start', '')}-{chunk.get('source_page_end', '')}",
                "excerpt": excerpt,
                "score": f"{chunk.get('score', 0.0):.4f}",
                "articles": ", ".join(articles_short),
                "docs": ", ".join(docs_short),
            }
        )

        if len(cards) >= max(1, max_sources):
            break

    for article_id, rows in article_support.items():
        if len(cards) >= max(1, max_sources):
            break
        if not rows:
            continue
        row = rows[0]
        excerpt = row.get("content", "").strip().replace("\n", " ")
        if len(excerpt) > 360:
            excerpt = excerpt[:360] + "..."
        cards.append(
            {
                "doc_id": article_id.split("::article::")[0] if "::article::" in article_id else "",
                "chunk_id": row.get("chunk_id", ""),
                "pages": "",
                "excerpt": excerpt,
                "score": "n/a",
                "articles": article_id,
                "docs": "",
            }
        )

    return cards


def source_catalog_to_prompt(cards: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    source_labels: list[str] = []
    for index, card in enumerate(cards, start=1):
        sid = f"S{index}"
        source_labels.append(sid)
        lines.append(f"[{sid}]")
        lines.append(f"- doc_id: {card.get('doc_id', '')}")
        lines.append(f"- chunk_id: {card.get('chunk_id', '')}")
        lines.append(f"- pages: {card.get('pages', '')}")
        lines.append(f"- relevance_score: {card.get('score', '')}")
        if card.get("articles", ""):
            lines.append(f"- linked_articles: {card.get('articles', '')}")
        if card.get("docs", ""):
            lines.append(f"- linked_documents: {card.get('docs', '')}")
        lines.append(f"- extrait: {card.get('excerpt', '')}")
        lines.append("")
    return lines, source_labels


def build_llm_prompts(
    question: str,
    chunks: list[dict[str, Any]],
    by_chunk_articles: dict[str, list[dict[str, str]]],
    by_chunk_docs: dict[str, list[dict[str, str]]],
    article_support: dict[str, list[dict[str, str]]],
    max_context_chars: int,
    response_mode: str,
    response_style: str,
    language_code: str,
    max_sources: int,
) -> tuple[str, str, list[str]]:
    cards = build_source_catalog(
        chunks=chunks,
        by_chunk_articles=by_chunk_articles,
        by_chunk_docs=by_chunk_docs,
        article_support=article_support,
        max_sources=max_sources,
    )

    catalog_lines, source_labels = source_catalog_to_prompt(cards)
    required_sections = mode_sections(response_mode)
    output_language = language_name(language_code)

    system_prompt = (
        "You are a legal assistant for startup founders. "
        "Never fabricate laws, articles, deadlines, sanctions, or legal effects. "
        "Only use provided sources and clearly mark uncertainty when evidence is missing."
    )

    lines: list[str] = []
    lines.append("User question:")
    lines.append(question)
    lines.append("")
    lines.append(f"Answer mode: {response_mode}")
    lines.append(f"Answer style: {response_style}")
    lines.append(f"Output language: {output_language}")
    lines.append("")
    lines.append("Retrieved legal context:")
    lines.extend(catalog_lines if catalog_lines else ["(no source cards available)"])

    lines.append("Required response contract:")
    lines.append(f"- Write in {output_language}.")
    lines.append("- Keep a human, practical tone for startup operators.")
    lines.append(style_instructions(response_style))
    lines.append("- Do not mention technical backend internals.")
    lines.append("- Every factual bullet must include one or more citations like [S1], [S2].")
    lines.append("- First run a topic-alignment check: if retrieved sources are partially off-topic, explicitly say 'Information insuffisante' and avoid definitive legal conclusions.")
    lines.append("- If evidence is insufficient, write 'Information insuffisante' and list what is missing.")
    lines.append("- Do not infer legal sanctions or validity unless text evidence explicitly supports it.")
    lines.append("- Distinguish 'confirme' vs 'a verifier' claims.")
    lines.append("- Use the exact section headings below in order:")
    for heading in required_sections:
        lines.append(f"  * {heading}")

    user_prompt = "\n".join(lines)
    if len(user_prompt) > max_context_chars:
        user_prompt = user_prompt[: max_context_chars - 3] + "..."

    return system_prompt, user_prompt, source_labels


def fallback_human_answer(
    question: str,
    source_cards: list[dict[str, str]],
    response_mode: str,
    language_code: str,
    llm_error: str,
) -> str:
    if language_code == "en":
        header = "I could not produce the full AI synthesis, but here is a grounded fallback answer from retrieved legal context."
        short = "Short answer: uncertain, limited by available context."
        missing = "Missing information to confirm"
        sources_title = "Sources used"
    elif language_code == "ar":
        header = "تعذر توليد الصياغة الكاملة، لكن هذه إجابة بديلة مبنية على المصادر المتاحة."
        short = "الإجابة المختصرة: غير حاسمة بسبب محدودية السياق."
        missing = "معلومات ناقصة للتأكيد"
        sources_title = "المصادر المستخدمة"
    else:
        header = "Je n'ai pas pu generer la synthese complete du modele, mais voici une reponse de secours fondee sur les sources recuperees."
        short = "Reponse courte: incertain, contexte limite."
        missing = "Informations manquantes pour confirmer"
        sources_title = "Sources utilisees"

    lines = [header, "", short, ""]

    if response_mode == "yes_no":
        lines.append("Reponse courte (Oui/Non/Incertain): Incertain")
    elif response_mode == "comparison":
        lines.append("Resume de comparaison: contexte insuffisant pour trancher completement.")
    elif response_mode == "procedure":
        lines.append("Etapes pratiques: verification documentaire supplementaire necessaire avant execution.")
    elif response_mode == "risk":
        lines.append("Matrice des risques: les risques exacts ne peuvent pas etre qualifies sans contexte additionnel.")
    elif response_mode == "definition":
        lines.append("Definition simple: information partielle disponible uniquement.")
    elif response_mode == "timeline":
        lines.append("Echeances: non determinables de maniere fiable avec le contexte actuel.")
    else:
        lines.append("Points cles: les sources recuperées sont partielles, verification complementaire requise.")

    lines.append("")
    lines.append("Elements confirms dans le contexte:")
    if source_cards:
        for index, card in enumerate(source_cards[:6], start=1):
            lines.append(
                f"- [S{index}] {card.get('doc_id','')} / {card.get('chunk_id','')}"
            )
    else:
        lines.append("- Aucun extrait pertinent disponible.")

    lines.append("")
    lines.append(f"{missing}:")
    lines.append("- Seuils d'application et exceptions")
    lines.append("- Champ exact des obligations demandees")
    lines.append("- Confirmation officielle complete du texte cible")

    lines.append("")
    lines.append(f"{sources_title}:")
    if source_cards:
        for index, card in enumerate(source_cards[:6], start=1):
            lines.append(f"- [S{index}] {card.get('doc_id','')} - {card.get('chunk_id','')}")
    else:
        lines.append("- (aucune)")

    if llm_error:
        lines.append("")
        lines.append("Note: fallback active suite a indisponibilite temporaire du modele.")

    return "\n".join(lines)


def fetch_graph_relations(runner: Neo4jRunner, chunk_ids: list[str]) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    if not chunk_ids:
        return {}, {}

    ids_literal = "[" + ", ".join(cypher_quote(chunk_id) for chunk_id in chunk_ids) + "]"

    article_rows = runner.run(
        "WITH "
        + ids_literal
        + " AS ids "
        + "UNWIND ids AS chunk_id "
        + "MATCH (c:Chunk {chunk_id: chunk_id})-[r:REFERS_TO_ARTICLE]->(a:Article)<-[:HAS_ARTICLE]-(d:Document) "
        + "RETURN c.chunk_id AS chunk_id, a.article_id AS article_id, d.doc_id AS target_doc, "
        + "a.article_number AS article_no, r.evidence AS evidence"
    )

    doc_rows = runner.run(
        "WITH "
        + ids_literal
        + " AS ids "
        + "UNWIND ids AS chunk_id "
        + "MATCH (c:Chunk {chunk_id: chunk_id})-[r:REFERS_TO_DOC]->(d:Document) "
        + "RETURN c.chunk_id AS chunk_id, d.doc_id AS target_doc, r.evidence AS evidence"
    )

    by_chunk_articles: dict[str, list[dict[str, str]]] = {}
    for row in article_rows:
        by_chunk_articles.setdefault(row.get("chunk_id", ""), []).append(
            {
                "article_id": row.get("article_id", ""),
                "target_doc": row.get("target_doc", ""),
                "article_no": row.get("article_no", ""),
                "evidence": row.get("evidence", ""),
            }
        )

    by_chunk_docs: dict[str, list[dict[str, str]]] = {}
    for row in doc_rows:
        by_chunk_docs.setdefault(row.get("chunk_id", ""), []).append(
            {
                "target_doc": row.get("target_doc", ""),
                "evidence": row.get("evidence", ""),
            }
        )

    return by_chunk_articles, by_chunk_docs


def fetch_article_supporting_chunks(
    runner: Neo4jRunner,
    article_ids: list[str],
    per_article_limit: int,
) -> dict[str, list[dict[str, str]]]:
    if not article_ids:
        return {}

    support: dict[str, list[dict[str, str]]] = {}
    for article_id in article_ids[:20]:
        rows = runner.run(
            "MATCH (a:Article {article_id: "
            + cypher_quote(article_id)
            + "})<-[:PART_OF_ARTICLE]-(c:Chunk) "
            + "RETURN a.article_id AS article_id, c.chunk_id AS chunk_id, c.content AS content "
            + "ORDER BY c.chunk_index ASC LIMIT "
            + str(max(1, per_article_limit))
        )
        if rows:
            support[article_id] = rows
    return support


def fetch_keyword_chunks_from_neo4j(
    runner: Neo4jRunner,
    limit: int,
) -> list[dict[str, str]]:
    scoped_docs = [
        "code_droits_procedures_fiscaux_2023",
        "loi_2018_56_fr",
        "decret_2018_840_startup",
        "loi_2000_82",
        "loi_89_114",
        "loi_2007_70",
    ]
    docs_literal = "[" + ", ".join(cypher_quote(doc_id) for doc_id in scoped_docs) + "]"

    query = (
        "WITH "
        + docs_literal
        + " AS scoped_docs "
        + "MATCH (c:Chunk) "
        + "WHERE c.doc_id IN scoped_docs "
        + "WITH c, "
        + "(CASE WHEN toLower(c.content) CONTAINS 'impôt sur les sociétés' OR toLower(c.content) CONTAINS 'impot sur les societes' THEN 2 ELSE 0 END "
        + "+ CASE WHEN toLower(c.content) CONTAINS 'taxe sur la valeur ajoutée' OR toLower(c.content) CONTAINS 'taxe sur la valeur ajoutee' OR toLower(c.content) CONTAINS 'tva' THEN 2 ELSE 0 END "
        + "+ CASE WHEN toLower(c.content) CONTAINS 'retenue à la source' OR toLower(c.content) CONTAINS 'retenue a la source' THEN 1 ELSE 0 END "
        + "+ CASE WHEN toLower(c.content) CONTAINS 'cnss' OR toLower(c.content) CONTAINS 'sécurité sociale' OR toLower(c.content) CONTAINS 'securite sociale' THEN 1 ELSE 0 END "
        + "+ CASE WHEN toLower(c.content) CONTAINS 'déclaration' OR toLower(c.content) CONTAINS 'declaration' THEN 1 ELSE 0 END) AS kw_score "
        + "WHERE kw_score > 0 "
        + "RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, "
        + "replace(replace(replace(coalesce(c.content,''), '\\n', ' '), '\\r', ' '), ',', ' ') AS text, "
        + "replace(replace(coalesce(c.section_title,''), '\\n', ' '), ',', ' ') AS section_title, c.source_page_start AS source_page_start, "
        + "c.source_page_end AS source_page_end "
        + "ORDER BY kw_score DESC, c.source_page_start ASC "
        + "LIMIT "
        + str(max(1, limit))
    )
    return runner.run(query)


def keyword_rows_to_hits(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for row in rows:
        text = row.get("text", "")
        doc_id = row.get("doc_id", "")
        chunk_id = row.get("chunk_id", "")
        if not text or not doc_id or not chunk_id:
            continue

        theme_hits = _theme_hit_count(f"{doc_id} {text}")
        score = 0.45 + (theme_hits * 0.03)
        hits.append(
            {
                "score": score,
                "payload": {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": text,
                    "section_title": row.get("section_title", ""),
                    "source_page_start": row.get("source_page_start", ""),
                    "source_page_end": row.get("source_page_end", ""),
                },
            }
        )
    return hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a human GraphRAG answer (Qdrant retrieval + Neo4j relation expansion + Azure synthesis)"
    )
    parser.add_argument("question", help="User legal question")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("GRAPHRAG_TOP_K", "6")))
    parser.add_argument("--neo4j-container", default=os.getenv("NEO4J_TEST_CONTAINER", "neo4j-local"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j123"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    parser.add_argument("--qdrant-collection", default=os.getenv("QDRANT_COLLECTION_NAME", "complianceguard_chunks"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("PIPELINE_HTTP_TIMEOUT", "120")))
    parser.add_argument("--max-context-chars", type=int, default=15000)
    parser.add_argument("--article-support-per-ref", type=int, default=1)
    parser.add_argument(
        "--response-mode",
        choices=("auto", "general", "obligations", "risk", "procedure", "comparison", "yes_no", "definition", "timeline"),
        default=os.getenv("GRAPHRAG_RESPONSE_MODE", "auto"),
    )
    parser.add_argument(
        "--response-style",
        choices=("auto", "concise", "standard", "detailed"),
        default=os.getenv("GRAPHRAG_RESPONSE_STYLE", "auto"),
    )
    parser.add_argument("--max-sources", type=int, default=int(os.getenv("GRAPHRAG_MAX_SOURCES", "8")))
    parser.add_argument(
        "--llm-provider",
        choices=("auto", "azure", "groq"),
        default=os.getenv("LLM_PROVIDER", "auto"),
        help="Select chat model backend",
    )
    parser.add_argument("--strict-llm", action="store_true")
    parser.add_argument("--show-debug-context", action="store_true")
    return parser.parse_args()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        return default


def _load_env_candidates() -> None:
    current = Path(__file__).resolve()
    for env_path in (
        current.parent / ".env",
        current.parent.parent / ".env",
        current.parent.parent.parent / ".env",
    ):
        load_env(env_path)


def run_local_graph_rag(
    question: str,
    *,
    top_k: int | None = None,
    neo4j_container: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_collection: str | None = None,
    ollama_url: str | None = None,
    embed_model: str | None = None,
    timeout: int | None = None,
    max_context_chars: int | None = None,
    article_support_per_ref: int | None = None,
    response_mode: str = "auto",
    response_style: str = "auto",
    max_sources: int | None = None,
    llm_provider: str = "auto",
    strict_llm: bool = False,
    show_debug_context: bool = False,
) -> tuple[str, list[str], dict[str, Any]]:
    _load_env_candidates()

    resolved_top_k = max(1, int(top_k if top_k is not None else _env_int("GRAPHRAG_TOP_K", 6)))
    resolved_timeout = max(10, int(timeout if timeout is not None else _env_int("PIPELINE_HTTP_TIMEOUT", 120)))
    resolved_max_context = max(1000, int(max_context_chars if max_context_chars is not None else 15000))
    resolved_article_support = max(1, int(article_support_per_ref if article_support_per_ref is not None else 1))
    resolved_max_sources = max(1, int(max_sources if max_sources is not None else _env_int("GRAPHRAG_MAX_SOURCES", 8)))

    neo4j = Neo4jRunner(
        container=(neo4j_container or os.getenv("NEO4J_TEST_CONTAINER", "neo4j-local")).strip(),
        username=(neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")).strip(),
        password=(neo4j_password or os.getenv("NEO4J_PASSWORD", "neo4j123")).strip(),
        database=(neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")).strip(),
    )

    resolved_provider = resolve_llm_provider(llm_provider)
    if resolved_provider == "groq":
        chat = GroqChat(timeout_seconds=resolved_timeout)
    else:
        chat = AzureChat(timeout_seconds=resolved_timeout)

    resolved_qdrant_url = (qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")).strip()
    resolved_qdrant_api_key = (qdrant_api_key or os.getenv("QDRANT_API_KEY", "")).strip()
    resolved_qdrant_collection = (
        qdrant_collection
        or os.getenv("QDRANT_COLLECTION_NAME", "complianceguard_chunks")
    ).strip()
    resolved_ollama_url = (ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip()
    resolved_embed_model = (embed_model or os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")).strip()

    language_code = detect_language(question)
    resolved_response_mode = detect_response_mode(question, response_mode)
    resolved_response_style = detect_verbosity(question, response_style)
    required_themes = infer_required_themes(question)
    retrieval_target_k = resolved_top_k

    metadata: dict[str, Any] = {
        "provider": resolved_provider,
        "response_mode": resolved_response_mode,
        "response_style": resolved_response_style,
        "language": language_code,
        "source_type": "GraphRAG Local",
    }
    source_cards: list[dict[str, str]] = []

    try:
        retrieval_queries = build_retrieval_queries(question, resolved_response_mode)
        per_query_k = max(4, min(14, max(6, resolved_top_k)))
        desired_k = retrieval_target_k
        if looks_like_broad_fiscal_question(question, resolved_response_mode):
            desired_k = max(desired_k, 12)
        retrieval_target_k = desired_k

        query_hits: list[tuple[str, list[dict[str, Any]]]] = []
        for retrieval_query in retrieval_queries:
            try:
                query_vector = embed_query(resolved_ollama_url, resolved_embed_model, retrieval_query, resolved_timeout)
                partial_hits = qdrant_search(
                    qdrant_url=resolved_qdrant_url,
                    api_key=resolved_qdrant_api_key,
                    collection=resolved_qdrant_collection,
                    vector=query_vector,
                    top_k=per_query_k,
                    timeout=resolved_timeout,
                )
                if partial_hits:
                    query_hits.append((retrieval_query, partial_hits))
            except Exception:
                continue

        if not query_hits:
            raise RuntimeError("No retrieval query returned results from Qdrant.")

        all_candidate_hits = [hit for _, partial in query_hits for hit in partial]
        hits = merge_hits_across_queries(query_hits, question, desired_k=desired_k)

        if looks_like_broad_fiscal_question(question, resolved_response_mode):
            try:
                keyword_rows = fetch_keyword_chunks_from_neo4j(neo4j, limit=max(8, desired_k * 2))
                keyword_hits = keyword_rows_to_hits(keyword_rows)
                if keyword_hits:
                    all_candidate_hits.extend(keyword_hits)
                    hits = merge_hits_across_queries(
                        [("semantic", hits), ("keyword", keyword_hits)],
                        question,
                        desired_k=desired_k,
                    )
            except Exception:
                pass

        hits = enforce_theme_coverage(
            selected_hits=hits,
            candidate_hits=all_candidate_hits + hits,
            question=question,
            desired_k=desired_k,
        )
    except Exception as error:  # noqa: BLE001
        answer = fallback_human_answer(
            question=question,
            source_cards=[],
            response_mode=resolved_response_mode,
            language_code=language_code,
            llm_error=str(error),
        )
        metadata["fallback_reason"] = str(error)
        metadata["chunks_retrieved"] = 0
        return answer, [], metadata

    chunks = build_chunk_context(hits)
    if not chunks:
        answer = fallback_human_answer(
            question=question,
            source_cards=[],
            response_mode=resolved_response_mode,
            language_code=language_code,
            llm_error="No relevant chunks returned from Qdrant.",
        )
        metadata["fallback_reason"] = "No relevant chunks returned from Qdrant."
        metadata["chunks_retrieved"] = 0
        return answer, [], metadata

    chunk_ids = [item["chunk_id"] for item in chunks]
    try:
        by_chunk_articles, by_chunk_docs = fetch_graph_relations(neo4j, chunk_ids)
    except Exception:
        by_chunk_articles, by_chunk_docs = {}, {}

    article_ids: list[str] = []
    seen_articles: set[str] = set()
    for chunk_id in chunk_ids:
        for ref in by_chunk_articles.get(chunk_id, []):
            aid = ref.get("article_id", "")
            if aid and aid not in seen_articles:
                seen_articles.add(aid)
                article_ids.append(aid)

    try:
        article_support = fetch_article_supporting_chunks(
            neo4j,
            article_ids=article_ids,
            per_article_limit=resolved_article_support,
        )
    except Exception:
        article_support = {}

    effective_max_sources = max(
        resolved_max_sources,
        min(retrieval_target_k, 12) if required_themes else resolved_max_sources,
    )

    source_cards = build_source_catalog(
        chunks=chunks,
        by_chunk_articles=by_chunk_articles,
        by_chunk_docs=by_chunk_docs,
        article_support=article_support,
        max_sources=effective_max_sources,
    )

    system_prompt, user_prompt, source_labels = build_llm_prompts(
        question=question,
        chunks=chunks,
        by_chunk_articles=by_chunk_articles,
        by_chunk_docs=by_chunk_docs,
        article_support=article_support,
        max_context_chars=resolved_max_context,
        response_mode=resolved_response_mode,
        response_style=resolved_response_style,
        language_code=language_code,
        max_sources=effective_max_sources,
    )

    if show_debug_context:
        metadata["debug_prompt_user"] = user_prompt

    try:
        answer = chat.complete(system_prompt, user_prompt)
    except Exception as error:  # noqa: BLE001
        if strict_llm:
            raise
        answer = fallback_human_answer(
            question=question,
            source_cards=source_cards,
            response_mode=resolved_response_mode,
            language_code=language_code,
            llm_error=str(error),
        )
        metadata["llm_error"] = str(error)

    sources: list[str] = []
    for card in source_cards[:effective_max_sources]:
        doc_id = card.get("doc_id", "").strip()
        chunk_id = card.get("chunk_id", "").strip()
        if doc_id and chunk_id:
            sources.append(f"{doc_id}:{chunk_id}")
        elif chunk_id:
            sources.append(chunk_id)
        elif doc_id:
            sources.append(doc_id)

    if not sources:
        sources = source_labels

    metadata["chunks_retrieved"] = len(chunks)
    metadata["source_count"] = len(sources)
    return answer, sources, metadata


def main() -> None:
    args = parse_args()
    answer, _, _ = run_local_graph_rag(
        question=args.question,
        top_k=args.top_k,
        neo4j_container=args.neo4j_container,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection=args.qdrant_collection,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        timeout=args.timeout,
        max_context_chars=args.max_context_chars,
        article_support_per_ref=args.article_support_per_ref,
        response_mode=args.response_mode,
        response_style=args.response_style,
        max_sources=args.max_sources,
        llm_provider=args.llm_provider,
        strict_llm=args.strict_llm,
        show_debug_context=args.show_debug_context,
    )
    print(answer)


if __name__ == "__main__":
    main()
