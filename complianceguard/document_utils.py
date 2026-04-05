from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(min_value, int(raw))
    except Exception:
        return default


def _env_csv_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values if values else list(default)


# ── Document Converter (Unstructured-based) ───────────────────────────────────

class _UnstructuredConverter:
    """
    Converter basé sur Unstructured pour extraire le texte des documents uploadés.
    Le fallback pypdf reste disponible plus bas pour les pages PDF en échec.
    """

    def convert(self, file_path: str, raises_on_error: bool = False) -> "_ConversionResult":
        file_path = Path(file_path)
        errors: list[str] = []
        full_text = ""

        try:
            full_text = self._extract_with_unstructured(file_path)
        except Exception as exc:
            if raises_on_error:
                raise
            errors.append(str(exc))

        return _ConversionResult(text=full_text, errors=errors)

    def _extract_with_unstructured(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        # Les formats texte simples restent en lecture directe pour eviter
        # des dependances inutiles et conserver la compatibilite historique.
        if suffix in (".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm"):
            return file_path.read_text(encoding="utf-8", errors="ignore").strip()

        if suffix == ".pdf":
            return self._extract_pdf_with_unstructured(file_path)

        if suffix == ".docx":
            return self._extract_docx_with_unstructured(file_path)

        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise ImportError(
                "unstructured n'est pas installe. Installez: pip install \"unstructured[pdf,docx]\""
            ) from exc

        elements = partition(filename=str(file_path))
        return self._elements_to_text(elements)

    def _extract_pdf_with_unstructured(self, file_path: Path) -> str:
        """
        Extraction PDF adaptative selon le style du document:
        - auto/fast pour PDF texte,
        - hi_res/ocr_only pour PDF scanne ou mise en page complexe.
        """
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise ImportError(
                "partition_pdf indisponible. Installez: pip install \"unstructured[pdf]\""
            ) from exc

        default_strategies = ["auto", "fast", "hi_res", "ocr_only"]
        strategies = _env_csv_list("UNSTRUCTURED_PDF_STRATEGIES", default_strategies)
        preferred_strategy = os.getenv("UNSTRUCTURED_PDF_STRATEGY", "").strip().lower()
        if preferred_strategy:
            ordered = [preferred_strategy] + [s for s in strategies if s != preferred_strategy]
            seen: set[str] = set()
            strategies = [s for s in ordered if not (s in seen or seen.add(s))]

        languages = _env_csv_list("UNSTRUCTURED_PDF_LANGUAGES", ["fra", "eng"])
        include_page_breaks = _env_bool("UNSTRUCTURED_PDF_INCLUDE_PAGE_BREAKS", False)
        min_text_chars = _env_int("UNSTRUCTURED_PDF_MIN_TEXT_CHARS", 120, min_value=1)
        ocr_max_partition = _env_int("UNSTRUCTURED_PDF_MAX_PARTITION", 1500, min_value=50)

        errors: list[str] = []
        best_text = ""

        for strategy in strategies:
            kwargs: dict[str, Any] = {
                "filename": str(file_path),
                "strategy": strategy,
                "languages": languages,
                "include_page_breaks": include_page_breaks,
            }
            if strategy == "ocr_only":
                kwargs["max_partition"] = ocr_max_partition

            try:
                elements = partition_pdf(**kwargs)
                text = self._elements_to_text(elements)
                if len(text) > len(best_text):
                    best_text = text
                if text and len(text) >= min_text_chars:
                    return text
            except Exception as exc:
                errors.append(f"{strategy}: {exc}")

        if best_text:
            return best_text

        if errors:
            raise RuntimeError(
                "Echec extraction PDF Unstructured. "
                "Strategies testees=" + ",".join(strategies) + " | " + " ; ".join(errors)
            )

        return ""

    def _extract_docx_with_unstructured(self, file_path: Path) -> str:
        try:
            from unstructured.partition.docx import partition_docx
        except ImportError as exc:
            raise ImportError(
                "partition_docx indisponible. Installez: pip install \"unstructured[docx]\""
            ) from exc

        include_page_breaks = _env_bool("UNSTRUCTURED_DOCX_INCLUDE_PAGE_BREAKS", True)
        elements = partition_docx(
            filename=str(file_path),
            include_page_breaks=include_page_breaks,
        )
        return self._elements_to_text(elements)

    @staticmethod
    def _elements_to_text(elements: list[Any] | None) -> str:
        chunks: list[str] = []
        for element in elements or []:
            text = (getattr(element, "text", "") or "").strip()
            if text:
                chunks.append(text)
        return "\n\n".join(chunks).strip()


class _ConversionResult:
    """Résultat de conversion."""

    def __init__(self, text: str, errors: list[str]):
        self.document = _ParsedDocument(text)
        self.errors = errors if errors else []


class _ParsedDocument:
    """Document parsé qui peut exporter en markdown."""

    def __init__(self, text: str):
        self._text = text

    def export_to_markdown(self) -> str:
        return self._text


def build_document_converter() -> _UnstructuredConverter:
    """
    Construit et retourne un converter base sur Unstructured
    pour l'extraction des documents uploades (PDF, DOCX, TXT, etc.).
    """
    return _UnstructuredConverter()


# ── Extraction des pages en échec ─────────────────────────────────────────────

def extract_failed_pages(errors: list | None) -> list[int]:
    """
    Analyse les erreurs de conversion et retourne les numéros
    de pages qui ont échoué (1-indexed).
    """
    if not errors:
        return []

    failed_pages: list[int] = []
    page_pattern = re.compile(r"page\s*[\#:]?\s*(\d+)", re.IGNORECASE)

    for error in errors:
        error_str = str(error)
        matches = page_pattern.findall(error_str)
        for m in matches:
            page_num = int(m)
            if page_num not in failed_pages:
                failed_pages.append(page_num)

    return sorted(failed_pages)


# ── Fallback pypdf ────────────────────────────────────────────────────────────

def fallback_extract_pdf_pages(
    file_path: str | Path,
    pages: list[int],
) -> list[tuple[int, str]]:
    """
    Extrait le texte brut des pages spécifiées (1-indexed) via pypdf.
    Utilisé en fallback quand le parsing principal échoue sur certaines pages.
    """
    from pypdf import PdfReader

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {file_path}")

    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)

    results: list[tuple[int, str]] = []
    for page_num in pages:
        idx = page_num - 1
        if idx < 0 or idx >= total_pages:
            continue
        text = reader.pages[idx].extract_text() or ""
        text = text.strip()
        if text:
            results.append((page_num, text))

    return results


# ── Découpe sémantique ────────────────────────────────────────────────────────

_ARTICLE_RE = re.compile(
    r"^\s*Art(?:icle)?\.?\s*\d+",
    re.IGNORECASE | re.MULTILINE,
)
_CHAPTER_RE = re.compile(
    r"^\s*(?:Chapitre|Titre|Section|Partie)\s+",
    re.IGNORECASE | re.MULTILINE,
)
_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s+",
    re.MULTILINE,
)


def split_semantic_chunks(
    text: str,
    max_chars: int = 2000,
    overlap: int = 300,
) -> list[dict[str, str]]:
    """
    Découpe un texte en chunks sémantiques en respectant les frontières
    naturelles (articles, chapitres, paragraphes).

    Chaque chunk est un dict : {"content": str, "chunk_type": str}
    """
    if not text or not text.strip():
        return []

    # 1. Tenter de découper par articles juridiques
    chunks = _split_by_pattern(_ARTICLE_RE, text, max_chars, overlap, "article")
    if chunks:
        return chunks

    # 2. Tenter de découper par chapitres / titres / sections
    chunks = _split_by_pattern(_CHAPTER_RE, text, max_chars, overlap, "chapitre")
    if chunks:
        return chunks

    # 3. Tenter de découper par headings markdown
    chunks = _split_by_pattern(_HEADING_RE, text, max_chars, overlap, "section")
    if chunks:
        return chunks

    # 4. Fallback : découpe par paragraphes avec respect de max_chars
    return _split_by_paragraphs(text, max_chars, overlap)


def _split_by_pattern(
    pattern: re.Pattern,
    text: str,
    max_chars: int,
    overlap: int,
    chunk_type: str,
) -> list[dict[str, str]]:
    """Découpe le texte aux positions matchant le pattern."""
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return []

    segments: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()
        if segment:
            segments.append(segment)

    preamble = text[: matches[0].start()].strip()

    chunks: list[dict[str, str]] = []
    if preamble:
        for sub in _enforce_max_chars(preamble, max_chars, overlap):
            chunks.append({"content": sub, "chunk_type": "preambule"})

    for segment in segments:
        for sub in _enforce_max_chars(segment, max_chars, overlap):
            chunks.append({"content": sub, "chunk_type": chunk_type})

    return chunks


def _split_by_paragraphs(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[dict[str, str]]:
    """Découpe par double saut de ligne, puis regroupe pour respecter max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return [{"content": text.strip(), "chunk_type": "preambule"}] if text.strip() else []

    chunks: list[dict[str, str]] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > max_chars:
            if current_parts:
                chunks.append({
                    "content": "\n\n".join(current_parts),
                    "chunk_type": "preambule",
                })
                current_parts = []
                current_len = 0

            for sub in _enforce_max_chars(para, max_chars, overlap):
                chunks.append({"content": sub, "chunk_type": "preambule"})
            continue

        if current_len + para_len + 2 <= max_chars:
            current_parts.append(para)
            current_len += para_len + 2
        else:
            if current_parts:
                chunks.append({
                    "content": "\n\n".join(current_parts),
                    "chunk_type": "preambule",
                })

                if overlap > 0 and current_parts:
                    overlap_text = current_parts[-1]
                    if len(overlap_text) > overlap:
                        overlap_text = overlap_text[-overlap:]
                    current_parts = [overlap_text, para]
                    current_len = len(overlap_text) + para_len + 2
                else:
                    current_parts = [para]
                    current_len = para_len
            else:
                current_parts = [para]
                current_len = para_len

    if current_parts:
        chunks.append({
            "content": "\n\n".join(current_parts),
            "chunk_type": "preambule",
        })

    return chunks


def _enforce_max_chars(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Garantit qu'aucun morceau ne dépasse max_chars."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            pieces.append(text[start:].strip())
            break

        best_break = -1
        search_start = max(start, end - 400)
        for i in range(end, search_start, -1):
            if text[i] in ".!?\n":
                best_break = i + 1
                break

        if best_break > start:
            pieces.append(text[start:best_break].strip())
            start = max(start + 1, best_break - overlap)
        else:
            space_pos = text.rfind(" ", search_start, end)
            if space_pos > start:
                pieces.append(text[start:space_pos].strip())
                start = max(start + 1, space_pos - overlap)
            else:
                pieces.append(text[start:end].strip())
                start = end - overlap

    return [p for p in pieces if p]
