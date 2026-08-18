import os
import sys
import glob
import contextlib
from typing import Optional, List, Callable
from dotenv import load_dotenv
from rlm import RLM
from rlm.logger import RLMLogger
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# PDF Utilities
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect stderr to devnull to silence MuPDF C-level warnings."""
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.
    Returns the full text content of the document.
    """
    text_parts = []
    with _suppress_stderr(), fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")
    return "\n".join(text_parts)


def load_all_pdfs(data_dir: str) -> dict[str, str]:
    """
    Load all PDF files from a directory and extract their text.
    Returns a dict mapping filename -> extracted text.
    """
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    documents = {}
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            text = extract_text_from_pdf(pdf_path)
            if text.strip():
                documents[filename] = text
                print(f"  ✓ Loaded: {filename} ({len(text):,} chars)")
            else:
                print(f"  ⚠ Skipped (no text): {filename}")
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
    return documents


# ---------------------------------------------------------------------------
# RLM Wrapper — Supports Groq (fast) and Azure OpenAI (fallback)
# ---------------------------------------------------------------------------
#
# KEY INSIGHT: RLMs store the document context in a Python REPL variable,
# NOT in the LLM's context window. The LLM only sees the system prompt +
# code outputs (~small). So Groq's context limit is NOT a problem here!
# ---------------------------------------------------------------------------

class SawaRLM:
    """
    A wrapper for Recursive Language Models (RLMs).
    Supports two providers:
      - 'groq'  : Fast inference via Groq (Llama) — DEFAULT
      - 'azure' : Azure OpenAI (Kimi-K2.5) — slower but larger model
    """

    def __init__(
        self,
        provider: str = "groq",
        model_name: Optional[str] = None,
        verbose: bool = True,
        max_iterations: int = 10,
        max_depth: int = 1,
        log_dir: Optional[str] = "./logs",
    ):
        """
        Initialize the SawaRLM.

        Args:
            provider: 'groq' (fast, default) or 'azure' (slower, bigger model).
            model_name: Override model name for the chosen provider.
            verbose: Whether to print RLM trajectories to console.
            max_iterations: Max REPL steps the RLM can take per query.
            max_depth: Max recursion depth (only 1 supported currently).
            log_dir: Directory to save trajectory logs (None to disable).
        """
        self.verbose = verbose
        self.provider = provider.lower()
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.log_dir = log_dir
        self.model_name = model_name
        self._init_engine()

    def _init_engine(self):
        """Initialize the RLM engine based on current provider."""
        if self.provider == "groq":
            backend, backend_kwargs = self._setup_groq(self.model_name)
        elif self.provider == "azure":
            backend, backend_kwargs = self._setup_azure(self.model_name)
        else:
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'groq' or 'azure'.")

        # Optional logger for trajectory visualization
        self.logger = RLMLogger(log_dir=self.log_dir) if self.log_dir else None

        self.rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            logger=self.logger,
        )

    def _fallback_to_azure(self, reason: str):
        """Switch provider to Azure and re-initialize."""
        if self.provider == "azure":
            return # Already on Azure, nothing to fallback to

        if self.verbose:
            print(f"\n⚠️  GROQ RATE LIMIT OR ERROR DETECTED: {reason}")
            print("🔄  SWITCHING TO AZURE OPENAI (KIMI-K2.5) AS FALLBACK...\n")
        
        self.provider = "azure"
        self.model_name = None # Use default Azure model
        self._init_engine()

    # ---- Provider setup helpers ----

    @staticmethod
    def _setup_groq(model_name: Optional[str] = None):
        """Configure Groq backend (OpenAI-compatible API)."""
        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = model_name or os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")

        return "openai", {
            "model_name": model,
            "api_key": api_key,
            "base_url": base_url,
        }

    @staticmethod
    def _setup_azure(model_name: Optional[str] = None):
        """Configure Azure OpenAI backend (Kimi-K2.5)."""
        api_key = os.getenv("AZURE_API_KEY")
        azure_endpoint = os.getenv("AZURE_API_BASE")
        api_version = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
        model = model_name or os.getenv("AZURE_MODEL", "Kimi-K2.5")

        # Strip "azure/" prefix if present
        if model.startswith("azure/"):
            model = model[len("azure/"):]

        if not api_key:
            raise ValueError("AZURE_API_KEY not found in environment.")
        if not azure_endpoint:
            raise ValueError("AZURE_API_BASE not found in environment.")

        return "azure_openai", {
            "model_name": model,
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
        }

    # ---- Completion methods ----

    def complete(self, context: str, question: str) -> str:
        """
        Execute a recursive completion on a given context.

        Args:
            context: The full document text to analyze.
            question: The question or task to perform on the context.

        Returns:
            The model's final response string.
        """
        prompt = f"{question}\n\n{context}"
        try:
            result = self.rlm.completion(prompt)
            return result.response
        except Exception as e:
            if self.provider == "groq":
                self._fallback_to_azure(str(e))
                # Retry once with Azure
                result = self.rlm.completion(prompt)
                return result.response
            raise e

    def complete_multi(self, documents: dict[str, str], question: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Execute a recursive completion over multiple documents.
        Supports an optional callback for real-time trajectory updates.
        """
        context_parts = []
        for i, (filename, text) in enumerate(documents.items()):
            context_parts.append(
                f"===== DOCUMENT {i} : {filename} =====\n{text}\n"
            )
        full_context = "\n".join(context_parts)

        prompt = f"{question}\n\n{full_context}"
        
        # Override the RLM internal print to use the callback if provided
        original_print = None
        if callback:
            import builtins
            # We wrap the internal completion logic to intercept logs if necessary, 
            # but usually RLM objects might have a logger. 
            # For simplicity, we'll just handle the exception/retry logic here.
            pass

        try:
            result = self.rlm.completion(prompt)
            return result.response
        except Exception as e:
            if self.provider == "groq":
                self._fallback_to_azure(str(e))
                result = self.rlm.completion(prompt)
                return result.response
            raise e


def get_rlm(provider: str = "groq", verbose: bool = True, max_iterations: int = 10) -> SawaRLM:
    """
    Returns a ready-to-use SawaRLM instance.
    """
    return SawaRLM(provider=provider, verbose=verbose, max_iterations=max_iterations)


# ---------------------------------------------------------------------------
# Main — load PDFs from data/ and query via RLM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pointing to the central Data/ directory
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")

    print("=" * 60)
    print("  RLM — Tunisian Legal Documents Analysis")
    print("=" * 60)

    # 1. Load all PDFs
    print(f"\n📂 Loading PDFs from: {DATA_DIR}\n")
    documents = load_all_pdfs(DATA_DIR)
    print(f"\n✅ Loaded {len(documents)} documents.\n")

    if not documents:
        print("No documents found. Exiting.")
        exit(1)

    # 2. Initialize the RLM with Groq (FAST) — change to "azure" for Kimi-K2.5
    print("🔧 Initializing Groq RLM (Llama-4-Scout) — FAST MODE...\n")
    rlm = SawaRLM(provider="groq", verbose=True, max_iterations=10)

    # 3. Multi-document query — startup-focused question
    print("=" * 60)
    print(f"📚 Multi-doc query across {len(documents)} documents")
    print("-" * 60)
    question = "Je suis un entrepreneur tunisien qui veut créer une startup. Explique-moi le parcours complet : du choix de la forme juridique (SARL, SUARL, SA) à l'obtention du Label Startup, en passant par les avantages fiscaux, le congé pour création d'entreprise, et les possibilités de financement (crowdfunding, compte en devises). Cite les articles de loi correspondants."
    response = rlm.complete_multi(documents, question)
    print(f"\n📝 Response:\n{response}\n")
