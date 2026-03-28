import os
import re
import yaml
from pathlib import Path

from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent  # Compliance-Startup/
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_serper_key = os.getenv("SERPER_API_KEY")
search_wrapper = None
if _serper_key:
    search_wrapper = GoogleSerperAPIWrapper(serper_api_key=_serper_key, k=10)


@tool
def serper_search(query: str) -> str:
    """Recherche Google via Serper API. Utilise cet outil pour trouver des
    informations sur le web, notamment sur les sites tunisiens de conformité
    juridique comme tunisieindustrie.nat.tn et startup.gov.tn.

    Args:
        query: La requête de recherche à effectuer.
    """
    if search_wrapper is None:
        return "Erreur: SERPER_API_KEY non configuré dans .env"
    return search_wrapper.run(query)


@tool
def scrape_website(url: str) -> str:
    """Extraire le contenu textuel d'une page web. Utilise cet outil pour lire
    le contenu détaillé d'une page trouvée via la recherche.

    Args:
        url: L'URL complète de la page web à scraper.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = "\n\n".join(doc.page_content for doc in docs)
        return content[:15000] if len(content) > 15000 else content
    except Exception as e:
        return f"Erreur lors du scraping de {url}: {e}"


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent / "config"


def _load_yaml(filename: str) -> dict:
    with open(_CONFIG_DIR / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Chain / Agent builder
# ---------------------------------------------------------------------------

# Map tool names to tool functions for invocation
_TOOLS = [serper_search, scrape_website]
_TOOL_MAP = {t.name: t for t in _TOOLS}
_URL_RE = re.compile(r"https?://[^\s\]\)>'\"`]+")


class ComplianceGuardChain:
    """LangChain agent for Tunisian legal compliance research.

    Uses ChatOpenAI with bind_tools() and a manual tool-calling loop,
    compatible with langchain >= 1.0.
    """

    def __init__(self, temperature: float = 0.2):
        # Load YAML configs
        agents_cfg = _load_yaml("agents.yaml")
        tasks_cfg = _load_yaml("tasks.yaml")

        agent_info = agents_cfg["compliance_agent"]
        task_info = tasks_cfg["compliance_research_task"]

        # Build the system prompt from agent role, goal, and backstory
        self.system_prompt = (
            f"Tu es un {agent_info['role'].strip()}\n\n"
            f"Ton objectif: {agent_info['goal'].strip()}\n\n"
            f"Contexte: {agent_info['backstory'].strip()}\n\n"
            f"Instructions de la tâche:\n{task_info['description'].strip()}\n\n"
            f"Format de sortie attendu:\n{task_info['expected_output'].strip()}"
        )

        # LLM — Azure-hosted model via OpenAI-compatible API
        raw_model = os.getenv("model", "Llama-4-Maverick-17B-128E-Instruct-FP8")
        model_name = raw_model.replace("azure/", "")

        azure_base = os.getenv("AZURE_API_BASE", "").rstrip("/")
        azure_key = os.getenv("AZURE_API_KEY", "")
        azure_api_version = os.getenv("AZURE_API_VERSION", "")

        # Normalize endpoint style
        if "services.ai.azure.com" in azure_base and not azure_base.endswith("/models"):
            api_base = f"{azure_base}/models"
        elif "openai.azure.com" in azure_base:
            if "/openai/v1" in azure_base:
                api_base = azure_base
            elif azure_base.endswith("/openai"):
                api_base = f"{azure_base}/v1"
            else:
                api_base = f"{azure_base}/openai/v1"
        elif azure_base.endswith("/v1") or azure_base.endswith("/models"):
            api_base = azure_base
        else:
            api_base = f"{azure_base}/v1"

        llm_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": azure_key,
            "base_url": api_base,
        }
        if azure_api_version:
            llm_kwargs["default_query"] = {"api-version": azure_api_version}

        self.llm = ChatOpenAI(**llm_kwargs)

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(_TOOLS)

        self.output_file = "report.md"
        self.max_iterations = 15

    def _extract_urls(self, text: str) -> list[str]:
        urls = _URL_RE.findall(text or "")
        # Keep first occurrence order while removing duplicates.
        return list(dict.fromkeys(urls))

    def _validate_url(self, url: str) -> tuple[bool, str, int | None]:
        try:
            resp = requests.get(
                url,
                timeout=12,
                allow_redirects=True,
                headers={"User-Agent": "ComplianceGuard/1.0"},
            )
            final_url = str(resp.url)
            return (resp.status_code < 400, final_url, resp.status_code)
        except Exception:
            return (False, url, None)

    def _build_link_validation_section(self, output: str) -> str:
        urls = self._extract_urls(output)
        if not urls:
            return "\n\n### Validation automatique des liens\n- Aucun lien détecté dans le rapport."

        lines = ["", "", "### Validation automatique des liens"]
        for url in urls:
            ok, final_url, status = self._validate_url(url)
            if ok:
                lines.append(
                    f"- OK ({status}): {url}" + (f" -> {final_url}" if final_url != url else "")
                )
            else:
                status_txt = str(status) if status is not None else "injoignable"
                lines.append(f"- KO ({status_txt}): {url}")
        return "\n".join(lines)

    def _search_verified_urls(self, query: str, limit: int = 5) -> list[str]:
        serper_key = os.getenv("SERPER_API_KEY", "")
        if not serper_key:
            return []

        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": 12,
                    "gl": "tn",
                    "hl": "fr",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        candidate_urls = []
        for item in data.get("organic", []):
            link = item.get("link")
            if link:
                candidate_urls.append(link)

        # Keep first occurrence order while removing duplicates.
        candidate_urls = list(dict.fromkeys(candidate_urls))

        verified_urls = []
        for url in candidate_urls:
            ok, final_url, _ = self._validate_url(url)
            if ok:
                verified_urls.append(final_url)
            if len(verified_urls) >= limit:
                break

        return verified_urls

    def run(self, query: str, current_year: str) -> str:
        """Run the compliance research agent.

        Args:
            query: The compliance question to research.
            current_year: The current year string for context.

        Returns:
            The agent's final response as a string.
        """
        human_input = (
            f"Question: {query}\n"
            f"Année courante: {current_year}\n\n"
            f"Réponds en français avec un rapport structuré en markdown."
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_input),
        ]

        # Grounding step: seed the model with verified reachable URLs.
        verified_urls = []
        try:
            verified_urls = self._search_verified_urls(
                f"{query} site:startup.gov.tn OR site:tunisieindustrie.nat.tn",
                limit=5,
            )

            if verified_urls:
                verified_block = "\n".join(f"- {u}" for u in verified_urls)
                messages.append(
                    SystemMessage(
                        content=(
                            "Utilise uniquement les URLs suivantes (déjà vérifiées HTTP 2xx/3xx). "
                            "N'ajoute pas d'autres liens.\n"
                            "URLs validées:\n"
                            f"{verified_block}"
                        )
                    )
                )
        except Exception:
            # Keep generation resilient if search/validation fails.
            pass

        # Agentic tool-calling loop
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i + 1} ---")
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, the agent is done
            if not response.tool_calls:
                print("Agent finished (no more tool calls).")
                break

            # Execute each tool call
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                print(f"  Calling tool: {tool_name}({tool_args})")

                if tool_name in _TOOL_MAP:
                    result = _TOOL_MAP[tool_name].invoke(tool_args)
                else:
                    result = f"Erreur: outil '{tool_name}' inconnu."

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )

        # Extract final text
        output = response.content if response.content else "Aucun résultat généré."

        if verified_urls:
            output += "\n\n### Sources vérifiées (pré-validées)\n"
            output += "\n".join(f"- {u}" for u in verified_urls)

        output += self._build_link_validation_section(output)

        # Save report to file
        report_path = _PROJECT_ROOT / self.output_file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"\n Rapport sauvegardé dans '{report_path}'")
        return output
