# ComplianceGuard — Architecture Deep Dive

> A context document for understanding and visualizing the ComplianceGuard system architecture.

---

## 🧭 The Big Picture

ComplianceGuard is not a simple chatbot. It is an **AI-powered legal compliance engine** built for Tunisian startups navigating the Startup Act (Law No. 2018-20). Under the hood, the system is split into two interconnected hemispheres that work in concert:

| | Part A | Part B |
|---|--------|--------|
| **Name** | The Legal AI Chatbot | The Compliance & Agent Ecosystem |
| **Purpose** | Answer complex legal questions with precision | Score compliance, monitor regulations, generate documents |
| **Core Tech** | GraphRAG · CRAG · RLM | Triangulation Scoring · Veille Agents · A2A Orchestration |
| **Personality** | Deep analyst — dives into 60+ laws | Action-oriented — scores, monitors, generates |

Both hemispheres share the same foundational infrastructure: a **Knowledge Graph** (Neo4j), a **Vector Database** (Qdrant), **Local Embeddings** (Ollama), and **Multi-LLM Inference** (Azure OpenAI + Groq).

---

## 🔷 PART A — The Legal AI Chatbot

This hemisphere handles the core Q&A experience. When a startup founder asks a legal question, the system chains three advanced AI paradigms sequentially to construct a precise, cited answer.

---

### 🔹 A1. GraphRAG — Hybrid Knowledge Retrieval

> *"Don't just search for keywords — traverse the law."*

**What it does:** Combines semantic vector search with knowledge graph traversal to find the exact legal context a question requires.

**The mechanism:**

1. **Vectorize the question** — The user's question is transformed into a mathematical vector using a local embedding model (Ollama, qwen3-embedding).

2. **Dual vector search** — The system searches across **two separate vector collections**:
   - 📚 **Legal Corpus Collection** — Contains 60+ pre-ingested Tunisian legal texts (laws, decrees, circulars, codes), chunked semantically and embedded.
   - 📄 **User Uploads Collection** — Contains any documents the user uploaded during their session.

3. **Knowledge Graph traversal** — Simultaneously, the system queries the Neo4j Knowledge Graph. This graph stores legal entities (Laws, Decrees, Articles, Benefits, Conditions, Organizations) connected by meaningful relationships. The system performs **multi-hop relational traversal** — for example:
   ```
   [BCT Circular 2019-01] ──APPLIES──→ [Startup Act Law 2018-20] ──PROVIDES──→ [FX Account Benefit]
   ```
   This surfaces context that pure keyword or vector search would completely miss.

4. **Merge & Enrich** — Vector results are cross-referenced with the Knowledge Graph. Each result is enriched with linked entities (related articles, conditions, benefits), and a statistical summary of graph relations is injected to help the LLM understand the full legal landscape.

5. **Three search modes:**
   - `all` — Searches the legal corpus + user uploads + graph (default).
   - `kb` — Knowledge base only (internal legal texts + graph, no user uploads).
   - `notebook` — User-uploaded documents only (no graph, no corpus).

---

### 🔹 A2. CRAG — Corrective Retrieval-Augmented Generation

> *"Don't blindly trust what you found — verify it first."*

**What it does:** A self-correcting layer that evaluates whether retrieved documents are actually relevant and trustworthy before using them to generate an answer.

**The mechanism:**

1. **Retrieve** — Documents are fetched from the GraphRAG hybrid retriever.

2. **Grade with Algorithmic Triangulation** — Each document is scored using a **deterministic, non-LLM triangulation**. This is critical: the grading is mathematical, not hallucinatable. It fuses three signals:
   - 🎯 **Semantic Relevance** (30%) — How close is the document's vector to the question's vector?
   - 🔗 **Graph Veracity** (40%) — Does the cited legal reference actually exist in the Neo4j Knowledge Graph? If a document says "Article 12 of Law 2018-20" but that article doesn't exist as a node, the score drops.
   - 🌐 **Web Freshness** (30%) — Is the cited law still in force? A quick search on Google Serper checks for recent amendments or repeals.
   - Final score is normalized to **[-1, 1]**.

3. **Decision Gate** — Based on the score, the system dynamically chooses a strategy:
   - ✅ Score **≥ 0.6** → **Use Documents** — the retrieved context is trustworthy.
   - ❌ Score **< -0.2** → **Web Search** — the context is irrelevant; fall back entirely to live web results.
   - ⚖️ **Otherwise** → **Combine** — merge internal documents with fresh web results for a balanced answer.

4. **Refinement via Knowledge Strips** — Documents that pass the gate are broken into tiny 2-sentence strips (max 600 characters each). Each strip is independently re-scored by the LLM. Only strips that are genuinely relevant (score ≥ 0.0) survive. This eliminates noise at the sentence level, not just the document level.

5. **Query Rewriting** — If web search is triggered, the original question is automatically rewritten into 3 concise search keywords optimized for Google to maximize result quality.

6. **Generation** — The filtered, refined context is passed to the LLM for final answer synthesis.

**Special mode — Notebook:** When the question targets an uploaded document, web fallback is completely disabled. Only the user's uploaded collection is searched.

---

### 🔹 A3. RLM — Recursive Language Model

> *"When the document is too big for the AI's brain — let the AI program its own thinking."*

**What it does:** Enables the LLM to analyze massive legal documents (far beyond its context window) by offloading document storage to a code execution environment.

**The mechanism:**

1. **Load outside the brain** — All relevant legal PDFs are loaded into a variable inside a sandboxed code execution environment (Python REPL). The documents are NOT placed inside the LLM's context window.

2. **Think in code** — The LLM receives only a system prompt. To analyze the documents, it writes code — filtering, searching, cross-referencing, extracting — against the data stored in the execution environment.

3. **Execute and learn** — The code runs, and its output is fed back to the LLM.

4. **Iterate recursively** — The LLM reads the code output, writes new, refined code, executes again... This loop continues for up to **10 recursive iterations** until a complete, synthesized answer emerges.

5. **Dual-provider resilience** — Primary inference runs on **Groq** (Llama-4-Scout) for speed. If Groq hits a rate limit, the system automatically falls back to **Azure OpenAI** (Kimi-K2.5) with zero downtime.

**The key insight:** The LLM never sees the full 60+ legal documents. It only sees the outputs of its own code. This elegantly bypasses context window limitations while enabling deep multi-document analysis.

---

## 🟣 PART B — The Compliance & Agent Ecosystem

This hemisphere handles proactive intelligence: scoring a startup's legal compliance, monitoring the regulatory landscape for changes, generating ready-to-use legal documents, and orchestrating multi-agent collaboration.

---

### 🟪 B1. Triangulation Compliance Scoring Engine

> *"One source of truth is never enough — triangulate."*

**What it does:** Produces a definitive **0-100% compliance score** for a startup by fusing three independent truth sources into a weighted analysis.

**The mechanism:**

1. **Input** — The startup's profile: activity description, sector, share capital, location.

2. **Source 1 — Knowledge Graph Intelligence** — Cypher queries against Neo4j retrieve hard-coded legal thresholds from the Startup Act:
   - "Share capital must be < 15 million TND"
   - "Company must be less than 8 years old"
   - "Workforce must be under 100 employees"
   - These are matched against the startup's profile.

3. **Source 2 — Web Freshness Verification** — The system queries the Serper API to check if cited laws are still in force. It searches official Tunisian portals for recent amendments, repeals, or new circulars that might invalidate the analysis.

4. **Source 3 — Semantic LLM Analysis** — The startup's description is analyzed by the LLM to reliably extract structured intelligence:
   - Innovation score (0-100)
   - Payment activity detection (triggers BCT authorization requirements)
   - Data sensitivity level (critical / standard / minimal → triggers INPDP obligations)
   - E-commerce detection (triggers electronic commerce regulations)
   - Detected sectors (fintech, healthtech, edtech, etc.)

5. **Source 4 — Interactive Audit Quiz (optional)** — If the user completed a compliance quiz in the UI, those answers are integrated as an additional truth source.

6. **Weighted scoring across 6 criteria:**

   | Criterion | Weight | Primary Source |
   |-----------|--------|----------------|
   | Startup Label Eligibility | 25% | LLM + Knowledge Graph |
   | Share Capital Compliance | 15% | Knowledge Graph |
   | BCT Authorizations | 20% | LLM (payment detection) |
   | Data Protection (INPDP) | 15% | LLM (data sensitivity) |
   | E-commerce Compliance | 10% | LLM |
   | Legislative Freshness | 15% | Web (Serper) |

7. **Final verdict:**
   - ✅ **≥ 75%** — Compliant
   - ⚠️ **50–74%** — Compliant with reservations
   - ❌ **< 50%** — Non-compliant
   - Plus a detailed per-criterion breakdown with actionable recommendations.

---

### 🟪 B2. Regulatory Monitoring Agent ("Veille")

> *"The law changed yesterday — does your AI know?"*

**What it does:** An asynchronous agent that continuously monitors official Tunisian government websites to detect regulatory changes that could affect startup compliance.

**The mechanism:**

1. **Target portals monitored:**
   - 🏛️ **startup.gov.tn** — The official Startup Act portal
   - 🏦 **bct.gov.tn** — Central Bank of Tunisia circulars
   - 🏭 **apii.tn** — Agency for the Promotion of Industry and Innovation

2. **Asynchronous scraping** — Uses async HTTP requests + HTML parsing to extract the full text content of each portal's key pages.

3. **Change detection via hashing** — The scraped content is normalized (whitespace, formatting stripped) and a SHA-256 hash is computed. This hash is compared against the locally cached hash from the previous scan.

4. **Change reporting** — If a hash mismatch is detected (meaning something changed), the agent generates a detailed Markdown change report identifying added, modified, or removed sections.

5. **Web Search Agent (secondary)** — A separate agentic loop that uses Google search and website scraping as tools. It runs up to 15 iterations of tool-calling to research a topic. Before generating its final report, it **pre-validates every URL** via HTTP HEAD requests — only links returning 2xx/3xx status codes are included, preventing hallucinated links.

---

### 🟪 B3. A2A Multi-Agent Orchestrator (Agent-to-Agent)

> *"One agent writes. Another agent reviews. Only perfection passes."*

**What it does:** An Agent-to-Agent workflow where specialized AI agents collaborate in a directed graph, each with a distinct persona and role, to produce validated legal outputs.

**The mechanism — 6 nodes in the workflow graph:**

1. **CRAG Analysis Node** — If the user uploaded a PDF, this node analyzes it using the CRAG pipeline in notebook mode (user documents only, no web).

2. **GraphRAG Node** — Queries the internal Neo4j/Qdrant knowledge base for relevant legal context from the 60+ ingested Tunisian laws.

3. **Evaluator Node** — An **Evaluator Agent** (persona: "Critical Analyst") judges whether the internal context gathered from steps 1-2 is sufficient to answer the question. If the context is weak or ambiguous → triggers web search.

4. **Web Search Node** — Performs live web research to supplement internal knowledge with fresh, real-time information.

5. **Drafter Node** — A **Drafter Agent** (persona: "Consultant Jurist") synthesizes all gathered context — from the graph, the vector store, and the web — into a structured legal draft. The draft must contain 3 mandatory sections:
   - Direct Answer
   - Main Conditions
   - Practical Steps

6. **Reviewer Node** — A **Reviewer Agent** (persona: "Senior Lawyer") critically reviews the draft. It checks for:
   - Presence of all 3 mandatory sections
   - Legal accuracy and proper citation
   - Completeness and actionability
   
   **If approved** → The draft is marked `APPROVED` and returned as the final answer.
   
   **If rejected** → Explicit, structured feedback is sent back to the Drafter, which must rewrite. This review loop runs a **maximum of 2 revision cycles** to prevent infinite loops.

---

### 🟪 B4. Legal Document Generation Agent

> *"Don't just advise — deliver ready-to-sign documents."*

**What it does:** Automatically generates 4 types of legal documents from parameterized templates, optionally enriched by the LLM for customization.

**Document types generated:**
1. 📜 **Company Statutes** (SUARL / SARL / SA) — Full articles of incorporation with Startup Act–specific clauses baked in.
2. 📋 **Terms of Service (CGU)** — Including data protection clauses compliant with Tunisian Law No. 2004-63 (the data protection act).
3. 💰 **Investment Contract** — With fiscal incentive clauses for investors per the Startup Act, including tax deduction provisions.
4. 🏷️ **Startup Label Application** — A complete candidacy form with eligibility checklist per Decree No. 2018-840.

**How it works:** Templates are pre-filled with project-specific data (startup name, capital, founders, activity). When the user provides supplementary instructions, the LLM adapts and enriches the template while preserving compliance with Tunisian law. A full document pack can be generated and exported in one click.

---

## 🔌 Shared Infrastructure

| Layer | Technology | Role in the System |
|-------|------------|---------------------|
| 🗄️ Knowledge Graph | Neo4j Aura | Stores legal entities (Law, Decree, Article, Benefit, Condition) and their multi-hop relationships |
| 📊 Vector Database | Qdrant | Two collections: legal corpus chunks + user-uploaded document chunks |
| 🧮 Embeddings | Ollama (local) | qwen3-embedding:0.6b — converts text to vectors locally, no API calls |
| 🧠 Primary LLM | Azure OpenAI | Kimi-K2.5 — main inference for generation, extraction, and analysis |
| ⚡ Fast LLM | Groq | Llama-4-Scout — high-speed inference for RLM and rapid tasks |
| 🌐 Web Search | Google Serper API | Real-time web search for freshness checks and fallback |
| 🕷️ Web Scraping | Crawl4AI | Deep content extraction from web pages |
| 🔗 Orchestration | LangChain + LangGraph | RAG chains, tool-calling loops, and multi-agent state graphs |

---

## 🌊 End-to-End Data Flow

```
                    ┌──────────────────┐
                    │   User Question  │
                    └────────┬─────────┘
                             │
              ╔══════════════▼══════════════╗
              ║   PART A: LEGAL CHATBOT     ║
              ║                             ║
              ║  ┌─────────┐                ║
              ║  │ GraphRAG│ Neo4j + Qdrant ║
              ║  │ (Hybrid │ dual search    ║
              ║  │ Search) │                ║
              ║  └────┬────┘                ║
              ║       │ retrieved docs      ║
              ║  ┌────▼────┐                ║
              ║  │  CRAG   │ Grade → Filter ║
              ║  │(Correct)│ → Refine       ║
              ║  └────┬────┘                ║
              ║       │ verified context    ║
              ║  ┌────▼────┐                ║
              ║  │   RLM   │ Recursive REPL ║
              ║  │(Reason) │ loop analysis  ║
              ║  └────┬────┘                ║
              ╚═══════╪════════════════════ ╝
                      │ deep legal context
              ╔═══════▼════════════════════ ╗
              ║  PART B: COMPLIANCE ENGINE  ║
              ║                             ║
              ║  Scoring ◄── Veille Agent   ║
              ║  Engine      (Monitoring)   ║
              ║    │                        ║
              ║    ▼                        ║
              ║  A2A Orchestrator           ║
              ║  (Drafter ↔ Reviewer)       ║
              ║    │                        ║
              ║    ▼                        ║
              ║  Document Generator         ║
              ║  (Statutes·TOS·Contracts)   ║
              ╚═══════╪════════════════════ ╝
                      │
              ┌───────▼───────┐
              │ Final Output  │
              │ • Legal Report│
              │ • Score 0-100 │
              │ • Documents   │
              └───────────────┘
```
