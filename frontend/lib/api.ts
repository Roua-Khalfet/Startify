export type BackendMode = "kb" | "notebook"

export interface UploadResponse {
  status: string
  file_name?: string
  chunks_indexed?: number
  collection?: string
  message?: string
  error?: string
}

export interface ChatResponse {
  response: string
  sources: string[]
  source_type: string
  metadata?: Record<string, unknown>
}

export interface ConformiteResult {
  score_global: number
  status: string
  criteres: Array<{
    label: string
    score: number
    status: string
    article: string
    article_source: string
    details: string
    category: string
    recommendation: string | null
  }>
  risk_profile: {
    niveau: string
    autorisations_requises: string[]
    capital_recommande: number
    delai_conformite: string
  }
  recommendations: string[]
  lois_applicables: string[]
}

export interface DocumentResult {
  doc_type: string
  content: string
  filename: string
}

export interface VeilleItem {
  url: string
  nom: string
  last_check: string
  has_changed: boolean
  status: string
}

export interface VeilleResponse {
  items: VeilleItem[]
  last_update: string
}

export interface GraphData {
  nodes: Array<{ id: string; label: string; type: string; properties: Record<string, unknown> }>
  edges: Array<{ source: string; target: string; relation: string }>
}

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api").replace(/\/$/, "")

export async function uploadSourceFile(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_BASE}/upload/`, {
    method: "POST",
    body: formData,
  })

  let payload: UploadResponse
  try {
    payload = (await response.json()) as UploadResponse
  } catch {
    payload = { status: "error", error: "Réponse serveur invalide." }
  }

  if (!response.ok) {
    throw new Error(payload.error || payload.message || "Echec upload document")
  }
  if (payload.status && payload.status !== "success") {
    throw new Error(payload.error || payload.message || "Echec ingestion document")
  }
  if (typeof payload.chunks_indexed === "number" && payload.chunks_indexed <= 0) {
    throw new Error("Aucun chunk extrait du document.")
  }
  return payload
}

export async function sendChatMessage(params: {
  message: string
  mode: BackendMode
  projectContext?: string
  knowledgeOnly?: boolean
}): Promise<ChatResponse> {
  const endpoint = params.knowledgeOnly ? `${API_BASE}/chat/knowledge/` : `${API_BASE}/chat/`
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: params.message,
      mode: params.mode,
      project_context: params.projectContext || "",
    }),
  })

  let payload: ChatResponse | { error?: string }
  try {
    payload = (await response.json()) as ChatResponse
  } catch {
    payload = { error: "Réponse serveur invalide." }
  }

  if (!response.ok) {
    throw new Error((payload as { error?: string }).error || "Echec chat backend")
  }
  return payload as ChatResponse
}

export async function analyzeConformite(params: {
  project_description: string
  sector: string
  capital?: number | null
  type_societe?: string
}): Promise<ConformiteResult> {
  const response = await fetch(`${API_BASE}/conformite/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
  if (!response.ok) throw new Error("Echec analyse conformité")
  return (await response.json()) as ConformiteResult
}

export async function generateDocuments(params: {
  doc_type: string
  nom_startup: string
  activite: string
  fondateurs?: string[]
  capital_social?: number
  siege_social?: string
  type_societe?: string
}): Promise<DocumentResult[]> {
  const response = await fetch(`${API_BASE}/documents/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
  if (!response.ok) throw new Error("Echec génération documents")
  return (await response.json()) as DocumentResult[]
}

export async function fetchVeille(): Promise<VeilleResponse> {
  const response = await fetch(`${API_BASE}/veille/`)
  if (!response.ok) throw new Error("Echec récupération veille")
  return (await response.json()) as VeilleResponse
}

export async function fetchGraph(): Promise<GraphData> {
  const response = await fetch(`${API_BASE}/graph/`)
  if (!response.ok) throw new Error("Echec récupération graphe")
  return (await response.json()) as GraphData
}
