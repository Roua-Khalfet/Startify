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
    payload = { error: "Réponse serveur invalide." }
  }

  if (!response.ok) {
    throw new Error(payload.error || payload.message || "Echec upload document")
  }

  return payload
}

export async function sendChatMessage(params: {
  message: string
  mode: BackendMode
  projectContext?: string
}): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat/`, {
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
