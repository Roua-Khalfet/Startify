'use client'

import { useState, useRef, type ChangeEvent } from 'react'
import { Send, Upload, Sparkles, BookOpen, X, FileText, ToggleLeft, ToggleRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { sendChatMessage, uploadSourceFile, type BackendMode } from '@/lib/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sourceType?: string
  sources?: string[]
}

interface UploadedFile {
  name: string
  chunks: number
  status: 'uploading' | 'ready' | 'error'
  error?: string
}

export default function ChatSection() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState<BackendMode>('kb')
  const [knowledgeOnly, setKnowledgeOnly] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [showUpload, setShowUpload] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSend = async () => {
    if (!input.trim() || isLoading) return
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input, timestamp: new Date() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const result = await sendChatMessage({ message: userMsg.content, mode, knowledgeOnly })
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(), role: 'assistant', content: result.response,
        timestamp: new Date(), sourceType: result.source_type, sources: result.sources,
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(), role: 'assistant',
        content: err instanceof Error ? err.message : 'Erreur inconnue.',
        timestamp: new Date(), sourceType: 'error',
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (!files) return
    const fileList = Array.from(files)
    // Clear input immediately before async work (e.currentTarget becomes null after await)
    if (fileInputRef.current) fileInputRef.current.value = ''
    for (const file of fileList) {
      const idx = uploadedFiles.length
      setUploadedFiles(prev => [...prev, { name: file.name, chunks: 0, status: 'uploading' }])
      try {
        const result = await uploadSourceFile(file)
        setUploadedFiles(prev => prev.map((f, i) => i === idx ? { ...f, chunks: result.chunks_indexed || 0, status: 'ready' } : f))
        setMode('notebook')
      } catch (err) {
        setUploadedFiles(prev => prev.map((f, i) => i === idx ? { ...f, status: 'error', error: err instanceof Error ? err.message : 'Erreur' } : f))
      }
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-gradient-to-r from-indigo-500/5 via-violet-500/5 to-transparent">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-foreground">Chat Juridique</h2>
            <p className="text-xs text-muted-foreground">GraphRAG + CRAG • Posez vos questions</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              const next = !knowledgeOnly
              setKnowledgeOnly(next)
              if (next) {
                setMode('kb')
                setShowUpload(false)
              }
            }}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
              knowledgeOnly
                ? 'border-emerald-400 bg-emerald-50 text-emerald-700'
                : 'border-border hover:bg-secondary'
            }`}
          >
            {knowledgeOnly ? <ToggleRight className="w-4 h-4 text-emerald-600" /> : <ToggleLeft className="w-4 h-4 text-muted-foreground" />}
            {knowledgeOnly ? 'Knowledge uniquement' : 'Knowledge mixte'}
          </button>
          <button
            onClick={() => setMode(mode === 'kb' ? 'notebook' : 'kb')}
            disabled={knowledgeOnly}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border border-border hover:bg-secondary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {mode === 'kb' ? <ToggleLeft className="w-4 h-4 text-indigo-500" /> : <ToggleRight className="w-4 h-4 text-violet-500" />}
            {mode === 'kb' ? 'Base juridique' : 'Mes documents'}
          </button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowUpload(!showUpload)}
            disabled={knowledgeOnly}
            className="gap-1.5 text-xs disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Upload className="w-3.5 h-3.5" /> Upload PDF
          </Button>
        </div>
      </div>

      {/* Upload area */}
      {showUpload && (
        <div className="px-6 py-3 border-b border-border bg-indigo-500/5 space-y-2">
          <label className="cursor-pointer flex items-center justify-center gap-2 py-4 border-2 border-dashed border-indigo-300/50 rounded-xl hover:border-indigo-400 transition-colors bg-white/50">
            <Upload className="w-4 h-4 text-indigo-400" />
            <span className="text-xs font-medium text-muted-foreground">Glissez un PDF ou cliquez ici</span>
            <input ref={fileInputRef} type="file" multiple accept=".pdf" onChange={handleUpload} className="hidden" />
          </label>
          {uploadedFiles.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {uploadedFiles.map((f, i) => (
                <span key={i} className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${
                  f.status === 'ready' ? 'bg-emerald-100 text-emerald-700' :
                  f.status === 'uploading' ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
                }`}>
                  <FileText className="w-3 h-3" />
                  {f.name} {f.status === 'ready' && `(${f.chunks} chunks)`}
                  {f.status === 'uploading' && '...'}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-6">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/10 to-violet-500/10 flex items-center justify-center animate-float">
              <BookOpen className="w-10 h-10 text-indigo-500/60" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-bold gradient-text">Assistant Juridique IA</h3>
              <p className="text-sm text-muted-foreground max-w-md">
                Posez une question sur le Startup Act, la fiscalité, le droit des sociétés, ou uploadez un PDF pour l&apos;analyser.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full">
              {['Quels avantages fiscaux du Startup Act ?', 'Comment obtenir le label startup ?', 'Documents pour le congé startup ?', 'Capital minimum pour une SARL ?'].map((q, i) => (
                <button key={i} onClick={() => { setInput(q); }} className="text-left text-xs p-3 rounded-xl border border-border hover:border-indigo-300 hover:bg-indigo-50/50 transition-all text-muted-foreground hover:text-foreground">
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}>
                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span className={`text-[10px] ${msg.role === 'user' ? 'text-primary-foreground/60' : 'text-muted-foreground'}`}>
                    {msg.timestamp.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                  </span>
                  {msg.sourceType && msg.role === 'assistant' && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-indigo-100 text-indigo-600 font-medium">{msg.sourceType}</span>
                  )}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {msg.sources.slice(0, 3).map((s, i) => (
                      <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-violet-100 text-violet-600">{s.length > 40 ? s.slice(0, 40) + '...' : s}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="chat-bubble-assistant">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 rounded-full bg-fuchsia-400 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-border bg-card/80 backdrop-blur-sm p-4">
        <div className="max-w-3xl mx-auto flex gap-2">
          <Textarea
            placeholder="Posez votre question juridique..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
            className="resize-none min-h-[44px] max-h-[120px] rounded-xl"
            rows={1}
          />
          <Button onClick={handleSend} disabled={!input.trim() || isLoading} size="icon"
            className="h-[44px] w-[44px] rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-600 hover:to-violet-700 shadow-lg shadow-indigo-500/20">
            <Send className="h-4 w-4 text-white" />
          </Button>
        </div>
      </div>
    </div>
  )
}
