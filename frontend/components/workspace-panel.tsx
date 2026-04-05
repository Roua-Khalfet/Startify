'use client'

import { useState } from 'react'
import { Send, BookOpen, Scale, TrendingUp, Shield } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { sendChatMessage, type BackendMode } from '@/lib/api'
import type { SelectedSource } from '@/components/source-panel'

interface WorkspacePanelProps {
  selectedSources: SelectedSource[]
}

const STARTUP_STEPS = [
  {
    id: 'legal',
    title: 'Cadre Légal',
    icon: Scale,
    description: 'Comprendre les fondamentaux juridiques',
    color: 'bg-blue-50 text-blue-700',
  },
  {
    id: 'structure',
    title: 'Structure Juridique',
    icon: Shield,
    description: 'Choisir la bonne forme',
    color: 'bg-purple-50 text-purple-700',
  },
  {
    id: 'business',
    title: 'Plan Affaires',
    icon: TrendingUp,
    description: 'Stratégie et objectifs',
    color: 'bg-green-50 text-green-700',
  },
]

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sourceType?: string
  sources?: string[]
}

export default function WorkspacePanel({ selectedSources }: WorkspacePanelProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [activeStep, setActiveStep] = useState<string | null>(null)

  const effectiveMode: BackendMode = selectedSources.some((source) => source.mode === 'notebook')
    ? 'notebook'
    : 'kb'

  const handleSendMessage = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const result = await sendChatMessage({
        message: userMessage.content,
        mode: effectiveMode,
      })

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: result.response,
        timestamp: new Date(),
        sourceType: result.source_type,
        sources: result.sources,
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: err instanceof Error ? err.message : 'Erreur inconnue lors de la requête backend.',
        timestamp: new Date(),
        sourceType: 'error',
        sources: [],
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-1 flex-col bg-muted/30 border-l border-border">
      {/* Workspace Header */}
      <div className="border-b border-border px-8 py-4 bg-card">
        <h2 className="text-sm font-semibold text-foreground">Startup Assistant</h2>
        <p className="text-xs text-muted-foreground mt-1">
          Guidage personnalisé pour votre création ({effectiveMode === 'notebook' ? 'mode notebook' : 'mode kb'})
        </p>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto flex flex-col">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center px-8 py-12">
            {/* Welcome Section */}
            <div className="w-full max-w-2xl space-y-8">
              <div className="text-center space-y-3">
                <div className="w-16 h-16 rounded-full bg-primary/10 mx-auto flex items-center justify-center">
                  <BookOpen className="w-8 h-8 text-primary" />
                </div>
                <h3 className="text-2xl font-semibold text-foreground">Bienvenue dans Startup Assistant</h3>
                <p className="text-sm text-muted-foreground max-w-md mx-auto">
                  Obtenez des conseils personnalisés pour créer votre startup en conformité avec la législation
                </p>
              </div>

              {/* Legal Guidelines Section */}
              <div className="space-y-4">
                <h4 className="text-xs font-semibold text-foreground uppercase tracking-wide">Étapes Principales</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {STARTUP_STEPS.map((step) => {
                    const Icon = step.icon
                    return (
                      <button
                        key={step.id}
                        onClick={() => setActiveStep(activeStep === step.id ? null : step.id)}
                        className={`p-4 rounded-lg border-2 border-border transition-all duration-200 hover:border-primary/50 text-left group ${
                          activeStep === step.id ? 'bg-primary/5 border-primary/50' : 'bg-card hover:bg-muted/50'
                        }`}
                      >
                        <div className={`w-10 h-10 rounded-lg ${step.color} flex items-center justify-center mb-3 group-hover:scale-110 transition-transform`}>
                          <Icon className="w-5 h-5" />
                        </div>
                        <h5 className="font-semibold text-sm text-foreground mb-1">{step.title}</h5>
                        <p className="text-xs text-muted-foreground">{step.description}</p>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Active Step Details */}
              {activeStep && (
                <div className="bg-card rounded-lg border border-border p-6 space-y-4 animate-in fade-in duration-300">
                  <h5 className="font-semibold text-foreground">
                    {STARTUP_STEPS.find(s => s.id === activeStep)?.title}
                  </h5>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {activeStep === 'legal' && "Le cadre légal définit les règles qui gouvernent votre entreprise. Comprendre ces fondamentaux est essentiel pour éviter les problèmes futurs."}
                    {activeStep === 'structure' && "Choisir entre SARL, SAS, Auto-entreprise etc. dépend de vos objectifs, de vos associés et de votre activité."}
                    {activeStep === 'business' && "Un plan affaires solide attire les investisseurs et guide votre développement. Il doit inclure votre marché, votre proposition de valeur et vos projections."}
                  </p>
                  <Button className="mt-4 bg-primary text-primary-foreground hover:bg-primary/90">
                    En savoir plus
                  </Button>
                </div>
              )}

              {/* Sources Indicator */}
              {selectedSources.length === 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
                  <p className="text-xs text-blue-700 font-medium">
                    Ajoutez des fichiers PDF ou des URLs pour obtenir des réponses plus précises basées sur vos documents
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-6 p-8">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-xl rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-card border border-border text-foreground'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.content}</p>
                  <p className={`text-xs mt-2 ${
                    message.role === 'user'
                      ? 'text-primary-foreground/70'
                      : 'text-muted-foreground'
                  }`}>
                    {message.timestamp.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                  </p>
                  {message.role === 'assistant' && (
                    <div className="mt-2 space-y-1">
                      {message.sourceType && (
                        <p className="text-xs text-muted-foreground">Source: {message.sourceType}</p>
                      )}
                      {message.sources && message.sources.length > 0 && (
                        <p className="text-xs text-muted-foreground">
                          Références: {message.sources.slice(0, 3).join(' | ')}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-card border border-border rounded-2xl px-4 py-3">
                  <div className="flex gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card p-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {selectedSources.length > 0 && (
            <div className="flex gap-2 flex-wrap">
              {selectedSources.map((source) => (
                <span key={source.id} className="text-xs bg-primary/10 text-primary px-3 py-1 rounded-full">
                  {source.title}
                </span>
              ))}
            </div>
          )}
          <div className="flex gap-3">
            <Textarea
              placeholder="Posez une question sur votre création de startup..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                  handleSendMessage()
                }
              }}
              className="resize-none"
              rows={3}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading}
              size="icon"
              className="h-auto bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
