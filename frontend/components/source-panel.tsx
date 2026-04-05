'use client'

import { useEffect, useState, type ChangeEvent } from 'react'
import { Plus, FileText, ChevronRight, Upload, Globe, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { uploadSourceFile } from '@/lib/api'

export interface SelectedSource {
  id: string
  title: string
  type: 'pdf' | 'link' | 'text'
  mode: 'kb' | 'notebook'
  chunksIndexed?: number
}

interface SourceItem {
  id: string
  title: string
  type: 'pdf' | 'link' | 'text'
  date: string
  selected: boolean
  mode: 'kb' | 'notebook'
  chunksIndexed?: number
  status?: 'ready' | 'uploading' | 'error'
  error?: string
}

interface SourcePanelProps {
  isCollapsed: boolean
  onToggleCollapse: () => void
  selectedSources: SelectedSource[]
  onSourcesChange: (sources: SelectedSource[]) => void
}

export default function SourcePanel({ isCollapsed, onToggleCollapse, selectedSources, onSourcesChange }: SourcePanelProps) {
  const [sources, setSources] = useState<SourceItem[]>([])
  const [showUpload, setShowUpload] = useState(false)
  const [showUrlScraper, setShowUrlScraper] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [scrapingUrls, setScrapingUrls] = useState<string[]>([])

  useEffect(() => {
    const selected = sources
      .filter(s => s.selected)
      .map((s): SelectedSource => ({
        id: s.id,
        title: s.title,
        type: s.type,
        mode: s.mode,
        chunksIndexed: s.chunksIndexed,
      }))
    onSourcesChange(selected)
  }, [sources, onSourcesChange])

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'pdf':
        return <FileText className="h-4 w-4" />
      case 'link':
        return <Globe className="h-4 w-4" />
      case 'text':
        return <FileText className="h-4 w-4" />
      default:
        return <FileText className="h-4 w-4" />
    }
  }

  const toggleSourceSelection = (id: string) => {
    const updated = sources.map(s => s.id === id ? { ...s, selected: !s.selected } : s)
    setSources(updated)
  }

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const inputEl = e.currentTarget
    const files = inputEl.files
    if (!files) return

    for (const file of Array.from(files)) {
      const pendingId = `${Date.now()}-${Math.random().toString(16).slice(2)}`
      setSources(prev => [
        {
          id: pendingId,
          title: file.name,
          type: 'pdf',
          date: 'En cours...',
          selected: false,
          mode: 'notebook',
          status: 'uploading',
        },
        ...prev,
      ])

      try {
        const result = await uploadSourceFile(file)
        const chunksIndexed = Number(result.chunks_indexed || 0)
        setSources(prev => prev.map(source => {
          if (source.id !== pendingId) return source
          return {
            ...source,
            date: `Ingesté (${chunksIndexed} chunks)`,
            chunksIndexed,
            selected: true,
            status: 'ready',
            mode: 'notebook',
          }
        }))
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : 'Erreur upload inconnue'
        setSources(prev => prev.map(source => {
          if (source.id !== pendingId) return source
          return {
            ...source,
            date: 'Erreur upload',
            selected: false,
            status: 'error',
            error: errMsg,
          }
        }))
      }
    }

    setShowUpload(false)
    inputEl.value = ''
  }

  const handleAddUrl = () => {
    if (!urlInput.trim()) return

    let parsedUrl: URL
    try {
      parsedUrl = new URL(urlInput)
    } catch {
      return
    }

    setScrapingUrls(prev => [...prev, urlInput])

    // Simulate scraping
    setTimeout(() => {
      const newSource: SourceItem = {
        id: Date.now().toString(),
        title: parsedUrl.hostname,
        type: 'link',
        date: 'Just now',
        selected: false,
        mode: 'kb',
      }
      setSources(prev => [newSource, ...prev])
      setScrapingUrls(prev => prev.filter(u => u !== urlInput))
      setUrlInput('')
      setShowUrlScraper(false)
    }, 1500)
  }

  return (
    <div
      className={`flex flex-col border-r border-border bg-card transition-all duration-300 ${
        isCollapsed ? 'w-16' : 'w-72'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        {!isCollapsed && (
          <h2 className="text-sm font-semibold text-foreground">Sources</h2>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapse}
          className="h-8 w-8 text-foreground hover:bg-muted"
        >
          <ChevronRight
            className={`h-4 w-4 transition-transform ${isCollapsed ? '' : 'rotate-180'}`}
          />
        </Button>
      </div>

      {!isCollapsed && (
        <>
          {/* Upload & Scraper Options */}
          <div className="px-3 py-3 space-y-2">
            <div className="flex gap-2">
              <Button
                variant={showUpload ? 'default' : 'outline'}
                size="sm"
                onClick={() => {
                  setShowUpload(!showUpload)
                  setShowUrlScraper(false)
                }}
                className="flex-1 gap-1 text-xs h-8"
              >
                <Upload className="h-3 w-3" />
                PDF
              </Button>
              <Button
                variant={showUrlScraper ? 'default' : 'outline'}
                size="sm"
                onClick={() => {
                  setShowUrlScraper(!showUrlScraper)
                  setShowUpload(false)
                }}
                className="flex-1 gap-1 text-xs h-8"
              >
                <Globe className="h-3 w-3" />
                URL
              </Button>
            </div>

            {/* PDF Upload Area */}
            {showUpload && (
              <div className="bg-primary/5 rounded-lg p-4 space-y-3 border border-primary/20">
                <label className="cursor-pointer flex items-center justify-center gap-2 py-6 border-2 border-dashed border-primary/30 rounded-lg hover:border-primary/50 transition-colors">
                  <Upload className="h-4 w-4 text-primary/60" />
                  <span className="text-xs font-medium text-muted-foreground hover:text-foreground">
                    Cliquez ou déposez des PDFs (ingestion backend)
                  </span>
                  <input
                    type="file"
                    multiple
                    accept=".pdf"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>

                <p className="text-[11px] text-muted-foreground">
                  Les PDFs sont envoyés vers /api/upload/ et disponibles en mode notebook.
                </p>
              </div>
            )}

            {/* URL Scraper Area */}
            {showUrlScraper && (
              <div className="bg-blue-50 rounded-lg p-3 space-y-2 border border-blue-200">
                <div className="flex gap-2">
                  <Input
                    placeholder="https://..."
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddUrl()}
                    className="h-8 text-xs"
                  />
                  <Button
                    onClick={handleAddUrl}
                    disabled={!urlInput.trim() || scrapingUrls.includes(urlInput)}
                    size="sm"
                    className="h-8 gap-1 px-2"
                  >
                    <Plus className="h-3 w-3" />
                  </Button>
                </div>

                {scrapingUrls.length > 0 && (
                  <div className="space-y-2">
                    {scrapingUrls.map((url) => (
                      <div key={url} className="flex items-center gap-2 text-xs">
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                        <span className="truncate text-blue-700 text-xs">{url}</span>
                      </div>
                    ))}
                  </div>
                )}
                <p className="text-[11px] text-blue-700">
                  Les URLs ajoutées servent de contexte local UI (pas d'ingestion backend pour le moment).
                </p>
              </div>
            )}
          </div>
        </>
      )}

      {/* Sources List */}
      <div className="flex-1 overflow-y-auto">
        <div className={isCollapsed ? 'space-y-2 px-2 py-4' : 'space-y-1 px-3 py-4'}>
          {sources.map((source) => (
            <button
              key={source.id}
              onClick={() => toggleSourceSelection(source.id)}
              className={`w-full group flex items-center gap-3 rounded-lg p-3 transition-all ${
                source.selected
                  ? 'bg-primary/10 border border-primary/30'
                  : 'hover:bg-muted'
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <div className="flex-shrink-0 text-foreground">
                {source.selected ? (
                  <Check className="h-4 w-4 text-primary" />
                ) : (
                  getTypeIcon(source.type)
                )}
              </div>
              {!isCollapsed && (
                <>
                  <div className="flex-1 min-w-0 text-left">
                    <p className="truncate text-sm font-medium text-foreground">
                      {source.title}
                    </p>
                    <p className="text-xs text-muted-foreground">{source.date}</p>
                    {source.error && (
                      <p className="text-[11px] text-red-600 truncate">{source.error}</p>
                    )}
                  </div>
                </>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Footer */}
      {!isCollapsed && (
        <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
          <p>{sources.filter(s => s.selected).length} source(s) sélectionnée(s)</p>
          <p>{sources.filter(s => s.mode === 'notebook' && s.status === 'ready').length} PDF(s) prêts pour CRAG</p>
          <p>{selectedSources.length} source(s) transmises au workspace</p>
        </div>
      )}
    </div>
  )
}
