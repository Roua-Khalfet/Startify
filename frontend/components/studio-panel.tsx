'use client'

import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface StudioPanelProps {
  onClose: () => void
}

const outputFormats = [
  { id: 'audio', label: 'Audio summary', icon: '🎵' },
  { id: 'presentation', label: 'Presentation', icon: '📊' },
  { id: 'video', label: 'Video summary', icon: '🎬' },
  { id: 'mindmap', label: 'Mind map', icon: '🧠' },
  { id: 'reports', label: 'Reports', icon: '📄' },
  { id: 'flashcards', label: 'Flashcards', icon: '🎓' },
  { id: 'quiz', label: 'Quiz', icon: '❓' },
  { id: 'infographic', label: 'Infographic', icon: '📈' },
]

export default function StudioPanel({ onClose }: StudioPanelProps) {
  return (
    <div className="flex w-80 flex-col border-l border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-3">
        <h2 className="text-sm font-semibold text-foreground">Studio</h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
        >
          <X className="h-5 w-5" />
        </Button>
      </div>

      {/* Content Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-2 gap-3">
          {outputFormats.map((format) => (
            <button
              key={format.id}
              className="flex flex-col items-center justify-center gap-2 rounded-lg bg-muted/40 hover:bg-muted/70 p-4 transition-colors group cursor-not-allowed"
            >
              <div className="text-2xl">{format.icon}</div>
              <span className="text-xs font-medium text-foreground text-center leading-tight group-hover:text-primary">
                {format.label}
              </span>
            </button>
          ))}
        </div>

        {/* Info Section */}
        <div className="mt-8 space-y-4 px-2">
          <div className="flex gap-3">
            <div className="text-2xl flex-shrink-0">🔧</div>
            <div>
              <p className="text-xs font-medium text-foreground">Studio outputs will appear here</p>
              <p className="text-xs text-muted-foreground mt-1">
                After adding sources and having a conversation, click on any format to create and customize your output.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Add Note Button */}
      <div className="border-t border-border px-4 py-3">
        <Button 
          className="w-full bg-foreground text-background hover:bg-foreground/90 rounded-full h-10 font-medium text-sm gap-2"
        >
          <span className="text-lg">+</span>
          Add a note
        </Button>
      </div>
    </div>
  )
}
