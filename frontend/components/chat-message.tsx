import { Copy, ThumbsUp, ThumbsDown } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface ChatMessageProps {
  message: {
    role: 'user' | 'assistant'
    content: string
    citations?: number[]
  }
  isLast?: boolean
}

export default function ChatMessage({ message, isLast }: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-2xl rounded-2xl bg-primary px-4 py-3 text-primary-foreground">
          <p className="text-sm leading-relaxed">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-4">
      {/* Avatar */}
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary/10">
        <span className="text-xs font-semibold text-primary">AI</span>
      </div>

      {/* Content */}
      <div className="flex-1 max-w-2xl">
        <div className="rounded-2xl bg-card p-4 text-foreground">
          <p className="text-sm leading-relaxed">{message.content}</p>

          {/* Citations */}
          {message.citations && message.citations.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {message.citations.map((citation) => (
                <button
                  key={citation}
                  className="inline-flex items-center gap-1 rounded-full bg-primary/5 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/10 transition-colors"
                >
                  <span className="text-[10px] font-bold">{citation}</span>
                  <span>Source</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        {isLast && (
          <div className="mt-3 flex items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
            >
              <Copy className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
            >
              <ThumbsUp className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-muted-foreground hover:text-foreground"
            >
              <ThumbsDown className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
