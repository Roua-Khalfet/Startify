import { Lightbulb, Plus, Share2, Settings, User, HelpCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function Header() {
  return (
    <header className="border-b border-border bg-card shadow-sm">
      <div className="flex h-14 items-center justify-between px-6">
        {/* Left: Logo and Platform Name */}
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center rounded-lg bg-primary/10 p-1.5">
            <Lightbulb className="h-5 w-5 text-primary" fill="currentColor" />
          </div>
          <div>
            <h1 className="text-base font-semibold text-foreground">Startup Assistant</h1>
            <p className="text-xs text-muted-foreground hidden sm:block">Votre guide légal complet</p>
          </div>
        </div>

        {/* Center: Create Button (Primary Action) */}
        <div className="flex items-center gap-3">
          <Button 
            className="rounded-full bg-primary hover:bg-primary/90 text-primary-foreground px-4 h-9 text-sm font-medium gap-2"
          >
            <Plus className="h-4 w-4" />
            Nouveau projet
          </Button>
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          <Button 
            variant="ghost" 
            size="icon" 
            className="text-muted-foreground hover:text-foreground h-9 w-9"
            title="Aide"
          >
            <HelpCircle className="h-5 w-5" />
          </Button>
          <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground text-sm gap-2 h-9">
            <Share2 className="h-4 w-4" />
            <span className="hidden sm:inline">Partager</span>
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            className="text-muted-foreground hover:text-foreground h-9 w-9"
            title="Paramètres"
          >
            <Settings className="h-5 w-5" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            className="text-muted-foreground hover:text-foreground h-9 w-9 rounded-full"
            title="Profil"
          >
            <User className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </header>
  )
}
