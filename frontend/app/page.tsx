'use client'

import { useState } from 'react'
import Header from '@/components/header'
import SourcePanel from '@/components/source-panel'
import WorkspacePanel from '@/components/workspace-panel'
import type { SelectedSource } from '@/components/source-panel'

export default function Page() {
  const [isSourceCollapsed, setIsSourceCollapsed] = useState(false)
  const [selectedSources, setSelectedSources] = useState<SelectedSource[]>([])

  return (
    <div className="flex h-screen flex-col bg-background">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        {/* Sources Panel - PDF Upload & URL Scraping */}
        <SourcePanel
          isCollapsed={isSourceCollapsed}
          onToggleCollapse={() => setIsSourceCollapsed(!isSourceCollapsed)}
          selectedSources={selectedSources}
          onSourcesChange={setSelectedSources}
        />

        {/* Main Workspace - Startup Guidance & Chat */}
        <WorkspacePanel selectedSources={selectedSources} />
      </div>
    </div>
  )
}
