import { useState, useEffect } from 'react'
import type { Agent } from './lib/types'
import { AgentsListView } from './views/AgentsListView'
import { AgentEditorView } from './views/AgentEditorView'
import { CallsHistoryView } from './views/CallsHistoryView'
import { DesignSystemView } from './views/DesignSystemView'
import { TestCallModal } from './views/TestCallModal'
import { Moon, Sun, Layers, PhoneCall, Bot } from 'lucide-react'

type ViewMode = 'agents' | 'editor' | 'calls' | 'design-system'

export function App() {
  const [view, setView] = useState<ViewMode>('agents')
  const [editingAgentId, setEditingAgentId] = useState<string | undefined>(undefined)
  const [activeCallAgent, setActiveCallAgent] = useState<Agent | null>(null)
  const [isDarkMode, setIsDarkMode] = useState(false)

  // Dark mode effect
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.setAttribute('data-theme', 'dark')
    } else {
      document.documentElement.removeAttribute('data-theme')
    }
  }, [isDarkMode])

  const navigateToEditor = (agentId?: string) => {
    setEditingAgentId(agentId)
    setView('editor')
  }

  const navigateToAgents = () => {
    setEditingAgentId(undefined)
    setView('agents')
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] text-[var(--color-text-primary)] flex flex-col transition-colors duration-200">
      {/* Top Application Bar */}
      <header className="sticky top-0 z-40 bg-[var(--color-bg-surface)]/85 backdrop-blur-md border-b border-[var(--color-hairline)] px-6 py-3.5">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          {/* Brand Logo & Title */}
          <div
            className="flex items-center gap-2.5 cursor-pointer select-none"
            onClick={navigateToAgents}
          >
            <div className="w-8 h-8 rounded-lg bg-[#C2603F] text-white flex items-center justify-center font-bold text-[15px] shadow-xs">
              P
            </div>
            <div>
              <span className="font-semibold text-[15px] tracking-tight block">
                PersonaPlex
              </span>
              <span className="text-[11px] text-[var(--color-text-tertiary)] -mt-1 block">
                Voice Agent Platform
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1">
            <button
              onClick={() => {
                setEditingAgentId(undefined)
                setView('agents')
              }}
              className={`px-3 py-1.5 rounded-lg text-[13px] font-medium flex items-center gap-1.5 transition-colors ${
                view === 'agents' || view === 'editor'
                  ? 'bg-[var(--color-bg-sunken)] text-[var(--color-text-primary)] font-semibold'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
              }`}
            >
              <Bot className="w-4 h-4" />
              Agents
            </button>

            <button
              onClick={() => setView('calls')}
              className={`px-3 py-1.5 rounded-lg text-[13px] font-medium flex items-center gap-1.5 transition-colors ${
                view === 'calls'
                  ? 'bg-[var(--color-bg-sunken)] text-[var(--color-text-primary)] font-semibold'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
              }`}
            >
              <PhoneCall className="w-4 h-4" />
              Calls
            </button>

            <button
              onClick={() => setView('design-system')}
              className={`px-3 py-1.5 rounded-lg text-[13px] font-medium flex items-center gap-1.5 transition-colors ${
                view === 'design-system'
                  ? 'bg-[var(--color-bg-sunken)] text-[var(--color-text-primary)] font-semibold'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
              }`}
            >
              <Layers className="w-4 h-4" />
              Style Guide
            </button>
          </nav>

          {/* Theme Toggle */}
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="p-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] rounded-lg hover:bg-[var(--color-bg-sunken)] transition-colors"
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>
      </header>

      {/* Main View Router */}
      <main className="flex-1 pb-16">
        {view === 'agents' && (
          <AgentsListView
            onSelectAgent={(id) => navigateToEditor(id)}
            onCreateNew={() => navigateToEditor(undefined)}
            onStartCall={(agent) => setActiveCallAgent(agent)}
          />
        )}

        {view === 'editor' && (
          <AgentEditorView
            agentId={editingAgentId}
            onBack={navigateToAgents}
            onStartCall={(agent) => setActiveCallAgent(agent)}
          />
        )}

        {view === 'calls' && (
          <CallsHistoryView onBack={navigateToAgents} />
        )}

        {view === 'design-system' && (
          <DesignSystemView onBack={navigateToAgents} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--color-hairline)] py-6 text-center text-[12px] text-[var(--color-text-tertiary)]">
        PersonaPlex 7B Speech-to-Speech Voice Agent Platform • 16 kHz Linear PCM • Dual-Decoder Mimi Audio
      </footer>

      {/* Live Voice Call Modal */}
      {activeCallAgent && (
        <TestCallModal
          agent={activeCallAgent}
          onClose={() => setActiveCallAgent(null)}
        />
      )}
    </div>
  )
}

export default App
