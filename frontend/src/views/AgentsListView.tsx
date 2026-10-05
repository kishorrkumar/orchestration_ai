import React, { useState, useEffect } from 'react'
import type { Agent } from '../lib/types'
import { api } from '../lib/api'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Monogram } from '../components/ui/Monogram'
import { Plus, Phone, Search, Trash2, Edit3, Loader2 } from 'lucide-react'

export interface AgentsListViewProps {
  onSelectAgent: (agentId: string) => void
  onCreateNew: () => void
  onStartCall: (agent: Agent) => void
}

export const AgentsListView: React.FC<AgentsListViewProps> = ({
  onSelectAgent,
  onCreateNew,
  onStartCall,
}) => {
  const [agents, setAgents] = useState<Agent[]>([])
  const [search, setSearch] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const loadAgents = () => {
    setIsLoading(true)
    api
      .listAgents()
      .then(setAgents)
      .catch(console.error)
      .finally(() => setIsLoading(false))
  }

  useEffect(() => {
    loadAgents()
  }, [])

  const handleDelete = async (agentId: string) => {
    try {
      await api.deleteAgent(agentId)
      setDeleteConfirmId(null)
      loadAgents()
    } catch (err) {
      console.error('Failed to delete agent', err)
    }
  }

  const filtered = agents.filter((a) => {
    const q = search.toLowerCase()
    return (
      a.name.toLowerCase().includes(q) ||
      a.system_prompt.toLowerCase().includes(q) ||
      a.voice_id.toLowerCase().includes(q)
    )
  })

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[rgba(31,30,29,0.08)] pb-6">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#1F1E1D]">
            Voice Agents
          </h1>
          <p className="text-[14px] text-[#6B6963] mt-1">
            Real-time speech-to-speech agents powered by PersonaPlex 7B
          </p>
        </div>

        <Button
          variant="primary"
          size="regular"
          leftIcon={<Plus className="w-4 h-4" />}
          onClick={onCreateNew}
        >
          New Agent
        </Button>
      </div>

      {/* Search Bar */}
      <div className="relative">
        <Search className="w-4 h-4 text-[#9E9B93] absolute left-3.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          placeholder="Filter agents by name, prompt or voice..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full h-11 pl-10 pr-4 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.1)] rounded-xl text-[#1F1E1D] placeholder:text-[#9E9B93] focus:outline-none focus:ring-2 focus:ring-[#C2603F] focus:border-transparent transition-all shadow-2xs"
        />
      </div>

      {/* Agents List / Inset Group */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20 text-[#6B6963] space-y-3">
          <Loader2 className="w-6 h-6 animate-spin text-[#C2603F]" />
          <span className="text-[14px]">Loading voice agents...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-16 bg-[#FFFFFF] rounded-2xl border border-[rgba(31,30,29,0.08)] p-8 space-y-3 shadow-xs">
          <p className="text-[15px] font-medium text-[#1F1E1D]">
            {search ? 'No agents match your filter.' : 'No agents created yet.'}
          </p>
          <p className="text-[13px] text-[#6B6963] max-w-sm mx-auto">
            {search
              ? 'Try adjusting your search keywords.'
              : 'Create your first voice agent with an opening greeting and prompt.'}
          </p>
          {!search && (
            <div className="pt-2">
              <Button variant="primary" onClick={onCreateNew}>
                Create Voice Agent
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-2xl divide-y divide-[rgba(31,30,29,0.06)] shadow-xs overflow-hidden">
          {filtered.map((agent) => (
            <div
              key={agent.id}
              className="p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 hover:bg-[#FAF9F5] transition-colors"
            >
              {/* Agent Meta Info */}
              <div
                className="flex items-start gap-3.5 cursor-pointer flex-1"
                onClick={() => onSelectAgent(agent.id)}
              >
                <Monogram name={agent.name} size="lg" />
                <div className="space-y-1">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <span className="text-[16px] font-semibold text-[#1F1E1D] hover:text-[#C2603F] transition-colors">
                      {agent.name}
                    </span>
                    <Badge
                      variant={agent.status === 'published' ? 'published' : 'draft'}
                      dot
                    >
                      {agent.status === 'published'
                        ? `Published (v${agent.published_version})`
                        : `Draft (v${agent.current_version})`}
                    </Badge>
                  </div>
                  <p className="text-[13px] text-[#6B6963] line-clamp-1 max-w-xl">
                    {agent.system_prompt}
                  </p>
                  <div className="flex items-center gap-3 text-[12px] text-[#9E9B93] pt-0.5">
                    <span>Voice: <strong className="text-[#6B6963] font-medium">{agent.voice_id}</strong></span>
                    <span>•</span>
                    <span>Zone: <strong className="text-[#6B6963] font-medium">{agent.timezone_str}</strong></span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 self-end sm:self-center shrink-0">
                <Button
                  variant="primary"
                  size="compact"
                  leftIcon={<Phone className="w-3.5 h-3.5" />}
                  onClick={() => onStartCall(agent)}
                >
                  Talk
                </Button>

                <Button
                  variant="secondary"
                  size="compact"
                  leftIcon={<Edit3 className="w-3.5 h-3.5" />}
                  onClick={() => onSelectAgent(agent.id)}
                >
                  Edit
                </Button>

                {deleteConfirmId === agent.id ? (
                  <div className="flex items-center gap-1.5 bg-[#F9ECEB] p-1 rounded-lg">
                    <Button
                      variant="destructive"
                      size="compact"
                      onClick={() => handleDelete(agent.id)}
                    >
                      Confirm
                    </Button>
                    <Button
                      variant="quiet"
                      size="compact"
                      onClick={() => setDeleteConfirmId(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Button
                    variant="quiet"
                    size="compact"
                    className="text-[#9E9B93] hover:text-[#A63A38]"
                    onClick={() => setDeleteConfirmId(agent.id)}
                    title="Delete Agent"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
