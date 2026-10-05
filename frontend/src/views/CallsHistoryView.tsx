import React, { useState, useEffect } from 'react'
import type { CallSession, CallTurn } from '../lib/types'
import { api } from '../lib/api'
import { Badge } from '../components/ui/Badge'
import { formatDuration, formatTimeAgo } from '../lib/utils'
import { Phone, Loader2, ArrowLeft } from 'lucide-react'

export interface CallsHistoryViewProps {
  onBack: () => void
}

export const CallsHistoryView: React.FC<CallsHistoryViewProps> = ({ onBack }) => {
  const [calls, setCalls] = useState<CallSession[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedCall, setSelectedCall] = useState<CallSession | null>(null)
  const [turns, setTurns] = useState<CallTurn[]>([])
  const [isLoadingTurns, setIsLoadingTurns] = useState(false)

  useEffect(() => {
    setIsLoading(true)
    api
      .listCalls()
      .then(setCalls)
      .catch(console.error)
      .finally(() => setIsLoading(false))
  }, [])

  const handleSelectCall = (call: CallSession) => {
    setSelectedCall(call)
    setIsLoadingTurns(true)
    api
      .getCallTurns(call.id)
      .then(setTurns)
      .catch(console.error)
      .finally(() => setIsLoadingTurns(false))
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[rgba(31,30,29,0.08)] pb-6">
        <div>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-1.5 text-[13px] text-[#6B6963] hover:text-[#1F1E1D] mb-1 font-medium"
          >
            <ArrowLeft className="w-4 h-4" /> All Agents
          </button>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#1F1E1D]">
            Call History
          </h1>
          <p className="text-[14px] text-[#6B6963] mt-1">
            Persisted call sessions, timestamps, durations, and turn transcripts
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
        {/* Calls Table / Inset List (2 cols) */}
        <div className="md:col-span-2 bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-2xl shadow-xs overflow-hidden divide-y divide-[rgba(31,30,29,0.06)]">
          {isLoading ? (
            <div className="p-12 text-center text-[#6B6963] flex flex-col items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-[#C2603F]" />
              <span className="text-[13px]">Loading call records...</span>
            </div>
          ) : calls.length === 0 ? (
            <div className="p-12 text-center text-[#6B6963] space-y-2">
              <Phone className="w-6 h-6 mx-auto text-[#9E9B93]" />
              <p className="text-[14px] font-medium text-[#1F1E1D]">No calls recorded yet</p>
              <p className="text-[12px] text-[#9E9B93]">Completed calls will appear here automatically.</p>
            </div>
          ) : (
            calls.map((call) => {
              const isSelected = selectedCall?.id === call.id
              return (
                <div
                  key={call.id}
                  onClick={() => handleSelectCall(call)}
                  className={`p-4 flex items-center justify-between cursor-pointer transition-colors ${
                    isSelected ? 'bg-[#FAF9F5]' : 'hover:bg-[#FAF9F5]/60'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[13px] font-semibold text-[#1F1E1D]">
                        {call.id.slice(0, 18)}
                      </span>
                      <Badge
                        variant={call.status === 'completed' ? 'published' : 'warning'}
                        className="text-[11px]"
                      >
                        {call.end_reason || call.status}
                      </Badge>
                    </div>
                    <div className="text-[12px] text-[#9E9B93] flex items-center gap-3">
                      <span>Agent: {call.agent_id.slice(0, 14)}</span>
                      <span>•</span>
                      <span>{formatTimeAgo(call.created_at)}</span>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-[14px] font-mono tabular-nums font-medium text-[#1F1E1D]">
                      {formatDuration(call.duration_sec)}
                    </div>
                    <div className="text-[11px] text-[#9E9B93]">{call.total_turns} turns</div>
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* Selected Transcript Drawer (1 col) */}
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-2xl p-5 shadow-xs space-y-4">
          <div className="border-b border-[rgba(31,30,29,0.06)] pb-3">
            <h3 className="text-[14px] font-semibold text-[#1F1E1D]">
              Call Transcript
            </h3>
            <p className="text-[12px] text-[#6B6963]">
              {selectedCall ? `Session ${selectedCall.id.slice(0, 14)}` : 'Select a call to view dialogue'}
            </p>
          </div>

          {!selectedCall ? (
            <div className="py-12 text-center text-[#9E9B93] text-[13px]">
              No call selected
            </div>
          ) : isLoadingTurns ? (
            <div className="py-12 text-center text-[#6B6963] flex flex-col items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-[#C2603F]" />
              <span className="text-[12px]">Loading transcript...</span>
            </div>
          ) : turns.length === 0 ? (
            <div className="py-8 text-center text-[#9E9B93] text-[13px] italic">
              No turns captured during this call
            </div>
          ) : (
            <div className="space-y-3 max-h-[450px] overflow-y-auto pr-1">
              {turns.map((turn) => {
                const isAgent = turn.speaker === 'agent'
                return (
                  <div
                    key={turn.id}
                    className={`p-3 rounded-xl text-[13px] leading-relaxed space-y-1 ${
                      isAgent
                        ? 'bg-[#FAF9F5] border border-[rgba(31,30,29,0.06)] text-[#1F1E1D]'
                        : 'bg-[#FBEFEA]/60 border border-[rgba(194,96,63,0.15)] text-[#1F1E1D]'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] font-medium text-[#9E9B93]">
                      <span className={isAgent ? 'text-[#6B6963]' : 'text-[#C2603F]'}>
                        {isAgent ? 'Agent' : 'Caller'}
                      </span>
                      <span className="font-mono tabular-nums">
                        {turn.started_at_sec.toFixed(1)}s
                      </span>
                    </div>
                    <p>{turn.text}</p>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
