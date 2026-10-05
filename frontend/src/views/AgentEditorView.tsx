import React, { useState, useEffect } from 'react'
import type { Agent, AgentVersion, CompilePromptResult, VoicePreset } from '../lib/types'
import { api, ApiError } from '../lib/api'
import { AGENT_TEMPLATES, type AgentTemplate } from '../lib/templates'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Input } from '../components/ui/Input'
import { TextArea } from '../components/ui/TextArea'
import { TokenMeter } from '../components/ui/TokenMeter'
import {
  ArrowLeft,
  Check,
  History,
  Phone,
  AlertTriangle,
  Info,
  Clock,
  Volume2,
  Sparkles,
  BookOpen,
  X,
} from 'lucide-react'

export interface AgentEditorViewProps {
  agentId?: string // undefined if creating new
  initialTemplate?: AgentTemplate | null
  onBack: () => void
  onStartCall: (agent: Agent) => void
  onOpenGuide?: () => void
}

const COMMON_TIMEZONES = [
  'UTC',
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'Europe/London',
  'Europe/Paris',
  'Asia/Dubai',
  'Asia/Kolkata',
  'Asia/Kathmandu',
  'Asia/Singapore',
  'Asia/Tokyo',
  'Australia/Sydney',
]

export const AgentEditorView: React.FC<AgentEditorViewProps> = ({
  agentId,
  initialTemplate,
  onBack,
  onStartCall,
  onOpenGuide,
}) => {
  const isNew = !agentId

  // Agent State
  const [agent, setAgent] = useState<Agent | null>(null)
  const [name, setName] = useState('')
  const [voiceId, setVoiceId] = useState('natural_calm')
  const [greeting, setGreeting] = useState('')
  const [agentSpeaksFirst, setAgentSpeaksFirst] = useState(true)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [ending, setEnding] = useState('')
  const [timezoneStr, setTimezoneStr] = useState('UTC')
  const [silenceTimeoutSec, setSilenceTimeoutSec] = useState(12.0)
  const [maxDurationSec, setMaxDurationSec] = useState(600.0)

  // Meta & Aux State
  const [voices, setVoices] = useState<VoicePreset[]>([])
  const [versions, setVersions] = useState<AgentVersion[]>([])
  const [showVersions, setShowVersions] = useState(false)
  const [showTemplatesModal, setShowTemplatesModal] = useState(false)
  const [compileResult, setCompileResult] = useState<CompilePromptResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isPublishing, setIsPublishing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  // 1. Load Voices & Agent (if editing)
  useEffect(() => {
    api.getVoices().then(setVoices).catch(console.error)

    if (agentId) {
      setIsLoading(true)
      Promise.all([api.getAgent(agentId), api.listVersions(agentId)])
        .then(([ag, vers]) => {
          setAgent(ag)
          setName(ag.name)
          setVoiceId(ag.voice_id)
          setGreeting(ag.greeting)
          setAgentSpeaksFirst(ag.agent_speaks_first)
          setSystemPrompt(ag.system_prompt)
          setEnding(ag.ending)
          setTimezoneStr(ag.timezone_str)
          setSilenceTimeoutSec(ag.silence_timeout_sec)
          setMaxDurationSec(ag.max_duration_sec)
          setVersions(vers)
        })
        .catch((err) => {
          setErrorMessage(err.message || 'Failed to load agent details')
        })
        .finally(() => setIsLoading(false))
    } else if (initialTemplate) {
      setName(initialTemplate.name)
      setVoiceId(initialTemplate.voice_id)
      setGreeting(initialTemplate.greeting)
      setAgentSpeaksFirst(initialTemplate.agent_speaks_first)
      setSystemPrompt(initialTemplate.system_prompt)
      setEnding(initialTemplate.ending)
      setTimezoneStr(initialTemplate.timezone_str)
      setSuccessMessage(`Loaded "${initialTemplate.name}" template.`)
    } else {
      // Default template for new agent
      setName('New Voice Agent')
      setGreeting('Hello, thank you for calling. How can I assist you today?')
      setSystemPrompt('You are a helpful and polite voice assistant. Speak concisely and naturally.')
      setEnding('Thank you so much for your time. Have a wonderful day, goodbye!')
      setTimezoneStr('UTC')
    }
  }, [agentId, initialTemplate])

  const applyTemplate = (tmpl: AgentTemplate) => {
    setName(tmpl.name)
    setVoiceId(tmpl.voice_id)
    setGreeting(tmpl.greeting)
    setAgentSpeaksFirst(tmpl.agent_speaks_first)
    setSystemPrompt(tmpl.system_prompt)
    setEnding(tmpl.ending)
    setTimezoneStr(tmpl.timezone_str)
    setShowTemplatesModal(false)
    setSuccessMessage(`Applied "${tmpl.name}" template.`)
  }

  // 2. Debounced Prompt Compilation & Linter Check
  useEffect(() => {
    if (!systemPrompt.trim()) return

    const timer = setTimeout(() => {
      api
        .compilePrompt({
          system_prompt: systemPrompt,
          greeting,
          ending,
          agent_name: name || 'Agent',
          timezone_str: timezoneStr,
        })
        .then(setCompileResult)
        .catch(console.error)
    }, 250)

    return () => clearTimeout(timer)
  }, [systemPrompt, greeting, ending, name, timezoneStr])

  // Save Draft
  const handleSaveDraft = async () => {
    setIsSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)

    try {
      if (isNew) {
        const created = await api.createAgent({
          name: name.trim(),
          voice_id: voiceId,
          greeting: greeting.trim(),
          agent_speaks_first: agentSpeaksFirst,
          system_prompt: systemPrompt.trim(),
          ending: ending.trim(),
          timezone_str: timezoneStr,
          silence_timeout_sec: silenceTimeoutSec,
          max_duration_sec: maxDurationSec,
        })
        setSuccessMessage('Agent created successfully.')
        setAgent(created)
        const vers = await api.listVersions(created.id)
        setVersions(vers)
      } else if (agent) {
        const updated = await api.updateAgent(agent.id, {
          name: name.trim(),
          voice_id: voiceId,
          greeting: greeting.trim(),
          agent_speaks_first: agentSpeaksFirst,
          system_prompt: systemPrompt.trim(),
          ending: ending.trim(),
          timezone_str: timezoneStr,
          silence_timeout_sec: silenceTimeoutSec,
          max_duration_sec: maxDurationSec,
        })
        setAgent(updated)
        setSuccessMessage('Draft saved successfully.')
        const vers = await api.listVersions(updated.id)
        setVersions(vers)
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setErrorMessage(err.problem.detail || err.problem.title)
      } else {
        setErrorMessage('Failed to save draft.')
      }
    } finally {
      setIsSaving(false)
    }
  }

  // Publish Version
  const handlePublish = async () => {
    if (!agent) return
    setIsPublishing(true)
    setErrorMessage(null)
    setSuccessMessage(null)

    try {
      // First save any unsaved changes
      await handleSaveDraft()
      const published = await api.publishVersion(agent.id, 'Published via editor')
      setAgent(published)
      setSuccessMessage(`Version ${published.published_version} is now published and active!`)
      const vers = await api.listVersions(agent.id)
      setVersions(vers)
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setErrorMessage(err.problem.detail || err.problem.title)
      } else {
        setErrorMessage('Failed to publish version.')
      }
    } finally {
      setIsPublishing(false)
    }
  }

  // Revert to Version
  const handleRevert = async (verNum: number) => {
    if (!agent) return
    setIsLoading(true)
    try {
      const reverted = await api.revertVersion(agent.id, verNum)
      setAgent(reverted)
      setName(reverted.name)
      setVoiceId(reverted.voice_id)
      setGreeting(reverted.greeting)
      setAgentSpeaksFirst(reverted.agent_speaks_first)
      setSystemPrompt(reverted.system_prompt)
      setEnding(reverted.ending)
      setTimezoneStr(reverted.timezone_str)
      setSuccessMessage(`Configuration reverted to version ${verNum}.`)
      setShowVersions(false)
      const vers = await api.listVersions(agent.id)
      setVersions(vers)
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to revert version.')
    } finally {
      setIsLoading(false)
    }
  }

  const tokenCount = compileResult?.token_count ?? 0
  const canPublish = tokenCount > 0 && tokenCount <= 350 && !isNew

  if (isLoading && !agent && !isNew) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-[14px] text-[#6B6963] animate-pulse">Loading agent details...</div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
      {/* Top Bar Navigation & Actions */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[rgba(31,30,29,0.08)] pb-6">
        <div>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-1.5 text-[13px] text-[#6B6963] hover:text-[#1F1E1D] mb-1 font-medium transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> All Agents
          </button>
          <div className="flex items-center gap-3">
            <h1 className="text-[26px] font-semibold tracking-tight text-[#1F1E1D]">
              {name || 'Untitled Agent'}
            </h1>
            {agent && (
              <Badge variant={agent.status === 'published' ? 'published' : 'draft'} dot>
                {agent.status === 'published'
                  ? `Published (v${agent.published_version})`
                  : `Draft (v${agent.current_version})`}
              </Badge>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          {agent && (
            <>
              <Button
                variant="quiet"
                size="regular"
                leftIcon={<History className="w-4 h-4" />}
                onClick={() => setShowVersions(!showVersions)}
              >
                History ({versions.length})
              </Button>
              <Button
                variant="primary"
                size="regular"
                leftIcon={<Phone className="w-4 h-4" />}
                onClick={() => onStartCall(agent)}
              >
                Test Call
              </Button>
            </>
          )}

          <Button
            variant="secondary"
            size="regular"
            isLoading={isSaving}
            onClick={handleSaveDraft}
          >
            Save Draft
          </Button>

          {agent && (
            <Button
              variant="secondary"
              size="regular"
              isLoading={isPublishing}
              disabled={!canPublish}
              onClick={handlePublish}
              title={tokenCount > 350 ? 'Hard limit exceeded (>350 tokens)' : undefined}
            >
              Publish
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
      {errorMessage && (
        <div className="p-3.5 bg-[#F9ECEB] border border-[rgba(166,58,56,0.2)] rounded-lg text-[#A63A38] text-[13px] flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{errorMessage}</span>
        </div>
      )}
      {successMessage && (
        <div className="p-3.5 bg-[#EDF4EE] border border-[rgba(56,102,65,0.2)] rounded-lg text-[#386641] text-[13px] flex items-start gap-2.5">
          <Check className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{successMessage}</span>
        </div>
      )}

      {/* Version History Drawer (Conditional) */}
      {showVersions && (
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.1)] rounded-xl p-5 shadow-xs space-y-3">
          <div className="flex items-center justify-between border-b border-[rgba(31,30,29,0.06)] pb-3">
            <h3 className="text-[14px] font-semibold text-[#1F1E1D] flex items-center gap-2">
              <History className="w-4 h-4 text-[#6B6963]" />
              Immutable Version Snapshots
            </h3>
            <button
              onClick={() => setShowVersions(false)}
              className="text-[12px] text-[#6B6963] hover:text-[#1F1E1D]"
            >
              Close
            </button>
          </div>
          <div className="divide-y divide-[rgba(31,30,29,0.06)] max-h-60 overflow-y-auto">
            {versions.map((ver) => (
              <div key={ver.version_id} className="py-2.5 flex items-center justify-between text-[13px]">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-[#1F1E1D]">Version {ver.version_number}</span>
                    {agent?.published_version === ver.version_number && (
                      <Badge variant="published" className="text-[11px] py-0 px-2">Published</Badge>
                    )}
                    <span className="text-[#9E9B93] text-[12px]">{ver.change_note || 'Snapshot'}</span>
                  </div>
                  <div className="text-[12px] text-[#6B6963] mt-0.5">
                    Tokens: <span className="tabular-nums font-mono">{ver.token_count}</span> | Voice: {ver.voice_id} | Timezone: {ver.timezone_str}
                  </div>
                </div>
                {agent?.current_version !== ver.version_number && (
                  <Button
                    variant="quiet"
                    size="compact"
                    onClick={() => handleRevert(ver.version_number)}
                  >
                    Revert
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Templates Quick Selector Modal */}
      {showTemplatesModal && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl max-w-xl w-full max-h-[85vh] flex flex-col shadow-xl overflow-hidden animate-fadeIn">
            <div className="p-5 border-b border-[var(--color-hairline)] flex items-center justify-between">
              <div>
                <h3 className="text-[16px] font-semibold text-[var(--color-text-primary)] flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-[#C2603F]" />
                  Production Voice Agent Templates
                </h3>
                <p className="text-[12px] text-[var(--color-text-secondary)] mt-0.5">
                  Select a template to prefill Name, Voice, Greeting, System Prompt, and Ending.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowTemplatesModal(false)}
                className="p-1 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] rounded-lg cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 space-y-3 overflow-y-auto divide-y divide-[var(--color-hairline)]">
              {AGENT_TEMPLATES.map((t) => (
                <div key={t.id} className="pt-3 first:pt-0 space-y-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-[14px] text-[var(--color-text-primary)]">
                          {t.name}
                        </span>
                        <Badge variant="published" className="text-[10px] py-0 px-1.5">
                          ~{t.approx_tokens} tokens
                        </Badge>
                        <Badge variant="neutral" className="text-[10px] py-0 px-1.5">
                          {t.voice_id}
                        </Badge>
                      </div>
                      <span className="text-[12px] text-[#C2603F] font-medium block mt-0.5">
                        {t.role}
                      </span>
                    </div>
                    <Button
                      variant="primary"
                      size="compact"
                      onClick={() => applyTemplate(t)}
                    >
                      Apply
                    </Button>
                  </div>
                  <p className="text-[12px] text-[var(--color-text-secondary)] italic">
                    "{t.system_prompt}"
                  </p>
                </div>
              ))}
            </div>

            <div className="p-4 border-t border-[var(--color-hairline)] bg-[var(--color-bg-sunken)]/50 flex items-center justify-between">
              {onOpenGuide ? (
                <button
                  type="button"
                  onClick={() => {
                    setShowTemplatesModal(false)
                    onOpenGuide()
                  }}
                  className="text-[12px] text-[#C2603F] hover:underline flex items-center gap-1 font-medium cursor-pointer"
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  View Full Prompting Guide
                </button>
              ) : (
                <div />
              )}
              <Button
                variant="secondary"
                size="compact"
                onClick={() => setShowTemplatesModal(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Main Form: The 6 Lean Agent Fields */}
      <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-2xl p-6 sm:p-8 space-y-8 shadow-xs">
        {/* Field 1: Name */}
        <div className="space-y-1.5">
          <Input
            label="Agent Name"
            hint="Display name & interpolates into {{agent_name}}"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Elena — Receptionist"
            maxLength={128}
          />
        </div>

        {/* Field 2: Voice Preset */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[13px] font-medium text-[#1F1E1D]">
              Voice Preset (PersonaPlex 18 Upstream Embeddings)
            </label>
            <span className="text-[12px] text-[#9E9B93]">Zero hallucinated voices</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
            {voices.map((v) => {
              const isSelected = voiceId === v.id
              return (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => setVoiceId(v.id)}
                  className={`text-left p-3 rounded-xl border transition-all text-[13px] ${
                    isSelected
                      ? 'border-[#C2603F] bg-[#FBEFEA] shadow-xs'
                      : 'border-[rgba(31,30,29,0.08)] bg-[#FAF9F5] hover:border-[rgba(31,30,29,0.18)]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-[#1F1E1D] flex items-center gap-1.5">
                      <Volume2 className={`w-3.5 h-3.5 ${isSelected ? 'text-[#C2603F]' : 'text-[#6B6963]'}`} />
                      {v.name}
                    </span>
                    <span className="text-[11px] text-[#9E9B93] uppercase font-mono">{v.gender[0]}</span>
                  </div>
                  <div className="text-[12px] text-[#6B6963] truncate">{v.speaking_style}</div>
                  <div className="text-[11px] text-[#9E9B93] mt-1">{v.accent}</div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Field 3: Greeting */}
        <div className="space-y-3">
          <TextArea
            label="Spoken Greeting"
            hint="Spoken line delivered immediately upon call connection"
            value={greeting}
            onChange={(e) => setGreeting(e.target.value)}
            rows={2}
            placeholder="Hello! Thank you for calling Metro Health. How can I help you today?"
          />
          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-2 cursor-pointer select-none text-[13px] text-[#1F1E1D]">
              <input
                type="checkbox"
                checked={agentSpeaksFirst}
                onChange={(e) => setAgentSpeaksFirst(e.target.checked)}
                className="w-4 h-4 rounded border-[rgba(31,30,29,0.2)] text-[#C2603F] focus:ring-[#C2603F]"
              />
              <span>Agent speaks first (delivers greeting automatically)</span>
            </label>
          </div>
        </div>

        {/* Field 4: System Prompt */}
        <div className="space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <label className="text-[13px] font-medium text-[#1F1E1D] block">
                System Prompt (Spoken Persona Instructions)
              </label>
              <span className="text-[12px] text-[#9E9B93]">
                Wrapped as <span className="font-mono text-[11px]">&lt;system&gt; &#123;prompt&#125; &lt;system&gt;</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              {onOpenGuide && (
                <button
                  type="button"
                  onClick={onOpenGuide}
                  className="text-[12px] text-[#C2603F] hover:underline flex items-center gap-1 font-medium"
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  Prompting Guide
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowTemplatesModal(true)}
                className="text-[12px] bg-[#FAF0EC] text-[#C2603F] border border-[rgba(194,96,63,0.2)] px-2.5 py-1 rounded-lg hover:bg-[#F5E6E0] flex items-center gap-1.5 font-medium transition-colors shadow-2xs cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5" />
                Templates ({AGENT_TEMPLATES.length})
              </button>
            </div>
          </div>

          <TextArea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={7}
            placeholder="You are an empathetic, calm, and concise clinic assistant. Help the patient schedule or check their appointment status. Conclude warmly once finished."
          />

          {/* Token Meter Live Feedback */}
          <div className="p-4 bg-[#FAF9F5] border border-[rgba(31,30,29,0.06)] rounded-xl space-y-2">
            <TokenMeter
              tokenCount={tokenCount}
              recommendedLimit={150}
              hardLimit={350}
            />
            {compileResult && (
              <div className="text-[12px] text-[#6B6963] flex items-center justify-between font-mono pt-1">
                <span>{compileResult.local_time_line}</span>
                <span>Fast start: {compileResult.is_fast_start ? '✓ YES' : '✗ SLOWER'}</span>
              </div>
            )}
          </div>

          {/* Voice Linter Warnings */}
          {compileResult?.warnings && compileResult.warnings.length > 0 && (
            <div className="space-y-2 pt-1">
              {compileResult.warnings.map((warn, i) => (
                <div
                  key={i}
                  className={`p-3 rounded-lg text-[13px] flex items-start gap-2.5 ${
                    warn.severity === 'warning'
                      ? 'bg-[#FCF4EB] text-[#995D1A] border border-[rgba(153,93,26,0.2)]'
                      : 'bg-[#F3F1EA] text-[#6B6963] border border-[rgba(31,30,29,0.08)]'
                  }`}
                >
                  <Info className="w-4 h-4 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">{warn.rule}:</span> {warn.message}
                    {warn.match && (
                      <span className="block font-mono text-[11px] mt-0.5 opacity-80">
                        Found: "{warn.match}"
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Field 5: Ending Phrase */}
        <div className="space-y-1.5">
          <Input
            label="Call Ending Phrase"
            hint="Trigger phrase detected by EndOfCallDetector to gracefully hang up"
            value={ending}
            onChange={(e) => setEnding(e.target.value)}
            placeholder="e.g. Thanks for calling, have a wonderful day! Goodbye!"
          />
        </div>

        {/* Field 6: Timezone & Call Timeouts */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-[rgba(31,30,29,0.06)]">
          <div className="space-y-1.5">
            <label className="text-[13px] font-medium text-[#1F1E1D] flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-[#6B6963]" /> Timezone
            </label>
            <select
              value={timezoneStr}
              onChange={(e) => setTimezoneStr(e.target.value)}
              className="w-full h-10 px-3 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-[#C2603F] focus:outline-none"
            >
              {COMMON_TIMEZONES.map((tz) => (
                <option key={tz} value={tz}>
                  {tz}
                </option>
              ))}
            </select>
            <span className="text-[11px] text-[#9E9B93] block">Supports half-hour offsets</span>
          </div>

          <div className="space-y-1.5">
            <label className="text-[13px] font-medium text-[#1F1E1D]">
              Silence Hangup
            </label>
            <input
              type="number"
              value={silenceTimeoutSec}
              onChange={(e) => setSilenceTimeoutSec(parseFloat(e.target.value))}
              min={3}
              max={60}
              step={1}
              className="w-full h-10 px-3 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-[#C2603F] focus:outline-none tabular-nums"
            />
            <span className="text-[11px] text-[#9E9B93] block">Seconds of silence before end</span>
          </div>

          <div className="space-y-1.5">
            <label className="text-[13px] font-medium text-[#1F1E1D]">
              Max Duration
            </label>
            <input
              type="number"
              value={maxDurationSec}
              onChange={(e) => setMaxDurationSec(parseFloat(e.target.value))}
              min={30}
              max={3600}
              step={30}
              className="w-full h-10 px-3 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-[#C2603F] focus:outline-none tabular-nums"
            />
            <span className="text-[11px] text-[#9E9B93] block">Hard cap in seconds (default 600s)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
