import React, { useState, useEffect } from 'react'
import type {
  Agent,
  AgentVersion,
  CascadedPipelineSpec,
  CompilePromptResult,
  CredentialSummary,
  EngineType,
  ProviderManifest,
  VoicePreset,
} from '../lib/types'
import { api, ApiError } from '../lib/api'
import { AGENT_TEMPLATES, type AgentTemplate } from '../lib/templates'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Input } from '../components/ui/Input'
import { TextArea } from '../components/ui/TextArea'
import { TokenMeter } from '../components/ui/TokenMeter'
import { VoiceCloneModal } from './VoiceCloneModal'
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
  Cpu,
  Layers,
  Globe,
  Sliders,
  Key,
  CheckCircle2,
  AlertCircle,
  Trash2,
  Play,
  Pause,
} from 'lucide-react'

export interface AgentEditorViewProps {
  agentId?: string // undefined if creating new
  initialTemplate?: AgentTemplate | null
  onBack: () => void
  onStartCall: (agent: Agent) => void
  onOpenGuide?: () => void
  onOpenProviders?: () => void
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

const SUPPORTED_LANGUAGES = [
  { code: 'en', label: 'English (US / Global)', flag: '🇺🇸' },
  { code: 'en-IN', label: 'English (India)', flag: '🇮🇳' },
  { code: 'ta-IN', label: 'Tamil (தமிழ்)', flag: '🇮🇳' },
  { code: 'hi-IN', label: 'Hindi (हिंदी)', flag: '🇮🇳' },
  { code: 'te-IN', label: 'Telugu (తెలుగు)', flag: '🇮🇳' },
  { code: 'kn-IN', label: 'Kannada (ಕನ್ನಡ)', flag: '🇮🇳' },
  { code: 'ml-IN', label: 'Malayalam (മലയാളം)', flag: '🇮🇳' },
  { code: 'bn-IN', label: 'Bengali (বাংলা)', flag: '🇮🇳' },
  { code: 'mr-IN', label: 'Marathi (मराठी)', flag: '🇮🇳' },
]

const INDIAN_ENGLISH_GUIDELINES = `\n\n[Indian English Conversational Guidelines]
- Keep sentences concise, conversational, and direct.
- Never use robotic fillers such as 'Regarding your query' or 'Kindly revert back'.
- Confirm critical names, dates, or numbers respectfully and clearly.`

export const AgentEditorView: React.FC<AgentEditorViewProps> = ({
  agentId,
  initialTemplate,
  onBack,
  onStartCall,
  onOpenGuide,
  onOpenProviders,
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

  // Engine Architecture & Cascaded Cloud State
  const [engine, setEngine] = useState<EngineType>('personaplex_s2s')
  const [language, setLanguage] = useState('en')
  const [sttProvider, setSttProvider] = useState('fake_stt')
  const [sttModel, setSttModel] = useState('mock-fast-stt')
  const [llmProvider, setLlmProvider] = useState('fake_llm')
  const [llmModel, setLlmModel] = useState('mock-stream-llm')
  const [llmTemperature, setLlmTemperature] = useState(0.7)
  const [ttsProvider, setTtsProvider] = useState('fake_tts')
  const [ttsModel, setTtsModel] = useState('mock-fast-tts')
  const [ttsVoice, setTtsVoice] = useState('mock-alex')
  const [turnStrategy, setTurnStrategy] = useState<'auto' | 'provider_eot' | 'vad_smart_turn'>('auto')
  const [indianEnglishRules, setIndianEnglishRules] = useState(false)

  // Meta & Catalog State
  const [voices, setVoices] = useState<VoicePreset[]>([])
  const [catalog, setCatalog] = useState<ProviderManifest[]>([])
  const [credentials, setCredentials] = useState<CredentialSummary[]>([])
  const [versions, setVersions] = useState<AgentVersion[]>([])
  const [showVersions, setShowVersions] = useState(false)
  const [showTemplatesModal, setShowTemplatesModal] = useState(false)
  const [compileResult, setCompileResult] = useState<CompilePromptResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isPublishing, setIsPublishing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [isCloneModalOpen, setIsCloneModalOpen] = useState(false)
  const [playingVoiceId, setPlayingVoiceId] = useState<string | null>(null)
  const previewAudioRef = React.useRef<HTMLAudioElement | null>(null)

  const handlePlayVoicePreview = (v: VoicePreset, e: React.MouseEvent) => {
    e.stopPropagation()
    if (playingVoiceId === v.id) {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause()
      }
      setPlayingVoiceId(null)
      return
    }
    if (previewAudioRef.current) {
      previewAudioRef.current.pause()
    }
    const audioUrl = v.preview_url || `/v2/agents/voices/${encodeURIComponent(v.id)}/preview`
    const audio = new Audio(audioUrl)
    previewAudioRef.current = audio
    audio.onended = () => setPlayingVoiceId(null)
    audio.onerror = () => setPlayingVoiceId(null)
    audio.play().then(() => setPlayingVoiceId(v.id)).catch(() => setPlayingVoiceId(null))
  }

  const handleDeleteClonedVoice = async (v: VoicePreset, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm(`Are you sure you want to delete the cloned voice "${v.name}"?`)) return
    try {
      await api.deleteVoice(v.id)
      setVoices((prev) => prev.filter((item) => item.id !== v.id))
      if (voiceId === v.id) {
        setVoiceId('NATM1.pt')
      }
    } catch (err) {
      console.error('Failed to delete cloned voice:', err)
    }
  }

  const handleVoiceCloned = (newVoice: VoicePreset) => {
    setVoices((prev) => [newVoice, ...prev.filter((item) => item.id !== newVoice.id)])
    setVoiceId(newVoice.id)
    setSuccessMessage(`Voice "${newVoice.name}" cloned and applied successfully!`)
  }

  // 1. Load Voices, Catalog, Credentials & Agent (if editing)
  useEffect(() => {
    api.getVoices().then(setVoices).catch(console.error)
    api.listProviderCatalog().then(setCatalog).catch(console.error)
    api.listProviderCredentials().then(setCredentials).catch(console.error)

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

          // Load Engine B fields
          if (ag.engine) setEngine(ag.engine)
          if (ag.language) setLanguage(ag.language)
          if (ag.pipeline_json && ag.pipeline_json !== '{}') {
            try {
              const spec = JSON.parse(ag.pipeline_json)
              if (spec.stt?.provider_id) setSttProvider(spec.stt.provider_id)
              if (spec.stt?.model) setSttModel(spec.stt.model)
              if (spec.llm?.provider_id) setLlmProvider(spec.llm.provider_id)
              if (spec.llm?.model) setLlmModel(spec.llm.model)
              if (spec.llm?.temperature !== undefined) setLlmTemperature(spec.llm.temperature)
              if (spec.tts?.provider_id) setTtsProvider(spec.tts.provider_id)
              if (spec.tts?.model) setTtsModel(spec.tts.model)
              if (spec.tts?.voice) setTtsVoice(spec.tts.voice)
              if (spec.turn?.strategy) setTurnStrategy(spec.turn.strategy)
            } catch (err) {
              console.error('Failed to parse pipeline_json', err)
            }
          }
          if (ag.system_prompt.includes('[Indian English Conversational Guidelines]')) {
            setIndianEnglishRules(true)
          }
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

  // Helper: check if provider has active credential
  const hasCredential = (providerId: string) => {
    if (providerId.startsWith('fake_')) return true
    return credentials.some((c) => c.provider_id === providerId)
  }

  // Filter manifests by category
  const sttManifests = catalog.filter((p) => p.kind === 'stt')
  const llmManifests = catalog.filter((p) => p.kind === 'llm')
  const ttsManifests = catalog.filter((p) => p.kind === 'tts')

  // Handle Indian English Prompt Rules toggle
  const toggleIndianEnglishRules = (enabled: boolean) => {
    setIndianEnglishRules(enabled)
    if (enabled) {
      if (!systemPrompt.includes('[Indian English Conversational Guidelines]')) {
        setSystemPrompt((prev) => prev.trim() + INDIAN_ENGLISH_GUIDELINES)
      }
    } else {
      setSystemPrompt((prev) =>
        prev.replace(INDIAN_ENGLISH_GUIDELINES, '').replace(/\n*\[Indian English Conversational Guidelines\][\s\S]*?clearly\./, '').trim()
      )
    }
  }

  // Handle STT Provider Change
  const handleSttProviderChange = (pid: string) => {
    setSttProvider(pid)
    const manifest = catalog.find((p) => p.id === pid)
    if (manifest) {
      setSttModel(manifest.default_model || manifest.models[0]?.id || '')
    }
  }

  // Handle LLM Provider Change
  const handleLlmProviderChange = (pid: string) => {
    setLlmProvider(pid)
    const manifest = catalog.find((p) => p.id === pid)
    if (manifest) {
      setLlmModel(manifest.default_model || manifest.models[0]?.id || '')
    }
  }

  // Handle TTS Provider Change
  const handleTtsProviderChange = (pid: string) => {
    setTtsProvider(pid)
    const manifest = catalog.find((p) => p.id === pid)
    if (manifest) {
      setTtsModel(manifest.default_model || manifest.models[0]?.id || '')
      setTtsVoice(manifest.default_voice || manifest.voices[0]?.id || 'default')
    }
  }

  // Save Draft
  const handleSaveDraft = async () => {
    setIsSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)

    const pipelineSpec: CascadedPipelineSpec = {
      stt: {
        provider_id: sttProvider,
        model: sttModel,
        language: language,
      },
      llm: {
        provider_id: llmProvider,
        model: llmModel,
        temperature: llmTemperature,
      },
      tts: {
        provider_id: ttsProvider,
        model: ttsModel,
        voice: engine === 'cascaded_cloud' ? ttsVoice : voiceId,
      },
      turn: {
        strategy: turnStrategy,
      },
    }
    const pipelineJsonStr = JSON.stringify(pipelineSpec)
    const effectiveVoiceId = engine === 'cascaded_cloud' ? ttsVoice : voiceId

    try {
      if (isNew) {
        const created = await api.createAgent({
          name: name.trim(),
          voice_id: effectiveVoiceId,
          greeting: greeting.trim(),
          agent_speaks_first: agentSpeaksFirst,
          system_prompt: systemPrompt.trim(),
          ending: ending.trim(),
          timezone_str: timezoneStr,
          silence_timeout_sec: silenceTimeoutSec,
          max_duration_sec: maxDurationSec,
          engine: engine,
          language: language,
          pipeline_json: pipelineJsonStr,
        })
        setSuccessMessage('Agent created successfully.')
        setAgent(created)
        const vers = await api.listVersions(created.id)
        setVersions(vers)
      } else if (agent) {
        const updated = await api.updateAgent(agent.id, {
          name: name.trim(),
          voice_id: effectiveVoiceId,
          greeting: greeting.trim(),
          agent_speaks_first: agentSpeaksFirst,
          system_prompt: systemPrompt.trim(),
          ending: ending.trim(),
          timezone_str: timezoneStr,
          silence_timeout_sec: silenceTimeoutSec,
          max_duration_sec: maxDurationSec,
          engine: engine,
          language: language,
          pipeline_json: pipelineJsonStr,
        })
        setAgent(updated)
        setSuccessMessage('Draft saved successfully.')
        const vers = await api.listVersions(updated.id)
        setVersions(vers)
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        const p = err.problem
        const invalidStr = p.invalid_params?.map((ip) => `${ip.name}: ${ip.reason}`).join('; ')
        setErrorMessage(invalidStr ? `Validation failed: ${invalidStr}` : (p.detail || p.title))
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
        const p = err.problem
        const invalidStr = p.invalid_params?.map((ip) => `${ip.name}: ${ip.reason}`).join('; ')
        setErrorMessage(invalidStr ? `Validation failed: ${invalidStr}` : (p.detail || p.title))
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
      if (reverted.engine) setEngine(reverted.engine)
      if (reverted.language) setLanguage(reverted.language)
      if (reverted.pipeline_json && reverted.pipeline_json !== '{}') {
        try {
          const spec = JSON.parse(reverted.pipeline_json)
          if (spec.stt?.provider_id) setSttProvider(spec.stt.provider_id)
          if (spec.stt?.model) setSttModel(spec.stt.model)
          if (spec.llm?.provider_id) setLlmProvider(spec.llm.provider_id)
          if (spec.llm?.model) setLlmModel(spec.llm.model)
          if (spec.llm?.temperature !== undefined) setLlmTemperature(spec.llm.temperature)
          if (spec.tts?.provider_id) setTtsProvider(spec.tts.provider_id)
          if (spec.tts?.model) setTtsModel(spec.tts.model)
          if (spec.tts?.voice) setTtsVoice(spec.tts.voice)
          if (spec.turn?.strategy) setTurnStrategy(spec.turn.strategy)
        } catch (err) {
          console.error('Failed to parse pipeline_json on revert', err)
        }
      }
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
  const isEngineB = engine === 'cascaded_cloud'
  const effectiveVoiceId = isEngineB ? ttsVoice : voiceId
  // On Engine B, 350-token hard limit is relaxed; on Engine A, it is strictly enforced
  const canPublish = !isNew && (isEngineB ? name.trim().length > 0 : (tokenCount > 0 && tokenCount <= 350))

  if (isLoading && !agent && !isNew) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-[14px] text-[#6B6963] animate-pulse">Loading agent details...</div>
      </div>
    )
  }

  // Selected manifests for current selections
  const currentSttManifest = catalog.find((p) => p.id === sttProvider)
  const currentLlmManifest = catalog.find((p) => p.id === llmProvider)
  const currentTtsManifest = catalog.find((p) => p.id === ttsProvider)

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
            {isEngineB ? (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                Cascaded Cloud
              </span>
            ) : (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-purple-50 text-purple-700 border border-purple-200">
                PersonaPlex S2S
              </span>
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
                onClick={() => onStartCall({ ...agent, voice_id: effectiveVoiceId, engine: engine })}
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
              title={!isEngineB && tokenCount > 350 ? 'Hard limit exceeded (>350 tokens)' : undefined}
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
                    Engine: <strong className="font-medium text-[#1F1E1D]">{ver.engine || 'personaplex_s2s'}</strong> | Tokens: <span className="tabular-nums font-mono">{ver.token_count}</span> | Voice: {ver.voice_id} | Lang: {ver.language || 'en'}
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

      {/* Engine Architecture Switcher */}
      <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-2xl p-5 shadow-xs space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-[15px] font-semibold text-[#1F1E1D]">
              Voice Engine Architecture
            </h2>
            <p className="text-[12px] text-[#6B6963]">
              Switch between native PersonaPlex GPU full-duplex speech-to-speech or modular cloud STT → LLM → TTS
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          <button
            type="button"
            onClick={() => setEngine('personaplex_s2s')}
            className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
              engine === 'personaplex_s2s'
                ? 'border-purple-500 bg-purple-50/40 ring-2 ring-purple-500/20 shadow-xs'
                : 'border-[rgba(31,30,29,0.08)] bg-[#FAF9F5] hover:border-[rgba(31,30,29,0.18)]'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-semibold text-[14px] text-[#1F1E1D] flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-600" /> PersonaPlex S2S (GPU)
              </span>
              <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-purple-100 text-purple-800">
                Sub-100ms
              </span>
            </div>
            <p className="text-[12px] text-[#6B6963]">
              Native speech-to-speech neural model on NVIDIA GPU. English only, 18 official voice conditioning presets, strict &le;350 token prompt limit.
            </p>
          </button>

          <button
            type="button"
            onClick={() => setEngine('cascaded_cloud')}
            className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
              engine === 'cascaded_cloud'
                ? 'border-emerald-500 bg-emerald-50/40 ring-2 ring-emerald-500/20 shadow-xs'
                : 'border-[rgba(31,30,29,0.08)] bg-[#FAF9F5] hover:border-[rgba(31,30,29,0.18)]'
            }`}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-semibold text-[14px] text-[#1F1E1D] flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-600" /> Cascaded Cloud (STT + LLM + TTS)
              </span>
              <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
                Multilingual
              </span>
            </div>
            <p className="text-[12px] text-[#6B6963]">
              Modular streaming pipeline using user API keys (Deepgram, Sarvam, OpenAI, Anthropic, Cartesia, ElevenLabs). Multilingual, Tamil/Hindi, no GPU required.
            </p>
          </button>
        </div>
      </div>

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

      {/* Main Form Fields */}
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

        {/* Engine A: Voice Presets & Cloned Voices */}
        {!isEngineB && (
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <label className="text-[13px] font-medium text-[#1F1E1D]">
                  Voice Conditioning Presets
                </label>
                <p className="text-[12px] text-[#9E9B93]">
                  Select an official neural preset or clone your own voice from a microphone recording
                </p>
              </div>
              <Button
                type="button"
                variant="secondary"
                size="compact"
                onClick={() => setIsCloneModalOpen(true)}
                className="flex items-center gap-1.5 self-start sm:self-auto border-[#C2603F]/30 text-[#C2603F] hover:bg-[#FBEFEA]"
              >
                <Sparkles className="w-3.5 h-3.5 text-[#C2603F]" />
                Clone Your Voice
              </Button>
            </div>

            {/* Cloned Voices Category */}
            {voices.some((v) => v.is_cloned) && (
              <div className="space-y-2 p-3.5 bg-[#FAF9F5] border border-[rgba(194,96,63,0.18)] rounded-xl">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-semibold text-[#1F1E1D] flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-[#C2603F]" />
                    Your Cloned Voices
                  </span>
                  <span className="text-[11px] text-[#C2603F] font-medium">Custom Reference Samples</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                  {voices
                    .filter((v) => v.is_cloned)
                    .map((v) => {
                      const isSelected =
                        voiceId === v.id ||
                        voiceId === v.id.replace(/\.(wav|pt)$/i, '') ||
                        v.id === `${voiceId}.wav` ||
                        v.id === `${voiceId}.pt`
                      const isPlaying = playingVoiceId === v.id
                      return (
                        <div
                          key={v.id}
                          onClick={() => setVoiceId(v.id)}
                          className={`text-left p-3 rounded-xl border transition-all text-[13px] cursor-pointer relative group ${
                            isSelected
                              ? 'border-[#C2603F] bg-[#FBEFEA] shadow-xs'
                              : 'border-[rgba(31,30,29,0.08)] bg-white hover:border-[rgba(31,30,29,0.18)]'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-semibold text-[#1F1E1D] flex items-center gap-1.5 truncate">
                              <Volume2 className={`w-3.5 h-3.5 shrink-0 ${isSelected ? 'text-[#C2603F]' : 'text-[#6B6963]'}`} />
                              <span className="truncate">{v.name}</span>
                            </span>
                            <div className="flex items-center gap-1 shrink-0">
                              {v.qa_score !== undefined && v.qa_score !== null ? (
                                <span
                                  className={`text-[9px] px-1.5 py-0.5 rounded-full font-mono font-medium ${
                                    v.qa_passed !== false
                                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                                      : 'bg-amber-50 text-amber-700 border border-amber-200'
                                  }`}
                                  title={`Acoustic similarity: ${v.qa_score}. Route: ${v.recommended_engine || 'PersonaPlex S2S'}`}
                                >
                                  {v.qa_passed !== false ? `QA ${v.qa_score}` : (v.recommended_engine === 'cascaded' ? 'Cascaded' : 'S2S')}
                                </span>
                              ) : (
                                <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono">
                                  PersonaPlex S2S
                                </span>
                              )}
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-[#C2603F]/10 text-[#C2603F] font-medium shrink-0">
                                Cloned
                              </span>
                            </div>
                          </div>
                          <div className="text-[12px] text-[#6B6963] truncate">
                            {v.duration_sec ? `${v.duration_sec}s sample` : v.speaking_style}
                          </div>
                          <div className="flex items-center justify-between mt-2 pt-1 border-t border-[rgba(31,30,29,0.06)]">
                            <button
                              type="button"
                              onClick={(e) => handlePlayVoicePreview(v, e)}
                              className="text-[11px] font-medium text-[#C2603F] hover:underline flex items-center gap-1 cursor-pointer"
                            >
                              {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                              {isPlaying ? 'Pause' : 'Preview'}
                            </button>
                            <button
                              type="button"
                              onClick={(e) => handleDeleteClonedVoice(v, e)}
                              className="text-[11px] text-[#9E9B93] hover:text-red-600 flex items-center gap-0.5 opacity-60 group-hover:opacity-100 transition-opacity cursor-pointer"
                              title="Delete cloned voice"
                            >
                              <Trash2 className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>
            )}

            {/* Official PersonaPlex Presets Category */}
            <div className="space-y-2">
              <span className="text-[12px] font-semibold text-[#1F1E1D] block">
                Official Built-in Presets (18 Voices)
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                {voices
                  .filter((v) => !v.is_cloned)
                  .map((v) => {
                    const isSelected =
                      voiceId === v.id ||
                      voiceId === v.id.replace(/\.(wav|pt)$/i, '') ||
                      v.id === `${voiceId}.wav` ||
                      v.id === `${voiceId}.pt`
                    return (
                      <button
                        key={v.id}
                        type="button"
                        onClick={() => setVoiceId(v.id)}
                        className={`text-left p-3 rounded-xl border transition-all text-[13px] cursor-pointer ${
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
          </div>
        )}

        {/* Engine B: Cascaded Cloud Pipeline Studio */}
        {isEngineB && (
          <div className="space-y-6 p-5 bg-[#FAF9F5] border border-[rgba(31,30,29,0.08)] rounded-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[rgba(31,30,29,0.08)] pb-3">
              <div>
                <h3 className="text-[15px] font-semibold text-[#1F1E1D] flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-emerald-600" />
                  Cascaded Cloud Pipeline Configuration
                </h3>
                <p className="text-[12px] text-[#6B6963]">
                  Configure your streaming STT, LLM, TTS providers, voice, and conversational turn strategy.
                </p>
              </div>
              {onOpenProviders && (
                <button
                  type="button"
                  onClick={onOpenProviders}
                  className="text-[12px] text-[#C2603F] hover:underline flex items-center gap-1 font-medium cursor-pointer self-start sm:self-auto"
                >
                  <Key className="w-3.5 h-3.5" /> Manage API Keys ({credentials.length})
                </button>
              )}
            </div>

            {/* Cloned Voice Switcher Banner */}
            {voices.some((v) => v.is_cloned) && (
              <div className="p-3 bg-[#FBEFEA] border border-[rgba(194,96,63,0.25)] rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="text-[12px] text-[#6B6963]">
                  <span className="font-semibold text-[#1F1E1D] flex items-center gap-1.5 mb-0.5">
                    <Sparkles className="w-3.5 h-3.5 text-[#C2603F]" />
                    Looking for your cloned voice ({voices.find((v) => v.is_cloned)?.name || 'Custom Voice'})?
                  </span>
                  Cloned neural voices run directly on the low-latency <strong>PersonaPlex S2S (GPU)</strong> architecture.
                </div>
                <Button
                  type="button"
                  variant="primary"
                  size="compact"
                  onClick={() => setEngine('personaplex_s2s')}
                  className="shrink-0 self-start sm:self-auto bg-[#C2603F] hover:bg-[#A84F32] text-white"
                >
                  Switch to PersonaPlex S2S
                </Button>
              </div>
            )}

            {/* Language & Indian English Rules Toggle */}
            <div className="space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1">
                  <label className="text-[13px] font-medium text-[#1F1E1D] flex items-center gap-1.5">
                    <Globe className="w-3.5 h-3.5 text-[#6B6963]" /> Target Language & Dialect
                  </label>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {SUPPORTED_LANGUAGES.map((lang) => {
                      const isSelected = language === lang.code
                      return (
                        <button
                          key={lang.code}
                          type="button"
                          onClick={() => setLanguage(lang.code)}
                          className={`px-3 py-1.5 rounded-lg text-[12px] font-medium border transition-all cursor-pointer flex items-center gap-1.5 ${
                            isSelected
                              ? 'border-emerald-600 bg-emerald-50 text-emerald-900 shadow-2xs font-semibold'
                              : 'border-[rgba(31,30,29,0.1)] bg-[#FFFFFF] text-[#6B6963] hover:border-[rgba(31,30,29,0.2)]'
                          }`}
                        >
                          <span>{lang.flag}</span>
                          <span>{lang.label}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </div>

              {/* Indian English Dialogue Rules Toggle */}
              <div className="p-3 bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-[13px] font-medium text-[#1F1E1D] block">
                    Indian English Conversational Rules
                  </span>
                  <span className="text-[12px] text-[#6B6963] block">
                    Suppresses mechanical phrasing like 'Regarding your query'; ensures warm, concise dialogue.
                  </span>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={indianEnglishRules}
                    onChange={(e) => toggleIndianEnglishRules(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-600"></div>
                </label>
              </div>
            </div>

            {/* Pipeline Stage 1: STT (Speech-to-Text) */}
            <div className="p-4 bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-semibold text-[#1F1E1D] flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-800 text-[11px] font-bold flex items-center justify-center">1</span>
                  Speech-to-Text (STT)
                </span>
                {hasCredential(sttProvider) ? (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3" /> Credential Active
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                    <AlertCircle className="w-3 h-3" /> Needs API Key
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">STT Provider</label>
                  <select
                    value={sttProvider}
                    onChange={(e) => handleSttProviderChange(e.target.value)}
                    className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                  >
                    {sttManifests.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.display_name} {p.streaming ? '(Streaming)' : ''}
                      </option>
                    ))}
                    {sttManifests.length === 0 && (
                      <option value="fake_stt">Pipecat Fake STT (Offline Mock)</option>
                    )}
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">STT Model</label>
                  {currentSttManifest && currentSttManifest.models.length > 0 ? (
                    <select
                      value={sttModel}
                      onChange={(e) => setSttModel(e.target.value)}
                      className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    >
                      {currentSttManifest.models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name} ({m.latency_profile || 'fast'})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={sttModel}
                      onChange={(e) => setSttModel(e.target.value)}
                      className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Pipeline Stage 2: LLM (Language Model) */}
            <div className="p-4 bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-semibold text-[#1F1E1D] flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-800 text-[11px] font-bold flex items-center justify-center">2</span>
                  LLM Intelligence & Routing
                </span>
                {hasCredential(llmProvider) ? (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3" /> Credential Active
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                    <AlertCircle className="w-3 h-3" /> Needs API Key
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">LLM Provider</label>
                  <select
                    value={llmProvider}
                    onChange={(e) => handleLlmProviderChange(e.target.value)}
                    className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                  >
                    {llmManifests.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.display_name}
                      </option>
                    ))}
                    {llmManifests.length === 0 && (
                      <option value="fake_llm">Pipecat Fake LLM (Offline Mock)</option>
                    )}
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">Model</label>
                  {currentLlmManifest && currentLlmManifest.models.length > 0 ? (
                    <select
                      value={llmModel}
                      onChange={(e) => setLlmModel(e.target.value)}
                      className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    >
                      {currentLlmManifest.models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={llmModel}
                      onChange={(e) => setLlmModel(e.target.value)}
                      className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    />
                  )}
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <label className="text-[12px] text-[#6B6963]">Temperature</label>
                    <span className="text-[12px] font-mono font-medium text-[#1F1E1D]">{llmTemperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={llmTemperature}
                    onChange={(e) => setLlmTemperature(parseFloat(e.target.value))}
                    className="w-full accent-emerald-600 mt-1"
                  />
                </div>
              </div>
            </div>

            {/* Pipeline Stage 3: TTS (Text-to-Speech) */}
            <div className="p-4 bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-semibold text-[#1F1E1D] flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-800 text-[11px] font-bold flex items-center justify-center">3</span>
                  Text-to-Speech (TTS)
                </span>
                {hasCredential(ttsProvider) ? (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    <CheckCircle2 className="w-3 h-3" /> Credential Active
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                    <AlertCircle className="w-3 h-3" /> Needs API Key
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">TTS Provider</label>
                  <select
                    value={ttsProvider}
                    onChange={(e) => handleTtsProviderChange(e.target.value)}
                    className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                  >
                    {ttsManifests.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.display_name}
                      </option>
                    ))}
                    {ttsManifests.length === 0 && (
                      <option value="fake_tts">Pipecat Fake TTS (Offline Mock)</option>
                    )}
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">Voice</label>
                  {currentTtsManifest && currentTtsManifest.voices.length > 0 ? (
                    <select
                      value={ttsVoice}
                      onChange={(e) => setTtsVoice(e.target.value)}
                      className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    >
                      {currentTtsManifest.voices.map((v) => (
                        <option key={v.id} value={v.id}>
                          {v.name} ({v.gender || 'neutral'})
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={ttsVoice}
                      onChange={(e) => setTtsVoice(e.target.value)}
                      className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                    />
                  )}
                </div>

                <div className="space-y-1">
                  <label className="text-[12px] text-[#6B6963]">Turn Detection Strategy</label>
                  <select
                    value={turnStrategy}
                    onChange={(e) => setTurnStrategy(e.target.value as any)}
                    className="w-full h-9 px-2.5 text-[13px] bg-[#FAF9F5] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                  >
                    <option value="auto">Adaptive Auto-Turn</option>
                    <option value="provider_eot">Provider Native End-of-Turn</option>
                    <option value="vad_smart_turn">Silero VAD + Smart Turn</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Field 3: Greeting */}
        <div className="space-y-3">
          <TextArea
            label="Spoken Greeting"
            hint="Spoken line delivered immediately upon call connection"
            value={greeting}
            onChange={(e) => setGreeting(e.target.value)}
            rows={2}
            placeholder={
              language === 'ta-IN'
                ? 'Vanakkam! Metro Health-ku azhaithadharku nandri. Naan ungaluku epdi udhavalam?'
                : language === 'hi-IN'
                ? 'Namaste! Metro Health me call karne ke liye dhanyavad. Main aapki kya madad kar sakta hoon?'
                : 'Hello! Thank you for calling Metro Health. How can I help you today?'
            }
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
                {isEngineB
                  ? 'Fed directly to streaming LLM. Large prompts supported (up to 128k context).'
                  : 'Wrapped as <system> {prompt} <system>. Hard limit of 350 tokens.'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {onOpenGuide && (
                <button
                  type="button"
                  onClick={onOpenGuide}
                  className="text-[12px] text-[#C2603F] hover:underline flex items-center gap-1 font-medium cursor-pointer"
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
            {isEngineB ? (
              <div className="flex items-center justify-between text-[13px]">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-[#1F1E1D]">Prompt Tokens:</span>
                  <span className="font-mono tabular-nums font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                    {tokenCount} tokens
                  </span>
                </div>
                <span className="text-[12px] text-emerald-700 font-medium flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> 350-token hard limit relaxed for Cloud LLMs
                </span>
              </div>
            ) : (
              <TokenMeter
                tokenCount={tokenCount}
                recommendedLimit={150}
                hardLimit={350}
              />
            )}
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
            placeholder={
              language === 'ta-IN'
                ? 'Nandri, vanakkam!'
                : language === 'hi-IN'
                ? 'Aapka din shubh ho, dhanyavad!'
                : 'e.g. Thanks for calling, have a wonderful day! Goodbye!'
            }
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
              min={1}
              max={7200}
              step={1}
              className="w-full h-10 px-3 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-[#C2603F] focus:outline-none tabular-nums"
            />
            <span className="text-[11px] text-[#9E9B93] block">Seconds of silence before end (1 - 7200s)</span>
          </div>

          <div className="space-y-1.5">
            <label className="text-[13px] font-medium text-[#1F1E1D]">
              Max Duration
            </label>
            <input
              type="number"
              value={maxDurationSec}
              onChange={(e) => setMaxDurationSec(parseFloat(e.target.value))}
              min={1}
              max={14400}
              step={30}
              className="w-full h-10 px-3 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] focus:ring-2 focus:ring-[#C2603F] focus:outline-none tabular-nums"
            />
            <span className="text-[11px] text-[#9E9B93] block">Hard cap in seconds (1 - 14400s)</span>
          </div>
        </div>
      </div>

      {/* Voice Clone Modal */}
      <VoiceCloneModal
        isOpen={isCloneModalOpen}
        onClose={() => setIsCloneModalOpen(false)}
        onVoiceCloned={handleVoiceCloned}
      />
    </div>
  )
}
