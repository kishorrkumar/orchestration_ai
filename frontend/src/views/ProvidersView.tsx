import { useState, useEffect } from 'react'
import { api } from '../lib/api'
import type { ProviderManifest, CredentialSummary, ProviderTestResult } from '../lib/types'
import {
  Key,
  CheckCircle2,
  AlertCircle,
  Eye,
  EyeOff,
  ChevronRight,
  Trash2,
  ShieldCheck,
  RefreshCw,
  X,
} from 'lucide-react'

export function ProvidersView() {
  const [catalog, setCatalog] = useState<ProviderManifest[]>([])
  const [credentials, setCredentials] = useState<Record<string, CredentialSummary>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Active modal state
  const [selectedProvider, setSelectedProvider] = useState<ProviderManifest | null>(null)
  const [apiKeyInput, setApiKeyInput] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [customBaseUrl, setCustomBaseUrl] = useState('')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<ProviderTestResult | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [catList, credList] = await Promise.all([
        api.listProviderCatalog(),
        api.listProviderCredentials(),
      ])
      setCatalog(catList)
      const credMap: Record<string, CredentialSummary> = {}
      for (const c of credList) {
        credMap[c.provider_id] = c
      }
      setCredentials(credMap)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load providers')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const handleOpenProvider = (provider: ProviderManifest) => {
    setSelectedProvider(provider)
    setApiKeyInput('')
    setShowApiKey(false)
    setTestResult(null)
    setSaveError(null)
    setConfirmDelete(false)

    const existing = credentials[provider.id]
    if (existing && existing.config?.base_url) {
      setCustomBaseUrl(existing.config.base_url)
    } else {
      setCustomBaseUrl('')
    }
  }

  const handleCloseModal = () => {
    setSelectedProvider(null)
    setTestResult(null)
    setSaveError(null)
  }

  const handleTestConnection = async () => {
    if (!selectedProvider) return
    setTesting(true)
    setTestResult(null)
    setSaveError(null)

    const existing = credentials[selectedProvider.id]
    const keyToTest = apiKeyInput.trim() || undefined

    if (!keyToTest && !existing) {
      setSaveError('Please enter an API key to test.')
      setTesting(false)
      return
    }

    try {
      const config: Record<string, any> = {}
      if (customBaseUrl.trim()) {
        config.base_url = customBaseUrl.trim()
      }
      const res = await api.testProviderCredential(selectedProvider.id, {
        api_key: keyToTest,
        config: Object.keys(config).length > 0 ? config : undefined,
      })
      setTestResult(res)
    } catch (err: any) {
      setTestResult({
        status: 'error',
        latency_ms: 0,
        error_message: err.message || 'Connection test failed',
      })
    } finally {
      setTesting(false)
    }
  }

  const handleSaveCredential = async () => {
    if (!selectedProvider) return
    const keyToSave = apiKeyInput.trim()
    const existing = credentials[selectedProvider.id]

    if (!keyToSave && !existing) {
      setSaveError('Please enter an API key.')
      return
    }

    setSaving(true)
    setSaveError(null)
    try {
      const config: Record<string, any> = {}
      if (customBaseUrl.trim()) {
        config.base_url = customBaseUrl.trim()
      }

      // If key was not modified but user edited config, require re-entry or keep existing
      if (!keyToSave && existing) {
        // Just re-test
        setSaveError('Enter your API key again to update configuration.')
        setSaving(false)
        return
      }

      const saved = await api.saveProviderCredential(selectedProvider.id, {
        api_key: keyToSave,
        config,
      })

      setCredentials((prev) => ({
        ...prev,
        [saved.provider_id]: saved,
      }))
      handleCloseModal()
    } catch (err: any) {
      setSaveError(err.message || 'Failed to save credential')
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteCredential = async () => {
    if (!selectedProvider) return
    setSaving(true)
    try {
      await api.deleteProviderCredential(selectedProvider.id)
      setCredentials((prev) => {
        const next = { ...prev }
        delete next[selectedProvider.id]
        return next
      })
      handleCloseModal()
    } catch (err: any) {
      setSaveError(err.message || 'Failed to remove credential')
    } finally {
      setSaving(false)
    }
  }

  // Group catalog by kind
  const sttProviders = catalog.filter((p) => p.kind === 'stt')
  const llmProviders = catalog.filter((p) => p.kind === 'llm')
  const ttsProviders = catalog.filter((p) => p.kind === 'tts')

  const totalConnected = Object.keys(credentials).length

  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-text-primary)]">
          Providers
        </h1>
        <p className="text-[14px] text-[var(--color-text-secondary)] mt-1.5">
          Connect speech recognition, language models, and voices with your own API keys.
        </p>
      </div>

      {/* Empty State Banner if no providers connected */}
      {!loading && totalConnected === 0 && (
        <div className="mb-8 p-4 rounded-xl bg-[var(--color-bg-sunken)] border border-[var(--color-hairline)] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Key className="w-5 h-5 text-[var(--color-text-tertiary)] shrink-0" />
            <span className="text-[13px] text-[var(--color-text-secondary)]">
              Connect at least one speech recognition, language model, and voice provider to build a pipeline agent.
            </span>
          </div>
          <button
            onClick={() => {
              const first = catalog[0]
              if (first) handleOpenProvider(first)
            }}
            className="px-3 py-1.5 rounded-lg text-[13px] font-medium bg-[var(--color-accent)] text-white hover:opacity-90 transition-opacity"
          >
            Connect First Provider
          </button>
        </div>
      )}

      {error && (
        <div className="mb-6 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 text-[13px]">
          {error}
        </div>
      )}

      {/* Inset Grouped Lists */}
      <div className="space-y-8">
        {/* 1. Speech Recognition (STT) */}
        <section>
          <div className="flex items-center justify-between mb-2.5 px-1">
            <h2 className="text-[12px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
              Speech Recognition (STT)
            </h2>
            <span className="text-[12px] text-[var(--color-text-tertiary)]">
              {sttProviders.filter((p) => credentials[p.id]).length} connected
            </span>
          </div>
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl overflow-hidden divide-y divide-[var(--color-hairline)] shadow-xs">
            {sttProviders.map((provider) => {
              const cred = credentials[provider.id]
              return (
                <ProviderRow
                  key={provider.id}
                  provider={provider}
                  credential={cred}
                  onSelect={() => handleOpenProvider(provider)}
                />
              )
            })}
          </div>
        </section>

        {/* 2. Language Models (LLM) */}
        <section>
          <div className="flex items-center justify-between mb-2.5 px-1">
            <h2 className="text-[12px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
              Language Models (LLM)
            </h2>
            <span className="text-[12px] text-[var(--color-text-tertiary)]">
              {llmProviders.filter((p) => credentials[p.id]).length} connected
            </span>
          </div>
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl overflow-hidden divide-y divide-[var(--color-hairline)] shadow-xs">
            {llmProviders.map((provider) => {
              const cred = credentials[provider.id]
              return (
                <ProviderRow
                  key={provider.id}
                  provider={provider}
                  credential={cred}
                  onSelect={() => handleOpenProvider(provider)}
                />
              )
            })}
          </div>
        </section>

        {/* 3. Voices (TTS) */}
        <section>
          <div className="flex items-center justify-between mb-2.5 px-1">
            <h2 className="text-[12px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)]">
              Voices & Speech Synthesis (TTS)
            </h2>
            <span className="text-[12px] text-[var(--color-text-tertiary)]">
              {ttsProviders.filter((p) => credentials[p.id]).length} connected
            </span>
          </div>
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl overflow-hidden divide-y divide-[var(--color-hairline)] shadow-xs">
            {ttsProviders.map((provider) => {
              const cred = credentials[provider.id]
              return (
                <ProviderRow
                  key={provider.id}
                  provider={provider}
                  credential={cred}
                  onSelect={() => handleOpenProvider(provider)}
                />
              )
            })}
          </div>
        </section>
      </div>

      {/* Provider Configuration Modal Sheet */}
      {selectedProvider && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs">
          <div
            className="w-full max-w-lg bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-150"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="px-6 py-4.5 border-b border-[var(--color-hairline)] flex items-start justify-between">
              <div>
                <h3 className="text-[17px] font-semibold text-[var(--color-text-primary)]">
                  {selectedProvider.display_name}
                </h3>
                <p className="text-[12px] text-[var(--color-text-secondary)] mt-0.5">
                  {selectedProvider.description}
                </p>
              </div>
              <button
                onClick={handleCloseModal}
                className="p-1.5 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] rounded-lg hover:bg-[var(--color-bg-sunken)] transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-4">
              {saveError && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 text-[12px]">
                  {saveError}
                </div>
              )}

              {/* Status Header if already connected */}
              {credentials[selectedProvider.id] && (
                <div className="p-3 rounded-xl bg-[var(--color-bg-sunken)] border border-[var(--color-hairline)] flex items-center justify-between text-[12px]">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                    <span className="font-medium text-[var(--color-text-primary)]">
                      Connected · •••• {credentials[selectedProvider.id].last4}
                    </span>
                  </div>
                  {credentials[selectedProvider.id].last_latency_ms != null && (
                    <span className="text-[var(--color-text-tertiary)]">
                      {Math.round(credentials[selectedProvider.id].last_latency_ms!)} ms latency
                    </span>
                  )}
                </div>
              )}

              {/* API Key Input */}
              <div>
                <label className="block text-[13px] font-medium text-[var(--color-text-primary)] mb-1.5">
                  {selectedProvider.auth_fields[0]?.label || 'API Key'}
                </label>
                <div className="relative">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    placeholder={
                      credentials[selectedProvider.id]
                        ? `•••• •••• •••• ${credentials[selectedProvider.id].last4} (leave empty to keep)`
                        : selectedProvider.auth_fields[0]?.placeholder || 'Enter API key'
                    }
                    className="w-full px-3.5 py-2.5 pr-10 text-[13px] font-mono bg-[var(--color-bg-base)] border border-[var(--color-hairline)] rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/30 focus:border-[var(--color-accent)]"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] p-1"
                  >
                    {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                {selectedProvider.auth_fields[0]?.description && (
                  <p className="text-[11px] text-[var(--color-text-tertiary)] mt-1">
                    {selectedProvider.auth_fields[0].description}
                  </p>
                )}
              </div>

              {/* Optional Custom Base URL (e.g. for OpenAI / Groq / Ollama) */}
              {selectedProvider.id === 'openai_llm' && (
                <div>
                  <label className="block text-[13px] font-medium text-[var(--color-text-primary)] mb-1.5">
                    Custom Base URL (Optional)
                  </label>
                  <input
                    type="text"
                    value={customBaseUrl}
                    onChange={(e) => setCustomBaseUrl(e.target.value)}
                    placeholder="https://api.openai.com/v1"
                    className="w-full px-3.5 py-2.5 text-[13px] font-mono bg-[var(--color-bg-base)] border border-[var(--color-hairline)] rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/30 focus:border-[var(--color-accent)]"
                  />
                  <p className="text-[11px] text-[var(--color-text-tertiary)] mt-1">
                    Leave blank for default OpenAI, or enter Groq, OpenRouter, Cerebras, or Ollama URL.
                  </p>
                </div>
              )}

              {/* Test Connection Probe Feedback */}
              {testResult && (
                <div
                  className={`p-3 rounded-xl border text-[12px] flex items-center justify-between ${
                    testResult.status === 'valid'
                      ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                      : testResult.status === 'rate_limited'
                      ? 'bg-amber-500/10 border-amber-500/20 text-amber-600 dark:text-amber-400'
                      : 'bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {testResult.status === 'valid' ? (
                      <CheckCircle2 className="w-4 h-4 shrink-0" />
                    ) : (
                      <AlertCircle className="w-4 h-4 shrink-0" />
                    )}
                    <span>
                      {testResult.status === 'valid'
                        ? `Valid connection · ${Math.round(testResult.latency_ms)} ms latency`
                        : testResult.error_message || 'Verification failed'}
                    </span>
                  </div>
                </div>
              )}

              {/* Security Footer Note */}
              <div className="pt-2 flex items-center gap-2 text-[11px] text-[var(--color-text-tertiary)]">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                <span>Keys are encrypted at rest with AES-256-GCM and never returned by the server.</span>
              </div>
            </div>

            {/* Modal Footer Controls */}
            <div className="px-6 py-4 border-t border-[var(--color-hairline)] bg-[var(--color-bg-sunken)]/50 flex items-center justify-between">
              <div>
                {credentials[selectedProvider.id] && !confirmDelete && (
                  <button
                    type="button"
                    onClick={() => setConfirmDelete(true)}
                    className="text-[12px] text-red-500 hover:text-red-600 font-medium flex items-center gap-1 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    Remove Key
                  </button>
                )}

                {confirmDelete && (
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] text-red-500 font-medium">Confirm remove?</span>
                    <button
                      type="button"
                      onClick={handleDeleteCredential}
                      disabled={saving}
                      className="px-2 py-1 rounded text-[11px] font-semibold bg-red-500 text-white hover:bg-red-600"
                    >
                      Yes, Remove
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmDelete(false)}
                      className="px-2 py-1 rounded text-[11px] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-base)]"
                    >
                      Cancel
                    </button>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2.5">
                <button
                  type="button"
                  onClick={handleTestConnection}
                  disabled={testing}
                  className="px-3.5 py-2 rounded-xl text-[13px] font-medium border border-[var(--color-hairline)] bg-[var(--color-bg-surface)] hover:bg-[var(--color-bg-sunken)] text-[var(--color-text-primary)] transition-colors flex items-center gap-1.5"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${testing ? 'animate-spin' : ''}`} />
                  Test
                </button>

                <button
                  type="button"
                  onClick={handleSaveCredential}
                  disabled={saving}
                  className="px-4 py-2 rounded-xl text-[13px] font-medium bg-[var(--color-accent)] text-white hover:opacity-90 transition-opacity"
                >
                  {saving ? 'Saving...' : 'Save Key'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function ProviderRow({
  provider,
  credential,
  onSelect,
}: {
  provider: ProviderManifest
  credential?: CredentialSummary
  onSelect: () => void
}) {
  const isConnected = !!credential
  const isError = credential?.status === 'invalid'

  return (
    <div
      onClick={onSelect}
      className="p-4 flex items-center justify-between hover:bg-[var(--color-bg-sunken)]/60 cursor-pointer transition-colors group"
    >
      <div className="flex-1 pr-4">
        <div className="flex items-center gap-2">
          <span className="font-medium text-[15px] text-[var(--color-text-primary)]">
            {provider.display_name}
          </span>
          {provider.native_eot && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
              Native EOT
            </span>
          )}
        </div>
        <p className="text-[12px] text-[var(--color-text-secondary)] mt-0.5 line-clamp-1">
          {provider.description}
        </p>
      </div>

      <div className="flex items-center gap-4 shrink-0">
        {/* Status Pill */}
        <div className="flex items-center gap-2 text-[12px]">
          {isConnected ? (
            isError ? (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
                <span className="text-red-500 font-medium">Key not working</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                <span className="text-[var(--color-text-secondary)] font-medium">
                  Connected · •••• {credential.last4}
                </span>
              </>
            )
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-[var(--color-text-tertiary)] inline-block opacity-40" />
              <span className="text-[var(--color-text-tertiary)]">Not connected</span>
            </>
          )}
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation()
            onSelect()
          }}
          className={`px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors ${
            isConnected
              ? 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-sunken)]'
              : 'bg-[var(--color-accent)] text-white hover:opacity-90'
          }`}
        >
          {isConnected ? 'Configure' : 'Connect'}
        </button>

        <ChevronRight className="w-4 h-4 text-[var(--color-text-tertiary)] group-hover:translate-x-0.5 transition-transform" />
      </div>
    </div>
  )
}
