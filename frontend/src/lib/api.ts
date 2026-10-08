import type {
  Agent,
  AgentVersion,
  CallSession,
  CallTurn,
  CompilePromptResult,
  ProblemDetail,
  VoicePreset,
} from './types'

export class ApiError extends Error {
  problem: ProblemDetail

  constructor(problem: ProblemDetail) {
    super(problem.detail || problem.title)
    this.problem = problem
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!res.ok) {
    let problem: ProblemDetail
    try {
      problem = await res.json()
    } catch {
      problem = {
        type: 'https://errors.personaplex.ai/unexpected-error',
        title: res.statusText || 'Error',
        status: res.status,
        detail: `HTTP ${res.status}: ${res.statusText}`,
        code: 'HTTP_ERROR',
        error_id: 'err_client',
      }
    }
    throw new ApiError(problem)
  }

  if (res.status === 204) {
    return {} as T
  }

  return res.json()
}

export const api = {
  // Voice Presets
  getVoices: () => request<VoicePreset[]>('/v2/agents/voices'),
  cloneVoice: async (formData: FormData): Promise<VoicePreset> => {
    const res = await fetch('/v2/agents/voices/clone', {
      method: 'POST',
      body: formData,
    })
    if (!res.ok) {
      let problem: ProblemDetail
      try {
        problem = await res.json()
      } catch {
        problem = {
          type: 'https://errors.personaplex.ai/unexpected-error',
          title: res.statusText || 'Error',
          status: res.status,
          detail: `HTTP ${res.status}: ${res.statusText}`,
          code: 'HTTP_ERROR',
          error_id: 'err_client',
        }
      }
      throw new ApiError(problem)
    }
    return res.json()
  },
  deleteVoice: (voiceId: string) =>
    request<{ success: boolean; id: string }>(`/v2/agents/voices/${encodeURIComponent(voiceId)}`, {
      method: 'DELETE',
    }),

  // Agents CRUD
  listAgents: () => request<Agent[]>('/v2/agents'),
  getAgent: (id: string) => request<Agent>(`/v2/agents/${id}`),
  createAgent: (data: Partial<Agent>) =>
    request<Agent>('/v2/agents', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateAgent: (id: string, data: Partial<Agent> & { change_note?: string }) =>
    request<Agent>(`/v2/agents/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  deleteAgent: (id: string) =>
    request<{ success: boolean }>(`/v2/agents/${id}`, {
      method: 'DELETE',
    }),

  // Versions
  listVersions: (agentId: string) =>
    request<AgentVersion[]>(`/v2/agents/${agentId}/versions`),
  publishVersion: (agentId: string, change_note?: string) =>
    request<Agent>(`/v2/agents/${agentId}/publish`, {
      method: 'POST',
      body: JSON.stringify({ change_note }),
    }),
  revertVersion: (agentId: string, versionNumber: number) =>
    request<Agent>(`/v2/agents/${agentId}/revert/${versionNumber}`, {
      method: 'POST',
    }),

  // Prompt compiler & linter
  compilePrompt: (data: {
    system_prompt: string
    greeting?: string
    ending?: string
    agent_name?: string
    customer_name?: string
    company?: string
    timezone_str?: string
  }) =>
    request<CompilePromptResult>('/v2/prompts/compile', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Calls
  listCalls: (agentId?: string) => {
    const query = agentId ? `?agent_id=${encodeURIComponent(agentId)}` : ''
    return request<CallSession[]>(`/v2/calls${query}`)
  },
  getCall: (sessionId: string) =>
    request<CallSession>(`/v2/calls/${sessionId}`),
  getCallTurns: (sessionId: string) =>
    request<CallTurn[]>(`/v2/calls/${sessionId}/turns`),

  // Engine B Providers & Encrypted Vault
  listProviderCatalog: () =>
    request<import('./types').ProviderManifest[]>('/v1/providers/catalog'),
  getProviderModels: (providerId: string) =>
    request<import('./types').ModelDescriptor[]>(`/v1/providers/${providerId}/models`),
  getProviderVoices: (providerId: string) =>
    request<import('./types').VoiceDescriptor[]>(`/v1/providers/${providerId}/voices`),
  listProviderCredentials: () =>
    request<import('./types').CredentialSummary[]>('/v1/providers/credentials'),
  saveProviderCredential: (
    providerId: string,
    data: { api_key: string; label?: string; config?: Record<string, any> }
  ) =>
    request<import('./types').CredentialSummary>(`/v1/providers/${providerId}/credentials`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  testProviderCredential: (
    providerId: string,
    data?: { api_key?: string; config?: Record<string, any> }
  ) =>
    request<import('./types').ProviderTestResult>(`/v1/providers/${providerId}/credentials/test`, {
      method: 'POST',
      body: JSON.stringify(data || {}),
    }),
  deleteProviderCredential: (providerId: string) =>
    request<{ deleted: boolean }>(`/v1/providers/${providerId}/credentials`, {
      method: 'DELETE',
    }),
}
