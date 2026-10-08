export interface VoicePreset {
  id: string
  name: string
  gender: string
  speaking_style: string
  accent: string
  recommended_for: string
  is_cloned?: boolean
  preview_url?: string | null
  duration_sec?: number | null
}

export type EngineType = 'personaplex_s2s' | 'cascaded_cloud'

export interface Agent {
  id: string
  name: string
  voice_id: string
  greeting: string
  agent_speaks_first: boolean
  system_prompt: string
  ending: string
  end_call_timeout_sec: number
  silence_timeout_sec: number
  max_duration_sec: number
  timezone_str: string
  engine?: EngineType
  language?: string
  pipeline_json?: string
  status: 'draft' | 'published' | 'archived'
  current_version: number
  published_version?: number | null
  created_at: string
  updated_at: string
}

export interface AgentVersion {
  version_id: string
  agent_id: string
  version_number: number
  name: string
  voice_id: string
  greeting: string
  agent_speaks_first: boolean
  system_prompt: string
  ending: string
  end_call_timeout_sec: number
  silence_timeout_sec: number
  max_duration_sec: number
  timezone_str: string
  engine?: EngineType
  language?: string
  pipeline_json?: string
  token_count: number
  created_at: string
  change_note: string
}

export interface STTPipelineConfig {
  provider_id: string
  model: string
  language: string
  credential_id?: string | null
  extra_params?: Record<string, any>
}

export interface LLMPipelineConfig {
  provider_id: string
  model: string
  temperature: number
  max_tokens?: number | null
  credential_id?: string | null
  extra_params?: Record<string, any>
}

export interface TTSPipelineConfig {
  provider_id: string
  model: string
  voice: string
  speed?: number
  credential_id?: string | null
  extra_params?: Record<string, any>
}

export interface TurnPipelineConfig {
  strategy: 'auto' | 'provider_eot' | 'vad_smart_turn'
  stop_words?: string[]
  eot_timeout_ms?: number
}

export interface CascadedPipelineSpec {
  stt: STTPipelineConfig
  llm: LLMPipelineConfig
  tts: TTSPipelineConfig
  turn: TurnPipelineConfig
}

export interface LintWarning {
  rule: string
  message: string
  severity: 'warning' | 'info'
  match?: string | null
}

export interface CompilePromptResult {
  compiled_text: string
  wrapped_text: string
  token_count: number
  hard_limit: number
  recommended_limit: number
  is_within_hard_limit: boolean
  is_fast_start: boolean
  local_time_line: string
  warnings: LintWarning[]
}

export interface CallSession {
  id: string
  agent_id: string
  agent_version: number
  status: 'initiating' | 'connected' | 'completed' | 'failed' | 'busy'
  client_type: string
  duration_sec: number
  end_reason: string | null
  total_turns: number
  created_at: string
  ended_at: string | null
}

export interface CallTurn {
  id: string
  turn_index: number
  speaker: 'caller' | 'agent' | 'system'
  text: string
  started_at_sec: number
  ended_at_sec: number
  latency_ms: number | null
  created_at: string
}

export interface ProblemDetail {
  type: string
  title: string
  status: number
  detail: string
  code: string
  error_id: string
  invalid_params?: Array<{
    name: string
    reason: string
    type: string
  }>
}

export type ProviderKind = 'stt' | 'llm' | 'tts'

export interface AuthFieldSpec {
  name: string
  label: string
  field_type: 'password' | 'text' | 'select'
  required: boolean
  placeholder?: string
  description?: string
  options?: string[]
  default_value?: string
}

export interface ModelDescriptor {
  id: string
  name: string
  description?: string
  context_window?: number
  latency_profile?: string
  last_verified?: string
}

export interface VoiceDescriptor {
  id: string
  name: string
  gender?: string
  language: string
  preview_url?: string
}

export interface ProviderManifest {
  id: string
  kind: ProviderKind
  display_name: string
  description: string
  docs_url: string
  auth_fields: AuthFieldSpec[]
  supported_languages: string[]
  native_sample_rates: number[]
  streaming: boolean
  native_eot: boolean
  default_model: string
  default_voice?: string
  model_list_type: 'static' | 'live_api'
  models: ModelDescriptor[]
  voices: VoiceDescriptor[]
  pipecat_service: string
}

export interface CredentialSummary {
  id: string
  workspace_id: string
  provider_id: string
  label: string
  last4: string
  status: 'untested' | 'valid' | 'invalid' | 'rate_limited'
  last_tested_at?: string
  last_latency_ms?: number
  config: Record<string, any>
  created_at: string
}

export interface ProviderTestResult {
  status: 'valid' | 'invalid' | 'rate_limited' | 'error'
  latency_ms: number
  error_message?: string
}

