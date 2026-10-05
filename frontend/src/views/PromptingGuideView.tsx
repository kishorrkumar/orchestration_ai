import { useState } from 'react'
import { AGENT_TEMPLATES, type AgentTemplate } from '../lib/templates'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import {
  BookOpen,
  Check,
  Copy,
  ArrowRight,
  Sparkles,
  Zap,
  Clock,
  Mic,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
} from 'lucide-react'

interface PromptingGuideViewProps {
  onBack: () => void
  onUseTemplate: (template: AgentTemplate) => void
}

export function PromptingGuideView({ onBack, onUseTemplate }: PromptingGuideViewProps) {
  const [activeTab, setActiveTab] = useState<
    'templates' | 'principles' | 'fields' | 'variables' | 'rules' | 'tokens'
  >('templates')
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--color-hairline)] pb-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <Badge variant="neutral" className="gap-1 px-2.5 py-0.5">
              <BookOpen className="w-3.5 h-3.5 text-[#C2603F]" />
              Official Documentation
            </Badge>
            <span className="text-[12px] text-[var(--color-text-tertiary)]">PersonaPlex 7B</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[var(--color-text-primary)]">
            Speech-to-Speech Prompting Guide
          </h1>
          <p className="text-[14px] text-[var(--color-text-secondary)] mt-1 max-w-2xl">
            How to craft natural, ultra-low latency voice agent prompts for the PersonaPlex 7B dual-decoder model.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <Button variant="secondary" onClick={onBack}>
            Back to Agents
          </Button>
        </div>
      </div>

      {/* Navigation Pills */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-[var(--color-hairline)]">
        {[
          { id: 'templates', label: 'Ready-to-Use Templates', icon: Sparkles },
          { id: 'principles', label: 'S2S vs. LLM Principles', icon: Zap },
          { id: 'fields', label: 'The 6 Lean Fields', icon: Mic },
          { id: 'variables', label: 'Variables & Time', icon: Clock },
          { id: 'rules', label: "Do's & Don'ts", icon: CheckCircle2 },
          { id: 'tokens', label: 'Token Budgets & Latency', icon: HelpCircle },
        ].map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`px-3.5 py-2 rounded-xl text-[13px] font-medium flex items-center gap-2 whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-[var(--color-bg-sunken)] text-[var(--color-text-primary)] font-semibold shadow-xs'
                  : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-sunken)]/50'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-[#C2603F]' : ''}`} />
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* TAB 1: READY-TO-USE TEMPLATES */}
      {activeTab === 'templates' && (
        <div className="space-y-6">
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6">
            <h2 className="text-[16px] font-semibold text-[var(--color-text-primary)] mb-1">
              Production Voice Agent Templates
            </h2>
            <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed">
              Every template is pre-tuned to stay under the recommended <strong>135-token budget</strong> for fast start (&lt;200ms Time-to-First-Audio), with natural spoken phrasing, clean ending triggers, and zero markdown or emojis.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-6">
            {AGENT_TEMPLATES.map((tmpl) => (
              <div
                key={tmpl.id}
                className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6 space-y-4 hover:border-[var(--color-border)] transition-all shadow-xs"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--color-hairline)] pb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-[16px] font-semibold text-[var(--color-text-primary)]">
                        {tmpl.name}
                      </h3>
                      <Badge variant="published" className="text-[11px]">
                        ~{tmpl.approx_tokens} tokens
                      </Badge>
                      <Badge variant="neutral" className="text-[11px]">
                        {tmpl.voice_id}
                      </Badge>
                    </div>
                    <p className="text-[13px] text-[#C2603F] font-medium mt-0.5">
                      {tmpl.role}
                    </p>
                    <p className="text-[12px] text-[var(--color-text-tertiary)] mt-0.5">
                      {tmpl.description}
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      variant="secondary"
                      size="compact"
                      onClick={() =>
                        handleCopy(
                          `Name: ${tmpl.name}\nVoice: ${tmpl.voice_id}\nGreeting: ${tmpl.greeting}\nSystem Prompt: ${tmpl.system_prompt}\nEnding: ${tmpl.ending}\nTimezone: ${tmpl.timezone_str}`,
                          tmpl.id
                        )
                      }
                    >
                      {copiedId === tmpl.id ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-[#386641]" />
                          Copied
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          Copy All
                        </>
                      )}
                    </Button>
                    <Button
                      variant="primary"
                      size="compact"
                      onClick={() => onUseTemplate(tmpl)}
                    >
                      Use in Editor
                      <ArrowRight className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[13px]">
                  <div className="space-y-1.5 p-3.5 bg-[var(--color-bg-sunken)]/60 rounded-xl">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)] block">
                      Greeting (Spoken Opening Line)
                    </span>
                    <p className="text-[var(--color-text-primary)] italic">
                      "{tmpl.greeting}"
                    </p>
                  </div>

                  <div className="space-y-1.5 p-3.5 bg-[var(--color-bg-sunken)]/60 rounded-xl">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)] block">
                      Ending Text (Auto-Hangup Trigger)
                    </span>
                    <p className="text-[var(--color-text-primary)] italic">
                      "{tmpl.ending}"
                    </p>
                  </div>
                </div>

                <div className="space-y-1.5 p-3.5 bg-[var(--color-bg-sunken)]/40 rounded-xl">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)] block">
                      System Prompt (Spoken Persona Instructions)
                    </span>
                    <span className="text-[11px] text-[var(--color-text-tertiary)]">
                      Timezone: {tmpl.timezone_str}
                    </span>
                  </div>
                  <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed font-sans">
                    {tmpl.system_prompt}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 2: PRINCIPLES (S2S VS LLM) */}
      {activeTab === 'principles' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-[#F3F1EA] flex items-center justify-center text-[var(--color-text-primary)] font-semibold">
                Text
              </div>
              <h3 className="text-[16px] font-semibold text-[var(--color-text-primary)]">
                Text LLMs (ChatGPT / Claude)
              </h3>
              <ul className="text-[13px] text-[var(--color-text-secondary)] space-y-2 list-disc pl-4 leading-relaxed">
                <li>Generates text characters to be read by human eyes on a screen.</li>
                <li>Punctuation provides visual sentence boundaries.</li>
                <li>Markdown headings, bold words, and bullet points structure layout.</li>
                <li>Long negative constraint lists ("NEVER do X, NEVER say Y") are followed reliably.</li>
                <li>Tolerates thousands of prompt tokens with nominal latency penalty.</li>
              </ul>
            </div>

            <div className="bg-[var(--color-bg-surface)] border border-[rgba(194,96,63,0.3)] rounded-2xl p-6 space-y-3 shadow-xs">
              <div className="w-10 h-10 rounded-xl bg-[#FAF0EC] text-[#C2603F] flex items-center justify-center font-semibold">
                Voice
              </div>
              <h3 className="text-[16px] font-semibold text-[var(--color-text-primary)]">
                PersonaPlex 7B (Speech-to-Speech)
              </h3>
              <ul className="text-[13px] text-[var(--color-text-secondary)] space-y-2 list-disc pl-4 leading-relaxed">
                <li><strong>Directly synthesizes spoken audio waveforms</strong> (24 kHz) and vocal intonation.</li>
                <li><strong>Punctuation dictates vocal breath, pausing, and cadence.</strong> Commas add natural pauses; question marks raise pitch.</li>
                <li><strong>Markdown and bullet points are destructive.</strong> The acoustic model attempts to pronounce symbols like "star star bold".</li>
                <li><strong>Positive conversational style works best.</strong> Tell the agent how to converse rather than listing dozens of bans.</li>
                <li><strong>Prompt length strictly dictates latency.</strong> Keep prompt &le; 135 tokens for &lt;200ms response time.</li>
              </ul>
            </div>
          </div>

          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6 space-y-3">
            <h3 className="text-[15px] font-semibold text-[var(--color-text-primary)]">
              Exact Upstream Delimiter Syntax
            </h3>
            <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed">
              PersonaPlex conditions on the system prompt using an exact delimiter token verified against the upstream runtime:
            </p>
            <div className="bg-[var(--color-bg-sunken)] p-3.5 rounded-xl font-mono text-[13px] text-[var(--color-text-primary)] select-all">
              &lt;system&gt; &#123;system_prompt&#125; &lt;system&gt;
            </div>
            <p className="text-[12px] text-[var(--color-text-tertiary)]">
              Note: Both opening and closing tags are literally <code className="font-mono text-[#C2603F]">&lt;system&gt;</code>, NOT <code className="font-mono">&lt;s&gt;</code> and NOT <code className="font-mono">&lt;/system&gt;</code>. The studio prompt compiler wraps this automatically.
            </p>
          </div>
        </div>
      )}

      {/* TAB 3: THE 6 LEAN FIELDS */}
      {activeTab === 'fields' && (
        <div className="space-y-4">
          {[
            {
              num: 1,
              title: 'Name',
              desc: 'Display name for the agent in lists, logs, and call records. Also automatically interpolates into {{agent_name}} in prompts and greetings.',
              example: 'Elena — Front Desk Receptionist',
            },
            {
              num: 2,
              title: 'Voice Preset',
              desc: 'One of the 18 official upstream PersonaPlex voice conditioning embeddings (NATF0.pt to NATF8.pt, NATM0.pt to NATM8.pt). Provides zero-shot vocal timbre, pitch, and cadence without synthetic cloning.',
              example: 'NATF2.pt (Warm, professional female)',
            },
            {
              num: 3,
              title: 'Greeting Text',
              desc: 'The exact opening line spoken by the agent when the call connects (when Greeting Mode is agent_first). Kept short and welcoming.',
              example: 'Hi, thank you for calling Horizon Dental! This is Elena. How can I help you today?',
            },
            {
              num: 4,
              title: 'System Prompt',
              desc: 'The core spoken persona instructions defining identity, conversational tone, primary goal, and scope guardrails. Must be written in natural spoken English prose.',
              example: 'You are Elena, a courteous front desk assistant. Your goal is to help callers book dental visits. Speak in 1-2 short sentences per turn.',
            },
            {
              num: 5,
              title: 'Ending Text',
              desc: 'The natural wrap-up closing line spoken by the agent when ending the call. Monitored by the EndOfCallDetector state machine to trigger clean, automatic hangup after a 1.5-second audio drain window.',
              example: 'Thanks for calling Horizon Dental. Have a wonderful day, goodbye!',
            },
            {
              num: 6,
              title: 'Timezone',
              desc: 'An IANA timezone identifier (e.g. America/New_York, Asia/Kolkata). The gateway calculates the caller\'s local weekday, time, and day-part and injects it into the prompt context at call start.',
              example: 'Asia/Kolkata (supports half-hour offsets UTC+05:30)',
            },
          ].map((field) => (
            <div
              key={field.num}
              className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-5 flex flex-col sm:flex-row sm:items-start gap-4"
            >
              <div className="w-8 h-8 rounded-lg bg-[var(--color-bg-sunken)] text-[var(--color-text-primary)] font-bold text-[14px] flex items-center justify-center shrink-0 border border-[var(--color-hairline)]">
                {field.num}
              </div>
              <div className="space-y-1.5 flex-1">
                <div className="flex items-center justify-between">
                  <h3 className="text-[15px] font-semibold text-[var(--color-text-primary)]">
                    {field.title}
                  </h3>
                </div>
                <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed">
                  {field.desc}
                </p>
                <div className="pt-1">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-tertiary)] block">
                    Example:
                  </span>
                  <p className="text-[13px] text-[#C2603F] font-mono mt-0.5">
                    {field.example}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* TAB 4: VARIABLES & TIME */}
      {activeTab === 'variables' && (
        <div className="space-y-6">
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6 space-y-4">
            <h2 className="text-[16px] font-semibold text-[var(--color-text-primary)]">
              Dynamic Template Variables
            </h2>
            <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed">
              Use double curly braces <code className="font-mono text-[#C2603F]">&#123;&#123;variable&#125;&#125;</code> in your System Prompt or Greeting. The prompt compiler renders them dynamically when a call connects:
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-[13px]">
                <thead>
                  <tr className="border-b border-[var(--color-hairline)] text-[12px] font-semibold text-[var(--color-text-tertiary)]">
                    <th className="py-2.5 pr-4">Variable</th>
                    <th className="py-2.5 px-4">Description</th>
                    <th className="py-2.5 pl-4">Example Output</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--color-hairline)]">
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;agent_name&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">Agent's configured display name</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">Sarah</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;current_time&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">Local 12-hour time without leading zero</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">3:15 PM</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;weekday&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">Current day of week in timezone</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">Monday</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;day_part&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">morning (5-12), afternoon (12-17), evening (17-21), night</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">afternoon</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;date&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">Formatted calendar date</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">October 05, 2026</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;caller_name&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">Name of the caller (if passed in query / CRM)</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">Alex</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-mono text-[#C2603F]">&#123;&#123;phone_number&#125;&#125;</td>
                    <td className="py-2.5 px-4 text-[var(--color-text-secondary)]">Caller telephone number</td>
                    <td className="py-2.5 pl-4 font-mono text-[12px]">+1-555-0199</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6 space-y-3">
            <h3 className="text-[15px] font-semibold text-[var(--color-text-primary)]">
              Automatic Local Time Injection
            </h3>
            <p className="text-[13px] text-[var(--color-text-secondary)] leading-relaxed">
              You do not need to manually write time rules. The gateway automatically injects an authoritative local time line into the compiled prompt:
            </p>
            <div className="p-3 bg-[var(--color-bg-sunken)] rounded-xl font-mono text-[13px] text-[var(--color-text-primary)]">
              It is Monday, 3:15 PM (afternoon) for the caller.
            </div>
            <p className="text-[12px] text-[var(--color-text-tertiary)]">
              This prevents temporal hallucination and enables the agent to naturally say "Good afternoon" on Monday at 3 PM without manual prompting.
            </p>
          </div>
        </div>
      )}

      {/* TAB 5: DO'S & DON'TS */}
      {activeTab === 'rules' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* DOs */}
          <div className="bg-[var(--color-bg-surface)] border border-[rgba(56,102,65,0.3)] rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2 text-[#386641] font-semibold text-[15px]">
              <CheckCircle2 className="w-5 h-5" />
              Voice Prompting DO's
            </div>

            <ul className="space-y-3 text-[13px] text-[var(--color-text-secondary)]">
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-[#386641] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">Write for the ear:</strong> Read your prompt aloud. If it sounds like a written document, simplify it into spoken dialogue.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-[#386641] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">Use short sentences:</strong> Keep conversational turns to 1–2 brief sentences. Long monologues cause the user to interrupt.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-[#386641] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">Use natural contractions:</strong> Write <em>"I'm"</em>, <em>"we'll"</em>, <em>"don't"</em>, <em>"you're"</em> instead of formal uncontracted forms.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-[#386641] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">Spell numbers simply:</strong> Write <em>"$25"</em> or <em>"twenty-five dollars"</em> and <em>"9 AM"</em> instead of <em>"USD 25.00"</em> or <em>"09:00:00"</em>.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <Check className="w-4 h-4 text-[#386641] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">State the goal clearly:</strong> Explicitly state: <em>"Your goal is to help callers schedule appointments."</em>
                </div>
              </li>
            </ul>
          </div>

          {/* DONTs */}
          <div className="bg-[var(--color-bg-surface)] border border-[rgba(166,58,56,0.3)] rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2 text-[#A63A38] font-semibold text-[15px]">
              <XCircle className="w-5 h-5" />
              Voice Prompting DON'Ts
            </div>

            <ul className="space-y-3 text-[13px] text-[var(--color-text-secondary)]">
              <li className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-[#A63A38] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">No Markdown formatting:</strong> Never use <code className="font-mono">**bold**</code>, <code className="font-mono"># headers</code>, or <code className="font-mono">- bullet lists</code>. The neural model will attempt to vocalize symbols.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-[#A63A38] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">No Emojis:</strong> Never include emojis (😊, 🚀, 👍). The acoustic tokenizer will produce vocal distortion or garbled sound.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-[#A63A38] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">No long "NEVER" rule lists:</strong> Avoid repeating lists of negative constraints. Positive conversational style instructions work significantly better.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-[#A63A38] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">English only:</strong> PersonaPlex weights are strictly trained on English speech. Foreign words cause phonetic hallucinations.
                </div>
              </li>
              <li className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-[#A63A38] shrink-0 mt-0.5" />
                <div>
                  <strong className="text-[var(--color-text-primary)]">No prompt knowledge dumps:</strong> Do not paste full policy manuals into the system prompt. Keep it under 135 tokens.
                </div>
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* TAB 6: TOKEN BUDGETS & LATENCY */}
      {activeTab === 'tokens' && (
        <div className="space-y-6">
          <div className="bg-[var(--color-bg-surface)] border border-[var(--color-hairline)] rounded-2xl p-6 space-y-6">
            <div>
              <h2 className="text-[16px] font-semibold text-[var(--color-text-primary)]">
                SentencePiece Token Budgets & Latency Impact
              </h2>
              <p className="text-[13px] text-[var(--color-text-secondary)] mt-1">
                Measured with the official 32k PersonaPlex SentencePiece model. The model conditions its KV-cache on the prompt text before processing speech frames.
              </p>
            </div>

            {/* Visual Bar */}
            <div className="space-y-2">
              <div className="h-4 w-full bg-[var(--color-bg-sunken)] rounded-full overflow-hidden flex">
                <div className="bg-[#386641] h-full w-[38%]" title="Fast Start (0 - 135 tokens)" />
                <div className="bg-[#588157] h-full w-[5%]" title="Acceptable (135 - 150 tokens)" />
                <div className="bg-[#DDA15E] h-full w-[57%]" title="Amber Warning (150 - 350 tokens)" />
              </div>

              <div className="flex justify-between text-[11px] font-mono text-[var(--color-text-tertiary)] px-1">
                <span>0 tokens</span>
                <span className="text-[#386641] font-semibold">135 (Fast Start)</span>
                <span>150 (Ideal)</span>
                <span className="text-[#A63A38] font-semibold">350 (Hard Limit)</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-[var(--color-bg-sunken)]/60 rounded-xl space-y-1">
                <div className="flex items-center gap-1.5 text-[#386641] font-semibold text-[13px]">
                  <Check className="w-4 h-4" />
                  &le; 135 tokens
                </div>
                <div className="text-[20px] font-mono font-bold text-[var(--color-text-primary)]">
                  &lt; 200ms
                </div>
                <div className="text-[12px] text-[var(--color-text-secondary)]">
                  Instant, conversational speech start. Highly recommended for production.
                </div>
              </div>

              <div className="p-4 bg-[var(--color-bg-sunken)]/60 rounded-xl space-y-1">
                <div className="flex items-center gap-1.5 text-[#588157] font-semibold text-[13px]">
                  <Check className="w-4 h-4" />
                  136 - 150 tokens
                </div>
                <div className="text-[20px] font-mono font-bold text-[var(--color-text-primary)]">
                  200 - 280ms
                </div>
                <div className="text-[12px] text-[var(--color-text-secondary)]">
                  Acceptable for roles requiring slightly more persona context.
                </div>
              </div>

              <div className="p-4 bg-[var(--color-bg-sunken)]/60 rounded-xl space-y-1">
                <div className="flex items-center gap-1.5 text-[#DDA15E] font-semibold text-[13px]">
                  <AlertTriangle className="w-4 h-4" />
                  151 - 350 tokens
                </div>
                <div className="text-[20px] font-mono font-bold text-[var(--color-text-primary)]">
                  280 - 450ms
                </div>
                <div className="text-[12px] text-[var(--color-text-secondary)]">
                  Triggers amber warning. Noticeable connection handshake delay.
                </div>
              </div>

              <div className="p-4 bg-[var(--color-bg-sunken)]/60 rounded-xl space-y-1">
                <div className="flex items-center gap-1.5 text-[#A63A38] font-semibold text-[13px]">
                  <XCircle className="w-4 h-4" />
                  &gt; 350 tokens
                </div>
                <div className="text-[20px] font-mono font-bold text-[var(--color-text-primary)]">
                  Blocked
                </div>
                <div className="text-[12px] text-[var(--color-text-secondary)]">
                  Publish blocked by gateway RFC 9457 linter. Shorten prompt to proceed.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
export default PromptingGuideView
