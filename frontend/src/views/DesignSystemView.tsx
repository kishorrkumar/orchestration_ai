import React, { useState } from 'react'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { AudioIndicator } from '../components/ui/AudioIndicator'
import { TokenMeter } from '../components/ui/TokenMeter'
import { Monogram } from '../components/ui/Monogram'
import { Input } from '../components/ui/Input'
import { TextArea } from '../components/ui/TextArea'
import { Phone, ArrowLeft } from 'lucide-react'

export const DesignSystemView: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [rms, setRms] = useState(0.4)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [tokenCount, setTokenCount] = useState(132)
  const [btnLoading, setBtnLoading] = useState(false)

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-12">
      {/* Header */}
      <div className="border-b border-[rgba(31,30,29,0.08)] pb-6 flex items-center justify-between">
        <div>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-1.5 text-[13px] text-[#6B6963] hover:text-[#1F1E1D] mb-2 font-medium"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Platform
          </button>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#1F1E1D]">
            Design System & Living Spec
          </h1>
          <p className="text-[14px] text-[#6B6963] mt-1">
            Apple HIG Precision + Claude Calm & Warmth — Zero AI Clichés
          </p>
        </div>
        <Badge variant="published" dot>
          Living Style Guide
        </Badge>
      </div>

      {/* 1. Typography Scale */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          1. Typography Specimen
        </h2>
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl p-6 space-y-6">
          <div>
            <span className="text-[12px] text-[#9E9B93] block mb-1">Large Title (34px/41px, Semibold -0.02em)</span>
            <div className="text-[34px] leading-[41px] font-semibold tracking-[-0.02em] text-[#1F1E1D]">
              Conversational Voice Platform
            </div>
          </div>
          <div className="border-t border-[rgba(31,30,29,0.06)] pt-4">
            <span className="text-[12px] text-[#9E9B93] block mb-1">Title 1 (24px/30px, Semibold -0.015em)</span>
            <div className="text-[24px] leading-[30px] font-semibold tracking-[-0.015em] text-[#1F1E1D]">
              Elena — Medical Receptionist
            </div>
          </div>
          <div className="border-t border-[rgba(31,30,29,0.06)] pt-4">
            <span className="text-[12px] text-[#9E9B93] block mb-1">Body (15px/22px, 1.6 line-height)</span>
            <p className="text-[15px] leading-[22px] text-[#1F1E1D] max-w-2xl">
              You are an empathetic, calm, and professional clinic assistant. Greet the patient warmly, confirm their preferred appointment time, and conclude naturally.
            </p>
          </div>
          <div className="border-t border-[rgba(31,30,29,0.06)] pt-4">
            <span className="text-[12px] text-[#9E9B93] block mb-1">Tabular Monospace (Meters, Timers & Delimiters)</span>
            <div className="font-mono text-[14px] tabular-nums text-[#6B6963] bg-[#F3F1EA] p-3 rounded-lg border border-[rgba(31,30,29,0.06)]">
              &lt;system&gt; 00:14.28 | TTFA: 198ms | Tokens: 132/350 &lt;system&gt;
            </div>
          </div>
        </div>
      </section>

      {/* 2. Color Tokens */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          2. Color Palette & Roles
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          <div className="p-3 rounded-lg border border-[rgba(31,30,29,0.08)] bg-[#FAF9F5]">
            <div className="w-full h-8 rounded bg-[#FAF9F5] border border-[rgba(31,30,29,0.1)] mb-2" />
            <div className="text-[12px] font-medium text-[#1F1E1D]">Canvas Base</div>
            <div className="text-[11px] text-[#9E9B93]">#FAF9F5</div>
          </div>
          <div className="p-3 rounded-lg border border-[rgba(31,30,29,0.08)] bg-white">
            <div className="w-full h-8 rounded bg-[#F3F1EA] mb-2" />
            <div className="text-[12px] font-medium text-[#1F1E1D]">Sunken Inset</div>
            <div className="text-[11px] text-[#9E9B93]">#F3F1EA</div>
          </div>
          <div className="p-3 rounded-lg border border-[rgba(31,30,29,0.08)] bg-white">
            <div className="w-full h-8 rounded bg-[#C2603F] mb-2" />
            <div className="text-[12px] font-medium text-[#1F1E1D]">Terracotta</div>
            <div className="text-[11px] text-[#9E9B93]">#C2603F</div>
          </div>
          <div className="p-3 rounded-lg border border-[rgba(31,30,29,0.08)] bg-white">
            <div className="w-full h-8 rounded bg-[#386641] mb-2" />
            <div className="text-[12px] font-medium text-[#1F1E1D]">Forest Green</div>
            <div className="text-[11px] text-[#9E9B93]">#386641</div>
          </div>
          <div className="p-3 rounded-lg border border-[rgba(31,30,29,0.08)] bg-white">
            <div className="w-full h-8 rounded bg-[#995D1A] mb-2" />
            <div className="text-[12px] font-medium text-[#1F1E1D]">Warm Amber</div>
            <div className="text-[11px] text-[#9E9B93]">#995D1A</div>
          </div>
          <div className="p-3 rounded-lg border border-[rgba(31,30,29,0.08)] bg-white">
            <div className="w-full h-8 rounded bg-[#A63A38] mb-2" />
            <div className="text-[12px] font-medium text-[#1F1E1D]">Muted Brick</div>
            <div className="text-[11px] text-[#9E9B93]">#A63A38</div>
          </div>
        </div>
      </section>

      {/* 3. Buttons */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          3. Button Variants & States
        </h2>
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl p-6 space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" leftIcon={<Phone className="w-4 h-4" />}>
              Start Call
            </Button>
            <Button variant="secondary">
              Save Draft
            </Button>
            <Button variant="quiet">
              Cancel
            </Button>
            <Button variant="destructive">
              End Call
            </Button>
            <Button
              variant="secondary"
              isLoading={btnLoading}
              onClick={() => {
                setBtnLoading(true)
                setTimeout(() => setBtnLoading(false), 1500)
              }}
            >
              {btnLoading ? 'Publishing...' : 'Click for Loading State'}
            </Button>
          </div>
          <div className="flex items-center gap-3 pt-2">
            <Button size="compact" variant="secondary">Compact (32px)</Button>
            <Button size="regular" variant="secondary">Regular (40px)</Button>
            <Button size="prominent" variant="primary">Prominent (48px)</Button>
          </div>
        </div>
      </section>

      {/* 4. Badges & Monograms */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          4. Status Badges & Agent Monograms
        </h2>
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl p-6 space-y-6">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="published" dot>Published (v2)</Badge>
            <Badge variant="draft" dot>Draft (v3)</Badge>
            <Badge variant="connected" dot>Live Call</Badge>
            <Badge variant="warning" dot>Token Warning</Badge>
            <Badge variant="danger" dot>Exceeded Limit</Badge>
            <Badge variant="neutral">Asia/Kolkata</Badge>
          </div>
          <div className="flex items-center gap-4 pt-2">
            <div className="flex items-center gap-2">
              <Monogram name="Elena" size="sm" />
              <span className="text-[13px] text-[#6B6963]">Elena (sm)</span>
            </div>
            <div className="flex items-center gap-2">
              <Monogram name="Marcus Clinic" size="md" />
              <span className="text-[13px] text-[#6B6963]">Marcus Clinic (md)</span>
            </div>
            <div className="flex items-center gap-2">
              <Monogram name="Support Desk" size="lg" />
              <span className="text-[13px] text-[#6B6963]">Support Desk (lg)</span>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Restrained Audio Indicator (Anti-AI Look) */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          5. Restrained Call Audio Indicator (Zero Glowing Neon Blobs)
        </h2>
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl p-6 flex flex-col md:flex-row items-center gap-8">
          <div className="flex flex-col items-center justify-center p-6 bg-[#FAF9F5] rounded-xl border border-[rgba(31,30,29,0.06)] min-w-[200px]">
            <AudioIndicator rms={rms} isSpeaking={isSpeaking} size={80} />
            <span className="text-[12px] font-medium text-[#6B6963] mt-3">
              {isSpeaking ? 'Agent Speaking' : 'Listening...'}
            </span>
          </div>

          <div className="space-y-4 flex-1 w-full">
            <div>
              <label className="text-[13px] font-medium text-[#1F1E1D] flex justify-between">
                <span>Simulated Speech Energy (RMS Level)</span>
                <span className="tabular-nums text-[#6B6963]">{Math.round(rms * 100)}%</span>
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={rms}
                onChange={(e) => setRms(parseFloat(e.target.value))}
                className="w-full accent-[#C2603F] mt-2 cursor-pointer"
              />
            </div>

            <div className="flex items-center gap-3">
              <Button
                variant={isSpeaking ? 'primary' : 'secondary'}
                onClick={() => setIsSpeaking(!isSpeaking)}
              >
                {isSpeaking ? 'Switch to Listening' : 'Switch to Speaking'}
              </Button>
            </div>
            <p className="text-[12px] text-[#9E9B93] leading-relaxed">
              Monochrome disc scales subtly with volume (scale 1.00 to 1.16). Transitions to warm terracotta only when assistant audio is active. No multi-layer neon glow or particle waves.
            </p>
          </div>
        </div>
      </section>

      {/* 6. Token Budget Meter */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          6. Token Budget Meter
        </h2>
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl p-6 space-y-4">
          <TokenMeter tokenCount={tokenCount} recommendedLimit={150} hardLimit={350} />

          <div className="pt-3">
            <label className="text-[13px] font-medium text-[#1F1E1D] flex justify-between">
              <span>Adjust Simulated Prompt Tokens:</span>
              <span className="tabular-nums font-semibold">{tokenCount} tokens</span>
            </label>
            <input
              type="range"
              min="20"
              max="400"
              value={tokenCount}
              onChange={(e) => setTokenCount(parseInt(e.target.value, 10))}
              className="w-full accent-[#C2603F] mt-2 cursor-pointer"
            />
          </div>
        </div>
      </section>

      {/* 7. Form Fields & Inset Inputs */}
      <section className="space-y-4">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-[#9E9B93]">
          7. Form Inputs & Inset Controls
        </h2>
        <div className="bg-[#FFFFFF] border border-[rgba(31,30,29,0.08)] rounded-xl p-6 space-y-4">
          <Input
            label="Agent Name"
            hint="Display name & {{agent_name}}"
            defaultValue="Elena Care"
          />
          <TextArea
            label="System Prompt"
            hint="Spoken voice persona instructions"
            defaultValue="You are Elena, a calm and welcoming clinic receptionist."
            rows={3}
          />
        </div>
      </section>
    </div>
  )
}
