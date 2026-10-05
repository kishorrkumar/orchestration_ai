import React from 'react'
import { cn } from '../../lib/utils'

export interface AudioIndicatorProps {
  rms?: number // 0.0 to 1.0
  isSpeaking?: boolean
  isListening?: boolean
  className?: string
  size?: number
}

export const AudioIndicator: React.FC<AudioIndicatorProps> = ({
  rms = 0.0,
  isSpeaking = false,
  isListening = true,
  className,
  size = 80,
}) => {
  // Normalize and clamp RMS scale: 1.00 to 1.16 max
  const clampedRms = Math.min(Math.max(rms, 0), 1)
  const scale = 1.0 + clampedRms * 0.16
  const outerScale = 1.0 + clampedRms * 0.28

  // Palette: Stone grey when listening / idle, terracotta clay when speaking
  const coreBg = isSpeaking ? 'bg-[#C2603F]' : 'bg-[#1F1E1D]'
  const ringBorder = isSpeaking
    ? 'border-[#C2603F]/25 bg-[#C2603F]/10'
    : 'border-[#1F1E1D]/15 bg-[#1F1E1D]/5'

  return (
    <div
      className={cn('relative flex items-center justify-center select-none', className)}
      style={{ width: size * 1.5, height: size * 1.5 }}
      aria-label={isSpeaking ? 'Agent speaking' : isListening ? 'Listening' : 'Idle'}
      role="img"
    >
      {/* Outer subtle breathing aura ring */}
      <div
        className={cn(
          'absolute rounded-full border transition-transform duration-100 ease-out pointer-events-none',
          ringBorder
        )}
        style={{
          width: size * 1.25,
          height: size * 1.25,
          transform: `scale(${outerScale})`,
        }}
      />

      {/* Primary calm disc */}
      <div
        className={cn(
          'rounded-full shadow-xs transition-all duration-100 ease-out flex items-center justify-center',
          coreBg
        )}
        style={{
          width: size,
          height: size,
          transform: `scale(${scale})`,
        }}
      >
        {/* Subtle inner tactile accent */}
        <div
          className={cn(
            'w-2 h-2 rounded-full transition-opacity duration-200',
            isSpeaking ? 'bg-white/80' : 'bg-white/30'
          )}
        />
      </div>
    </div>
  )
}
