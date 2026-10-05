import React from 'react'
import { cn } from '../../lib/utils'

export interface TokenMeterProps {
  tokenCount: number
  recommendedLimit?: number
  hardLimit?: number
  className?: string
  showLabels?: boolean
}

export const TokenMeter: React.FC<TokenMeterProps> = ({
  tokenCount,
  recommendedLimit = 150,
  hardLimit = 350,
  className,
  showLabels = true,
}) => {
  // Percentage against hard limit (clamped to 100%)
  const percentage = Math.min(Math.max((tokenCount / hardLimit) * 100, 0), 100)
  const isOverHard = tokenCount > hardLimit
  const isOverRecommended = tokenCount > recommendedLimit

  // Color mapping
  let barColor = 'bg-[#1F1E1D]'
  let textColor = 'text-[#6B6963]'
  let statusText = 'Optimal speed (<150)'

  if (isOverHard) {
    barColor = 'bg-[#A63A38]'
    textColor = 'text-[#A63A38] font-medium'
    statusText = 'Exceeds limit (>350) — Cannot publish'
  } else if (isOverRecommended) {
    barColor = 'bg-[#995D1A]'
    textColor = 'text-[#995D1A]'
    statusText = 'Noticeable TTFA delay (>150)'
  }

  return (
    <div className={cn('space-y-1.5 w-full select-none', className)}>
      {showLabels && (
        <div className="flex items-center justify-between text-[13px]">
          <span className="text-[#6B6963] flex items-center gap-1.5">
            <span>Prompt Tokens:</span>
            <span className={cn('tabular-nums font-semibold', textColor)}>
              {tokenCount}
            </span>
            <span className="text-[#9E9B93]">/ {hardLimit}</span>
          </span>
          <span className={cn('text-[12px]', textColor)}>{statusText}</span>
        </div>
      )}

      {/* Track */}
      <div className="relative h-1.5 w-full bg-[#E8E6DF] rounded-full overflow-hidden">
        {/* Fill */}
        <div
          className={cn('h-full transition-all duration-300 ease-out rounded-full', barColor)}
          style={{ width: `${percentage}%` }}
        />

        {/* Recommended Limit Tick (150/350 = 42.8%) */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-[#FAF9F5] z-10"
          style={{ left: `${(recommendedLimit / hardLimit) * 100}%` }}
          title={`Recommended budget (${recommendedLimit} tokens)`}
        />
      </div>
    </div>
  )
}
