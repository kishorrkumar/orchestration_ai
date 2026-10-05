import React from 'react'
import { cn } from '../../lib/utils'

export interface MonogramProps {
  name: string
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const PALETTES = [
  { bg: 'bg-[#F2ECE6]', text: 'text-[#87553A]', border: 'border-[#DEC8BA]' }, // Terracotta / Ochre
  { bg: 'bg-[#EBF1EB]', text: 'text-[#386641]', border: 'border-[#C4D9C5]' }, // Sage / Olive
  { bg: 'bg-[#ECEFF3]', text: 'text-[#38536C]', border: 'border-[#CAD6E2]' }, // Slate / Blue Stone
  { bg: 'bg-[#F3EFE7]', text: 'text-[#7D632E]', border: 'border-[#E0D2B8]' }, // Sand / Ochre
  { bg: 'bg-[#EDEBF2]', text: 'text-[#584D72]', border: 'border-[#D0C9DD]' }, // Heather / Muted Plum
  { bg: 'bg-[#EEEEEC]', text: 'text-[#4E4D48]', border: 'border-[#D1D0CB]' }, // Warm Stone
]

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/)
  if (parts.length === 0 || !parts[0]) return 'A'
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return (parts[0][0] + parts[1][0]).toUpperCase()
}

function getPaletteIndex(name: string): number {
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
  }
  return Math.abs(hash) % PALETTES.length
}

export const Monogram: React.FC<MonogramProps> = ({
  name,
  size = 'md',
  className,
}) => {
  const initials = getInitials(name)
  const palette = PALETTES[getPaletteIndex(name)]

  const sizeClasses = {
    sm: 'w-7 h-7 text-[11px] font-medium border',
    md: 'w-9 h-9 text-[13px] font-medium border',
    lg: 'w-12 h-12 text-[16px] font-semibold border',
  }

  return (
    <div
      className={cn(
        'rounded-full flex items-center justify-center select-none shrink-0 tracking-wider',
        palette.bg,
        palette.text,
        palette.border,
        sizeClasses[size],
        className
      )}
      aria-hidden="true"
    >
      {initials}
    </div>
  )
}
