import React from 'react'
import { cn } from '../../lib/utils'

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'draft' | 'published' | 'connected' | 'warning' | 'danger' | 'neutral'
  dot?: boolean
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  dot = false,
  className,
  children,
  ...props
}) => {
  const variants = {
    neutral: 'bg-[#F3F1EA] text-[#6B6963] border-[rgba(31,30,29,0.08)]',
    draft: 'bg-[#F3F1EA] text-[#6B6963] border-[rgba(31,30,29,0.1)]',
    published: 'bg-[#EDF4EE] text-[#386641] border-[rgba(56,102,65,0.2)]',
    connected: 'bg-[#EDF4EE] text-[#386641] border-[rgba(56,102,65,0.2)]',
    warning: 'bg-[#FCF4EB] text-[#995D1A] border-[rgba(153,93,26,0.2)]',
    danger: 'bg-[#F9ECEB] text-[#A63A38] border-[rgba(166,58,56,0.2)]',
  }

  const dotColors = {
    neutral: 'bg-[#9E9B93]',
    draft: 'bg-[#9E9B93]',
    published: 'bg-[#386641]',
    connected: 'bg-[#386641]',
    warning: 'bg-[#995D1A]',
    danger: 'bg-[#A63A38]',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[12px] font-medium border tracking-wide select-none',
        variants[variant],
        className
      )}
      {...props}
    >
      {dot && <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', dotColors[variant])} />}
      {children}
    </span>
  )
}
