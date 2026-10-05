import React from 'react'
import { cn } from '../../lib/utils'
import { Loader2 } from 'lucide-react'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'quiet' | 'destructive'
  size?: 'compact' | 'regular' | 'prominent'
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'secondary',
      size = 'regular',
      isLoading = false,
      leftIcon,
      rightIcon,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center font-medium select-none transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C2603F] focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98]'

    const variants = {
      primary:
        'bg-[#C2603F] text-white shadow-xs hover:bg-[#AE5334] active:bg-[#9B482D]',
      secondary:
        'bg-[#F3F1EA] text-[#1F1E1D] border border-[rgba(31,30,29,0.08)] hover:bg-[#EBE8DF] active:bg-[#E2DFD5]',
      quiet:
        'text-[#6B6963] hover:text-[#1F1E1D] hover:bg-[rgba(31,30,29,0.05)] active:bg-[rgba(31,30,29,0.08)]',
      destructive:
        'bg-[#F9ECEB] text-[#A63A38] border border-[rgba(166,58,56,0.2)] hover:bg-[#F4DCDA] active:bg-[#EECBC8]',
    }

    const sizes = {
      compact: 'h-8 px-3 text-[13px] rounded-md gap-1.5',
      regular: 'h-10 px-4 text-[14px] rounded-lg gap-2',
      prominent: 'h-12 px-5 text-[15px] rounded-lg gap-2.5 shadow-sm',
    }

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        {...props}
      >
        {isLoading && <Loader2 className="w-4 h-4 animate-spin shrink-0" />}
        {!isLoading && leftIcon}
        <span>{children}</span>
        {!isLoading && rightIcon}
      </button>
    )
  }
)

Button.displayName = 'Button'
