import React from 'react'
import { cn } from '../../lib/utils'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  hint?: string
  error?: string
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, hint, error, id, ...props }, ref) => {
    const inputId = id || (label ? label.toLowerCase().replace(/\s+/g, '-') : undefined)

    return (
      <div className="space-y-1.5 w-full">
        {label && (
          <div className="flex items-center justify-between">
            <label
              htmlFor={inputId}
              className="text-[13px] font-medium text-[#1F1E1D] select-none"
            >
              {label}
            </label>
            {hint && <span className="text-[12px] text-[#9E9B93]">{hint}</span>}
          </div>
        )}
        <input
          id={inputId}
          ref={ref}
          className={cn(
            'w-full h-10 px-3 text-[14px] bg-[#FFFFFF] border border-[rgba(31,30,29,0.12)] rounded-lg text-[#1F1E1D] placeholder:text-[#9E9B93] focus:outline-none focus:ring-2 focus:ring-[#C2603F] focus:border-transparent transition-all disabled:opacity-50 disabled:bg-[#F3F1EA]',
            error && 'border-[#A63A38] focus:ring-[#A63A38]',
            className
          )}
          {...props}
        />
        {error && <p className="text-[12px] text-[#A63A38] font-medium">{error}</p>}
      </div>
    )
  }
)

Input.displayName = 'Input'
