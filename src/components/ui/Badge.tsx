import React from 'react';
import { cn } from '@/lib/utils';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'destructive' | 'outline';
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  const variants = {
    default: 'bg-black text-white',
    success: 'bg-green-500 text-black',
    warning: 'bg-yellow-400 text-black',
    destructive: 'bg-red-500 text-white',
    outline: 'bg-transparent border border-black text-black',
  };

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-widest border border-black",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
