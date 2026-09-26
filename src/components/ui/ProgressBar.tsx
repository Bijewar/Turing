import React from 'react';
import { cn } from '@/lib/utils';

export interface ProgressBarProps {
  progress: number; // 0 to 100
  className?: string;
  colorClass?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ 
  progress, 
  className,
  colorClass = "bg-orange-500"
}) => {
  const safeProgress = Math.max(0, Math.min(100, progress));
  
  return (
    <div className={cn("w-full h-4 bg-gray-200 border-2 border-black overflow-hidden relative", className)}>
      <div
        className={cn("absolute left-0 top-0 h-full transition-all duration-300 ease-out", colorClass)}
        style={{ width: `${safeProgress}%` }}
      />
      {/* Optional matrix texture overlay */}
      <div className="absolute inset-0 matrix-bg opacity-10 pointer-events-none" />
    </div>
  );
};
