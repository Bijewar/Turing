import React from 'react';

export interface ReportSectionProps {
  stepNumber: string;
  title: string;
  content?: string;
  isStreaming?: boolean;
  colorHex?: string;
}

export const ReportSection: React.FC<ReportSectionProps> = ({ 
  stepNumber,
  title,
  content = '', 
  isStreaming = false,
  colorHex = '#000000'
}) => {
  if (!content && !isStreaming) return null;

  return (
    <div className="mb-10 w-full brutal-box bg-white border-2 border-black p-8 shadow-[8px_8px_0px_rgba(0,0,0,0.1)]">
      <div className="flex justify-between items-center mb-6 border-b-2 border-black pb-4">
        <h2 className="font-mono text-xl font-bold tracking-widest text-black flex items-center gap-3 uppercase">
          <span style={{ color: colorHex }}>{stepNumber}{' //'}</span> {title}
        </h2>
        {isStreaming && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: colorHex }} />
            <span 
              className="font-mono text-xs font-bold uppercase tracking-widest"
              style={{ color: colorHex }}
            >
              Synthesizing...
            </span>
          </div>
        )}
      </div>

      <div className="font-sans text-gray-800 text-[15px] leading-relaxed whitespace-pre-wrap relative min-h-[100px]">
        {content}
        {isStreaming && (
          <span 
            className="inline-block w-2 h-4 ml-1 animate-pulse align-middle" 
            style={{ backgroundColor: colorHex }} 
          />
        )}
        
        {!content && isStreaming && (
          <div className="text-gray-400 font-mono text-xs italic tracking-widest mt-2">
            Awaiting synthesis from neural backend...
          </div>
        )}
      </div>
    </div>
  );
};
