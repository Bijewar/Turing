import React from 'react';

export interface ExperimentSectionProps {
  content?: string;
  isStreaming?: boolean;
}

export const ExperimentSection: React.FC<ExperimentSectionProps> = ({ 
  content = '', 
  isStreaming = false 
}) => {
  if (!content && !isStreaming) return null;

  return (
    <div className="mb-10 w-full brutal-box bg-white border-2 border-black p-8 shadow-[8px_8px_0px_rgba(0,0,0,0.1)]">
      <div className="flex justify-between items-center mb-6 border-b-2 border-black pb-4">
        <h2 className="font-mono text-xl font-bold tracking-widest text-black flex items-center gap-3">
          <span className="text-[#1A936F]">02 //</span> THE EXPERIMENT
        </h2>
        {isStreaming && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#1A936F] animate-pulse" />
            <span className="font-mono text-xs font-bold text-[#1A936F] uppercase tracking-widest">
              Designing Protocol...
            </span>
          </div>
        )}
      </div>

      <div className="font-sans text-gray-800 text-[15px] leading-relaxed whitespace-pre-wrap relative min-h-[100px]">
        {content}
        {isStreaming && (
          <span className="inline-block w-2 h-4 ml-1 bg-[#1A936F] animate-pulse align-middle" />
        )}
        
        {!content && isStreaming && (
          <div className="text-gray-400 font-mono text-xs italic tracking-widest mt-2">
            Awaiting experimental protocol synthesis...
          </div>
        )}
      </div>
    </div>
  );
};
