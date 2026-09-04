import React from 'react';
import { PipelineStep } from './D3GraphEngine';

export interface GraphControlsProps {
  step: PipelineStep;
  onRunDiscovery?: () => void;
  onIdentifyBottleneck?: () => void;
  onSearchCrossDomain?: () => void;
}

export const GraphControls: React.FC<GraphControlsProps> = ({
  step,
  onRunDiscovery,
  onIdentifyBottleneck,
  onSearchCrossDomain
}) => {
  return (
    <aside className="absolute z-10 bg-white border border-[#E5E5E5] p-4 shadow-[4px_4px_0px_rgba(0,0,0,0.05)] text-[13px] top-5 left-5 w-[250px]">
      <h3 className="font-mono font-bold text-xs text-gray-500 mb-3 uppercase tracking-widest border-b border-gray-100 pb-2">Pipeline Status</h3>
      
      <div className={`w-full border font-sans text-[13px] font-semibold py-2 px-3 mb-2 text-center transition-all duration-200 ${
        step === 'idle'
          ? 'border-[#E0E0E0] bg-white text-[#666]'
          : 'border-[#1A936F] text-[#1A936F] bg-[#E8F8F5] completed'
      }`}>
        1. Causal Discovery
      </div>
      
      <div className={`w-full border font-sans text-[13px] font-semibold py-2 px-3 mb-2 text-center transition-all duration-200 ${
        step === 'simulated' || step === 'idle'
          ? 'border-[#E0E0E0] bg-[#FAFAFA] text-[#999]'
          : 'border-[#C5283D] text-[#C5283D] bg-[#FFF0F5]'
      }`}>
        2. Identify Bottleneck
      </div>
      
      <div className={`w-full border font-sans text-[13px] font-semibold py-2 px-3 text-center transition-all duration-200 ${
        step === 'crossdomain'
          ? 'border-[#7B2D8E] text-[#7B2D8E] bg-[#F5EEF8]'
          : 'border-[#E0E0E0] bg-[#FAFAFA] text-[#999]'
      }`}>
        3. Search Cross-Domain
      </div>
    </aside>
  );
};
