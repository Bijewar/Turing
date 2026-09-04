import React from 'react';
import { BridgeResult } from '../../types/layer3';

interface CrossDomainBridgeProps {
  insightVisible: boolean;
  bridge?: BridgeResult | null;
}

export const CrossDomainBridge: React.FC<CrossDomainBridgeProps> = ({ 
  insightVisible,
  bridge
}) => {
  return (
    <div
      className="absolute top-5 w-[320px] bg-white border border-[#E5E5E5] p-4 shadow-[4px_4px_0px_rgba(0,0,0,0.05)] z-10 transition-all duration-[400ms] ease-[cubic-bezier(0.1,0.7,0.1,1)]"
      style={{
        right: insightVisible ? '20px' : '-400px',
      }}
    >
      <h2 className="text-[14px] text-[#7B2D8E] mb-3 font-bold border-b border-[#EEE] pb-2 uppercase">
        {bridge ? 'Cross-Domain Match' : 'Awaiting Isomorphism...'}
      </h2>
      <div className="text-[13px] leading-relaxed text-black">
        {bridge ? (
          <>
            <p className="mb-2 uppercase font-bold text-[11px] tracking-wider text-gray-500">
              {bridge.sourceDomain} → {bridge.targetDomain}
            </p>
            <p className="mb-3 text-black font-semibold">
              {bridge.title}
            </p>
            <div className="pl-2.5 border-l-2 border-[#7B2D8E]">
              <p className="text-gray-700 whitespace-pre-wrap">
                {bridge.description}
              </p>
              <div className="mt-2 text-[10px] font-mono font-bold bg-[#F5F5F5] inline-block px-1.5 py-0.5 border border-[#E0E0E0]">
                TIER: {bridge.evidenceTier} | SCORE: {(bridge.isomorphismScore > 1 ? bridge.isomorphismScore : bridge.isomorphismScore * 100).toFixed(1)}%
              </div>
            </div>
          </>
        ) : (
          <div className="text-gray-400 italic">
            Layer 3 engine searching vector space for structural isomorphisms...
          </div>
        )}
      </div>
    </div>
  );
};
