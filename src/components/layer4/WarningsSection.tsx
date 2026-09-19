import React from 'react';

export interface WarningsSectionProps {
  warnings?: string[];
}

export const WarningsSection: React.FC<WarningsSectionProps> = ({ warnings = [] }) => {
  // Only render if we have actual warnings from the LLM
  if (!warnings || warnings.length === 0) return null;

  return (
    <div className="mb-10 w-full brutal-box bg-[#FEF2F2] border-2 border-red-500 p-8 shadow-[8px_8px_0px_rgba(220,38,38,0.15)]">
      <div className="flex justify-between items-center mb-6 border-b-2 border-red-200 pb-4">
        <h2 className="font-mono text-xl font-bold tracking-widest text-red-600 flex items-center gap-3 uppercase">
          <span>04 //</span> WARNINGS & CONFLICTS
        </h2>
      </div>

      <div className="font-sans text-gray-800 text-[15px] leading-relaxed">
        <ul className="space-y-4">
          {warnings.map((warning, idx) => (
            <li key={idx} className="flex gap-4 items-start border-l-2 border-red-300 pl-3">
              <span className="font-mono text-red-500 font-bold">[{idx + 1}]</span>
              <span>{warning}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
