import React from 'react';

export interface ReportNavProps {
  modelName?: string;
  status?: string;
}

export const ReportNav: React.FC<ReportNavProps> = ({ 
  modelName = "CLAUDE_3.5_SONNET", 
  status = "AWAITING LAYER 4 SYNTHESIS..." 
}) => {
  return (
    <div className="font-mono text-xs text-gray-400 mb-10 border-b border-[#E5E5E5] pb-4 w-full uppercase tracking-widest font-bold flex justify-between items-center">
      <span>{modelName}{' // '}{status}</span>
      {status !== "REPORT COMPLETE" && (
        <span className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-ping" />
          STREAM ACTIVE
        </span>
      )}
    </div>
  );
};
