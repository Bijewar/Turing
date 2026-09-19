import React from 'react';

export interface BayesianOptimizerState {
  targetMaximizationPercentage: number;
  parameters: Record<string, string>;
  expectedImprovement: number;
  roundsComputed: number;
  maxRounds: number;
}

export interface BestFoundPanelProps {
  optimizerState?: BayesianOptimizerState | null;
}

export const BestFoundPanel: React.FC<BestFoundPanelProps> = ({ optimizerState }) => {
  // Graceful fallback for when the optimizer hasn't produced results yet
  if (!optimizerState) {
    return (
      <div className="brutal-box bg-white p-6 w-[350px] flex flex-col shrink-0 opacity-50">
        <div className="font-mono text-[10px] font-bold text-gray-400 tracking-widest mb-6">
          OPTIMIZATION // BEST FOUND
        </div>
        <div className="flex flex-col items-center justify-center h-full text-gray-400 font-mono text-xs tracking-widest text-center">
          AWAITING OPTIMIZER CONVERGENCE
        </div>
      </div>
    );
  }

  const {
    targetMaximizationPercentage,
    parameters,
    expectedImprovement,
    roundsComputed,
    maxRounds
  } = optimizerState;

  const progressPercent = maxRounds > 0 ? Math.min(100, Math.round((roundsComputed / maxRounds) * 100)) : 0;

  return (
    <div className="brutal-box bg-white p-6 w-[350px] flex flex-col shrink-0">
      <div className="font-mono text-[10px] font-bold text-gray-400 tracking-widest mb-6">
        OPTIMIZATION // BEST FOUND
      </div>
      
      <div className="flex flex-col gap-6 flex-1">
        <div>
          <div className="font-mono text-xs mb-1 uppercase">Target Maximization</div>
          <div className="text-4xl font-bold font-mono text-orange-600">
            {targetMaximizationPercentage.toFixed(1)}%
          </div>
        </div>
        
        <div className="border-t border-gray-200 pt-4">
          <div className="font-mono text-xs mb-2 text-gray-500 tracking-widest">PARAMETERS</div>
          {Object.entries(parameters).length > 0 ? (
            Object.entries(parameters).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center mb-1">
                <span className="font-mono text-sm">{key}</span>
                <span className="font-mono text-sm font-bold">{value}</span>
              </div>
            ))
          ) : (
            <div className="font-mono text-[10px] text-gray-400">NO PARAMETERS TUNED</div>
          )}
        </div>
        
        <div className="border-t border-gray-200 pt-4">
          <div className="font-mono text-[10px] text-gray-400 mb-1 tracking-widest">EXPECTED IMPROVEMENT (EI)</div>
          <div className="font-mono text-sm">{expectedImprovement.toFixed(5)}</div>
        </div>
      </div>
      
      <div className="mt-6 pt-4 border-t-2 border-black">
        <div className="font-mono text-[10px] font-bold flex justify-between tracking-widest">
          <span>ROUNDS COMPUTED</span>
          <span>{roundsComputed.toLocaleString()} / {maxRounds.toLocaleString()}</span>
        </div>
        <div className="w-full h-2 bg-gray-200 mt-2 relative">
          <div 
            className="h-full bg-black transition-all duration-300 ease-out" 
            style={{ width: `${progressPercent}%` }} 
          />
        </div>
      </div>
    </div>
  );
};
