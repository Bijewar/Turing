import React from 'react';

export interface ActionsSectionProps {
  actions?: string[];
  isStreaming?: boolean;
}

export const ActionsSection: React.FC<ActionsSectionProps> = ({
  actions = [],
  isStreaming = false,
}) => {
  if ((!actions || actions.length === 0) && !isStreaming) return null;

  const actionColors = ['#1A936F', '#004E89', '#C5283D'];
  const actionLabels = [
    'ACTION 1 — MOST IMPORTANT',
    'ACTION 2 — IF ACTION 1 CONFIRMS',
    'ACTION 3 — PARALLEL TRACK',
  ];

  return (
    <div className="mb-10 w-full brutal-box bg-white border-2 border-black p-8 shadow-[8px_8px_0px_rgba(0,0,0,0.1)]">
      <div className="flex justify-between items-center mb-6 border-b-2 border-black pb-4">
        <h2 className="font-mono text-xl font-bold tracking-widest text-black flex items-center gap-3">
          <span className="text-[#1A936F]">05 //</span> NEXT 3 ACTIONS
        </h2>
        {isStreaming && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#1A936F] animate-pulse" />
            <span className="font-mono text-xs font-bold text-[#1A936F] uppercase tracking-widest">
              Formulating...
            </span>
          </div>
        )}
      </div>

      {actions && actions.length > 0 ? (
        <div className="flex flex-col gap-4">
          {actions.slice(0, 3).map((action, idx) => (
            <div
              key={idx}
              className="border-l-4 pl-4 py-2"
              style={{ borderColor: actionColors[idx] ?? '#000' }}
            >
              <div
                className="font-mono text-[10px] font-bold uppercase tracking-widest mb-1"
                style={{ color: actionColors[idx] ?? '#000' }}
              >
                {actionLabels[idx] ?? `ACTION ${idx + 1}`}
              </div>
              <div className="font-sans text-gray-800 text-[15px] leading-relaxed">
                {action}
              </div>
            </div>
          ))}
        </div>
      ) : (
        isStreaming && (
          <div className="text-gray-400 font-mono text-xs italic tracking-widest mt-2">
            Awaiting action plan synthesis...
          </div>
        )
      )}
    </div>
  );
};
