import React from 'react';
import { HypothesisNode, ConnectorLine } from '@/types/layer2';

// ─── HypothesisNode Component ───────────────────────────────────────────────

const HypothesisNodeView: React.FC<HypothesisNode> = ({
  id, variant, label, title, description, x, y, mechanism, delayMs = 0
}) => {
  const baseClasses = "absolute opacity-0 animate-fade-node z-20";
  const style = { 
    left: `${x}px`, 
    top: `${y}px`,
    animationDelay: `${delayMs}ms`
  };

  if (variant === 'synthesized') {
    return (
      <div id={id} className={`${baseClasses} w-[600px] bg-white border-2 border-orange-500 p-8 shadow-[8px_8px_0px_rgba(255,69,0,0.1)]`} style={style}>
        <div className="absolute -top-3 -left-3 bg-orange-500 text-white font-mono text-[9px] px-2 py-1 font-bold">{label}</div>
        <div className="font-bold text-xl mb-4 text-black">{title}</div>
        <div className="font-mono text-sm text-gray-800 leading-relaxed bg-[#FFF8F5] p-3 border border-orange-200">
          <span className="font-bold text-orange-600">MECHANISM: </span>{mechanism || description}
        </div>
        <div className="mt-4 flex justify-end">
          <div className="font-mono text-[10px] font-bold text-black border border-black px-3 py-1 bg-[#FAFAFA]">➔ READY FOR CROSS-DOMAIN SEARCH</div>
        </div>
      </div>
    );
  }

  if (variant === 'falsified') {
    return (
      <div id={id} className={`${baseClasses} w-[500px] bg-[#FEF2F2] border border-red-200 p-6 shadow-[4px_4px_0px_rgba(220,38,38,0.1)]`} style={style}>
        <div className="absolute -top-3 -left-3 bg-red-600 text-white font-mono text-[9px] px-2 py-1 font-bold animate-redact">REDACT</div>
        <div className="font-bold text-lg text-red-700 mb-2 line-through">{title}</div>
        <div className="font-mono text-xs text-red-600 font-bold">{description}</div>
      </div>
    );
  }

  if (variant === 'confirmed') {
    return (
      <div id={id} className={`${baseClasses} w-[500px] bg-[#F0FDF4] border border-green-500 p-6 shadow-[4px_4px_0px_#22C55E]`} style={style}>
        <div className="absolute -top-3 -left-3 bg-green-600 text-white font-mono text-[9px] px-2 py-1 font-bold">{label}</div>
        <div className="font-bold text-lg mb-2 text-green-900">{title}</div>
        <div className="font-mono text-xs text-green-800 font-bold">{description}</div>
      </div>
    );
  }

  if (variant === 'exploiter') {
    return (
      <div id={id} className={`${baseClasses} w-[500px] bg-white border border-orange-500 p-6 shadow-sm`} style={style}>
        <div className="absolute -top-3 -left-3 bg-orange-500 text-white font-mono text-[9px] px-2 py-1 font-bold">{label}</div>
        <div className="font-bold text-lg mb-2 text-black">{title}</div>
        <div className="font-mono text-xs text-gray-500">{description}</div>
        {mechanism && (
          <div className="mt-3 font-mono text-xs text-orange-600 bg-orange-50 p-2 border border-orange-200">
            <span className="font-bold">MECHANISM: </span>{mechanism}
          </div>
        )}
      </div>
    );
  }

  return (
    <div id={id} className={`${baseClasses} w-[500px] bg-white border border-gray-300 p-6 shadow-sm`} style={style}>
      <div className="absolute -top-3 -left-3 bg-black text-white font-mono text-[9px] px-2 py-1 font-bold">{label}</div>
      <div className="font-bold text-lg mb-2">{title}</div>
      <div className="font-mono text-xs text-gray-500">{description}</div>
    </div>
  );
};

// ─── ConnectorLines Component ───────────────────────────────────────────────

const ConnectorLinesView: React.FC<{ lines: ConnectorLine[] }> = ({ lines }) => {
  return (
    <svg className="absolute inset-0 w-full h-[100%] min-h-screen pointer-events-none z-10" xmlns="http://www.w3.org/2000/svg">
      {lines.map((line) => {
        let strokeColor = "#000000";
        let strokeWidth = 1;
        let isDashed = false;
        if (line.variant === 'contrarian') strokeColor = "#DC2626";
        else if (line.variant === 'exploiter') { strokeColor = "#FF4500"; strokeWidth = 2; }
        else if (line.variant === 'tree') isDashed = true;

        const pathId = `line-${line.id}`;
        const d = line.variant === 'tree' 
          ? `M ${line.x1} ${line.y1} L ${line.x2} ${line.y2}`
          : `M ${line.x1} ${line.y1} L ${(line.x1 + line.x2) / 2} ${line.y1} L ${(line.x1 + line.x2) / 2} ${line.y2} L ${line.x2} ${line.y2}`;

        return (
          <path
            key={pathId} 
            d={d} 
            stroke={strokeColor} 
            strokeWidth={strokeWidth} 
            fill="none"
            strokeDasharray={isDashed ? "4,4" : "1000"} 
            strokeDashoffset={isDashed ? "0" : "1000"}
            className={isDashed ? "animate-fade-node" : "animate-draw-line"}
            style={{ animationDelay: `${line.delayMs || 0}ms` }}
          />
        );
      })}
    </svg>
  );
};

// ─── Main Heatmap Wrapper ───────────────────────────────────────────────────

export const Heatmap: React.FC<{ nodes: HypothesisNode[], lines: ConnectorLine[] }> = ({ nodes, lines }) => {
  return (
    <>
      <ConnectorLinesView lines={lines} />
      {nodes.map(node => <HypothesisNodeView key={node.id} {...node} />)}
    </>
  );
};
