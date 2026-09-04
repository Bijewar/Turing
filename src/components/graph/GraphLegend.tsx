import React from 'react';

export interface GraphLegendProps {
  types: string[];
  colorMap?: Record<string, string>;
}

const DEFAULT_COLOR_MAP: Record<string, string> = {
  controllable: '#FF6B35',
  mediator:     '#004E89',
  bottleneck:   '#C5283D',
  outcome:      '#1A936F',
  chemistry:    '#7B2D8E',
};

export const GraphLegend: React.FC<GraphLegendProps> = ({ 
  types, 
  colorMap = DEFAULT_COLOR_MAP 
}) => {
  if (!types || types.length === 0) return null;

  return (
    <div className="absolute bottom-5 left-5 bg-white p-3 px-4 border border-[#E5E5E5] shadow-[4px_4px_0px_rgba(0,0,0,0.05)] z-10">
      <span className="block text-[11px] font-bold text-[#E91E63] mb-2 uppercase tracking-[0.5px]">Entity Types</span>
      <div className="flex flex-wrap gap-x-4 gap-y-2 max-w-[320px]">
        {types.map((type) => {
          if (!type) return null;
          
          // If a type doesn't have a specific color mapped, fall back to neutral gray
          const color = colorMap[type] || '#999999';
          return (
            <div key={type} className="flex items-center gap-1.5 text-[12px] text-[#666] font-medium">
              <div className="w-2.5 h-2.5 rounded-full border border-black/10 shrink-0" style={{ background: color }} />
              {typeof type === 'string' ? type.charAt(0).toUpperCase() + type.slice(1) : String(type)}
            </div>
          );
        })}
      </div>
    </div>
  );
};
