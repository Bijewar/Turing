import React from 'react';
import { Badge } from '../ui/Badge';
import { CausalGraph } from '../../types/graph';

interface ConfidencePanelProps {
  graph: CausalGraph | null;
}

export const ConfidencePanel: React.FC<ConfidencePanelProps> = ({ graph }) => {
  if (!graph || !graph.edges || graph.edges.length === 0) {
    return (
      <div className="w-full h-full bg-white border-l-2 border-black p-6 flex flex-col justify-center items-center text-gray-400 font-mono text-sm font-bold tracking-widest">
        AWAITING CAUSAL GRAPH DATA...
      </div>
    );
  }

  // Calculate the overall confidence dynamically based on the backend data.
  // If the backend provides an overall confidence natively, use it. 
  // Otherwise, fallback to averaging the absolute edge weights (betas).
  const edgeCount = graph.edges.length;
  let overallConfidence = graph.overall_graph_confidence 
    ? Math.round(graph.overall_graph_confidence * 100) 
    : 0;
  
  if (!graph.overall_graph_confidence && edgeCount > 0) {
    const avgWeight = graph.edges.reduce((sum, edge) => sum + Math.abs(edge.weight || 0), 0) / edgeCount;
    overallConfidence = Math.round(avgWeight * 100);
  }

  return (
    <div className="w-full h-full bg-white border-l-2 border-black p-6 flex flex-col overflow-hidden">
      <div className="font-mono text-xs font-bold tracking-widest text-gray-500 mb-6 shrink-0">
        STEP 8: CONFIDENCE CHECK
      </div>
      
      {/* Scrollable container for dynamic edge iteration */}
      <div className="flex-1 flex flex-col gap-4 overflow-y-auto pr-2 custom-scrollbar">
        {graph.edges.map((edge, idx) => {
          const conf = Math.round(Math.abs(edge.weight || 0) * 100);
          
          let variant: 'success' | 'warning' | 'destructive' = 'success';
          let icon = '✅';
          
          if (conf < 50) {
            variant = 'destructive';
            icon = '❌';
          } else if (conf < 75) {
            variant = 'warning';
            icon = '⚠️';
          }

          return (
            <div key={`edge-${edge.source}-${edge.target}-${idx}`} className="flex items-center justify-between border-b border-gray-200 pb-2 shrink-0">
              <div className="font-mono text-sm truncate pr-4">{`${edge.source} → ${edge.target}`}</div>
              <Badge variant={variant} className="shrink-0">{`${conf}% ${icon}`}</Badge>
            </div>
          );
        })}
      </div>
      
      <div className="mt-6 border-t-2 border-black pt-4 shrink-0">
        <div className="font-mono text-xs font-bold mb-2">OVERALL CONFIDENCE</div>
        <div className="text-3xl font-bold font-mono">{overallConfidence}%</div>
        
        {/* Dynamic warning logic tied to real statistical confidence */}
        {overallConfidence < 75 && (
          <div className="mt-4 font-mono text-[10px] text-red-500 font-bold bg-red-50 p-2 border border-red-200 animate-pulse">
            WARNING: Low confidence detected. Layer 2 Adversarial Swarm required.
          </div>
        )}
      </div>
    </div>
  );
};
