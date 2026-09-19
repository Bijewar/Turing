// layer3/BridgeResultsPanel.tsx
// Official Layer 3 bridge results / abstraction panel
"use client";

import React from 'react';
import { useParams } from 'next/navigation';
import { useGraphAnimation } from '@/hooks/useGraphAnimation';
import { CausalNode, CausalEdge } from '@/types/graph';

// ─── SourceDomainColumn ────────────────────────────────────────────────────────
const SourceDomainColumn: React.FC<{ nodes: CausalNode[] }> = ({ nodes }) => {
  return (
    <div className="brutal-box flex-1 bg-white p-6 relative overflow-hidden flex flex-col h-full">
      <div className="absolute inset-0 matrix-bg opacity-30 pointer-events-none" />
      <div className="relative z-10 w-full flex flex-col h-full">
        <div className="font-mono text-xs font-bold mb-6 tracking-widest text-gray-500">
          PHASE 1 — SOURCE DOMAIN
        </div>
        <div className="flex-1 flex flex-col items-center justify-center w-full max-w-[280px] mx-auto overflow-y-auto custom-scrollbar">
          {nodes.length === 0 && (
            <div className="text-xs font-mono text-gray-400">AWAITING LAYER 1 DATA...</div>
          )}
          {nodes.slice(0, 5).map((node, i) => (
            <React.Fragment key={node.id}>
              <div className={`w-full border-2 border-black p-4 shadow-sm ${node.type === 'outcome' || node.type === 'bottleneck' ? 'bg-gray-50' : 'bg-white'}`}>
                <div className="text-[10px] text-gray-400 font-mono font-bold mb-1 uppercase">NODE: {node.type}</div>
                <div className={`text-xl font-bold font-sans animate-glitch ${node.type === 'bottleneck' ? 'text-red-600' : (node.type === 'outcome' ? 'text-orange-600' : 'text-black')}`}>
                  {node.label}
                </div>
                <div className="text-xs mt-2 text-gray-600 font-sans font-mono">Value: {(node.value || 0).toFixed(2)}</div>
              </div>
              {i < Math.min(nodes.length, 5) - 1 && (
                <div className="w-[2px] h-[40px] bg-black" />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─── SemanticStrippingColumn ───────────────────────────────────────────────────
const SemanticStrippingColumn: React.FC<{ nodes: CausalNode[] }> = ({ nodes }) => {
  return (
    <div
      className="brutal-box flex-[1.5] bg-gray-900 p-8 text-white relative overflow-hidden flex flex-col h-full border-gray-900"
      style={{
        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,69,0,0.1) 2px, rgba(255,69,0,0.1) 4px)'
      }}
    >
      <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
        <line x1="0" y1="100" x2="100%" y2="100" stroke="#FF4500" strokeWidth="1" strokeDasharray="4,4" opacity="0.2" />
        <line x1="0" y1="300" x2="100%" y2="300" stroke="#FF4500" strokeWidth="1" strokeDasharray="4,4" opacity="0.2" />
        <line x1="0" y1="500" x2="100%" y2="500" stroke="#FF4500" strokeWidth="1" strokeDasharray="4,4" opacity="0.2" />
      </svg>

      <div className="relative z-10 w-full h-full flex flex-col">
        <div className="font-mono text-xs font-bold mb-3 tracking-widest text-gray-400">
          PHASE 2 — SEMANTIC STRIPPING
        </div>

        <div className="border-b border-gray-700 pb-2 mb-8 flex justify-between items-end mt-4">
          <div className="font-mono text-xs font-bold text-gray-400">ISOLATING VARIABLES...</div>
          <div className="font-mono text-xs font-bold text-orange-500">100%</div>
        </div>

        <div className="flex-1 flex flex-col gap-6 justify-center overflow-y-auto custom-scrollbar pr-4">
          {nodes.length === 0 && (
            <div className="text-xs font-mono text-gray-500">AWAITING LAYER 1 DATA...</div>
          )}
          {nodes.map((node, idx) => (
            <div className="w-full" key={node.id}>
              <div className="font-mono text-sm leading-relaxed">
                <span className="text-gray-400 uppercase">[{node.type}]</span>{' '}
                <span className="animate-redact px-2 inline-block font-bold truncate max-w-[150px] align-bottom" style={{ animationDelay: `${idx * 0.5}s` }}>
                  {node.label}
                </span>{' '}
                <span className="text-orange-500 font-bold tracking-wider">➔ VAR_{node.id}</span>
              </div>
              <div className="text-sm text-gray-500 font-mono mt-1 border-l border-gray-700 pl-3 ml-2 uppercase">
                Properties: continuous, β={(node.beta || 0).toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─── StructuralFingerprintColumn ───────────────────────────────────────────────
const StructuralFingerprintColumn: React.FC<{ nodes: CausalNode[], edges: CausalEdge[] }> = ({ nodes, edges }) => {
  // Build dynamic adjacency matrix for the first 3 nodes to fit the UI cleanly
  const topNodes = nodes.slice(0, 3);
  
  return (
    <div className="brutal-box flex-1 bg-white p-6 relative flex flex-col h-full">
      <div className="absolute right-6 top-6 font-mono text-[10px] text-gray-400 text-right">
        GENERATED BY:<br/>CAUSAL_DISCOVERY_V4
      </div>

      <div className="font-mono text-xs font-bold mb-10 tracking-widest text-gray-500">
        PHASE 3 — STRUCTURAL FINGERPRINT
      </div>

      <div className="flex-1 flex flex-col justify-center">
        {topNodes.length > 0 ? (
          <>
            <div className="w-full mb-8">
              <div className="font-mono text-xs font-bold text-black mb-3">ADJACENCY MATRIX (W)</div>
              <div className="grid gap-1 font-mono text-[10px] font-bold text-center" style={{ gridTemplateColumns: `repeat(${topNodes.length + 1}, minmax(0, 1fr))` }}>
                <div className="py-2 bg-gray-100 border border-gray-300"></div>
                {topNodes.map(n => (
                  <div key={`col-${n.id}`} className="py-2 bg-black text-white truncate px-1">VAR_{n.id}</div>
                ))}
                
                {topNodes.map(rowNode => (
                  <React.Fragment key={`row-${rowNode.id}`}>
                    <div className="py-2 bg-black text-white truncate px-1">VAR_{rowNode.id}</div>
                    {topNodes.map(colNode => {
                      const edge = edges.find(e => e.source === rowNode.id && e.target === colNode.id);
                      const weight = edge ? Math.abs(edge.weight || 0).toFixed(2) : '0.00';
                      const isStrong = edge && Math.abs(edge.weight || 0) > 0.5;
                      return (
                        <div key={`cell-${rowNode.id}-${colNode.id}`} className={`py-2 border border-black ${rowNode.id === colNode.id ? 'bg-gray-50' : 'bg-white'} ${isStrong ? 'text-orange-500' : ''}`}>
                          {weight}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>

            <div className="w-full bg-orange-50 p-4 border-2 border-orange-500 shadow-brutal-medium mt-4">
              <div className="text-[11px] font-bold text-orange-600 font-mono mb-2">
                EXTRACTED CORE MECHANISM:
              </div>
              <div className="text-black font-bold font-mono text-sm leading-relaxed">
                &quot;A generic structural graph topology defined purely by its mathematical nodes and weighted adjacency matrix. All domain-specific semantics have been completely redacted for blind searching.&quot;
              </div>
            </div>
          </>
        ) : (
          <div className="text-xs font-mono text-gray-400 text-center">AWAITING LAYER 1 DATA...</div>
        )}
      </div>
    </div>
  );
};

// ─── BridgeResultsPanel ────────────────────────────────────────────────────────
export const BridgeResultsPanel: React.FC = () => {
  const params = useParams();
  const rawId = params?.runId;
  const runId = Array.isArray(rawId) ? rawId[0] : rawId || '';
  const { graph } = useGraphAnimation(runId);
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  return (
    <div className="w-full h-full flex flex-col items-center max-w-[1400px] mx-auto px-10 pt-6">
      <div className="w-full flex gap-0 flex-1" style={{ height: 'calc(100vh - 140px)' }}>
        <SourceDomainColumn nodes={nodes} />
        <SemanticStrippingColumn nodes={nodes} />
        <StructuralFingerprintColumn nodes={nodes} edges={edges} />
      </div>
    </div>
  );
};
