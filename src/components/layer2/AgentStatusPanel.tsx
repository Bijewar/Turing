"use client";

import React, { useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import { ExperimentHistoryTable } from './ExperimentHistoryTable';
import { Heatmap } from './Heatmap';
import { useAgentLoop } from '@/hooks/useAgentLoop';
import { Layer2State } from '@/types/layer2';

// ─── Sticky header bar (the original AgentStatusPanel role) ─────────────────

const CanvasHeader: React.FC<{ state: Layer2State | null }> = ({ state }) => (
  <div className="sticky top-0 left-0 w-full bg-white p-4 px-10 z-50 flex justify-between items-center shadow-[0_2px_10px_rgba(0,0,0,0.05)] border-b border-black">
    <div className="flex items-center space-x-2">
      <span className="text-gray-500 font-bold font-mono tracking-widest text-sm">/ LAYER 2 : </span>
      <span className="text-[#FF6B35] font-bold font-mono tracking-widest text-sm">TURING</span>
      <span className="text-[#E5E5E5]">|</span>
      <span className="text-black font-bold font-mono tracking-widest text-sm">LIVE INFERENCE CANVAS</span>
    </div>
    <div className="font-mono text-[10px] text-gray-500 flex space-x-6 items-center tracking-widest">
      <div>
        NODES COMPUTED: <span className="font-bold text-black text-[11px]">{state?.nodesComputed || 0}</span>
      </div>
      <div className="flex items-center space-x-2">
        <span>STATUS:</span>
        <span className={`font-bold ${state?.status === 'RUNNING' ? 'text-[#C5283D] animate-pulse' : 'text-black'}`}>
          {state?.status || 'AWAITING DISPATCH'}
        </span>
      </div>
    </div>
  </div>
);

// ─── AgentStatusPanel — main exported canvas orchestrator ────────────────────

export const AgentStatusPanel: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Connect to global backend orchestrator
  const params = useParams();
  const rawId = params?.runId;
  const runId = Array.isArray(rawId) ? rawId[0] : rawId || '';
  const { state, error } = useAgentLoop(runId);

  // Auto-scroll logic: gracefully scroll to bottom matching the CSS animation timeline
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !state?.agents?.length) return;

    // Find the total duration of the CSS animation sequence
    const nodes = state.heatmapNodes || [];
    const maxDelay = Math.max(...nodes.map(n => n.delayMs || 0), 0);
    if (maxDelay === 0) return;

    const targetScroll = container.scrollHeight - container.clientHeight;
    
    let startTime: number | null = null;
    let animationFrame: number;
    let userInterrupted = false;
    
    // Allow user to interrupt the animation by scrolling
    const handleUserInterrupt = () => {
      userInterrupted = true;
    };
    
    container.addEventListener('wheel', handleUserInterrupt, { passive: true });
    container.addEventListener('touchstart', handleUserInterrupt, { passive: true });
    container.addEventListener('mousedown', handleUserInterrupt, { passive: true });
    
    const animateScroll = (timestamp: number) => {
      if (userInterrupted || !container) return;
      
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      
      // Calculate scroll progress. Extended by 5000ms to ensure it keeps moving 
      // while the final agents are 'talking' (fading in and rendering).
      const duration = maxDelay + 5000;
      const percentage = Math.min(elapsed / duration, 1);
      
      // Linear scroll down
      container.scrollTop = percentage * targetScroll;
      
      if (percentage < 1) {
        animationFrame = requestAnimationFrame(animateScroll);
      }
    };
    
    // Start animation if we are at the top (fresh stream)
    if (container.scrollTop < 100) {
        animationFrame = requestAnimationFrame(animateScroll);
    }
    
    return () => {
      if (animationFrame) cancelAnimationFrame(animationFrame);
      container.removeEventListener('wheel', handleUserInterrupt);
      container.removeEventListener('touchstart', handleUserInterrupt);
      container.removeEventListener('mousedown', handleUserInterrupt);
    };
  }, [state?.agents?.length, state?.heatmapNodes]);

  const agents = state?.agents || [];
  const nodes = state?.heatmapNodes || [];
  const lines = state?.heatmapLines || [];

  // Calculate required canvas height dynamically based on the deepest graph node
  // to ensure absolute positioned elements don't overflow the container.
  const maxNodeY = Math.max(...nodes.map(n => n.y), ...agents.map(a => a.y), 0);
  const dynamicHeight = Math.max(maxNodeY + 500, 1000); 

  return (
    <div ref={containerRef} className="w-full h-full overflow-y-auto bg-[#FAFAFA] relative">
      <div 
        className="w-full matrix-bg relative transition-all duration-500 ease-out" 
        style={{ height: `${dynamicHeight}px` }}
      >
        <CanvasHeader state={state} />
        
        {error && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-50 bg-red-500 text-white px-4 py-2 text-sm font-mono font-bold shadow-[4px_4px_0px_rgba(0,0,0,0.2)] border-2 border-black">
            SYSTEM ERROR: {error.message || "Failed to sync Layer 2 State via WebSocket"}
          </div>
        )}

        {agents.length === 0 && !error ? (
          <div className="w-full flex items-center justify-center pt-32 text-gray-400 font-mono text-sm tracking-widest font-bold">
            AWAITING LAYER 2 ADVERSARIAL SWARM DATA...
          </div>
        ) : (
          <>
            <ExperimentHistoryTable entries={agents} />
            <Heatmap nodes={nodes} lines={lines} />
          </>
        )}
      </div>
    </div>
  );
};
