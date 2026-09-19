"use client";

import React from 'react';
import { useParams } from 'next/navigation';
import { useSearchStream } from '@/hooks/useSearchStream';
import { SearchSource, BridgeResult } from '@/types/layer3';

// ─── DataStream sub-component ────────────────────────────────────────────────

interface DataStreamProps {
  isLocked: boolean;
  sources: SearchSource[];
}

const DataStream: React.FC<DataStreamProps> = ({ isLocked, sources }) => {
  // Distribute incoming sources into 3 columns for the marquee effect
  const colA = sources.filter((_, i) => i % 3 === 0);
  const colB = sources.filter((_, i) => i % 3 === 1);
  const colC = sources.filter((_, i) => i % 3 === 2);

  const generateCards = (colSources: SearchSource[]) => {
    return colSources.map((item, i) => (
      <div key={i} className="border border-black p-3 bg-white font-mono text-[10px] mb-4 shadow-sm w-full">
        <span className="text-gray-500">[{item.src}]</span> {item.title}<br/>
        MATCH: {item.match}% ➔ <span className="text-red-500 font-bold">DISCARD</span>
      </div>
    ));
  };

  const cardsA = generateCards(colA);
  const cardsB = generateCards(colB);
  const cardsC = generateCards(colC);

  return (
    <div
      className={`absolute inset-0 top-[100px] w-full flex gap-4 px-10 pointer-events-none transition-all duration-500 z-10 ${
        isLocked ? 'opacity-20 blur-[2px]' : 'opacity-30'
      }`}
    >
      <div className="flex-1 relative overflow-hidden">
        <div className="w-full flex flex-col animate-fast-scroll" style={{ animationPlayState: isLocked ? 'paused' : 'running' }}>
          {cardsA}
          {/* Duplicate for infinite marquee illusion */}
          {cardsA} 
        </div>
      </div>
      <div className="flex-1 relative overflow-hidden">
        <div className="w-full flex flex-col animate-fast-scroll" style={{ animationDelay: '-2s', animationDuration: '4s', animationPlayState: isLocked ? 'paused' : 'running' }}>
          {cardsB}
          {cardsB}
        </div>
      </div>
      <div className="flex-1 relative overflow-hidden">
        <div className="w-full flex flex-col animate-fast-scroll" style={{ animationDelay: '-4s', animationDuration: '6s', animationPlayState: isLocked ? 'paused' : 'running' }}>
          {cardsC}
          {cardsC}
        </div>
      </div>
    </div>
  );
};

// ─── RadarLockCard sub-component ─────────────────────────────────────────────

interface RadarLockCardProps {
  isLocked: boolean;
  bridge?: BridgeResult | null;
  index: number;
}

const RadarLockCard: React.FC<RadarLockCardProps> = ({ isLocked, bridge, index }) => {
  if (!bridge) return null;

  return (
    <div
      className={`z-30 brutal-box min-w-[600px] max-w-[600px] bg-white border-4 border-black shadow-[16px_16px_0px_#FF4500] flex flex-col shrink-0 ${
        isLocked
          ? 'opacity-100 pointer-events-auto animate-slide-in-lock'
          : 'opacity-0 pointer-events-none'
      }`}
      style={{ animationDelay: `${index * 150}ms`, animationFillMode: 'both' }}
    >
      <div className="bg-black text-white p-4 flex justify-between items-center">
        <div className="font-mono font-bold text-lg tracking-widest text-orange-500 flex items-center gap-3">
          <div className="w-3 h-3 bg-orange-500 rounded-full animate-pulse" />
          TARGET LOCKED
        </div>
        <div className="font-mono text-xs text-gray-400 uppercase">SOURCE: {bridge.evidenceTier}</div>
      </div>

      <div className="p-8 flex flex-col">
        <div className="w-16 h-16 bg-orange-100 border-2 border-orange-500 mb-6 flex items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FF4500" strokeWidth="2" strokeLinecap="square" strokeLinejoin="miter">
            <path d="M12 2L2 12l10 10 10-10L12 2z" />
            <path d="M12 22V2" />
            <path d="M2 12h20" />
          </svg>
        </div>

        <div className="text-xs text-gray-500 font-mono font-bold mb-2 tracking-widest uppercase">
          {bridge.targetDomain}
        </div>
        <div className="text-2xl font-bold font-sans leading-tight mb-4 text-black truncate">
          {bridge.title}
        </div>
        <p className="text-sm text-gray-600 leading-relaxed font-sans whitespace-pre-wrap line-clamp-3">
          &quot;{bridge.description}&quot;
        </p>

        <div className="border-t-2 border-black pt-6 mt-6">
          <div className="flex justify-between items-end mb-2">
            <div className="font-mono text-xs font-bold text-gray-500 tracking-widest">STRUCTURAL ISOMORPHISM SCORE</div>
            <div className="font-mono text-lg font-bold text-black">{(bridge.isomorphismScore * 100).toFixed(1)}% MATCH</div>
          </div>

          <div className="w-full h-4 bg-gray-200 border border-black overflow-hidden relative">
            <div
              className="absolute left-0 top-0 h-full bg-orange-500 transition-all duration-1000 ease-out"
              style={{ width: isLocked ? `${Math.min(100, Math.round(bridge.isomorphismScore * 100))}%` : '0%', transitionDelay: `${index * 150 + 500}ms` }}
            />
            <div className="absolute inset-0 matrix-bg opacity-20" />
          </div>

          <div className="flex justify-between mt-3 text-[10px] font-mono text-gray-400 font-bold uppercase">
            <div className="truncate pr-4">{bridge.sourceDomain} ➔ {bridge.targetDomain}</div>
            <div className="shrink-0">TOPOLOGICAL ALIGNMENT CONFIRMED</div>
          </div>
        </div>

        <button
          className="mt-8 w-full bg-black text-white hover:bg-orange-500 hover:text-black transition-colors px-8 py-4 font-mono font-bold tracking-widest text-sm shadow-[4px_4px_0px_rgba(0,0,0,0.2)] border-2 border-black flex items-center justify-center gap-2 group"
        >
          GENERATE SOLUTION REPORT
          <span className="transform group-hover:translate-x-1 transition-transform">➔</span>
        </button>
      </div>
    </div>
  );
};

// ─── SearchStatusPanel (main export) ─────────────────────────────────────────

export const SearchStatusPanel: React.FC = () => {
  const params = useParams();
  const rawId = params?.runId;
  const runId = Array.isArray(rawId) ? rawId[0] : rawId || '';
  const { state, error } = useSearchStream(runId);

  const isLocked = state?.isLocked || false;
  const sources = state?.streamSources || [];
  const query = state?.query || 'WAITING FOR LAYER 2 ABSTRACTION...';
  const topBridges = state?.topBridges || [];

  return (
    <div className="w-full h-[700px] max-w-[1400px] mx-auto pt-6 relative overflow-hidden flex flex-col items-center">
      {error && (
        <div className="absolute top-0 left-1/2 -translate-x-1/2 z-50 bg-red-500 text-white px-4 py-2 text-sm font-mono font-bold shadow-[4px_4px_0px_rgba(0,0,0,0.2)] border-2 border-black">
          SYSTEM ERROR: {error.message || "Failed to sync Layer 3 State"}
        </div>
      )}

      {/* Query Uplink Banner */}
      <div className="w-full bg-white border-b-4 border-black p-6 z-20 flex justify-between items-center shadow-[0px_8px_0px_rgba(0,0,0,0.1)] px-10">
        <div>
          <div className="font-mono text-xs text-gray-500 font-bold tracking-widest mb-1">
            DOMAIN-BLIND QUERY UPLINK
          </div>
          <div className="text-xl font-bold font-mono text-black">
            &quot;{query}&quot;
          </div>
        </div>
        <div className={`font-mono text-sm font-bold ${isLocked ? 'text-green-500' : 'text-orange-500 animate-pulse'}`}>
          {isLocked ? `${topBridges.length} TARGETS ACQUIRED` : 'SCANNING EXTERNAL DATABASES...'}
        </div>
      </div>

      {/* Background Data Stream */}
      <DataStream isLocked={isLocked} sources={sources} />

      {/* Target Match Cards */}
      <div className="z-30 absolute inset-0 top-[100px] w-full overflow-x-auto flex gap-10 px-10 pb-10 pt-16 items-start custom-scrollbar pointer-events-none" style={{ pointerEvents: isLocked ? 'auto' : 'none' }}>
        {topBridges.map((bridge, idx) => (
          <RadarLockCard key={idx} isLocked={isLocked} bridge={bridge} index={idx} />
        ))}
      </div>
    </div>
  );
};
