import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useRunState } from '@/hooks/useRunState';
import { useGraphAnimation } from '@/hooks/useGraphAnimation';
import { useSearchStream } from '@/hooks/useSearchStream';
import { PipelineStep } from './D3GraphEngine';
import { useParams } from 'next/navigation';
import { apiUrl } from '@/lib/api';

// D3GraphEngine touches SVG/DOM APIs — must be client-only
const D3GraphEngine = dynamic(
  () => import('./D3GraphEngine').then(mod => mod.D3GraphEngine),
  { ssr: false }
);

export const GraphPane: React.FC = () => {
  const params = useParams();
  const rawId = params?.runId;
  const runId = Array.isArray(rawId) ? rawId[0] : rawId || '';

  const { runState } = useRunState(runId);
  const { graph } = useGraphAnimation(runId);
  const { state: layer3State } = useSearchStream(runId);

  const topBridge = layer3State?.topBridges?.[0] || null;

  const [step, setStep] = useState<PipelineStep>('idle');
  const [insightVisible, setInsightVisible] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Map the backend RunState strictly to the D3 Animation Sequence
  useEffect(() => {
    if (!runState) return;

    if (runState.currentLayer >= 3) {
      setStep('crossdomain');
      setInsightVisible(true);
    } else if (runState.currentLayer === 2) {
      setStep('bottleneck');
      setInsightVisible(false);
    } else if (runState.layer1Status === 'completed') {
      setStep('simulated');
      setInsightVisible(false);
    } else {
      setStep('idle');
      setInsightVisible(false);
    }
  }, [runState]);

  // Robust network calls for orchestrating the AI pipeline
  const handleRunDiscovery = async () => {
    try {
      setError(null);
      const res = await fetch(apiUrl(`/api/runs/${runId}/layer1`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!res.ok) throw new Error(`Layer 1 API failed: ${res.statusText}`);
    } catch (err: any) {
      setError(err.message || "Network error triggering Layer 1 Causal Discovery");
    }
  };

  const handleIdentifyBottleneck = async () => {
    try {
      setError(null);
      const res = await fetch(apiUrl(`/api/runs/${runId}/layer2`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!res.ok) throw new Error(`Layer 2 API failed: ${res.statusText}`);
    } catch (err: any) {
      setError(err.message || "Network error triggering Layer 2 Bottleneck Resolution");
    }
  };

  const handleSearchCrossDomain = async () => {
    try {
      setError(null);
      const res = await fetch(apiUrl(`/api/runs/${runId}/layer3`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!res.ok) throw new Error(`Layer 3 API failed: ${res.statusText}`);
    } catch (err: any) {
      setError(err.message || "Network error triggering Layer 3 Isomorphism Search");
    }
  };

  return (
    // position:relative + overflow:hidden so absolute children stay inside
    <div className="relative w-full h-full bg-[#F5F5F5] tab-pane-bg overflow-hidden flex-1">
      {error && (
        <div className="absolute top-5 left-1/2 -translate-x-1/2 z-50 bg-red-500 text-white px-4 py-2 text-sm font-mono font-bold shadow-[4px_4px_0px_rgba(0,0,0,0.2)] border-2 border-black">
          SYSTEM ERROR: {error}
        </div>
      )}

      {/* Show a centered spinner while waiting for the very first state poll */}
      {!runState && (
        <div className="absolute inset-0 flex items-center justify-center z-40 bg-[#F5F5F5]">
          <div className="flex flex-col items-center gap-4">
            <div className="w-8 h-8 border-4 border-black border-t-transparent rounded-full animate-spin" />
            <span className="font-mono text-xs text-gray-400 uppercase tracking-widest">Connecting...</span>
          </div>
        </div>
      )}

      {/* Show a clear error overlay when Layer 1 fails */}
      {runState?.layer1Status === 'error' && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-40 bg-[#F5F5F5]/95">
          <div className="bg-white border-2 border-red-500 p-8 shadow-[8px_8px_0px_rgba(220,38,38,0.3)] w-[480px]">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse shrink-0" />
              <h3 className="font-mono font-bold text-lg uppercase tracking-tight text-red-600">
                Layer 1 Failed
              </h3>
            </div>
            <p className="font-mono text-xs text-gray-500 mb-5 leading-relaxed">
              The causal graph could not be built from the uploaded file. This usually means:
            </p>
            <ul className="font-mono text-xs text-gray-600 space-y-1 mb-6 pl-4 border-l-2 border-red-200">
              <li>• The file has no numeric columns (only text)</li>
              <li>• The file format is not supported (use CSV with numbers)</li>
              <li>• The file is corrupted or too large to process</li>
            </ul>
            <a
              href="/"
              className="block w-full text-center font-mono font-bold text-sm uppercase tracking-widest bg-black text-white py-3 border-2 border-black hover:bg-red-600 hover:border-red-600 transition-colors"
            >
              ← Upload a Different File
            </a>
          </div>
        </div>
      )}

      {/* Show progress overlay while Layer 1 is building the graph */}
      {(runState?.layer1Status === 'running' || runState?.layer1Status === 'idle') && runState?.layer1Progress !== undefined && runState.layer1Progress >= 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-40 bg-[#F5F5F5]/90 backdrop-blur-sm">
          <div className="bg-white border-2 border-black p-8 shadow-[8px_8px_0px_rgba(0,0,0,1)] w-[420px]">
            {/* Title */}
            <h3 className="font-mono font-bold text-lg mb-1 text-center uppercase tracking-tight">
              [ Causal Graph Building ]
            </h3>

            {/* Step label */}
            <div className="text-center font-mono text-[10px] text-gray-400 mb-5 uppercase tracking-[0.2em]">
              {runState.layer1Progress <= 10 ? 'Detecting File Type...' :
                runState.layer1Progress <= 20 ? 'Extracting Data...' :
                  runState.layer1Progress <= 40 ? 'Running PC Algorithm...' :
                    runState.layer1Progress <= 60 ? 'Validating With AI...' :
                      runState.layer1Progress <= 75 ? 'Fitting Equations...' :
                        runState.layer1Progress <= 85 ? 'Classifying Nodes...' :
                          runState.layer1Progress <= 95 ? 'Detecting Ambiguities...' :
                            'Finalizing Graph...'}
            </div>

            {/* Progress bar */}
            <div className="w-full bg-gray-100 h-5 border-2 border-black relative overflow-hidden mb-3">
              <div
                className="bg-black h-full transition-all duration-700 ease-out"
                style={{ width: `${runState.layer1Progress}%` }}
              />
            </div>

            {/* Percentage */}
            <div className="flex justify-between items-center font-mono text-xs">
              <span className="text-gray-400">Progress</span>
              <span className="font-bold text-black text-sm">{runState.layer1Progress}%</span>
            </div>
          </div>
        </div>
      )}

      <D3GraphEngine
        graph={graph}
        bridge={topBridge}
        step={step}
        insightVisible={insightVisible}
        onRunDiscovery={handleRunDiscovery}
        onIdentifyBottleneck={handleIdentifyBottleneck}
        onSearchCrossDomain={handleSearchCrossDomain}
      />
    </div>
  );
};
