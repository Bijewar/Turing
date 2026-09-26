"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { FileUploader } from '@/components/layer1/FileUploader';
import { createRun, uploadDataset, listRuns, RunResponse } from '@/lib/api';
import { useSystemStatus } from '@/hooks/useSystemStatus';

type SubmitPhase = 'idle' | 'creating-run' | 'uploading' | 'done' | 'error';

const STATUS_CONFIG = {
  checking: { label: 'CHECKING', colour: 'text-yellow-500', dot: 'bg-yellow-400 animate-pulse' },
  online:   { label: 'ONLINE',   colour: 'text-green-500',  dot: 'bg-green-500' },
  offline:  { label: 'OFFLINE',  colour: 'text-red-500',    dot: 'bg-red-500' },
} as const;

export default function LandingPage() {
  const router = useRouter();
  const { health, version } = useSystemStatus();
  const [prompt, setPrompt] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<SubmitPhase>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [recentRuns, setRecentRuns] = useState<RunResponse[]>([]);
  const submittingRef = React.useRef(false);

  // Load recent runs for the sidebar with isMounted guard
  useEffect(() => {
    let isMounted = true;
    listRuns(10)
      .then((runs) => {
        if (isMounted) setRecentRuns(runs);
      })
      .catch(() => {/* backend may not be up yet — silently skip */});
    return () => {
      isMounted = false;
    };
  }, []);

  const isLocked = phase === 'creating-run' || phase === 'uploading' || phase === 'done';
  const isLoading = phase === 'creating-run' || phase === 'uploading';

  const phaseLabel: Record<SubmitPhase, string> = {
    idle: 'Launch Pipeline →',
    'creating-run': 'Initialising Run…',
    uploading: 'Uploading Dataset…',
    done: 'Redirecting…',
    error: 'Retry',
  };

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submittingRef.current) return;
    setErrorMsg(null);

    // Derive input_type from the selected file extension
    const ext = selectedFile?.name.split('.').pop()?.toLowerCase() ?? '';
    function deriveInputType(fileExt: string): string {
      if (fileExt === 'csv') return 'csv';
      if (fileExt === 'xlsx' || fileExt === 'xls') return 'excel';
      if (fileExt === 'json') return 'json';
      return 'text';
    }
    const inputType = deriveInputType(ext);
    const inputFile = selectedFile?.name ?? (prompt.trim() ? 'hypothesis.txt' : 'text-only');

    try {
      submittingRef.current = true;
      // Step 1: Create a run record in the DB with hypothesis prompt included
      setPhase('creating-run');
      const run = await createRun(inputFile, inputType, prompt.trim());

      // Step 2: Upload the dataset file (if one was provided)
      if (selectedFile) {
        setPhase('uploading');
        await uploadDataset(run.id, selectedFile);
      }

      // Step 3: Navigate to the live pipeline dashboard
      setPhase('done');
      router.push(`/run/${run.id}`);
    } catch (err) {
      submittingRef.current = false;
      setPhase('error');
      setErrorMsg(err instanceof Error ? err.message : 'An unexpected error occurred.');
    }
  };

  return (
    <main className="h-screen overflow-y-auto bg-[#F5F5F5] tab-pane-bg font-sans text-black selection:bg-orange-500 selection:text-white flex flex-col items-center py-20 px-10">

      {/* Header */}
      <div className="max-w-5xl w-full flex justify-between items-end border-b-4 border-black pb-6 mb-12 bg-white px-8 pt-8 brutal-box">
        <div>
          <h1 className="text-4xl font-bold tracking-tighter uppercase font-sans flex items-center gap-4">
            <div className="w-6 h-6 bg-orange-500 rounded-full animate-pulse" />
            Turing <span className="text-orange-500 ml-2">ENGINE</span>
          </h1>
          <p className="font-mono text-xs text-gray-500 mt-3 font-bold tracking-widest uppercase">
            Autonomous Causal Discovery &amp; Cross-Domain Abstraction
          </p>
        </div>
        <div className="font-mono text-[10px] font-bold text-gray-400 text-right flex flex-col items-end gap-1">
          <span className="flex items-center gap-1.5">
            SYSTEM STATUS:
            <span className={`flex items-center gap-1 ${STATUS_CONFIG[health].colour}`}>
              <span className={`inline-block w-1.5 h-1.5 rounded-full ${STATUS_CONFIG[health].dot}`} />
              {STATUS_CONFIG[health].label}
            </span>
          </span>
          <span>
            ENGINE:{' '}
            <span className="text-gray-500">
              {version ? `v${version}` : '—'}
            </span>
          </span>
        </div>
      </div>

      <div className="max-w-5xl w-full grid grid-cols-3 gap-10">

        {/* Main Interface */}
        <div className="col-span-2 flex flex-col gap-8">

          {/* Explanation */}
          <div className="brutal-box p-8 bg-white border-2 border-black shadow-[8px_8px_0px_rgba(0,0,0,1)]">
            <h2 className="font-mono font-bold text-sm tracking-widest mb-4 uppercase text-black">System Objective</h2>
            <p className="text-sm leading-relaxed text-gray-700 font-sans">
              TURING is a next-generation AI pipeline that ingests your raw datasets, autonomously maps the causal topology, mathematically strips away domain-specific semantics, and scans external domains (e.g., biology, supply chain, physics) to find isomorphic structural solutions to your bottlenecks. It also performs dynamic system simulations to validate the proposed interventions.
            </p>
          </div>

          {/* New Run Form */}
          <div className="brutal-box p-8 bg-white border-2 border-black shadow-[8px_8px_0px_#FF4500]">
            <h2 className="font-mono font-bold text-sm tracking-widest mb-6 uppercase flex items-center gap-3">
              Initialize Discovery Pipeline
            </h2>

            <form onSubmit={handleStart} className="flex flex-col gap-6">

              {/* File Upload */}
              <div>
                <label htmlFor="file-uploader" className="block font-mono text-xs font-bold text-gray-500 mb-2 uppercase">
                  Dataset / Text Upload (CSV/XLSX/TXT/JSON/PDF/MD)
                </label>
                <div className="w-full">
                  <FileUploader
                    onFileSelect={setSelectedFile}
                    selectedFileName={selectedFile?.name ?? null}
                  />
                </div>
              </div>

              {/* Text Prompt */}
              <div className="mt-4">
                <label htmlFor="research-prompt" className="block font-mono text-xs font-bold text-gray-500 mb-2 uppercase">
                  Research Hypothesis / Objective
                </label>
                <textarea
                  id="research-prompt"
                  className="w-full border-2 border-black p-4 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-orange-500 min-h-[120px] resize-y bg-gray-50 placeholder-gray-400"
                  placeholder="Describe the system bottleneck or optimization target..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  required
                />
              </div>

              {/* Error banner */}
              {errorMsg && (
                <div className="font-mono text-xs text-red-600 bg-red-50 border border-red-200 p-3">
                  ❌ {errorMsg}
                </div>
              )}

              {/* Progress indicator */}
              {isLoading && (
                <div className="font-mono text-xs text-orange-600 bg-orange-50 border border-orange-200 p-3 flex items-center gap-2">
                  <span className="inline-block w-3 h-3 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
                  {phaseLabel[phase]}
                </div>
              )}

              {/* Submit */}
              <button
                id="launch-pipeline-btn"
                type="submit"
                disabled={isLocked}
                className="mt-4 w-full bg-black text-white hover:bg-orange-500 hover:text-black transition-colors py-5 font-mono font-bold tracking-widest uppercase flex justify-center items-center gap-2 group border-2 border-black shadow-[4px_4px_0px_rgba(0,0,0,0.2)] disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {phaseLabel[phase]}
              </button>

            </form>
          </div>
        </div>

        {/* Sidebar: History */}
        <div className="col-span-1 flex flex-col gap-6">
          <div className="brutal-box p-6 bg-white border-2 border-black shadow-[4px_4px_0px_rgba(0,0,0,1)] h-full">
            <h2 className="font-mono font-bold text-xs tracking-widest mb-6 uppercase border-b-2 border-black pb-3">Research Logs</h2>

            <div className="flex flex-col gap-5">
              {recentRuns.length === 0 ? (
                <div className="text-left opacity-50 p-2 -ml-2">
                  <div className="font-mono text-[10px] text-gray-400 font-bold mb-1">WAITING FOR INGESTION</div>
                  <div className="font-bold font-sans text-sm text-black italic">No active runs</div>
                </div>
              ) : (
                recentRuns.map((run) => (
                  <button
                    key={run.id}
                    onClick={() => router.push(`/run/${run.id}`)}
                    className="text-left p-2 -ml-2 hover:bg-orange-50 transition-colors rounded group"
                  >
                    <div className="font-mono text-[10px] text-gray-400 font-bold mb-1 uppercase">
                      {run.status} · {new Date(run.created_at).toLocaleDateString()}
                    </div>
                    <div className="font-bold font-sans text-sm text-black group-hover:text-orange-600 truncate">
                      {run.input_file}
                    </div>
                    <div className="font-mono text-[9px] text-gray-400 mt-0.5 truncate">{run.id}</div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
