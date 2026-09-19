"use client";

import React from 'react';
import { useParams } from 'next/navigation';
import { useReportStream } from '@/hooks/useReportStream';
import { MechanismSection } from './MechanismSection';
import { ExperimentSection } from './ExperimentSection';
import { BridgesSection } from './BridgesSection';
import { WarningsSection } from './WarningsSection';
import { ActionsSection } from './ActionsSection';

export const StreamingReport: React.FC = () => {
  const params = useParams();
  const rawId = params?.runId;
  const runId = Array.isArray(rawId) ? rawId[0] : rawId || '';
  const { report, isConnected } = useReportStream(runId);
  const isLoading = report === null && isConnected;

  const isGlobalStreaming = report ? !report.isComplete : true;

  // Cascade streaming state: each section starts only after the previous has content
  const hasMech = !!report?.theMechanism?.length;
  const hasExp = !!report?.theExperiment?.length;
  const hasBridge = !!report?.whoSolvedThis?.length;
  const hasActions = (report?.next3Actions?.length ?? 0) > 0;

  const isMechStreaming = isGlobalStreaming && !hasExp;
  const isExpStreaming = isGlobalStreaming && hasMech && !hasBridge;
  const isBridgeStreaming = isGlobalStreaming && hasExp && !hasActions;
  const isActionsStreaming = isGlobalStreaming && hasBridge;

  return (
    <div className="w-full flex-1 flex flex-col items-center">
      <div className="w-full flex flex-col gap-2">

        {/* 01 // CAUSAL MECHANISM */}
        <MechanismSection
          content={report?.theMechanism || ''}
          isStreaming={isMechStreaming}
        />

        {/* 02 // THE EXPERIMENT */}
        <ExperimentSection
          content={report?.theExperiment || ''}
          isStreaming={isExpStreaming}
        />

        {/* 03 // WHO ALREADY SOLVED THIS */}
        <BridgesSection
          content={report?.whoSolvedThis || ''}
          isStreaming={isBridgeStreaming}
        />

        {/* 04 // WARNINGS & CONFLICTS */}
        <WarningsSection warnings={report?.warningsAndConflicts} />

        {/* 05 // NEXT 3 ACTIONS */}
        <ActionsSection
          actions={report?.next3Actions}
          isStreaming={isActionsStreaming}
        />

        {/* Confidence disclaimer + Export button */}
        {report?.isComplete && (
          <div className="mt-6 mb-2 w-full">
            {report.confidenceDisclaimer && (
              <p className="font-mono text-[10px] text-gray-400 text-center mb-6 tracking-wider px-4">
                {report.confidenceDisclaimer}
              </p>
            )}
            <button className="w-full border-2 border-black bg-black text-white font-bold tracking-widest font-mono px-6 py-4 text-sm hover:bg-white hover:text-black transition-colors">
              EXPORT PDF REPORT
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
