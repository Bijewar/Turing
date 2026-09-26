"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, useParams, notFound } from "next/navigation";
import { useRunState } from "@/hooks/useRunState";

// Official shell components
import { TopNav, TabType } from "@/components/shell/TopNav";
import { LayerProgress } from "@/components/shell/LayerProgress";

// Official graph components
import { GraphPane } from "@/components/graph/GraphPane";

// Official layer2 components — Agent Canvas
import { AgentStatusPanel } from "@/components/layer2/AgentStatusPanel";

// Official layer3 components
import { BridgeResultsPanel } from "@/components/layer3/BridgeResultsPanel";
import { SearchStatusPanel } from "@/components/layer3/SearchStatusPanel";

// Official layer4 components (report is composed from layer4 sub-components)
import { StreamingReport } from "@/components/layer4/StreamingReport";

export const isValidRunId = (id?: string) => Boolean(id && /^[0-9a-f-]{36}$/i.test(id));

export default function PipelineRunPage() {
  const router = useRouter();
  const params = useParams();

  // Resolve runId from dynamic segment.
  const rawId = params?.runId;
  const runId = Array.isArray(rawId) ? rawId[0] : rawId;

  const [activeTab, setActiveTab] = useState<TabType>("graph");
  const userOverrideRef = useRef(false);

  // Validate runId as UUID format; if invalid or missing trigger notFound
  useEffect(() => {
    if (!runId) {
      router.replace("/");
    } else if (!isValidRunId(runId)) {
      notFound();
    }
  }, [runId, router]);

  const { runState, error, isLoading } = useRunState(runId ?? "");

  const handleTabChange = (tab: TabType) => {
    userOverrideRef.current = true;
    setActiveTab(tab);
  };

  // Automatically advance tabs only if user has not manually clicked a tab
  useEffect(() => {
    if (!runState || userOverrideRef.current) return;

    switch (runState.currentLayer) {
      case 1:
        setActiveTab('graph');
        break;
      case 2:
        setActiveTab('canvas');
        break;
      case 3:
        if (runState.progressPercentage > 65) {
          setActiveTab('abstraction');
        } else {
          setActiveTab('search');
        }
        break;
      case 4:
        setActiveTab('report');
        break;
    }
  }, [runState?.currentLayer, runState?.progressPercentage]);

  if (!runId) {
    return null;
  }

  return (
    <main className="relative w-full h-screen flex flex-col overflow-hidden bg-white">
      {/* Global Shell Navigation */}
      <TopNav activeTab={activeTab} onTabChange={handleTabChange} />
      
      {/* Global Pipeline Progress Bar */}
      <div className="w-full border-b border-[#E5E5E5] px-10 py-3 bg-gray-50 flex items-center shadow-sm z-40 relative">
        <LayerProgress runState={runState} />
      </div>

      {/* Inline Error Notice */}
      {error && (
        <div className="w-full bg-red-50 border-b border-red-200 px-10 py-2 text-xs font-mono text-red-700 flex items-center gap-2 z-30">
          <span>⚠️ {error.message}</span>
        </div>
      )}

      {/* Loading Skeleton */}
      {isLoading && !runState && (
        <div className="flex-1 flex items-center justify-center font-mono text-sm text-gray-500 animate-pulse">
          FETCHING CAUSAL TOPOLOGY…
        </div>
      )}

      <div className="relative w-full flex-1 overflow-hidden bg-gray-50">

        {/* TAB 1: GRAPH */}
        {activeTab === "graph" && <GraphPane />}

        {/* TAB 2: AGENT CANVAS */}
        {activeTab === "canvas" && <AgentStatusPanel />}

        {/* TAB 3: ABSTRACTION — Layer 3 bridge results */}
        {activeTab === "abstraction" && <BridgeResultsPanel />}

        {/* TAB 4: SEARCH — Layer 3 domain-blind query uplink */}
        {activeTab === "search" && <SearchStatusPanel />}

        {/* TAB 5: REPORT — Layer 4 streaming report */}
        {activeTab === "report" && <StreamingReport />}

      </div>
    </main>
  );
}
