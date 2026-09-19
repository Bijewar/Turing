"use client";

import React from 'react';

export type TabType = 'graph' | 'canvas' | 'abstraction' | 'search' | 'report';

interface TopNavProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

export const TopNav: React.FC<TopNavProps> = ({ activeTab, onTabChange }) => {
  const tabs: { id: TabType; label: string }[] = [
    { id: 'graph', label: '[GRAPH]' },
    { id: 'canvas', label: '[AGENT CANVAS]' },
    { id: 'abstraction', label: '[ABSTRACTION]' },
    { id: 'search', label: '[SEARCH]' },
    { id: 'report', label: '[REPORT]' },
  ];

  return (
    <nav aria-label="Primary" className="w-full h-20 flex items-center justify-between px-10 border-b border-[#E5E5E5] bg-white z-50 relative">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-orange-500 rounded-sm" />
        <div className="text-xl tracking-widest font-bold text-black font-mono">TURING_</div>
      </div>
      <div className="flex space-x-12 text-sm tracking-widest font-mono">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              id={`nav-tab-${tab.id}`}
              aria-current={isActive ? "page" : undefined}
              onClick={() => onTabChange(tab.id)}
              className={`
                relative py-2.5 transition-all duration-200
                ${isActive ? 'text-black font-bold' : 'text-gray-400 font-medium hover:text-black'}
              `}
            >
              {tab.label}
              {isActive && (
                <div className="absolute -bottom-[2px] left-0 w-full h-[3px] bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]" />
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
};
