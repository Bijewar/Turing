'use client';

import React from 'react';
import Link from 'next/link';

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="h-screen w-full flex flex-col items-center justify-center bg-[#F5F5F5] font-mono text-black px-6">
      <div className="max-w-md w-full bg-white border-2 border-black p-8 brutal-box shadow-[8px_8px_0px_#FF4500] text-center flex flex-col items-center gap-4">
        <div className="text-2xl">⚠️</div>
        <h2 className="text-xl font-bold tracking-tight uppercase">Dashboard Pipeline Exception</h2>
        <p className="text-xs text-red-600 bg-red-50 p-3 border border-red-200 text-left w-full break-words">
          {error.message || 'An unhandled exception occurred during pipeline rendering.'}
        </p>
        <div className="flex gap-4 mt-2">
          <button
            onClick={reset}
            className="px-5 py-2.5 bg-black text-white hover:bg-orange-500 hover:text-black transition-colors font-bold text-xs uppercase tracking-widest border-2 border-black"
          >
            Retry Pipeline
          </button>
          <Link
            href="/"
            className="px-5 py-2.5 bg-white text-black hover:bg-gray-100 transition-colors font-bold text-xs uppercase tracking-widest border-2 border-black"
          >
            Home
          </Link>
        </div>
      </div>
    </main>
  );
}
