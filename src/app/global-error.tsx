'use client';

import React from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body className="h-screen w-full flex flex-col items-center justify-center bg-[#F5F5F5] font-mono text-black p-6">
        <div className="max-w-md w-full bg-white border-2 border-black p-8 shadow-[8px_8px_0px_rgba(0,0,0,1)] text-center flex flex-col items-center gap-4">
          <h2 className="text-xl font-bold tracking-tight uppercase text-red-600">Application Critical Error</h2>
          <p className="text-xs text-gray-700 bg-gray-100 p-3 border border-gray-300 text-left w-full break-words">
            {error.message || 'An unhandled application level error occurred.'}
          </p>
          <button
            onClick={reset}
            className="mt-2 px-6 py-3 bg-black text-white hover:bg-orange-500 hover:text-black transition-colors font-bold text-xs uppercase tracking-widest border-2 border-black"
          >
            Reload Application
          </button>
        </div>
      </body>
    </html>
  );
}
