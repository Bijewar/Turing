import Link from 'next/link';

export default function NotFound() {
  return (
    <main className="h-screen w-full flex flex-col items-center justify-center bg-[#F5F5F5] font-mono text-black px-6">
      <div className="max-w-md w-full bg-white border-2 border-black p-8 brutal-box shadow-[8px_8px_0px_rgba(0,0,0,1)] text-center flex flex-col items-center gap-4">
        <div className="w-8 h-8 bg-orange-500 rounded-full animate-pulse" />
        <h1 className="text-4xl font-bold tracking-tight">404</h1>
        <p className="text-xs text-gray-600 font-bold uppercase tracking-wider">
          Node not found in causal discovery topology
        </p>
        <Link
          href="/"
          className="mt-4 px-6 py-3 bg-black text-white hover:bg-orange-500 hover:text-black transition-colors font-bold text-xs uppercase tracking-widest border-2 border-black"
        >
          ← Return to Engine Home
        </Link>
      </div>
    </main>
  );
}
