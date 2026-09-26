export default function DashboardLoading() {
  return (
    <main className="h-screen w-full flex flex-col items-center justify-center bg-[#F5F5F5] font-mono text-black">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-4 border-black border-t-orange-500 rounded-full animate-spin" />
        <span className="text-xs font-bold tracking-widest uppercase text-gray-600 animate-pulse">
          INITIALIZING PIPELINE TOPOLOGY…
        </span>
      </div>
    </main>
  );
}
