// Stub for next/dynamic so components that import it can load in tests.
import React from 'react';

export default function dynamic<T>(
  loader: () => Promise<{ default: React.ComponentType<T> } | React.ComponentType<T>>,
  _opts?: { ssr?: boolean; loading?: () => React.ReactNode },
) {
  // Eagerly resolve and render in tests — simpler and deterministic.
  const Lazy = React.lazy(async () => {
    const mod = await loader();
    // @ts-expect-error - normalize default vs direct component
    return mod.default ? mod : { default: mod };
  });
  return function Dynamic(props: T) {
    return (
      <React.Suspense fallback={null}>
        <Lazy {...props} />
      </React.Suspense>
    );
  } as unknown as React.ComponentType<T>;
}