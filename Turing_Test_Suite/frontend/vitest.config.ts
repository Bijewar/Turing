import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createRequire } from 'node:module';

import fs from 'node:fs';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
let SRC = path.resolve(ROOT, '../../src');
if (!fs.existsSync(SRC)) {
  SRC = path.resolve(ROOT, '../../src');
}

// Resolve project source-file imports (d3, clsx, tailwind-merge, lucide-react, next/*)
// from THIS test package's node_modules, because the project at
// `` has no installed deps of its own.
const require = createRequire(import.meta.url);
const NM = path.resolve(ROOT, 'node_modules');
function nm(pkg: string) {
  // Resolve to the package's main entry directory so vite uses it directly.
  try {
    return path.dirname(require.resolve(`${pkg}/package.json`, { paths: [NM] }));
  } catch {
    return require.resolve(pkg, { paths: [NM] });
  }
}

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': SRC,
      'next/navigation': path.resolve(ROOT, 'stubs/next-navigation.ts'),
      'next/dynamic': path.resolve(ROOT, 'stubs/next-dynamic.ts'),
      'next/image': path.resolve(ROOT, 'stubs/next-image.ts'),
      'next/link': path.resolve(ROOT, 'stubs/next-link.ts'),
      // Map bare package imports the project source uses to our installed copies.
      d3: nm('d3'),
      clsx: nm('clsx'),
      'tailwind-merge': nm('tailwind-merge'),
      'lucide-react': nm('lucide-react'),
      // next/* falls through to our installed `next` package for any non-stubbed path.
      next: nm('next'),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./setup.ts'],
    include: ['test_*.tsx', 'test_*.ts', '*.test.ts'],
    css: false,
    coverage: { reporter: ['text'] },
  },
});