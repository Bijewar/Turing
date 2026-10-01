import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const topNavSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/shell/TopNav.tsx'),
  'utf8',
);
const layerProgressSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/shell/LayerProgress.tsx'),
  'utf8',
);

async function importTopNav() {
  return (await import('@/components/shell/TopNav')).TopNav;
}
async function importLayerProgress() {
  return (await import('@/components/shell/LayerProgress')).LayerProgress;
}

describe('TopNav', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-1 BUG #13 — active tab lacks `aria-current`. The fix should
   * add `aria-current="page"` to the active tab button.
   */
  test('test_has_aria_current_on_active_tab', async () => {
    const TopNav = await importTopNav();
    const { container } = render(
      <TopNav activeTab="graph" onTabChange={() => {}} />,
    );
    // Find the active tab button — its label is "[GRAPH]".
    // The active tab should have aria-current="page".
    const activeBtn = screen.getByText('[GRAPH]').closest('button');
    expect(activeBtn).toBeTruthy();
    expect(activeBtn!.getAttribute('aria-current')).toBeTruthy();
  });

  /**
   * SWARM-1 BUG #13 — the nav element should have an `aria-label`.
   */
  test('test_has_aria_label', async () => {
    const TopNav = await importTopNav();
    const { container } = render(<TopNav activeTab="graph" onTabChange={() => {}} />);
    const nav = container.querySelector('nav');
    expect(nav).toBeTruthy();
    expect(nav!.getAttribute('aria-label')).toBeTruthy();
  });
});

describe('LayerProgress', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-1 BUG #12 — the progress indicator has no ARIA role for progressbar.
   * The fix should add `role="progressbar"`.
   *
   * We mock useRunState to return a controlled state and assert the rendered
   * component has at least one element with `role="progressbar"`.
   */
  test('test_has_role_progressbar', async () => {
    vi.doMock('@/hooks/useRunState', () => ({
      useRunState: () => ({
        runState: {
          runId: 'test',
          currentLayer: 2,
          layer1Status: 'completed',
          layer2Status: 'processing',
          layer3Status: 'pending',
          layer4Status: 'pending',
          progressPercentage: 50,
        },
        error: null,
        isLoading: false,
      }),
    }));
    vi.doMock('next/navigation', () => ({
      useParams: () => ({ runId: 'test-run-id' }),
      useRouter: () => ({ replace: () => {}, push: () => {} }),
    }));

    const LayerProgress = await importLayerProgress();
    const { container } = render(<LayerProgress />);
    const pb = container.querySelector('[role="progressbar"]');
    expect(pb).toBeTruthy();
  });

  /**
   * SWARM-1 BUG #12 — `aria-valuenow`, `aria-valuemin`, `aria-valuemax`
   * must be set on the progressbar element.
   */
  test('test_has_aria_valuenow', async () => {
    vi.doMock('@/hooks/useRunState', () => ({
      useRunState: () => ({
        runState: {
          runId: 'test',
          currentLayer: 2,
          layer1Status: 'completed',
          layer2Status: 'processing',
          layer3Status: 'pending',
          layer4Status: 'pending',
          progressPercentage: 50,
        },
        error: null,
        isLoading: false,
      }),
    }));
    vi.doMock('next/navigation', () => ({
      useParams: () => ({ runId: 'test-run-id' }),
      useRouter: () => ({ replace: () => {}, push: () => {} }),
    }));

    const LayerProgress = await importLayerProgress();
    const { container } = render(<LayerProgress />);
    const pb = container.querySelector('[role="progressbar"]');
    expect(pb).toBeTruthy();
    expect(pb!.getAttribute('aria-valuenow')).toBeTruthy();
    expect(pb!.getAttribute('aria-valuemin')).toBeTruthy();
    expect(pb!.getAttribute('aria-valuemax')).toBeTruthy();
  });
});