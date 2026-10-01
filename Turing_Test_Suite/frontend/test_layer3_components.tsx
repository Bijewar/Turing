import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const searchStatusSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer3/SearchStatusPanel.tsx'),
  'utf8',
);
const bridgeResultsSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer3/BridgeResultsPanel.tsx'),
  'utf8',
);

async function importSearchStatusPanel() {
  return (await import('@/components/layer3/SearchStatusPanel')).SearchStatusPanel;
}
async function importBridgeResultsPanel() {
  return (await import('@/components/layer3/BridgeResultsPanel')).BridgeResultsPanel;
}

describe('SearchStatusPanel', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #44 — `generateCards` uses `key={i}` (the array index) for
   * marquee cards. If two cards are rendered with the same source, React
   * may emit duplicate-key warnings OR — worse — duplicate cards because
   * the column distribution re-uses indices.
   *
   * Static-source check: the `generateCards` function must use a unique
   * stable key (e.g., `key={item.src + ':' + i}` or `key={item.title}`),
   * NOT just `key={i}`.
   */
  test('test_marquee_cards_have_unique_keys', () => {
    // Find the generateCards function block.
    const idx = searchStatusSrc.indexOf('generateCards');
    expect(idx).toBeGreaterThan(-1);
    const block = searchStatusSrc.slice(idx, idx + 500);
    // Buggy: `key={i}` — bare array index.
    // Fixed: `key={item.src + ...}` or `key={item.title + ...}` etc.
    const hasBareIndexKey = /key=\{i\}/.test(block);
    expect(hasBareIndexKey).toBe(false);
  });

  /**
   * SWARM-2 BUG #45 — the "GENERATE SOLUTION REPORT" button has no onClick
   * handler. The fix should wire it to a real handler.
   *
   * Static-source check: the button text "GENERATE SOLUTION REPORT" must
   * be followed by (or preceded by) an `onClick` attribute.
   */
  test('test_generate_solution_report_button_has_onclick', () => {
    // Find the button block in the source.
    const idx = searchStatusSrc.indexOf('GENERATE SOLUTION REPORT');
    expect(idx).toBeGreaterThan(-1);
    // The button JSX spans ~200 chars on either side.
    const block = searchStatusSrc.slice(Math.max(0, idx - 400), idx + 100);
    // The buggy code has no `onClick` attribute on this button.
    // The fix should add one.
    expect(block).toMatch(/onClick\s*=\s*[\{"]?/);
  });

  /**
   * SWARM-2 BUG #46 — `RadarLockCard` uses `key={idx}` (bare index).
   * The fix should use a stable id (e.g., `key={bridge.title}` or
   * `key={bridge.sourceDomain + ':' + bridge.targetDomain}`).
   */
  test('test_radar_lock_card_key_is_stable_id_not_index', () => {
    // Find the RadarLockCard mapping inside SearchStatusPanel.
    const idx = searchStatusSrc.indexOf('topBridges.map');
    expect(idx).toBeGreaterThan(-1);
    const block = searchStatusSrc.slice(idx, idx + 200);
    // Buggy: `key={idx}`. Fixed: `key={bridge.title}` or similar.
    const hasBareIndexKey = /key=\{idx\}/.test(block);
    expect(hasBareIndexKey).toBe(false);
  });

  /**
   * SWARM-2 BUG #47 — `isomorphismScore * 100` assumes 0-1 range. If the
   * backend sends a percentage (0-100), the display multiplies by 100 →
   * "7500.0% MATCH". The fix should detect values > 1 and use as-is.
   *
   * Static-source check: the source should not use `isomorphismScore * 100`
   * directly. It should clamp or detect range.
   *
   * We also do a runtime test by mocking useSearchStream to return a bridge
   * with score=75 and asserting the displayed value is "75.0% MATCH" (not
   * "7500.0% MATCH").
   */
  test('test_isomorphism_score_handles_percentage_input', async () => {
    vi.doMock('@/hooks/useSearchStream', () => ({
      useSearchStream: () => ({
        state: {
          status: 'LOCKED',
          query: 'test query',
          streamSources: [],
          topBridges: [
            {
              sourceDomain: 'Biology',
              targetDomain: 'Chemistry',
              isomorphismScore: 75, // percentage, not fraction
              description: 'test',
              title: 'Bridge 1',
              evidenceTier: 'A',
            },
          ],
          isLocked: true,
        },
        error: null,
        isConnected: true,
      }),
    }));
    vi.doMock('next/navigation', () => ({
      useParams: () => ({ runId: 'test-run-id' }),
      useRouter: () => ({ replace: () => {}, push: () => {} }),
      notFound: () => { throw new Error('NF'); },
    }));

    const SearchStatusPanel = await importSearchStatusPanel();
    render(<SearchStatusPanel />);
    // The buggy code displays (75 * 100).toFixed(1) = "7500.0% MATCH".
    // The fix should display "75.0% MATCH".
    expect(screen.queryByText(/7500\.0%/)).toBeNull();
    expect(screen.getByText(/75\.0%/)).toBeTruthy();
  });
});

describe('BridgeResultsPanel', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #48 — the SemanticStrippingColumn has scan lines at fixed
   * y positions (100, 300, 500) that don't scale with the container height.
   * On shorter containers, lines may overflow; on taller containers, lines
   * don't cover the full area.
   *
   * The fix should use percentages or compute from container height.
   *
   * Static-source check: the SVG line elements should use percentage
   * coordinates OR a viewBox that scales. The buggy code uses bare `y1="100"`
   * `y2="100"`.
   */
  test('test_svg_lines_scale_with_container', () => {
    // Find the SVG with the three scan lines.
    const svgIdx = bridgeResultsSrc.indexOf('SemanticStrippingColumn');
    expect(svgIdx).toBeGreaterThan(-1);
    // Look for the line elements.
    const lineRegex = /<line\s+x1=["']0["']\s+y1=["'](\d+)["']\s+x2=["']100%["']\s+y2=["']\d+["']/g;
    const matches: string[] = [];
    let m: RegExpExecArray | null;
    while ((m = lineRegex.exec(bridgeResultsSrc)) !== null) matches.push(m[1]);
    // The buggy code uses y1="100", y1="300", y1="500" (fixed pixel values).
    // The fix should use percentages or viewBox-based scaling.
    const hasFixedPixelY = matches.some(y => /^[1-9]\d+$/.test(y) && !y.endsWith('%'));
    expect(hasFixedPixelY).toBe(false);
  });
});