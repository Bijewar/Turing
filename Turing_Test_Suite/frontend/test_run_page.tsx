import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const runPageSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'app/(dashboard)/run/[runId]/page.tsx'),
  'utf8',
);
const landingPageSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'app/page.tsx'),
  'utf8',
);

/**
 * test_run_page.tsx — Vitest tests for the run/[runId] page.
 *
 * Most tests are STATIC source checks because:
 *   1. The page imports next/dynamic, next/navigation, and many components
 *      that pull D3, hooks, and React 18 features; rendering the entire
 *      tree in jsdom is brittle.
 *   2. Several of these bugs (auto-advance override, error swallowing, hidden
 *      vs unmounted) are about the SHAPE of the source code — they're easier
 *      to verify via source inspection than via brittle render+interact.
 *
 * For tests where runtime rendering IS valuable (e.g. invalid runId redirect
 * behavior), we mock the hooks/params and render the page.
 */

describe('run/[runId] page', () => {
  /**
   * SWARM-2 BUG #54 — the auto-advance useEffect overrides user's manual tab
   * selection every 1.5s. The fix should track user override (e.g. via a ref)
   * and not auto-advance when the user has manually selected a tab.
   *
   * Static-source check: the source should track user override (e.g. via
   * `userOverrideRef`, `userOverrodeRef`, `lastManualTab`, or similar)
   * BEFORE calling setActiveTab in the effect. The effect's dependency
   * should also be `[runState?.currentLayer, ...]` (specific fields) not
   * `[runState]` (whole object — re-fires every poll).
   */
  test('test_auto_advance_does_not_override_user_selection', () => {
    // Buggy: `useEffect(() => { ... setActiveTab(...) }, [runState])` with no
    // user-override tracking. Fixed: uses a ref AND depends on currentLayer.
    const hasUserOverrideRef =
      /userOverrideRef|userOverrodeRef|userOverride|manualTabRef/.test(runPageSrc);
    expect(hasUserOverrideRef).toBe(true);

    // The effect's dependency should NOT be the bare `[runState]` — should
    // be a specific field (e.g. `[runState?.currentLayer]`).
    const hasBareRunStateDeps = /\}, \[runState\]\);/.test(runPageSrc);
    expect(hasBareRunStateDeps).toBe(false);
  });

  /**
   * SWARM-1 BUG #4 — invalid runId (non-empty but not a UUID) renders a blank
   * shell. The fix should detect invalid runIds and call `notFound()` or
   * render an error banner.
   *
   * Static-source check: the source should validate runId as a UUID (via
   * regex or a helper) and either call notFound() or render an error UI.
   */
  test('test_invalid_runid_shows_error', () => {
    // The buggy source only checks `if (!runId) router.replace('/')` — no
    // UUID validation. The fix should validate UUID format.
    // Look for either:
    //   - A UUID regex test: /^[0-9a-f-]{36}$/i.test(runId)
    //   - A notFound() call
    //   - A runId validation helper
    const hasUuidCheck = /uuid|UUID|notFound|isValidRunId|^[0-9a-f-]{36}/.test(runPageSrc);
    expect(hasUuidCheck).toBe(true);

    // And the redirect-on-empty must NOT be the only check.
    const onlyFalsyCheck = /if \(!runId\)\s*\{[^}]*router\.replace/.test(runPageSrc)
      && !/notFound/.test(runPageSrc)
      && !/isValidRunId/.test(runPageSrc)
      && !/uuid/i.test(runPageSrc);
    expect(onlyFalsyCheck).toBe(false);
  });

  /**
   * SWARM-1 BUG #5 — `error` from useRunState is destructured away and never
   * surfaced. The fix should destructure `error` and render an error banner.
   *
   * Static-source check: the source should destructure `error` from
   * useRunState AND render it in the JSX.
   */
  test('test_backend_error_surfaces_to_ui', () => {
    // Buggy: `const { runState } = useRunState(runId ?? "")`.
    // Fixed: `const { runState, error } = useRunState(runId ?? "")` (and
    // render the error in JSX).
    const destructuresError = /\{\s*runState[^}]*\berror\b[^}]*\}\s*=\s*useRunState/.test(
      runPageSrc,
    );
    expect(destructuresError).toBe(true);
  });

  /**
   * SWARM-2 BUG #55 — inactive tabs are mounted (via CSS `hidden` class),
   * not unmounted. This means every tab's hooks (useRunState, useSearchStream,
   * useReportStream, useAgentLoop) run simultaneously, causing duplicate
   * network traffic.
   *
   * Static-source check: the source should use conditional rendering
   * (e.g. `{activeTab === 'graph' && <GraphPane />}`) rather than always
   * rendered with CSS `hidden`.
   *
   * Look for the pattern: `{activeTab === "graph" ? <GraphPane /> : null}` or
   * `{activeTab === "graph" && <GraphPane />}`.
   * Buggy pattern: `<div className={... ? "flex" : "hidden"}><GraphPane /></div>`
   */
  test('test_inactive_tabs_unmounted_not_hidden', () => {
    // Buggy: `<div className={... ? "flex" : "hidden"}>` with <GraphPane /> always rendered.
    // Fixed: conditional render: `{activeTab === "graph" && <GraphPane />}`
    //        (i.e., the component is NOT rendered when its tab isn't active).
    //
    // Count the buggy "hidden" pattern in the source — should be 0 after fix.
    const hiddenCount = (runPageSrc.match(/["'`]hidden["'`]/g) || []).length;
    expect(hiddenCount).toBe(0);

    // And there should be conditional-render patterns like `{activeTab === 'X' && <X />}`.
    const conditionalCount = (runPageSrc.match(/activeTab\s*===\s*["']\w+["']\s*&&\s*</g) || []).length;
    expect(conditionalCount).toBeGreaterThan(0);
  });

  /**
   * SWARM-1 BUG #1 — the research hypothesis (textarea) is captured in state
   * but never sent to the backend. The `createRun` call in handleStart
   * doesn't include the prompt.
   *
   * Static-source check: the landing page's handleStart should pass `prompt`
   * (or similar) to createRun. AND `createRun` in api.ts should accept it.
   *
   * We check the landing page source for a createRun call that includes the
   * prompt variable.
   */
  test('test_user_prompt_passed_to_backend', () => {
    // Buggy: `await createRun(inputFile, inputType)` — no prompt arg.
    // Fixed: `await createRun(inputFile, inputType, prompt)` (or similar).
    // Look at the createRun call inside handleStart.
    const createRunIdx = landingPageSrc.indexOf('await createRun(');
    expect(createRunIdx).toBeGreaterThan(-1);
    const block = landingPageSrc.slice(createRunIdx, createRunIdx + 100);
    // The fixed version should include `prompt` as an argument.
    expect(block).toMatch(/prompt/);

    // Also check api.ts createRun signature accepts prompt.
    const apiSrc = fs.readFileSync(
      path.join(SRC_ROOT, 'lib/api.ts'),
      'utf8',
    );
    // Buggy: `createRun(inputFile, inputType)` — 2 args.
    // Fixed: `createRun(inputFile, inputType, prompt?: string)` — 3 args.
    const sigMatch = apiSrc.match(
      /export\s+async\s+function\s+createRun\s*\(([^)]*)\)/,
    );
    expect(sigMatch).toBeTruthy();
    const params = sigMatch![1];
    expect(params).toMatch(/prompt/);
  });
});