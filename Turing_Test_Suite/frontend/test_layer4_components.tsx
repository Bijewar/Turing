import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const streamingReportSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer4/StreamingReport.tsx'),
  'utf8',
);
const warningsSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer4/WarningsSection.tsx'),
  'utf8',
);
const actionsSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer4/ActionsSection.tsx'),
  'utf8',
);
const reportNavSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer4/ReportNav.tsx'),
  'utf8',
);

async function importStreamingReport() {
  return (await import('@/components/layer4/StreamingReport')).StreamingReport;
}
async function importWarningsSection() {
  return (await import('@/components/layer4/WarningsSection')).WarningsSection;
}
async function importActionsSection() {
  return (await import('@/components/layer4/ActionsSection')).ActionsSection;
}
async function importReportNav() {
  return (await import('@/components/layer4/ReportNav')).ReportNav;
}

describe('StreamingReport', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #50 — `isMechStreaming = isGlobalStreaming && !hasExp`.
   * This is wrong: `isMechStreaming` should be `isGlobalStreaming && !hasMech`
   * (i.e., mechanism is streaming UNTIL the mechanism content arrives, not
   * until the experiment arrives).
   *
   * We render with a report that has mechanism content but NO experiment
   * yet (a real intermediate state), and assert that `isMechStreaming` is
   * FALSE (the mechanism section should NOT be marked as streaming because
   * it already has content).
   *
   * To check this, we render the StreamingReport and inspect the
   * MechanismSection's `isStreaming` prop. The MechanismSection shows a
   * "Formulating..." indicator when `isStreaming` is true.
   *
   * Static-source check: the source should have `isMechStreaming = isGlobalStreaming && !hasMech`,
   * not `!hasExp`.
   */
  test('test_is_mech_streaming_correct', () => {
    // Buggy: `const isMechStreaming = isGlobalStreaming && !hasExp;`
    // Fixed: `const isMechStreaming = isGlobalStreaming && !hasMech;`
    expect(streamingReportSrc).not.toMatch(/isMechStreaming\s*=\s*isGlobalStreaming\s*&&\s*!hasExp/);
    // And should have the correct logic:
    expect(streamingReportSrc).toMatch(/isMechStreaming\s*=\s*isGlobalStreaming\s*&&\s*!hasMech/);
  });

  /**
   * SWARM-2 BUG #51 — the "EXPORT PDF REPORT" button has no onClick handler.
   *
   * Static-source check: the button text "EXPORT PDF REPORT" must be
   * preceded by (or followed by) an `onClick` attribute.
   */
  test('test_export_pdf_button_has_onclick', () => {
    const idx = streamingReportSrc.indexOf('EXPORT PDF REPORT');
    expect(idx).toBeGreaterThan(-1);
    // The button JSX spans ~200 chars on either side.
    const block = streamingReportSrc.slice(Math.max(0, idx - 400), idx + 100);
    // The buggy code has no `onClick` attribute on this button.
    expect(block).toMatch(/onClick\s*=\s*[\{"]?/);
  });
});

describe('WarningsSection', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #49 — `warnings.map((warning, idx) => <li key={idx}>...)`
   * uses the array index as the key. If warnings are reordered or inserted,
   * React can re-use the wrong DOM nodes.
   *
   * Static-source check: the source should NOT use `key={idx}` for warnings.
   */
  test('test_uses_stable_key_not_index', () => {
    // Buggy: `warnings.map((warning, idx) => <li key={idx}>...)`.
    // Fixed: `key={warning}` or `key={warning + ':' + idx}` or use a stable id.
    const mapIdx = warningsSrc.indexOf('warnings.map');
    expect(mapIdx).toBeGreaterThan(-1);
    const block = warningsSrc.slice(mapIdx, mapIdx + 300);
    const hasBareIndexKey = /key=\{idx\}/.test(block);
    expect(hasBareIndexKey).toBe(false);
  });
});

describe('ActionsSection', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #52 — same as WarningsSection but for actions.
   */
  test('test_uses_stable_key_not_index', () => {
    const mapIdx = actionsSrc.indexOf('actions.slice');
    expect(mapIdx).toBeGreaterThan(-1);
    const block = actionsSrc.slice(mapIdx, mapIdx + 300);
    const hasBareIndexKey = /key=\{idx\}/.test(block);
    expect(hasBareIndexKey).toBe(false);
  });
});

describe('ReportNav', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #53 — the model name is hardcoded to "CLAUDE_3.5_SONNET"
   * as the default. It should come from props/run state.
   *
   * We render with a different `modelName` prop and assert the rendered
   * text uses that prop, not the hardcoded default.
   */
  test('test_model_name_not_hardcoded', async () => {
    const ReportNav = await importReportNav();
    render(<ReportNav modelName="GPT-4" status="REPORT COMPLETE" />);
    // The rendered output should include "GPT-4", not "CLAUDE_3.5_SONNET".
    const text = document.body.textContent || '';
    expect(text).toContain('GPT-4');
    expect(text).not.toContain('CLAUDE_3.5_SONNET');
  });

  /** And: the default in the source should not be a hardcoded Anthropic model. */
  test('test_model_name_default_not_anthropic', () => {
    // The buggy source has `modelName = "CLAUDE_3.5_SONNET"` as the default.
    // The fix should either make it required (no default) or use a generic
    // fallback like "MODEL".
    expect(reportNavSrc).not.toMatch(/CLAUDE_3\.5_SONNET/i);
  });
});