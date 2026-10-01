import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const fileUploaderSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer1/FileUploader.tsx'),
  'utf8',
);
const confidencePanelSrc = fs.readFileSync(
  path.join(SRC_ROOT, 'components/layer1/ConfidencePanel.tsx'),
  'utf8',
);

async function importFileUploader() {
  return (await import('@/components/layer1/FileUploader')).FileUploader;
}
async function importConfidencePanel() {
  return (await import('@/components/layer1/ConfidencePanel')).ConfidencePanel;
}

describe('FileUploader', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #32 — the label says "DRAG & DROP OR CLICK TO BROWSE" but the
   * component has NO onDrop/onDragOver handlers, so dragging a file onto the
   * area does nothing (or opens the file in a new tab).
   *
   * The fix should add onDrop/onDragOver handlers so dragging a file onto
   * the area selects it (calls onChange / onFileSelect).
   */
  test('test_drag_and_drop_works', async () => {
    const FileUploader = await importFileUploader();
    const onFileSelect = vi.fn();
    const { container } = render(<FileUploader onFileSelect={onFileSelect} />);

    // Find the wrapper div (the dashed border area).
    // The label text says "DRAG & DROP OR CLICK TO BROWSE".
    const label = screen.getByText(/DRAG & DROP/i);
    // The drop target should be the wrapper around the label.
    const dropZone = label.closest('div')!;
    expect(dropZone).toBeTruthy();

    // Build a File object to drop.
    const file = new File(['col1,col2\n1,2'], 'data.csv', { type: 'text/csv' });

    // Dispatch a 'drop' event with the file in dataTransfer.
    // jsdom doesn't natively support DataTransfer, so we attach a stub.
    const dt = {
      files: [file],
      items: [],
      types: ['Files'],
      getData: () => '',
      clearData: () => {},
    };
    const dropEvent = new Event('drop', { bubbles: true, cancelable: true });
    // @ts-expect-error - assign DataTransfer stub.
    dropEvent.dataTransfer = dt;
    fireEvent(dropZone, dropEvent);

    // The fix should call onFileSelect with the dropped file.
    expect(onFileSelect).toHaveBeenCalled();
    const arg = onFileSelect.mock.calls[0]?.[0];
    // The arg should be the File (or null on error). For a valid csv, it
    // should be the File object.
    expect(arg).toBeTruthy();
    // Note: in the buggy code onFileSelect is NEVER called for drops.
  });

  /**
   * SWARM-2 BUG #32 (typo) — the label should say "BROWSE", not "BROWSER".
   */
  test('test_typo_in_label_browse_not_browser', async () => {
    const FileUploader = await importFileUploader();
    render(<FileUploader onFileSelect={() => {}} />);
    // The buggy code has "BROWSER" (a typo). The fix should say "BROWSE".
    expect(screen.queryByText(/BROWSER/i)).toBeNull();
    expect(screen.getByText(/BROWSE/i)).toBeTruthy();
  });

  /**
   * SWARM-2 BUG #33 — the file input must have an `aria-label` so screen
   * readers announce its purpose.
   */
  test('test_input_has_aria_label', async () => {
    const FileUploader = await importFileUploader();
    render(<FileUploader onFileSelect={() => {}} />);
    const input = document.querySelector('input[type="file"]');
    expect(input).toBeTruthy();
    const ariaLabel = input!.getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
    expect(ariaLabel!.length).toBeGreaterThan(0);
  });
});

describe('ConfidencePanel', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #34 — when `overall_graph_confidence === 0`, the buggy code
   * treats it as missing (falsy check) and computes the average instead.
   * The fix should display 0% when the backend explicitly sends 0.
   *
   * We render with a graph that has `overall_graph_confidence: 0` and assert
   * the displayed "OVERALL CONFIDENCE" value is 0% (not the computed average).
   */
  test('test_overall_confidence_zero_not_treated_as_missing', async () => {
    const ConfidencePanel = await importConfidencePanel();
    const graph = {
      nodes: [
        { id: 'A', label: 'A', type: 'controllable' as const, value: 0, beta: 0 },
        { id: 'B', label: 'B', type: 'outcome' as const, value: 0, beta: 0 },
      ],
      edges: [
        { source: 'A', target: 'B', curvature: 0.2, weight: 0.85, relation: 'increases' },
        { source: 'B', target: 'A', curvature: 0.2, weight: 0.4, relation: 'increases' },
      ],
      overall_graph_confidence: 0,
    };
    const { container } = render(<ConfidencePanel graph={graph} />);
    // Find the "OVERALL CONFIDENCE" value.
    const overallLabel = screen.getByText(/OVERALL CONFIDENCE/i);
    // The percentage is the next text node sibling, look for the div containing it.
    const wrapper = overallLabel.closest('div')?.parentElement;
    expect(wrapper).toBeTruthy();
    // Look for the text content of the percentage (e.g., "0%" or "62%").
    const textContent = wrapper!.textContent || '';
    // The buggy code displays the computed average (e.g., (0.85+0.4)/2 * 100 = 62%).
    // The fix displays "0%".
    expect(textContent).toMatch(/(^|\s)0%/);
    // And NOT the computed average (62%).
    expect(textContent).not.toMatch(/62%/);
  });

  /**
   * SWARM-2 BUG #35 — edge weight > 1 (e.g., 2.5) is displayed as 250%.
   * The fix should clamp to 100%.
   */
  test('test_edge_confidence_clamped_to_100', async () => {
    const ConfidencePanel = await importConfidencePanel();
    const graph = {
      nodes: [
        { id: 'A', label: 'A', type: 'controllable' as const, value: 0, beta: 0 },
        { id: 'B', label: 'B', type: 'outcome' as const, value: 0, beta: 0 },
      ],
      edges: [
        { source: 'A', target: 'B', curvature: 0.2, weight: 2.5, relation: 'increases' },
      ],
    };
    render(<ConfidencePanel graph={graph} />);
    // The buggy code displays "250%" — the fix clamps to "100%".
    // Find the badge element containing the percentage.
    const allText = document.body.textContent || '';
    expect(allText).not.toMatch(/250%/);
    expect(allText).toMatch(/100%/);
  });
});