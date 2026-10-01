import { render } from '@testing-library/react';
import * as d3 from 'd3';

/**
 * test_D3GraphEngine.tsx — Vitest tests for src/components/graph/D3GraphEngine.tsx
 *
 * Strategy:
 * - We use the REAL d3 in jsdom (it mostly works — selections, forceSimulation, zoom
 *   behaviors attach handlers and dispatch events). jsdom supports SVG elements.
 * - We avoid rendering the entire <D3GraphEngine> via React when we need to inspect
 *   imperative d3 selection calls; instead we often mount once and assert on the
 *   DOM / call-spies attached to d3.selection.
 * - When a test wants to verify the *source code* uses a particular pattern
 *   (e.g. "scoped to svg" or "no inline <style>"), we read the source file
 *   directly. This lets us check the fix without depending on d3's runtime
 *   behavior, which is brittle in jsdom.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(
  HERE,
  '../../src/components/graph/D3GraphEngine.tsx',
);
const SOURCE = fs.readFileSync(SRC, 'utf8');

// Lazy-import the component — its module load creates d3 selections on import,
// so we only import inside individual tests where we mock what we need.
async function importD3GraphEngine() {
  // Re-import fresh for each call (vitest caches modules; vi.resetModules handled at test level).
  return (await import('@/components/graph/D3GraphEngine'))
    .D3GraphEngine as typeof import('@/components/graph/D3GraphEngine').D3GraphEngine;
}

const makeGraph = () => ({
  nodes: [
    { id: 'A', label: 'A', type: 'controllable' as const, value: 0, beta: 0 },
    { id: 'B', label: 'B', type: 'outcome' as const, value: 0, beta: 0 },
    { id: 'C', label: 'C', type: 'bottleneck' as const, value: 0, beta: 0 },
  ],
  edges: [
    { source: 'A', target: 'B', curvature: 0.2, weight: 0.5, relation: 'increases' },
    { source: 'B', target: 'C', curvature: 0.2, weight: 0.5, relation: 'increases' },
  ],
});

describe('D3GraphEngine', () => {
  beforeEach(() => {
    vi.resetModules();
    // Ensure clean slate for d3 zoom listeners between tests.
    document.body.innerHTML = '';
  });

  /**
   * SWARM-2 BUG #20 — zoom listeners accumulate on every graph rebuild.
   *
   * Each call to `svg.call(zoom)` attaches a NEW `wheel` (and `mousedown`, etc.)
   * listener to the SVG. The buggy version never removes the previous zoom
   * listeners, so after N graph rebuilds the SVG has N wheel listeners.
   *
   * We spy on the SVG's `addEventListener` to count how many `wheel` listeners
   * are attached. After 5 graph rebuilds the buggy code has ~5 wheel listeners;
   * the fixed code (which calls `svg.on('.zoom', null)` in cleanup) keeps it
   * at 1.
   */
  test('test_zoom_listeners_do_not_accumulate', async () => {
    // Patch d3.zoom so the listener doesn't actually fire d3's gesture code
    // (jsdom's SVG lacks a real layout box → defaultExtent throws).
    // We don't need d3.zoom to actually run; we only need to count how many
    // wheel listeners get attached to the SVG.
    vi.doMock('d3', async () => {
      const actual = await vi.importActual<typeof d3>('d3');
      // Replace zoom with a no-op behavior that still attaches a 'wheel' listener
      // (so the test can count it) but doesn't try to compute layout.
      const fakeZoom = () => {
        const behavior: any = (selection: any) => {
          selection.on('wheel.zoom', () => {});
        };
        behavior.scaleExtent = () => behavior;
        behavior.on = () => behavior;
        return behavior;
      };
      return { ...actual, zoom: fakeZoom as unknown as typeof d3.zoom };
    });

    const D3GraphEngine = await importD3GraphEngine();
    const { container, rerender } = render(<D3GraphEngine graph={makeGraph()} />);

    const svg = container.querySelector('svg')!;
    expect(svg).toBeTruthy();

    // Spy on addEventListener before triggering rebuilds.
    const addSpy = vi.spyOn(svg, 'addEventListener');

    // Re-render 5 times — each triggers a buildGraph (which calls svg.call(zoom)).
    for (let i = 0; i < 5; i++) {
      rerender(<D3GraphEngine graph={makeGraph()} />);
    }

    // Count how many wheel listeners were attached via d3's `svg.call(zoom)`.
    const wheelListenerCount = addSpy.mock.calls.filter(
      ([type]) => type === 'wheel' || type === 'wheel.zoom',
    ).length;

    // The buggy code attaches a new wheel listener on every rebuild and never
    // removes the old ones → expect ~5. The fixed code removes the previous
    // zoom listeners in cleanup → expect 1 (or at most 2 if the order is
    // add-then-remove-then-add).
    expect(wheelListenerCount).toBeLessThanOrEqual(2);
  });

  /**
   * SWARM-2 BUG #20 (cleanup) — `svg.on('.zoom', null)` must be called in cleanup.
   * Verified by reading the source; runtime cleanup in jsdom is brittle.
   */
  test('test_zoom_listeners_removed_on_unmount', () => {
    // Static check: the source file should call `svg.on('.zoom', null)`
    // somewhere in its useEffect cleanup path.
    // The buggy version does NOT contain this call.
    const hasZoomCleanup =
      SOURCE.includes("svg.on('.zoom', null)") ||
      SOURCE.includes('svg.on(".zoom", null)');
    expect(hasZoomCleanup).toBe(true);
  });

  /**
   * SWARM-2 BUG #21 — `simRef.current.stop()` should be called on unmount,
   * AND `simRef.current` should be nullified afterwards (no ghost simulation).
   *
   * We mount the component with a real graph (forceSimulation is created),
   * then unmount it. The cleanup must call `stop()` on the simulation.
   * To observe this we wrap `d3.forceSimulation` so we can spy on `.stop()`.
   */
  test('test_simulation_stopped_on_unmount', async () => {
    const stopSpy = vi.fn();
    vi.doMock('d3', async () => {
      const actual = await vi.importActual<typeof d3>('d3');
      const realForceSimulation = actual.forceSimulation;
      const wrapped = ((...args: any[]) => {
        const sim = realForceSimulation(...args as [any, any]);
        const origStop = sim.stop.bind(sim);
        sim.stop = () => {
          stopSpy();
          return origStop();
        };
        return sim;
      }) as typeof actual.forceSimulation;
      return { ...actual, forceSimulation: wrapped };
    });

    const D3GraphEngine = await importD3GraphEngine();
    const { unmount } = render(<D3GraphEngine graph={makeGraph()} />);
    unmount();
    expect(stopSpy).toHaveBeenCalled();
  });

  /**
   * SWARM-2 BUG #21 — `simRef.current` must be nulled after `stop()`.
   * Static-source check: the cleanup function in buildGraph's useEffect
   * should set `simRef.current = null` after `stop()`.
   */
  test('test_simulation_nulled_after_stop', () => {
    // Look for the pattern `simRef.current = null` somewhere in the cleanup
    // of the buildGraph useEffect. The buggy source does NOT have this.
    const hasNullAssignment = /simRef\.current\s*=\s*null/.test(SOURCE);
    expect(hasNullAssignment).toBe(true);
  });

  /**
   * SWARM-2 BUG #22 — selectors must be scoped to the SVG (use svg.selectAll(...)),
   * not d3.selectAll(...) (global selectors cross-contaminate instances).
   */
  test('test_selectall_scoped_to_svg', () => {
    // Find every call to d3.selectAll with 'g.node-bottleneck' or 'path.link'
    // in the source — they should be `svg.selectAll(...)`, not `d3.selectAll(...)`.
    // The buggy source has lines like:
    //   d3.selectAll<SVGGElement, SimulationNode>('g.node-bottleneck')
    //   const links = d3.selectAll<SVGPathElement, SimulationLink>('path.link');
    const globalNodeBottleneck = /d3\.selectAll<[^>]*>\(\s*['"]g\.node-bottleneck['"]\s*\)/.test(
      SOURCE,
    );
    const globalPathLink = /d3\.selectAll<[^>]*>\(\s*['"]path\.link['"]\s*\)/.test(SOURCE);
    // The buggy version has at least one of these; the fix removes them.
    expect(globalNodeBottleneck).toBe(false);
    expect(globalPathLink).toBe(false);
  });

  /**
   * SWARM-2 BUG #25 — forceLink must drop edges with unresolved endpoints
   * (i.e., edges referencing a non-existent node ID).
   *
   * The buggy code keeps the original string ID as `source`:
   *   source: nodes.find(...) || e.source   // <-- keeps the bad string
   * The fix should `.filter(...)` edges before mapping.
   *
   * Static check: the source should contain a `.filter(` call on
   * `graph.edges` (or `edges`) before/around the link mapping.
   */
  test('test_forceLink_drops_edges_with_unresolved_endpoints', () => {
    // Buggy source: `graph.edges.map(e => ({...e, source: nodes.find(...) || e.source, ...}))`
    // Fixed source: `graph.edges.filter(e => nodes.some(...) && nodes.some(...)).map(...)`
    const hasFilter = /\.filter\s*\(\s*[^)]*nodes\.some/.test(SOURCE);
    expect(hasFilter).toBe(true);

    // Also ensure the source DOES NOT use the buggy `|| e.source` fallback.
    const hasBugFallback = /\|\|\s*e\.source/.test(SOURCE);
    expect(hasBugFallback).toBe(false);
  });

  /**
   * SWARM-2 BUG #24 — clicking a node highlights it (pink #E91E63, width 4).
   * Hovering over a node then leaving it must NOT reset the clicked node's
   * highlight back to white. The buggy `mouseout` handler unconditionally
   * resets to `#fff` / `2.5`, destroying the click selection.
   *
   * We render the component, simulate a click then mouseover+mouseout on
   * the SAME node, then assert the stroke is still `#E91E63` (not `#fff`).
   *
   * jsdom supports SVG circle attributes, but D3's drag/tick handlers fire
   * async; for this test we read the `mouseout` handler from the source
   * — it must conditionally preserve the selected node's stroke.
   */
  test('test_click_highlight_preserved_after_hover', () => {
    // Buggy: `mouseout` handler unconditionally sets stroke to '#fff' and width 2.5.
    // Fixed: `mouseout` checks `selectedNodeRef.current === d.id` (or similar)
    //   and restores `#E91E63` / width 4 if the node is the selected one.
    //
    // Static check: extract ONLY the mouseout handler's body (between
    // `.on('mouseout',` and the next `.on(`) and assert the body contains
    // a conditional restoration of `#E91E63`. The buggy version only sets
    // `'#fff'` with no `#E91E63` reference inside this block.
    const mouseoutIdx = SOURCE.indexOf(".on('mouseout'");
    expect(mouseoutIdx).toBeGreaterThan(-1);
    // Cut at the next `.on(` after mouseout.
    const afterMouseout = SOURCE.slice(mouseoutIdx);
    const nextOnIdx = afterMouseout.indexOf(".on('", 5); // skip the first `.on('mouseout'`
    const block = nextOnIdx > 0 ? afterMouseout.slice(0, nextOnIdx) : afterMouseout.slice(0, 600);
    // The fix must preserve #E91E63 inside the mouseout block (conditional restoration).
    const preservesSelectionInMouseout = block.includes('#E91E63');
    expect(preservesSelectionInMouseout).toBe(true);
  });

  /**
   * SWARM-2 BUG #26 — `document.dispatchEvent(new CustomEvent('OPEN_INSIGHT', ...))`
   * must have a corresponding listener somewhere. The buggy code dispatches
   * but no component listens.
   *
   * Static check across the entire src tree: there must be an
   * `addEventListener('OPEN_INSIGHT'` (or `.on('OPEN_INSIGHT'`) somewhere.
   * If not, the dispatch is dead code.
   */
  test('test_open_insight_event_has_listener', () => {
    const srcRoot = path.resolve(HERE, '../../src');
    const files: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (entry.name.endsWith('.tsx') || entry.name.endsWith('.ts')) files.push(full);
      }
    };
    walk(srcRoot);

    let listenerFound = false;
    for (const f of files) {
      const content = fs.readFileSync(f, 'utf8');
      if (/addEventListener\(\s*['"]OPEN_INSIGHT['"]/.test(content) ||
          /\.on\(\s*['"]OPEN_INSIGHT['"]/.test(content)) {
        listenerFound = true;
        break;
      }
    }
    // The buggy code dispatches OPEN_INSIGHT but never listens for it.
    expect(listenerFound).toBe(true);
  });

  /**
   * SWARM-2 BUG #23 — `setTooltipNode` is called on EVERY mouseover/mouseout,
   * causing React re-render storms. The fix is to either debounce or
   * drive the tooltip imperatively (via tooltipRef).
   *
   * Static check: the mouseover handler should NOT call `setTooltipNode`.
   * (Or at least, the buggy version calls it directly with no debounce.)
   *
   * We look for `setTooltipNode(` inside the `.on('mouseover', ...)` block.
   */
  test('test_tooltip_does_not_trigger_re_render_storm', () => {
    const mouseoverIdx = SOURCE.indexOf(".on('mouseover'");
    expect(mouseoverIdx).toBeGreaterThan(-1);
    const block = SOURCE.slice(mouseoverIdx, mouseoverIdx + 600);
    // The buggy version calls setTooltipNode(d) directly in mouseover.
    const buggyCall = block.includes('setTooltipNode(');
    expect(buggyCall).toBe(false);
  });

  /**
   * SWARM-2 BUG #58 — inline `<style>{` block in JSX should be moved to
   * globals.css. The component re-renders this CSS on every React render.
   *
   * Static check: the source file should NOT contain `<style>{`.
   */
  test('test_no_inline_style_block_in_jsx', () => {
    const hasInlineStyle = SOURCE.includes('<style>{');
    expect(hasInlineStyle).toBe(false);
  });

  /**
   * Wheel event guard — wheel over the SVG must not propagate to the page
   * scroll. The component uses `svg.call(d3.zoom())`, which attaches a
   * `wheel.zoom` listener via d3-selection (non-passive by default) and
   * the real d3.zoom's wheeled handler calls `event.preventDefault()`.
   *
   * jsdom's SVG `width.baseVal` API isn't populated even with
   * `setAttribute('width', ...)`, so running real d3.zoom's wheeled()
   * crashes. We can't observe `defaultPrevented` reliably.
   *
   * Strategy: spy on the SVG's `addEventListener` and assert that:
   *   (a) at least one `wheel`-typed listener is attached by the component
   *       (proving `svg.call(zoom)` ran), AND
   *   (b) the listener is registered non-passive (so preventDefault CAN
   *       fire when d3.zoom's real wheeled handler runs in a browser).
   *
   * Combined with a static-source check that `svg.call(zoom)` exists,
   * this catches any "fix" that removes the zoom wiring or sets the wheel
   * listener to passive.
   */
  test('test_wheel_event_does_not_propagate_to_page_scroll', async () => {
    vi.doMock('d3', async () => {
      const actual = await vi.importActual<typeof d3>('d3');
      // Replace d3.zoom with a behavior that mimics d3-selection's
      // `selection.on` call but skips the layout-dependent gesture code
      // that crashes in jsdom.
      const fakeZoom = () => {
        const behavior: any = (selection: any) => {
          // d3-selection's `selection.on('wheel.zoom', ...)` calls
          // addEventListener('wheel', listener) WITHOUT the `passive` flag,
          // so preventDefault can be called.
          selection.on('wheel.zoom', () => {});
        };
        behavior.scaleExtent = () => behavior;
        behavior.on = () => behavior;
        return behavior;
      };
      return { ...actual, zoom: fakeZoom as unknown as typeof d3.zoom };
    });

    const D3GraphEngine = await importD3GraphEngine();
    const { container, rerender } = render(<D3GraphEngine graph={makeGraph()} />);

    const svg = container.querySelector('svg')!;
    expect(svg).toBeTruthy();

    // Spy on addEventListener BEFORE triggering a rebuild that re-attaches the
    // wheel listener. The first render's wheel listener was attached before
    // we installed the spy, so we re-render with a new graph prop to trigger
    // a new buildGraph call.
    const addSpy = vi.spyOn(svg, 'addEventListener');
    rerender(<D3GraphEngine graph={makeGraph()} />);

    // Find any wheel listener registration that occurred during the rebuild.
    const wheelRegistrations = addSpy.mock.calls.filter(
      ([type]) => type === 'wheel' || type === 'wheel.zoom',
    );
    // The buggy code DOES attach a wheel listener via svg.call(zoom), so
    // wheelRegistrations.length should be > 0. A fix that removes svg.call(zoom)
    // would fail this.
    expect(wheelRegistrations.length).toBeGreaterThan(0);
    // Also verify the wheel listener was registered non-passive (so
    // preventDefault can fire).
    const allNonPassive = wheelRegistrations.every(([_t, _l, opts]: any) => {
      if (!opts) return true; // no options = non-passive default
      return opts.passive !== true;
    });
    expect(allNonPassive).toBe(true);

    // Static-source regression check: the source uses `svg.call(zoom)`.
    expect(SOURCE).toMatch(/svg\.call\(zoom\)/);
  });
});