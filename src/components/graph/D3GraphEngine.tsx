"use client";

import React, { useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { GraphLegend } from './GraphLegend';
import { CrossDomainBridge } from './CrossDomainBridge';
import { GraphControls } from './GraphControls';
import { CausalGraph, CausalNode, CausalEdge, NodeType } from '../../types/graph';
import { BridgeResult } from '../../types/layer3';

// ─────────────────── Types ────────────────────────────────────────────────────
// D3 requires mutable coordinate properties for its simulation
interface SimulationNode extends CausalNode, d3.SimulationNodeDatum {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface SimulationLink extends d3.SimulationLinkDatum<SimulationNode> {
  curvature: number;
  crossDomain?: boolean;
  weight: number;
  label?: string;
  relation?: string;
}

export type PipelineStep = 'idle' | 'simulated' | 'bottleneck' | 'crossdomain';

export interface D3GraphEngineProps {
  graph?: CausalGraph | null;
  bridge?: BridgeResult | null;
  step?: PipelineStep;
  insightVisible?: boolean;
  onRunDiscovery?: () => void;
  onIdentifyBottleneck?: () => void;
  onSearchCrossDomain?: () => void;
}

// ─────────────────── Color Map ─────────────────────────────────────────────────
const COLOR_MAP: Record<NodeType, string> = {
  controllable: '#FF6B35',
  mediator:     '#004E89',
  bottleneck:   '#C5283D',
  outcome:      '#1A936F',
  chemistry:    '#7B2D8E',
};

// ─────────────────── Component ────────────────────────────────────────────────
export const D3GraphEngine: React.FC<D3GraphEngineProps> = ({
  graph,
  bridge,
  step = 'idle',
  insightVisible = false,
  onRunDiscovery,
  onIdentifyBottleneck,
  onSearchCrossDomain,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const simRef = useRef<d3.Simulation<SimulationNode, SimulationLink> | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const selectedNodeRef = useRef<string | null>(null);

  // Listen for OPEN_INSIGHT events
  useEffect(() => {
    const handleOpenInsight = (e: Event) => {
      const customEvent = e as CustomEvent;
      console.log('Insight opened:', customEvent.detail);
    };
    document.addEventListener('OPEN_INSIGHT', handleOpenInsight);
    return () => {
      document.removeEventListener('OPEN_INSIGHT', handleOpenInsight);
    };
  }, []);

  const zoomCallCountRef = useRef(0);

  // 1. Rebuild the entire graph topology when the 'graph' data prop changes
  const buildGraph = useCallback(() => {
    if (!svgRef.current) return;
    if (!graph || !graph.nodes || !graph.edges) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 900;
    const height = svgRef.current.clientHeight || 600;

    // Zoom Container
    const g = svg.append('g').attr('class', 'graph-container');

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => g.attr('transform', event.transform));
      
    if (zoomCallCountRef.current < 2) {
      zoomCallCountRef.current += 1;
      svg.call(zoom);
    }

    // Arrow Markers Definition
    const defs = svg.append('defs');
    [
      { id: 'arrow-normal',  fill: '#C0C0C0' },
      { id: 'arrow-dashed',  fill: '#7B2D8E' },
      { id: 'arrow-active',  fill: '#3498db' },
      { id: 'arrow-danger',  fill: '#C5283D' }
    ].forEach(({ id, fill }) => {
      defs.append('marker')
        .attr('id', id)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', fill);
    });

    // Deep copy data for D3 mutation to prevent polluting React props
    const nodes: SimulationNode[] = graph.nodes.map(n => ({ ...n }));
    
    // Resolve edge references from string IDs to actual node objects, filtering out dangling edges
    const links: SimulationLink[] = graph.edges
      .filter(e => nodes.some(n => n.id === e.source) && nodes.some(n => n.id === e.target))
      .map(e => ({
        ...e,
        source: nodes.find(n => n.id === e.source)!,
        target: nodes.find(n => n.id === e.target)!,
      })) as SimulationLink[];

    // Force Simulation Setup
    const simulation = d3.forceSimulation<SimulationNode>(nodes)
      .force('link', d3.forceLink<SimulationNode, SimulationLink>(links).id(d => d.id).distance(150))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(50));

    simRef.current = simulation;

    // Render Edges
    const linkGroup = g.append('g').attr('class', 'links');
    const linkEl = linkGroup.selectAll<SVGPathElement, SimulationLink>('path.link')
      .data(links)
      .enter()
      .append('path')
      .attr('class', d => `link ${d.crossDomain ? 'cross-domain' : ''}`)
      .attr('data-source', d => (d.source as SimulationNode).type)
      .attr('data-target', d => (d.target as SimulationNode).type)
      .attr('stroke', d => d.crossDomain ? '#7B2D8E' : '#C0C0C0')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', d => d.crossDomain ? '5,5' : null)
      .attr('marker-end', d => `url(#${d.crossDomain ? 'arrow-dashed' : 'arrow-normal'})`)
      .attr('fill', 'none');

    // Render Nodes
    const nodeGroup = g.append('g').attr('class', 'nodes');
    const nodeEl = nodeGroup.selectAll<SVGGElement, SimulationNode>('g.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', d => `node node-${d.type}`)
      .call(
        d3.drag<SVGGElement, SimulationNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      )
      .on('mouseover', (event, d) => {
        d3.select(event.currentTarget).select('circle')
          .attr('stroke', '#333')
          .attr('stroke-width', 3);
        if (tooltipRef.current) {
          tooltipRef.current.innerHTML = `
            <h3 class="text-[14px] font-bold mb-2.5 pb-2 font-sans" style="border-bottom: 2px solid ${COLOR_MAP[d.type] || '#999'}">
              ${d.label}
            </h3>
            <div class="flex justify-between my-1.5 leading-relaxed">
              <span class="text-[#666666] font-semibold">Value</span>
              <span class="text-black font-mono">${(d.value || 0).toFixed(2)}</span>
            </div>
            <div class="flex justify-between my-1.5 leading-relaxed">
              <span class="text-[#666666] font-semibold">β Coefficient</span>
              <span class="text-black font-mono">${(d.beta || 0).toFixed(3)}</span>
            </div>
            <div class="flex justify-between my-1.5 leading-relaxed">
              <span class="text-[#666666] font-semibold">Type</span>
              <span class="text-black font-mono uppercase">${d.type || 'unknown'}</span>
            </div>
          `;
          tooltipRef.current.style.display = 'block';
          tooltipRef.current.style.left = `${event.clientX + 15}px`;
          tooltipRef.current.style.top = `${event.clientY + 15}px`;
        }
      })
      .on('mousemove', (event) => {
        if (tooltipRef.current) {
          tooltipRef.current.style.left = `${event.clientX + 15}px`;
          tooltipRef.current.style.top = `${event.clientY + 15}px`;
        }
      })
      .on('mouseout', (event, d) => {
        const isSelected = selectedNodeRef.current === d.id;
        d3.select(event.currentTarget).select('circle')
          .attr('stroke', isSelected ? '#E91E63' : '#fff')
          .attr('stroke-width', isSelected ? 4 : 2.5);
        if (tooltipRef.current) {
          tooltipRef.current.style.display = 'none';
        }
      })
      .on('click', (event, d) => {
        selectedNodeRef.current = d.id;
        nodeEl.select('circle')
          .attr('stroke', '#fff')
          .attr('stroke-width', 2.5);
        d3.select(event.currentTarget).select('circle')
          .attr('stroke', '#E91E63')
          .attr('stroke-width', 4);
      });

    nodeEl.append('circle')
      .attr('r', d => d.type === 'bottleneck' ? 13 : 10)
      .attr('fill', d => COLOR_MAP[d.type] || '#999')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2.5);

    nodeEl.append('text')
      .attr('dx', 14)
      .attr('dy', 4)
      .attr('font-family', "'Space Grotesk', sans-serif")
      .attr('font-size', 11)
      .attr('fill', '#000000')
      .attr('font-weight', 600)
      .text(d => d.label);

    function linkPath(d: SimulationLink): string {
      const s = d.source as SimulationNode;
      const t = d.target as SimulationNode;
      return `M${s.x ?? 0},${s.y ?? 0} L${t.x ?? 0},${t.y ?? 0}`;
    }

    simulation.on('tick', () => {
      linkEl.attr('d', d => linkPath(d));
      nodeEl.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

  }, [graph]);

  // Boot the graph simulation when data changes
  useEffect(() => {
    buildGraph();
    return () => {
      if (svgRef.current) {
        const svg = d3.select(svgRef.current);
        svg.on('.zoom', null);
      }
      if (simRef.current) {
        simRef.current.stop();
        simRef.current = null;
      }
    };
  }, [buildGraph]);

  // 2. Handle 'step' prop changes
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const links = svg.selectAll<SVGPathElement, SimulationLink>('path.link');
    
    if (step === 'simulated') {
      simRef.current?.alpha(1).restart();
    }
  }, [step]);

  // Calculate unique node types present in the current graph
  const uniqueTypes = React.useMemo(() => {
    if (!graph || !graph.nodes) return [];
    const types = new Set<NodeType>();
    graph.nodes.forEach(n => types.add(n.type));
    return Array.from(types);
  }, [graph]);

  if (!graph) {
    return (
      <div className="flex items-center justify-center h-full text-[#999999] font-sans text-[14px]">
        No causal graph available. Run discovery pipeline to generate visualization.
      </div>
    );
  }

  return (
    <>
      {/* SVG Canvas */}
      <svg
        ref={svgRef}
        className="absolute inset-0 w-full h-full"
        style={{ zIndex: 1 }}
      />

      {/* Dynamic Hover Tooltip */}
      <div
        ref={tooltipRef}
        className="absolute bg-white border border-[#E5E5E5] p-4 pointer-events-none z-20 text-[12px] text-black shadow-[4px_4px_0px_rgba(0,0,0,0.05)] min-w-[220px]"
        style={{
          display: 'none',
          position: 'fixed',
        }}
      />

      <GraphControls 
        step={step}
        onRunDiscovery={onRunDiscovery}
        onIdentifyBottleneck={onIdentifyBottleneck}
        onSearchCrossDomain={onSearchCrossDomain}
      />

      <CrossDomainBridge insightVisible={insightVisible} bridge={bridge} />
      <GraphLegend types={uniqueTypes} colorMap={COLOR_MAP} />
    </>
  );
};
