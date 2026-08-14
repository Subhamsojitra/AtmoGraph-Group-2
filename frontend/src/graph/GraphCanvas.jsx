import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

/**
 * GraphCanvas Component
 * 
 * Renders a force-directed network graph using D3.js.
 * Consumes graph dataset passed via props from the host page.
 * Refined to use ref-based simulation persistence and D3 data join updates.
 */
export default function GraphCanvas({ data }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);

  const simulationRef = useRef(null);
  const resizeObserverRef = useRef(null);
  const gLinksRef = useRef(null);
  const gNodesRef = useRef(null);

  // Mount/Unmount cleanup effect
  useEffect(() => {
    const currentSvg = svgRef.current;
    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
        resizeObserverRef.current = null;
      }
      if (simulationRef.current) {
        simulationRef.current.stop();
        simulationRef.current = null;
      }
      if (currentSvg) {
        d3.select(currentSvg).selectAll("*").remove();
      }
    };
  }, []);

  // Data update and initialization effect
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    let svg = d3.select(svgRef.current);
    let simulation = simulationRef.current;
    let gLinks, gNodes;

    // First-time setup of SVG structural groups, simulation and ResizeObserver
    if (!simulation) {
      const initialWidth = containerRef.current.clientWidth || 800;
      const initialHeight = containerRef.current.clientHeight || 500;

      svg.attr("width", initialWidth).attr("height", initialHeight);

      // Create static containers once
      gLinks = svg.append("g").attr("class", "links");
      gNodes = svg.append("g").attr("class", "nodes");

      gLinksRef.current = gLinks;
      gNodesRef.current = gNodes;

      // Create D3 Force Simulation
      simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id).distance(120))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(initialWidth / 2, initialHeight / 2))
        .force("collide", d3.forceCollide().radius(40));

      simulationRef.current = simulation;

      // Initialize ResizeObserver
      const resizeObserver = new ResizeObserver((entries) => {
        if (!entries || entries.length === 0) return;
        const width = entries[0].contentRect.width || containerRef.current.clientWidth || 800;
        const height = entries[0].contentRect.height || containerRef.current.clientHeight || 500;

        // Update SVG canvas bounds
        svg.attr("width", width).attr("height", height);

        // Re-center force coordinates
        simulation.force("center", d3.forceCenter(width / 2, height / 2));

        // Re-heat simulation to adjust layout smoothly
        simulation.alpha(0.3).restart();
      });

      resizeObserver.observe(containerRef.current);
      resizeObserverRef.current = resizeObserver;
    } else {
      gLinks = gLinksRef.current;
      gNodes = gNodesRef.current;
    }

    // Process nodes and links data safely (basic robustness)
    const safeNodes = data?.nodes || [];
    const safeLinks = data?.links || [];

    // Map new nodes, preserving coordinates from existing nodes to avoid jarring jumps
    const previousNodes = simulation.nodes() || [];
    const previousNodesMap = new Map(previousNodes.map(n => [n.id, n]));

    const nodes = safeNodes
      .filter(n => n && n.id !== undefined && n.id !== null)
      .map(d => {
        const prev = previousNodesMap.get(d.id);
        if (prev) {
          return {
            ...d,
            x: prev.x,
            y: prev.y,
            vx: prev.vx,
            vy: prev.vy,
            fx: prev.fx,
            fy: prev.fy
          };
        }
        return { ...d };
      });

    const nodeIds = new Set(nodes.map(n => n.id));

    // Map links, validating both source and target references exist in nodeIds
    const links = safeLinks
      .filter(l => {
        if (!l || l.source === undefined || l.source === null || l.target === undefined || l.target === null) return false;
        const sourceId = (l.source && typeof l.source === 'object') ? l.source.id : l.source;
        const targetId = (l.target && typeof l.target === 'object') ? l.target.id : l.target;
        return nodeIds.has(sourceId) && nodeIds.has(targetId);
      })
      .map(d => {
        const sourceId = (d.source && typeof d.source === 'object') ? d.source.id : d.source;
        const targetId = (d.target && typeof d.target === 'object') ? d.target.id : d.target;
        return {
          ...d,
          source: sourceId,
          target: targetId
        };
      });

    // Draw links using data join
    const link = gLinks.selectAll("line")
      .data(links, d => {
        const s = (d.source && typeof d.source === 'object') ? d.source.id : d.source;
        const t = (d.target && typeof d.target === 'object') ? d.target.id : d.target;
        return `${s}->${t}`;
      })
      .join("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 2);

    // Draw node groups using data join
    const node = gNodes.selectAll("g.node-group")
      .data(nodes, d => d.id)
      .join(
        enter => {
          const g = enter.append("g")
            .attr("class", "node-group")
            .style("cursor", "grab");

          g.append("circle")
            .attr("r", 15)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5);

          g.append("text")
            .attr("x", 20)
            .attr("y", 5)
            .style("font-family", "sans-serif")
            .style("font-size", "12px")
            .style("fill", "#2d3748")
            .style("user-select", "none");

          return g;
        }
      );

    // Apply properties to merged selections (entering + updating)
    node.select("circle")
      .attr("fill", d => {
        switch (d.type) {
          case 'Supplier': return '#3182ce'; // blue
          case 'Factory': return '#38a169'; // green
          case 'Warehouse': return '#dd6b20'; // orange
          case 'Distribution': return '#805ad5'; // purple
          default: return '#718096'; // gray
        }
      });

    node.select("text")
      .text(d => d.label || d.id || "");

    // Drag Behavior
    const drag = d3.drag()
      .on("start", function(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        d3.select(this).style("cursor", "grabbing");
      })
      .on("drag", function(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", function(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
        d3.select(this).style("cursor", "grab");
      });

    node.call(drag);

    // Hover Interaction
    node
      .on("mouseenter", function() {
        d3.select(this).select("circle")
          .attr("stroke-width", 3);
        d3.select(this).select("text")
          .style("font-weight", "600");
      })
      .on("mouseleave", function() {
        d3.select(this).select("circle")
          .attr("stroke-width", 1.5);
        d3.select(this).select("text")
          .style("font-weight", "normal");
      });

    // Update positions on every tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("transform", d => `translate(${d.x}, ${d.y})`);
    });

    // Update simulation data and restart
    simulation.nodes(nodes);
    simulation.force("link").links(links);
    simulation.alpha(0.3).restart();
  }, [data]);

  return (
    <div 
      ref={containerRef} 
      className="graph-canvas-scaffold-boundary" 
      style={{ width: '100%', height: '500px', border: '1px solid #e2e8f0', borderRadius: '6px', background: '#f7fafc', overflow: 'hidden' }}
    >
      <svg ref={svgRef} style={{ display: 'block' }}></svg>
    </div>
  );
}
