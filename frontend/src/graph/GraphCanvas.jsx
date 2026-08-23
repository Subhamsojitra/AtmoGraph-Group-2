import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

/**
 * GraphCanvas Component
 * 
 * Renders a force-directed network graph using D3.js.
 * Consumes graph dataset passed via props from the host page.
 * Refined to use ref-based simulation persistence and D3 data join updates.
 */
export default function GraphCanvas({ data, selectedNodeId, onNodeClick }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);

  const simulationRef = useRef(null);
  const resizeObserverRef = useRef(null);
  const gLinksRef = useRef(null);
  const gNodesRef = useRef(null);
  const selectedNodeIdRef = useRef(null);

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

    selectedNodeIdRef.current = selectedNodeId;

    let svg = d3.select(svgRef.current);
    let simulation = simulationRef.current;
    let gLinks, gNodes;

    // First-time setup of SVG structural groups, simulation and ResizeObserver
    if (!simulation) {
      const initialWidth = containerRef.current.clientWidth || 800;
      const initialHeight = containerRef.current.clientHeight || 500;

      svg.attr("width", initialWidth).attr("height", initialHeight);

      // Create a parent group container to receive zoom/pan transforms
      const gMain = svg.append("g").attr("class", "graph-main-content");
      gLinks = gMain.append("g").attr("class", "links");
      gNodes = gMain.append("g").attr("class", "nodes");

      gLinksRef.current = gLinks;
      gNodesRef.current = gNodes;

      // Define zoom and pan behavior
      const zoom = d3.zoom()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
          gMain.attr("transform", event.transform);
        });

      // Bind zoom behavior to the SVG container
      svg.call(zoom);

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

    const LARGE_GRAPH_THRESHOLD = 500;
    const isLargeMode = safeNodes.length > LARGE_GRAPH_THRESHOLD;

    // Dynamically adjust forces and settings for large vs. normal graphs
    if (isLargeMode) {
      simulation
        .alphaDecay(0.08)
        .force("collide", null)
        .force("charge", d3.forceManyBody().strength(-80).distanceMax(250))
        .force("link", d3.forceLink().id(d => d.id).distance(80));
    } else {
      simulation
        .alphaDecay(1 - Math.pow(0.001, 1 / 300))
        .force("collide", d3.forceCollide().radius(40))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("link", d3.forceLink().id(d => d.id).distance(120));
    }

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

    // Extract unique node types and sort deterministically
    const uniqueTypes = Array.from(
      new Set(
        nodes
          .map(n => n.type)
          .filter(t => t !== undefined && t !== null && t !== '')
      )
    ).sort();

    // Premium dynamic color palette
    const colorPalette = [
      '#3182ce', // blue
      '#38a169', // green
      '#dd6b20', // orange
      '#805ad5', // purple
      '#e53e3e', // red
      '#319795', // teal
      '#d69e2e', // yellow
      '#b7791f', // gold
      '#4a5568'  // slate
    ];

    const typeColorMap = new Map();
    uniqueTypes.forEach((type, index) => {
      typeColorMap.set(type, colorPalette[index % colorPalette.length]);
    });

    const getNodeColor = (type) => {
      if (!type) return '#718096'; // default fallback for untyped
      return typeColorMap.get(type) || '#718096';
    };

    // Apply properties to merged selections (entering + updating)
    node.select("circle")
      .attr("fill", d => getNodeColor(d.type))
      .attr("stroke", d => (selectedNodeIdRef.current === d.id ? "#3182ce" : "#fff"))
      .attr("stroke-width", d => (selectedNodeIdRef.current === d.id ? 3 : 1.5));

    node.select("text")
      .text(d => d.label || d.id || "")
      .style("font-weight", d => (selectedNodeIdRef.current === d.id ? "600" : "normal"))
      .style("display", d => (isLargeMode ? (selectedNodeIdRef.current === d.id ? "block" : "none") : "block"));

    // Drag Behavior
    const drag = d3.drag()
      .on("start", function(event, d) {
        // Prevent background zoom/pan from capturing the drag
        if (event.sourceEvent) {
          event.sourceEvent.stopPropagation();
        }
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

    // Click & Hover Interaction
    node
      .on("click", (event, d) => {
        // Prevent click event from bubbling up to parent SVG zoom/pan listeners
        event.stopPropagation();
        if (event.defaultPrevented) return;

        const prevId = selectedNodeIdRef.current;
        selectedNodeIdRef.current = d.id;

        if (onNodeClick) {
          onNodeClick(d);
        }

        // Lightweight selection styling updates (only update previously selected and currently selected)
        gNodes.selectAll("g.node-group")
          .filter(n => n && (n.id === prevId || n.id === d.id))
          .each(function(n) {
            const isSelected = selectedNodeIdRef.current === n.id;
            d3.select(this).select("circle")
              .attr("stroke", isSelected ? "#3182ce" : "#fff")
              .attr("stroke-width", isSelected ? 3 : 1.5);
            d3.select(this).select("text")
              .style("font-weight", isSelected ? "600" : "normal")
              .style("display", isLargeMode ? (isSelected ? "block" : "none") : "block");
          });
      })
      .on("mouseenter", function() {
        d3.select(this).select("circle")
          .attr("stroke", "#3182ce")
          .attr("stroke-width", 3);
        d3.select(this).select("text")
          .style("font-weight", "600")
          .style("display", "block");
      })
      .on("mouseleave", function(_event, d) {
        const isSelected = selectedNodeIdRef.current === d.id;
        d3.select(this).select("circle")
          .attr("stroke", isSelected ? "#3182ce" : "#fff")
          .attr("stroke-width", isSelected ? 3 : 1.5);
        d3.select(this).select("text")
          .style("font-weight", isSelected ? "600" : "normal")
          .style("display", isLargeMode ? (isSelected ? "block" : "none") : "block");
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

    // Update simulation data
    simulation.nodes(nodes);
    simulation.force("link").links(links);

    if (isLargeMode) {
      // Experimental: Synchronous pre-ticking to settle positions quickly.
      // We will start with 40 ticks, then adjust/remove based on benchmarks.
      simulation.alpha(0.3);
      for (let i = 0; i < 40; i++) {
        simulation.tick();
      }
      simulation.restart();
    } else {
      simulation.alpha(0.3).restart();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, onNodeClick]);

  // Sync selection styling when selectedNodeId changes
  useEffect(() => {
    const prevId = selectedNodeIdRef.current;
    selectedNodeIdRef.current = selectedNodeId;
    if (!simulationRef.current) return;

    const safeNodes = data?.nodes || [];
    const LARGE_GRAPH_THRESHOLD = 500;
    const isLargeMode = safeNodes.length > LARGE_GRAPH_THRESHOLD;

    const svg = d3.select(svgRef.current);
    // Only update elements that changed selection state to avoid looping through thousands of nodes unnecessarily
    svg.selectAll("g.node-group")
      .filter(n => n && (n.id === prevId || n.id === selectedNodeId))
      .each(function(n) {
        const isSelected = selectedNodeId === n.id;
        d3.select(this).select("circle")
          .attr("stroke", isSelected ? "#3182ce" : "#fff")
          .attr("stroke-width", isSelected ? 3 : 1.5);
        d3.select(this).select("text")
          .style("font-weight", isSelected ? "600" : "normal")
          .style("display", isLargeMode ? (isSelected ? "block" : "none") : "block");
      });
  }, [selectedNodeId, data]);

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
