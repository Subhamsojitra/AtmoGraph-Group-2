import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

/**
 * GraphCanvas Component
 * 
 * Renders a force-directed network graph using D3.js.
 * Consumes graph dataset passed via props from the host page.
 * Refined on Day 2 to use ResizeObserver for container-aware responsive sizing
 * and to ensure clean simulation lifecycle cleanup.
 * Refined on Day 3 to add interactive drag & hover behaviors.
 */
export default function GraphCanvas({ data }) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !data) return;

    // Get dynamic width/height from the parent container bounds, falling back to defaults if not set
    const initialWidth = containerRef.current.clientWidth || 800;
    const initialHeight = containerRef.current.clientHeight || 500;

    const svg = d3.select(svgRef.current)
      .attr("width", initialWidth)
      .attr("height", initialHeight);

    // Clear previous contents of the SVG container to prevent duplicate drawings
    svg.selectAll("*").remove();

    // Deep copy nodes and links to prevent D3 from mutating original static mock/demo dataset objects
    const nodes = data.nodes.map(d => ({ ...d }));
    const links = data.links.map(d => ({ ...d }));

    // Create D3 Force Simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(initialWidth / 2, initialHeight / 2))
      .force("collide", d3.forceCollide().radius(40));

    // Groups for layout layering (links behind nodes)
    const gLinks = svg.append("g").attr("class", "links");
    const gNodes = svg.append("g").attr("class", "nodes");

    // Draw links
    const link = gLinks.selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", 2);

    // Draw node groups (circle + label)
    const node = gNodes.selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .style("cursor", "grab");

    // Render node circles
    node.append("circle")
      .attr("r", 15)
      .attr("fill", d => {
        switch (d.type) {
          case 'Supplier': return '#3182ce'; // blue
          case 'Factory': return '#38a169'; // green
          case 'Warehouse': return '#dd6b20'; // orange
          case 'Distribution': return '#805ad5'; // purple
          default: return '#718096'; // gray
        }
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5);

    // Render node labels
    node.append("text")
      .text(d => d.label)
      .attr("x", 20)
      .attr("y", 5)
      .style("font-family", "sans-serif")
      .style("font-size", "12px")
      .style("fill", "#2d3748")
      .style("user-select", "none");

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

    // Create ResizeObserver to handle dynamic width/height updates responsively
    const resizeObserver = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0) return;
      
      const width = entries[0].contentRect.width || containerRef.current.clientWidth || 800;
      const height = entries[0].contentRect.height || containerRef.current.clientHeight || 500;

      // Update SVG canvas bounds
      svg.attr("width", width).attr("height", height);

      // Re-center force coordinates
      simulation.force("center", d3.forceCenter(width / 2, height / 2));

      // Re-heat simulation to adjust layout smoothly to new dimensions
      simulation.alpha(0.3).restart();
    });

    // Start observing parent container dimensions
    resizeObserver.observe(containerRef.current);

    // Cleanup simulation, observer, and clear SVG on component unmount
    return () => {
      resizeObserver.disconnect();
      simulation.stop();
      svg.selectAll("*").remove();
    };
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

