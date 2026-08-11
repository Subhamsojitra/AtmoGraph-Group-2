import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { staticGraphData } from './data';

/**
 * GraphCanvas Component
 * 
 * Renders a static force-directed network graph using D3.js.
 * Consumes local dataset representing a supply chain network.
 * Handles D3 simulation cleanup on unmount.
 */
export default function GraphCanvas() {
  const containerRef = useRef(null);
  const svgRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    // Get dynamic width/height from the parent container bounds, falling back to defaults if not set
    const width = containerRef.current.clientWidth || 800;
    const height = containerRef.current.clientHeight || 500;

    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height);

    // Clear previous contents of the SVG container to prevent duplicate drawings
    svg.selectAll("*").remove();

    // Deep copy nodes and links to prevent D3 from mutating original static dataset objects
    const nodes = staticGraphData.nodes.map(d => ({ ...d }));
    const links = staticGraphData.links.map(d => ({ ...d }));

    // Create D3 Force Simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
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
      .append("g");

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

    // Cleanup simulation and clear SVG on component unmount
    return () => {
      simulation.stop();
      svg.selectAll("*").remove();
    };
  }, []);

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
