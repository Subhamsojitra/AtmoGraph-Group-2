/**
 * MOCK/DEMO DATA for Week 1 development.
 * 
 * NOTE: This is a temporary dataset and will be replaced by backend-provided
 * graph data (from FastAPI / WebSockets) in Week 2.
 * 
 * Predictable local dataset representing a supply chain network graph.
 * Composed of standard 'nodes' and 'links' format:
 * - Nodes: id (unique identifier), label (display name), type (category)
 * - Links: source (node id), target (node id)
 */
export const staticGraphData = {
  nodes: [
    { id: "node-1", label: "Lithium Mine A", type: "Supplier" },
    { id: "node-2", label: "Cathode Processor B", type: "Supplier" },
    { id: "node-3", label: "Anode Processor C", type: "Supplier" },
    { id: "node-4", label: "Gigafactory Assembly", type: "Factory" },
    { id: "node-5", label: "Regional Warehouse Hub", type: "Warehouse" },
    { id: "node-6", label: "EV Distribution Center", type: "Distribution" }
  ],
  links: [
    { source: "node-1", target: "node-2" },
    { source: "node-2", target: "node-4" },
    { source: "node-3", target: "node-4" },
    { source: "node-4", target: "node-5" },
    { source: "node-5", target: "node-6" }
  ]
};
