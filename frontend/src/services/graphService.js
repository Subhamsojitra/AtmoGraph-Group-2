import { staticGraphData } from '../graph/data';

/**
 * Transforms raw backend data structures into the standard { nodes, links } format for GraphCanvas.
 * Generic mapping - does not hardcode or rely on domain-specific type names.
 * 
 * Future integration flow:
 *   Backend nodes + relationships -> graphService transformation -> { nodes, links }
 */
export function transformBackendData(backendNodes, backendRelationships) {
  const seenIds = new Set();
  const nodes = [];

  if (Array.isArray(backendNodes)) {
    for (const node of backendNodes) {
      if (!node || typeof node !== 'object') continue;

      // Extract and sanitize ID
      const id = node.id !== undefined && node.id !== null ? String(node.id) : null;
      if (!id) continue;

      // Deduplicate node IDs
      if (seenIds.has(id)) continue;
      seenIds.add(id);

      // Determine the type:
      // If node.type is explicitly provided (mock data), use it.
      // Otherwise, map it to node.label (Neo4j node label in backend data).
      // Fallback to 'default' if neither is present.
      const type = node.type || node.label || 'default';

      // Determine label with safe fallback order:
      // If node.type is present (mock data), prioritize node.label as display label.
      // If node.type is NOT present (backend data), prioritize properties.name/title/label or node.name,
      // and only fall back to node.label (which represents the Neo4j type label) or id if name is missing.
      let label;
      if (node.type) {
        label = node.label || (node.properties && (node.properties.name || node.properties.title || node.properties.label)) || node.name || id;
      } else {
        label = (node.properties && (node.properties.name || node.properties.title || node.properties.label)) || node.name || node.label || id;
      }

      nodes.push({
        ...node,
        id,
        label: String(label),
        type: String(type),
        properties: node.properties || {}
      });
    }
  }

  const links = [];
  if (Array.isArray(backendRelationships)) {
    for (const rel of backendRelationships) {
      if (!rel || typeof rel !== 'object') continue;

      const source = rel.source || rel.startNodeId || rel.from;
      const target = rel.target || rel.endNodeId || rel.to;

      if (source === undefined || source === null || target === undefined || target === null) continue;

      const sourceStr = String(source);
      const targetStr = String(target);

      // Ensure that relationships only reference valid node IDs
      if (!seenIds.has(sourceStr) || !seenIds.has(targetStr)) continue;

      links.push({
        ...rel,
        source: sourceStr,
        target: targetStr,
        type: rel.type || 'default',
        properties: rel.properties || {}
      });
    }
  }

  return { nodes, links };
}


/**
 * Generates a development-only mock dataset of ~2,000 nodes and ~3,000 links
 * for benchmarking scalability.
 */
function generateLargeMockData() {
  const nodes = [];
  for (let i = 1; i <= 2000; i++) {
    nodes.push({
      id: `node-${i}`,
      label: `Node ${i}`,
      type: i % 4 === 0 ? 'Supplier' : (i % 4 === 1 ? 'Factory' : (i % 4 === 2 ? 'Warehouse' : 'Distribution')),
      properties: {
        name: `Node ${i}`,
        index: i,
        description: `Auto-generated large performance testing node #${i}`
      }
    });
  }

  const links = [];
  for (let i = 1; i <= 3000; i++) {
    const sourceIdx = (i % 2000) + 1;
    let targetIdx = (i * 7 + 13) % 2000 + 1;
    if (targetIdx === sourceIdx) {
      targetIdx = (targetIdx % 2000) + 1;
    }
    links.push({
      source: `node-${sourceIdx}`,
      target: `node-${targetIdx}`,
      type: 'TRANSPORT',
      properties: {
        cost: Math.round(Math.random() * 100)
      }
    });
  }

  return { nodes, links };
}

/**
 * Fetches graph data depending on the selected mode.
 * 
 * @param {string} mode - The active dataset mode ('mock', 'backend', 'large')
 * @returns {Promise<{ nodes: Array, links: Array }>}
 */
export async function getGraphData(mode = 'mock') {
  if (mode === 'backend') {
    const response = await fetch('/api/v1/graph/nodes');
    if (!response.ok) {
      throw new Error(`Backend request failed with status: ${response.status}`);
    }
    const backendNodes = await response.json();
    // Return empty links as Santanu's API does not provide bulk relationships.
    return transformBackendData(backendNodes, []);
  }

  if (mode === 'large') {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(generateLargeMockData());
      }, 300);
    });
  }

  // Default: 'mock' mode using staticGraphData
  return new Promise((resolve) => {
    setTimeout(() => {
      const mockBackendNodes = staticGraphData.nodes;
      const mockBackendRelationships = staticGraphData.links;
      resolve(transformBackendData(mockBackendNodes, mockBackendRelationships));
    }, 600);
  });
}

