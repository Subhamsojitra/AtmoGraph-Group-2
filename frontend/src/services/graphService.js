import { staticGraphData } from '../graph/data';

/**
 * Transforms raw backend data structures into the standard { nodes, links } format for GraphCanvas.
 * Generic mapping - does not hardcode or rely on domain-specific type names.
 * 
 * Future integration flow:
 *   Backend nodes + relationships -> graphService transformation -> { nodes, links }
 */
export function transformBackendData(backendNodes, backendRelationships) {
  const nodes = (backendNodes || []).map(node => ({
    id: node.id,
    label: node.label || node.name || node.id,
    type: node.type || 'default',
    ...node
  }));

  const links = (backendRelationships || []).map(rel => {
    // Support common backend property names for relationships
    const source = rel.source || rel.startNodeId || rel.from;
    const target = rel.target || rel.endNodeId || rel.to;
    return {
      source,
      target,
      type: rel.type || 'default',
      ...rel
    };
  });

  return { nodes, links };
}

/**
 * Fetches graph data from the backend.
 * Currently uses staticGraphData as the development/mock source.
 * 
 * @param {string} state - The simulated data state ('success', 'empty', 'error')
 * @returns {Promise<{ nodes: Array, links: Array }>}
 */
export async function getGraphData(state = 'success') {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (state === 'error') {
        reject(new Error('Failed to fetch graph data from database.'));
        return;
      }
      
      if (state === 'empty') {
        resolve(transformBackendData([], []));
        return;
      }

      // Default: success state using staticGraphData as mock source
      const mockBackendNodes = staticGraphData.nodes;
      const mockBackendRelationships = staticGraphData.links;
      
      resolve(transformBackendData(mockBackendNodes, mockBackendRelationships));
    }, 600); // 600ms network latency simulation
  });
}
