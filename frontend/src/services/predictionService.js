/**
 * Prediction Service Boundary
 * 
 * Isolates prediction API communication from React components and D3.
 * Establishes a normalized frontend adapter boundary.
 * 
 * NOTE: The backend/ML contract is NOT finalized because Shivangi has not
 * started the ML/GNN implementation yet. There is currently no prediction API
 * contract from Santanu, nor is there a finalized prediction schema.
 * 
 * The ONLY stable architectural relationship established today is:
 *   prediction.nodeId -> graph node.id
 * 
 * All other fields (e.g., predictedRisk, predictedLevel, confidence, timestamp)
 * are temporary development/mock fields only and MUST NOT be represented as the
 * final project contract. Do not hardcode these fields throughout the application.
 */

/**
 * Fetches prediction data for the current graph.
 * 
 * If no real prediction endpoint exists, backend mode returns { predictions: [] }.
 * The absence of prediction data must never block graph rendering.
 * 
 * @param {string} mode - The active dataset mode ('mock', 'backend', 'large')
 * @returns {Promise<{ predictions: Array }>}
 */
export async function getPredictionData(mode = 'mock') {
  if (mode === 'backend') {
    // Return empty prediction data in backend mode to support graceful failure
    // as no real prediction endpoint is implemented in the backend yet.
    return { predictions: [] };
  }

  if (mode === 'large') {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Generate mock predictions for testing performance with large graphs.
        // These fields are temporary and for validation only.
        const predictions = [];
        for (let i = 1; i <= 2000; i += 20) {
          predictions.push({
            nodeId: `node-${i}`,
            predictedRisk: (i * 17) % 100,
            predictedLevel: i % 3 === 0 ? "high" : "elevated",
            confidence: 0.8,
            timestamp: new Date().toISOString()
          });
        }
        resolve({ predictions });
      }, 300);
    });
  }

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        predictions: [
          {
            nodeId: "node-1",
            predictedRisk: 85,
            predictedLevel: "high",
            confidence: 0.92,
            timestamp: new Date().toISOString()
          },
          {
            nodeId: "node-2",
            predictedRisk: 45,
            predictedLevel: "elevated",
            confidence: 0.78,
            timestamp: new Date().toISOString()
          }
        ]
      });
    }, 600);
  });
}

/**
 * Maps a prediction object to a normalized risk visualization state.
 * Possible returned states:
 *   - 'high' (at-risk)
 *   - 'medium' (elevated)
 *   - 'low' (stable)
 *   - 'unknown' (no prediction / default)
 * 
 * @param {Object} prediction - The prediction object from the API.
 * @returns {string} One of: 'high', 'medium', 'low', 'unknown'
 */
export function getRiskState(prediction) {
  if (!prediction) return 'unknown';
  
  // Normalize based on temporary mock predictedLevel fields.
  // This centralizes prediction interpretation.
  const level = prediction.predictedLevel;
  if (level === 'high') {
    return 'high';
  }
  if (level === 'elevated' || level === 'medium') {
    return 'medium';
  }
  if (level === 'stable' || level === 'low') {
    return 'low';
  }
  return 'unknown';
}

