import { createWebSocketTransport } from './websocketService';

/**
 * Prediction Service Boundary
 * 
 * Isolates prediction API communication and WebSocket transport from React components and D3.
 * Establishes a normalized frontend adapter boundary.
 * 
 * NOTE: The backend exposes Module 14 GNN node regression (POST /api/v1/predictions)
 * and Module 15 WebSocket transport (/api/v1/ws).
 * 
 * The confirmed transport flow is:
 *   Confirmed WebSocket endpoint (/api/v1/ws)
 *     ↓
 *   WebSocket transport (lifecycle, ping/pong, errors, disconnects)
 *     ↓
 *   Raw message delivery
 *     ↓
 *   Prediction service boundary
 *     ↓
 *   Future finalized GNN payload
 * 
 * The stable architectural relationship is:
 *   prediction.nodeId -> graphNode.id
 * 
 * The adapter normalizes backend fields (e.g. node_id -> nodeId) and safely
 * handles raw scalar regression predictions without assuming unverified risk schemas.
 */

/**
 * Sanitizes and normalizes a list of prediction entries.
 * Supports both frontend camelCase (nodeId) and backend snake_case (node_id).
 * Filters out null/undefined entries, entries without nodeId, and resolves duplicate nodeIds.
 * Only normalizes numeric predictedRisk/confidence values when they are actually numeric.
 * Preserves non-numeric values and raw prediction values as-is.
 * 
 * @param {Array} predictions - Raw predictions array
 * @returns {Array} Sanitized predictions array
 */
export function sanitizePredictions(predictions) {
  if (!Array.isArray(predictions)) return [];

  const seenKeys = new Set();
  const sanitized = [];

  for (const p of predictions) {
    // 1. Skip null/undefined or non-object entries
    if (!p || typeof p !== 'object') {
      console.warn("Prediction service: ignored null or non-object prediction entry:", p);
      continue;
    }

    // 2. Skip entries missing nodeId or with invalid/empty nodeId (supporting both nodeId and node_id)
    const rawNodeId = p.nodeId !== undefined && p.nodeId !== null && p.nodeId !== ''
      ? p.nodeId
      : (p.node_id !== undefined && p.node_id !== null && p.node_id !== '' ? p.node_id : undefined);

    if (rawNodeId === undefined) {
      console.warn("Prediction service: ignored prediction entry missing nodeId/node_id:", p);
      continue;
    }

    // 3. Skip duplicate nodeIds for the same horizon (keep first seen)
    const nodeIdStr = String(rawNodeId);
    const dedupKey = p.horizon !== undefined && p.horizon !== null ? `${nodeIdStr}#${p.horizon}` : nodeIdStr;
    if (seenKeys.has(dedupKey)) {
      console.warn(`Prediction service: ignored duplicate prediction for nodeId "${nodeIdStr}":`, p);
      continue;
    }

    // 4. Copy entry properties safely with normalized nodeId
    const sanitizedEntry = { ...p, nodeId: nodeIdStr };

    // 5. Only normalize predictedRisk when it is actually a number
    if (p.predictedRisk !== undefined && p.predictedRisk !== null) {
      if (typeof p.predictedRisk === 'number' && !isNaN(p.predictedRisk)) {
        sanitizedEntry.predictedRisk = Math.max(0, Math.min(100, p.predictedRisk));
      } else {
        // Keep non-numeric values exactly as-is per user request
        sanitizedEntry.predictedRisk = p.predictedRisk;
      }
    }

    // 6. Only normalize confidence when it is actually a number
    if (p.confidence !== undefined && p.confidence !== null) {
      if (typeof p.confidence === 'number' && !isNaN(p.confidence)) {
        if (p.confidence > 1) {
          sanitizedEntry.confidence = Math.max(0, Math.min(100, p.confidence)) / 100;
        } else {
          sanitizedEntry.confidence = Math.max(0, Math.min(1, p.confidence));
        }
      } else {
        // Keep non-numeric values exactly as-is per user request
        sanitizedEntry.confidence = p.confidence;
      }
    }

    // 7. Safe fallback for level
    if (p.predictedLevel !== undefined && p.predictedLevel !== null) {
      if (typeof p.predictedLevel === 'string') {
        sanitizedEntry.predictedLevel = p.predictedLevel.toLowerCase();
      } else {
        sanitizedEntry.predictedLevel = p.predictedLevel;
      }
    }

    seenKeys.add(dedupKey);
    sanitized.push(sanitizedEntry);
  }

  return sanitized;
}

/**
 * Fetches prediction data for the current graph.
 * 
 * In backend mode, connects to POST /api/v1/predictions. If the backend or
 * checkpoint is unavailable (e.g. 503), it returns { predictions: [] } gracefully.
 * The absence of prediction data must never block graph rendering.
 * 
 * @param {string} mode - The active dataset mode ('mock', 'backend', 'large')
 * @returns {Promise<{ predictions: Array }>}
 */
export async function getPredictionData(mode = 'mock') {
  if (mode === 'backend') {
    try {
      const response = await fetch('/api/v1/predictions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });

      if (!response.ok) {
        // Backend returns 503 if no checkpoint configured or database is unreachable
        return { predictions: [] };
      }

      const data = await response.json();
      const rawPredictions = data && Array.isArray(data.predictions) ? data.predictions : [];
      return { predictions: sanitizePredictions(rawPredictions) };
    } catch (err) {
      console.warn("Prediction service: backend request failed or endpoint unavailable:", err);
      return { predictions: [] };
    }
  }

  if (mode === 'large') {
    return new Promise((resolve) => {
      setTimeout(() => {
        // Generate mock predictions for testing performance with large graphs.
        // The horizon and other fields are temporary development fields for validation.
        const predictions = [];
        const horizons = ["30", "60", "90"];
        for (let i = 1; i <= 2000; i += 20) {
          // Distribute predictions across horizons deterministically
          const horizon = horizons[Math.floor(i / 20) % horizons.length];
          predictions.push({
            nodeId: `node-${i}`,
            predictedRisk: (i * 17) % 100,
            predictedLevel: i % 3 === 0 ? "high" : "elevated",
            confidence: 0.8,
            timestamp: new Date().toISOString(),
            horizon: horizon
          });
        }
        resolve({ predictions: sanitizePredictions(predictions) });
      }, 300);
    });
  }

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        predictions: sanitizePredictions([
          // 30-day horizon predictions
          {
            nodeId: "node-1",
            predictedRisk: 85,
            predictedLevel: "high",
            confidence: 0.92,
            timestamp: new Date().toISOString(),
            horizon: "30"
          },
          {
            nodeId: "node-2",
            predictedRisk: 45,
            predictedLevel: "elevated",
            confidence: 0.78,
            timestamp: new Date().toISOString(),
            horizon: "30"
          },
          // 60-day horizon predictions
          {
            nodeId: "node-1",
            predictedRisk: 25,
            predictedLevel: "stable",
            confidence: 0.85,
            timestamp: new Date().toISOString(),
            horizon: "60"
          },
          {
            nodeId: "node-2",
            predictedRisk: 90,
            predictedLevel: "high",
            confidence: 0.88,
            timestamp: new Date().toISOString(),
            horizon: "60"
          },
          // 90-day horizon predictions
          {
            nodeId: "node-1",
            predictedRisk: 65,
            predictedLevel: "elevated",
            confidence: 0.75,
            timestamp: new Date().toISOString(),
            horizon: "90"
          },
          {
            nodeId: "node-2",
            predictedRisk: 10,
            predictedLevel: "stable",
            confidence: 0.90,
            timestamp: new Date().toISOString(),
            horizon: "90"
          }
        ])
      });
    }, 600);
  });
}

/**
 * Filters prediction data based on the selected prediction horizon.
 * 
 * NOTE: The horizon field (e.g. '30', '60', '90') is a temporary
 * development field used until the backend API contract and GNN prediction schema
 * are finalized by Santanu and Shivangi.
 * 
 * If a prediction lacks a horizon field, it is treated as matching the "30" horizon
 * to preserve backwards compatibility.
 * 
 * @param {Array} predictions - Sanitized predictions array
 * @param {string} horizon - The selected prediction horizon (e.g. 'current', '30', '60', '90')
 * @returns {Array} Predictions filtered for the selected horizon
 */
export function filterPredictionsByHorizon(predictions, horizon) {
  if (!Array.isArray(predictions)) return [];

  // 'current' represents the current state of the supply chain with no prediction overlay.
  // Returning an empty array ensures that no prediction overlay is rendered, testing the
  // predictions = [] boundary scenario gracefully.
  if (horizon === 'current') return [];

  const targetHorizon = horizon !== undefined && horizon !== null ? String(horizon) : '30';

  return predictions.filter(p => {
    // 1. Skip null/undefined or non-object entries
    if (!p || typeof p !== 'object') return false;

    // 2. Skip entries missing nodeId or with invalid/empty nodeId
    if (p.nodeId === undefined || p.nodeId === null || p.nodeId === '') return false;

    // 3. Normalize horizon values. Fallback to '30' if no horizon field is specified.
    const h = p.horizon !== undefined && p.horizon !== null ? String(p.horizon) : '30';
    return h === targetHorizon;
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
  if (!prediction || typeof prediction !== 'object') return 'unknown';
  
  // Normalize based on temporary mock predictedLevel fields.
  // This centralizes prediction interpretation.
  const level = prediction.predictedLevel;
  if (!level || typeof level !== 'string') return 'unknown';

  const normalized = level.toLowerCase();
  if (normalized === 'high') {
    return 'high';
  }
  if (normalized === 'elevated' || normalized === 'medium') {
    return 'medium';
  }
  if (normalized === 'stable' || normalized === 'low') {
    return 'low';
  }
  return 'unknown';
}

/**
 * Resolves and normalizes the risk state of a node.
 * Returns one of: 'high', 'medium', 'low', 'unknown'
 * 
 * @param {Object} node - The node object
 * @returns {string} One of: 'high', 'medium', 'low', 'unknown'
 */
export function getNodeRiskState(node) {
  if (!node) return 'unknown';

  // 1. If prediction data is available, it takes precedence.
  if (node.prediction) {
    return getRiskState(node.prediction);
  }

  // 2. Fall back to static risk fields on the node (e.g. from database or mock).
  const rawRisk = node.risk || node.properties?.risk;
  if (!rawRisk) return 'unknown';

  const normalized = String(rawRisk).toLowerCase();
  if (normalized === 'high') {
    return 'high';
  }
  if (normalized === 'medium' || normalized === 'elevated' || normalized === 'low') {
    // Note: raw node risk 'low' in Yashaswini's layout represents "Elevated"
    return 'medium';
  }
  if (normalized === 'stable' || normalized === 'none') {
    return 'low';
  }
  return 'unknown';
}

/**
 * Creates and connects a real-time prediction WebSocket stream.
 * 
 * Flow:
 *   Confirmed WebSocket endpoint (/api/v1/ws)
 *     ↓
 *   WebSocket transport
 *     ↓
 *   Raw message delivery
 *     ↓
 *   Prediction service boundary
 *     ↓
 *   Future finalized GNN payload
 * 
 * This boundary is payload-agnostic and safely delivers raw messages to
 * registered callbacks without assuming unfinalized GNN schemas.
 * 
 * @param {Object} [handlers]
 * @param {Function} [handlers.onMessage] - Receives raw parsed WebSocket messages
 * @param {Function} [handlers.onError] - Receives transport error events
 * @param {Function} [handlers.onStatusChange] - Receives status string ('connecting'|'connected'|'disconnected'|'error')
 * @param {Function} [handlers.onConnected] - Receives connection metadata payload
 * @param {Object} [options] - Additional transport options (e.g. autoReconnect, url)
 * @returns {Object} Transport control object { transport, disconnect, send, ping }
 */
export function connectPredictionStream(handlers = {}, options = {}) {
  const transport = createWebSocketTransport(options);

  if (handlers.onMessage) {
    transport.on('message', handlers.onMessage);
  }
  if (handlers.onError) {
    transport.on('error', handlers.onError);
  }
  if (handlers.onStatusChange) {
    transport.on('status', handlers.onStatusChange);
  }
  if (handlers.onConnected) {
    transport.on('connected', handlers.onConnected);
  }

  transport.connect();

  return {
    transport,
    disconnect: () => transport.disconnect(),
    send: (msg) => transport.send(msg),
    ping: (data) => transport.ping(data),
    getStatus: () => transport.getStatus(),
  };
}
