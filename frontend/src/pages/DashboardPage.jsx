import React, { useState, useEffect } from 'react';
import GraphCanvas from '../graph/GraphCanvas';
import { getGraphData } from '../services/graphService';

/**
 * DashboardPage Container Scaffold
 * 
 * High-level layout shell composing page and viewport boundaries.
 * Manages loading, success, empty, and error states for the graph data.
 */
export default function DashboardPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  // Retrieve development mode parameter from URL (?mode=mock|backend|large)
  const queryMode = new URLSearchParams(window.location.search).get('mode') || 'mock';

  useEffect(() => {
    let active = true;
    
    setLoading(true);
    setError(null);
    setData(null);
    setSelectedNode(null);

    getGraphData(queryMode)
      .then((res) => {
        if (!active) return;
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        setError(err.message || 'An error occurred while loading graph data.');
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [queryMode]);

  return (
    <div className="dashboard-page-container" style={{ padding: '20px', boxSizing: 'border-box' }}>
      <h2 style={{ marginBottom: '20px' }}>AtmoGraph Dashboard Page</h2>

      <div className="graph-container-wrapper" style={{ margin: '0 auto', maxWidth: '1000px', width: '100%' }}>
        {/* Isolated selected node details testing overlay */}
        {selectedNode && (
          <div 
            className="selected-node-panel"
            style={{
              background: 'var(--social-bg)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              padding: '12px 16px',
              marginBottom: '20px',
              textAlign: 'left',
              boxShadow: 'var(--shadow)',
              position: 'relative'
            }}
          >
            <button 
              onClick={() => setSelectedNode(null)}
              style={{
                position: 'absolute',
                top: '12px',
                right: '16px',
                background: 'none',
                border: 'none',
                color: 'var(--text)',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: 'bold'
              }}
            >
              ✕
            </button>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-h)' }}>Selected Node Details (Testing Overlay)</h4>
            <div style={{ fontSize: '14px', lineHeight: '1.5' }}>
              <div><strong>ID:</strong> {selectedNode.id}</div>
              <div><strong>Label:</strong> {selectedNode.label || selectedNode.id}</div>
              <div><strong>Type:</strong> {selectedNode.type || 'default'}</div>
              {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
                <div style={{ marginTop: '8px' }}>
                  <strong>Properties:</strong>
                  <pre style={{ margin: '4px 0 0 0', background: 'var(--code-bg)', padding: '6px', borderRadius: '4px', overflowX: 'auto', fontSize: '12px' }}>
                    {JSON.stringify(selectedNode.properties, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}

        {loading && (
          <div 
            className="graph-loading-overlay"
            style={{
              height: '500px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              background: 'var(--bg)',
              color: 'var(--text)',
              fontSize: '16px'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div 
                style={{
                  width: '20px',
                  height: '20px',
                  border: '3px solid var(--border)',
                  borderTop: '3px solid var(--accent)',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }} 
              />
              Loading graph data ({queryMode} mode)...
            </div>
            <style>{`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        )}

        {error && (
          <div 
            className="graph-error-overlay"
            style={{
              height: '500px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid #feb2b2',
              borderRadius: '6px',
              background: 'rgba(254, 178, 178, 0.1)',
              color: '#c53030',
              padding: '20px',
              textAlign: 'center'
            }}
          >
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>⚠️</div>
            <div style={{ fontWeight: '600', marginBottom: '4px' }}>Failed to Load Graph ({queryMode} mode)</div>
            <div style={{ fontSize: '14px' }}>{error}</div>
          </div>
        )}

        {!loading && !error && data && (data.nodes.length === 0) && (
          <div 
            className="graph-empty-overlay"
            style={{
              height: '500px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              background: 'var(--bg)',
              color: 'var(--text)',
              padding: '20px',
              textAlign: 'center'
            }}
          >
            <div style={{ fontSize: '24px', marginBottom: '8px' }}>🔍</div>
            <div style={{ fontWeight: '600', marginBottom: '4px' }}>No Graph Data Available ({queryMode} mode)</div>
            <div style={{ fontSize: '14px' }}>The database query returned zero nodes or links.</div>
          </div>
        )}

        {!loading && !error && data && data.nodes.length > 0 && (
          <GraphCanvas data={data} onNodeClick={setSelectedNode} />
        )}
      </div>
    </div>
  );
}


