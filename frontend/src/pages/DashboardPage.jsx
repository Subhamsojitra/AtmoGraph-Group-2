import React from 'react';
import GraphCanvas from '../graph/GraphCanvas';

/**
 * DashboardPage Container Scaffold
 * 
 * High-level layout shell composing page and viewport boundaries.
 * No layout elements (Header/Sidebar) are defined today.
 */
export default function DashboardPage() {
  return (
    <div className="dashboard-page-container">
      <h2>AtmoGraph Dashboard Page</h2>
      <GraphCanvas />
    </div>
  );
}
