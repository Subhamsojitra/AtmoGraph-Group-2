import React, { useState, useEffect, useRef } from "react";
import {
  Menu, X, Search, ZoomIn, ZoomOut, Maximize2, SlidersHorizontal, Hand,
  LayoutGrid, Bell, AlertTriangle, Activity, MapPin, Boxes, Clock,
  ChevronRight, Radio, FileText, Database, Settings, Wind, Flag, History, Tag,
} from "lucide-react";
import GraphCanvas from "../graph/GraphCanvas";
import { getGraphData } from "../services/graphService";
import { getPredictionData } from "../services/predictionService";
import "../DashboardPage.css";

/* ============================================================
   AtmoGraph — Week 2 (Yashaswini's scope)

   Added this pass:
   - Node click interaction UI: a selection chip in the controls
     bar reflects whatever node is selected (from search today,
     from the real graph once GraphCanvas exposes a click callback)
     with a one-click way to clear it.
   - Zoom/pan controls UI: zoom now has real local state (25–200%,
     a readout, a reset-to-100% "fit" action) and an onZoomChange
     callback so the architecture side can wire it to the actual
     graph transform during integration.
   - Selected-node styling: nodeStates.js + the "Graph node states"
     block in DashboardPage.css define the hover/selected/risk
     classes GraphCanvas should apply to its own nodes.
   - Supplier/node details panel: added category, tier, and lead
     time, plus a presentational actions row.
   - Search/filter interface: filters now actually narrow search
     results, with removable filter chips summarizing what's active.
   ============================================================ */

const TOKENS = {
  bg: "#0A0E13",
  surface: "#10151C",
  surface2: "#161D26",
  border: "#212A34",
  text: "#E7EDF3",
  textDim: "#8A96A3",
  flow: "#35D0BA",
  brand: "#6E8CFF",
  riskLow: "#F5B84C",
  riskHigh: "#E5484D",
};

const NAV_ITEMS = [
  { icon: LayoutGrid, label: "Network Map", active: true },
  { icon: AlertTriangle, label: "Disruption Alerts" },
  { icon: FileText, label: "Reports" },
  { icon: Database, label: "Data Sources" },
  { icon: Settings, label: "Settings" },
];

const RISK_OPTIONS = ["All", "High", "Elevated", "Stable"];
const RISK_VALUE_MAP = { All: "all", High: "high", Elevated: "low", Stable: "none" };
const TYPE_OPTIONS = ["Supplier", "Warehouse", "Port"];

// Local mock dataset — stands in for a real node-search API in Week 1/2.
const MOCK_NODES = [
  {
    id: "rtm-04",
    name: "Rotterdam Port Terminal 4",
    type: "Port",
    region: "Europe · Netherlands",
    risk: "high",
    connections: 214,
    updated: "2 min ago",
    category: "Logistics Hub",
    tier: "Tier 1",
    leadTime: "N/A",
    note: "Linked to a reported dockworker strike. Downstream electronics shipments to North America are the primary exposure.",
  },
  {
    id: "szn-12",
    name: "Shenzhen Manufacturing Plant 12",
    type: "Supplier",
    region: "Asia · China",
    risk: "low",
    connections: 138,
    updated: "9 min ago",
    category: "Electronics",
    tier: "Tier 1",
    leadTime: "18 days",
    note: "Minor customs delay reported. No production impact expected if cleared within 48 hours.",
  },
  {
    id: "lax-wh2",
    name: "Los Angeles Distribution Center",
    type: "Warehouse",
    region: "North America · USA",
    risk: "none",
    connections: 96,
    updated: "24 min ago",
    category: "Distribution",
    tier: "Tier 2",
    leadTime: "3 days",
    note: "Operating normally. No upstream disruptions currently mapped to this node.",
  },
  {
    id: "ham-07",
    name: "Hamburg Freight Yard 7",
    type: "Port",
    region: "Europe · Germany",
    risk: "low",
    connections: 172,
    updated: "17 min ago",
    category: "Logistics Hub",
    tier: "Tier 1",
    leadTime: "N/A",
    note: "Elevated congestion from rerouted Rotterdam traffic. Monitoring for further backlog.",
  },
];

function ContourBackground() {
  return (
    <svg className="contour-bg" viewBox="0 0 800 120" preserveAspectRatio="none">
      <defs>
        <style>{`
          .contour { animation: drift 22s linear infinite; }
          .contour:nth-child(2) { animation-duration: 30s; animation-direction: reverse; }
          .contour:nth-child(3) { animation-duration: 26s; }
          @keyframes drift {
            from { transform: translateX(0); }
            to { transform: translateX(-80px); }
          }
          @media (prefers-reduced-motion: reduce) {
            .contour { animation: none; }
          }
        `}</style>
      </defs>
      {[20, 55, 90].map((y, i) => (
        <path
          key={i}
          className="contour"
          d={`M-80 ${y} C 60 ${y - 18}, 140 ${y + 18}, 280 ${y} S 500 ${y - 18}, 640 ${y} S 860 ${y + 18}, 960 ${y}`}
          fill="none"
          stroke={TOKENS.flow}
          strokeWidth="1"
        />
      ))}
    </svg>
  );
}

function RiskBadge({ level }) {
  const map = {
    high: { color: TOKENS.riskHigh, label: "High risk" },
    low: { color: TOKENS.riskLow, label: "Elevated" },
    none: { color: TOKENS.flow, label: "Stable" },
  };
  const s = map[level] || map.none;
  return (
    <span
      className="risk-badge"
      style={{ backgroundColor: `${s.color}1A`, color: s.color, border: `1px solid ${s.color}40` }}
    >
      <span className="risk-dot" style={{ backgroundColor: s.color }} />
      {s.label}
    </span>
  );
}

function Header({ onMenuClick }) {
  return (
    <header className="header" style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}>
      <ContourBackground />
      <div className="header-left">
        <button
          onClick={onMenuClick}
          className="menu-toggle-btn"
          style={{ color: TOKENS.textDim }}
          aria-label="Toggle navigation"
        >
          <Menu size={20} />
        </button>
        <div className="brand-icon" style={{ backgroundColor: `${TOKENS.brand}22`, border: `1px solid ${TOKENS.brand}55` }}>
          <Wind size={18} style={{ color: TOKENS.brand }} />
        </div>
        <div>
          <h1 className="brand-text-title" style={{ color: TOKENS.text, fontFamily: "'Space Grotesk', sans-serif" }}>
            AtmoGraph
          </h1>
          <p className="brand-text-subtitle" style={{ color: TOKENS.textDim }}>
            Supply Chain Ripple Monitor
          </p>
        </div>
      </div>

      <div className="header-right">
        <div
          className="live-pill"
          style={{ backgroundColor: `${TOKENS.flow}14`, border: `1px solid ${TOKENS.flow}40`, color: TOKENS.flow }}
        >
          <Radio size={12} />
          Live feed synced
        </div>
        <button className="icon-btn" style={{ color: TOKENS.textDim }} aria-label="Notifications, 1 unread">
          <Bell size={18} />
          <span className="notif-dot" style={{ backgroundColor: TOKENS.riskHigh }} aria-hidden="true" />
        </button>
        <div
          className="avatar"
          style={{ backgroundColor: TOKENS.surface2, color: TOKENS.text, border: `1px solid ${TOKENS.border}` }}
          aria-label="Account: Yashaswini B"
          role="img"
        >
          YB
        </div>
      </div>
    </header>
  );
}

function Sidebar({ mobileOpen, onClose }) {
  const stats = [
    { icon: AlertTriangle, label: "Active disruptions", value: "3", color: TOKENS.riskHigh },
    { icon: Boxes, label: "Nodes at risk", value: "47", color: TOKENS.riskLow },
    { icon: Activity, label: "Nodes monitored", value: "6,204", color: TOKENS.flow },
  ];

  const body = (
    <>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.label}
            className="nav-item"
            style={{
              backgroundColor: item.active ? `${TOKENS.brand}1A` : "transparent",
              color: item.active ? TOKENS.text : TOKENS.textDim,
              border: item.active ? `1px solid ${TOKENS.brand}40` : "1px solid transparent",
            }}
          >
            <item.icon size={17} />
            {item.label}
            {item.active && <ChevronRight size={14} className="nav-item-chevron" />}
          </button>
        ))}
      </nav>

      <div className="stats-section">
        <p className="stats-label" style={{ color: TOKENS.textDim }}>Network snapshot</p>
        <div className="stats-list">
          {stats.map((s) => (
            <div
              key={s.label}
              className="stat-item"
              style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}` }}
            >
              <s.icon size={15} style={{ color: s.color }} />
              <span className="stat-label" style={{ color: TOKENS.textDim }}>{s.label}</span>
              <span className="stat-value" style={{ color: TOKENS.text }}>{s.value}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );

  return (
    <>
      <aside className="sidebar" style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}>
        {body}
      </aside>

      {mobileOpen && (
        <div className="mobile-drawer-overlay">
          <div className="mobile-drawer-backdrop" onClick={onClose} />
          <aside className="mobile-drawer-panel" style={{ backgroundColor: TOKENS.surface, borderRight: `1px solid ${TOKENS.border}` }}>
            <div className="mobile-drawer-header">
              <span style={{ fontSize: 14, fontWeight: 500, color: TOKENS.text }}>Menu</span>
              <button onClick={onClose} style={{ color: TOKENS.textDim }} aria-label="Close menu">
                <X size={18} />
              </button>
            </div>
            {body}
          </aside>
        </div>
      )}
    </>
  );
}

function SearchBar({ onSelect, risk, types }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) setFocused(false);
    }
    function handleKeyDown(e) {
      if (e.key === "Escape") setFocused(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const results = query.trim()
    ? MOCK_NODES.filter((n) => {
        const matchesQuery = n.name.toLowerCase().includes(query.trim().toLowerCase());
        const riskValue = RISK_VALUE_MAP[risk];
        const matchesRisk = riskValue === "all" || n.risk === riskValue;
        const matchesType = types.length === 0 || types.includes(n.type);
        return matchesQuery && matchesRisk && matchesType;
      })
    : [];
  const showDropdown = focused && query.trim().length > 0;

  return (
    <div ref={containerRef} className="search-container">
      <div className="search-box" style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${focused ? TOKENS.brand + "80" : TOKENS.border}` }}>
        <Search size={15} style={{ color: TOKENS.textDim }} aria-hidden="true" />
        <input
          id="node-search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setFocused(true)}
          placeholder="Search nodes, suppliers, routes…"
          aria-label="Search nodes, suppliers, and routes"
          role="combobox"
          aria-expanded={showDropdown}
          aria-controls="node-search-results"
          aria-autocomplete="list"
          className="search-input"
          style={{ color: TOKENS.text }}
        />
        {query && (
          <button onClick={() => setQuery("")} className="search-clear-btn" style={{ color: TOKENS.textDim }} aria-label="Clear search">
            <X size={14} />
          </button>
        )}
      </div>

      {showDropdown && (
        <div
          id="node-search-results"
          role="listbox"
          aria-label="Search results"
          className="search-dropdown"
          style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}` }}
        >
          {results.length === 0 ? (
            <p className="search-no-results" style={{ color: TOKENS.textDim }} role="status">
              No nodes match “{query.trim()}”.
            </p>
          ) : (
            results.map((n) => (
              <button
                key={n.id}
                role="option"
                aria-selected="false"
                onClick={() => { onSelect(n); setQuery(n.name); setFocused(false); }}
                className="search-result-item"
              >
                <span>
                  <span className="search-result-name" style={{ color: TOKENS.text }}>{n.name}</span>
                  <span className="search-result-region" style={{ color: TOKENS.textDim }}>{n.region}</span>
                </span>
                <RiskBadge level={n.risk} />
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function FilterPanel({ risk, setRisk, types, setTypes }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) setOpen(false);
    }
    function handleKeyDown(e) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const toggleType = (t) => {
    setTypes((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]));
  };

  const activeCount = (risk !== "All" ? 1 : 0) + types.length;

  return (
    <div ref={containerRef} className="filter-container">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="true"
        aria-expanded={open}
        aria-controls="filter-panel"
        className="filter-toggle-btn"
        style={{
          color: activeCount ? TOKENS.brand : TOKENS.textDim,
          border: `1px solid ${activeCount ? TOKENS.brand + "55" : TOKENS.border}`,
          backgroundColor: activeCount ? `${TOKENS.brand}14` : "transparent",
        }}
      >
        <SlidersHorizontal size={14} aria-hidden="true" />
        Filter
        {activeCount > 0 && (
          <span className="filter-badge-count" style={{ backgroundColor: TOKENS.brand, color: TOKENS.bg }}>
            {activeCount}
          </span>
        )}
      </button>

      {open && (
        <div id="filter-panel" className="filter-popover" style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}` }}>
          <fieldset className="filter-fieldset">
            <legend className="filter-legend" style={{ color: TOKENS.textDim }}>Risk</legend>
            <div className="filter-options">
              {RISK_OPTIONS.map((r) => (
                <label key={r} className="filter-option-label" style={{ color: TOKENS.text }}>
                  <input
                    type="radio"
                    name="risk"
                    checked={risk === r}
                    onChange={() => setRisk(r)}
                    style={{ accentColor: TOKENS.brand }}
                  />
                  {r}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="filter-fieldset">
            <legend className="filter-legend" style={{ color: TOKENS.textDim }}>Node type</legend>
            <div className="filter-options">
              {TYPE_OPTIONS.map((t) => (
                <label key={t} className="filter-option-label" style={{ color: TOKENS.text }}>
                  <input
                    type="checkbox"
                    checked={types.includes(t)}
                    onChange={() => toggleType(t)}
                    style={{ accentColor: TOKENS.brand }}
                  />
                  {t}
                </label>
              ))}
            </div>
          </fieldset>

          <div className="filter-actions" style={{ borderColor: TOKENS.border }}>
            <button onClick={() => { setRisk("All"); setTypes([]); }} className="filter-clear-btn" style={{ color: TOKENS.textDim }}>
              Clear
            </button>
            <button
              onClick={() => setOpen(false)}
              className="filter-apply-btn"
              style={{ backgroundColor: `${TOKENS.brand}22`, color: TOKENS.brand, border: `1px solid ${TOKENS.brand}55` }}
            >
              Apply
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function RiskLegend() {
  const items = [
    { label: "Stable", color: TOKENS.flow },
    { label: "Elevated", color: TOKENS.riskLow },
    { label: "High risk", color: TOKENS.riskHigh },
  ];
  return (
    <div className="risk-legend" aria-label="Risk color legend">
      {items.map((i) => (
        <span key={i.label} className="risk-legend-item" style={{ color: TOKENS.textDim }}>
          <span className="risk-legend-dot" style={{ backgroundColor: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

function ControlsBar({ 
  onSelectNode, 
  selectedNode, 
  onClearSelection, 
  zoom, 
  panActive, 
  onPanActiveChange,
  onZoomIn,
  onZoomOut,
  onResetZoom
}) {
  const [risk, setRisk] = useState("All");
  const [types, setTypes] = useState([]);

  const toggleType = (t) => setTypes((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]));

  const activeFilterChips = [
    ...(risk !== "All" ? [{ key: "risk", label: risk, clear: () => setRisk("All") }] : []),
    ...types.map((t) => ({ key: `type-${t}`, label: t, clear: () => toggleType(t) })),
  ];

  return (
    <>
      <div className="controls-bar" style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}>
        <SearchBar onSelect={onSelectNode} risk={risk} types={types} />

        {selectedNode && (
          <div className="selection-chip" style={{ backgroundColor: `${TOKENS.brand}14`, border: `1px solid ${TOKENS.brand}40`, color: TOKENS.brand }}>
            {selectedNode.name || selectedNode.label || selectedNode.id}
            <button onClick={onClearSelection} aria-label="Clear selected node" style={{ color: TOKENS.brand }}>
              <X size={13} />
            </button>
          </div>
        )}

        <div className="controls-zoom-group">
          <button
            onClick={() => onPanActiveChange(!panActive)}
            title="Pan"
            aria-label="Toggle pan mode"
            aria-pressed={panActive}
            className="pan-btn"
            style={{
              color: panActive ? TOKENS.brand : TOKENS.textDim,
              border: `1px solid ${panActive ? TOKENS.brand + "55" : TOKENS.border}`,
              backgroundColor: panActive ? `${TOKENS.brand}14` : "transparent",
            }}
          >
            <Hand size={15} />
          </button>
          <button
            onClick={onZoomOut}
            title="Zoom out"
            aria-label="Zoom out"
            className="zoom-btn"
            style={{ color: TOKENS.textDim, border: `1px solid ${TOKENS.border}` }}
          >
            <ZoomOut size={15} />
          </button>
          <span className="zoom-readout" style={{ color: TOKENS.text }} aria-live="polite">{zoom}%</span>
          <button
            onClick={onZoomIn}
            title="Zoom in"
            aria-label="Zoom in"
            className="zoom-btn"
            style={{ color: TOKENS.textDim, border: `1px solid ${TOKENS.border}` }}
          >
            <ZoomIn size={15} />
          </button>
          <button
            onClick={onResetZoom}
            title="Fit to screen (reset zoom)"
            aria-label="Fit to screen"
            className="zoom-btn"
            style={{ color: TOKENS.textDim, border: `1px solid ${TOKENS.border}` }}
          >
            <Maximize2 size={15} />
          </button>
        </div>

        <FilterPanel risk={risk} setRisk={setRisk} types={types} setTypes={setTypes} />
      </div>

      <div className="active-filters-row" style={{ borderBottom: `1px solid ${TOKENS.border}`, backgroundColor: TOKENS.surface }}>
        {activeFilterChips.map((chip) => (
          <span key={chip.key} className="filter-chip" style={{ backgroundColor: `${TOKENS.brand}14`, border: `1px solid ${TOKENS.brand}40`, color: TOKENS.brand }}>
            {chip.label}
            <button onClick={chip.clear} aria-label={`Remove ${chip.label} filter`} style={{ color: TOKENS.brand }}>
              <X size={11} />
            </button>
          </span>
        ))}
        <RiskLegend />
      </div>
    </>
  );
}

function NodeDetailsBody({ node }) {
  if (!node) {
    return (
      <div className="node-empty-state">
        <MapPin size={22} style={{ color: TOKENS.textDim }} />
        <p className="node-empty-title" style={{ color: TOKENS.text }}>No node selected</p>
        <p className="node-empty-desc" style={{ color: TOKENS.textDim }}>
          Search for a node above, or click one on the map once the graph is connected.
        </p>
      </div>
    );
  }

  return (
    <div className="node-details-content">
      <div>
        <p className="node-title" style={{ color: TOKENS.text }}>{node.name || node.label || node.id}</p>
        <p className="node-subtitle" style={{ color: TOKENS.textDim }}>{node.type || 'default'} · {node.region || node.properties?.region || 'Unknown region'}</p>
      </div>

      <RiskBadge 
        level={
          node.risk || 
          node.properties?.risk || 
          (node.prediction ? (node.prediction.predictedLevel === 'high' ? 'high' : (node.prediction.predictedLevel === 'elevated' ? 'low' : 'none')) : 'none')
        } 
      />

      <div className="node-note-box" style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}`, color: TOKENS.textDim }}>
        {node.note || node.properties?.description || node.properties?.note || 'No description available.'}
      </div>

      <dl className="node-meta-list">
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Tag size={13} /> Category</dt>
          <dd style={{ color: TOKENS.text }}>{node.category || node.properties?.category || node.type || 'N/A'}</dd>
        </div>
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Boxes size={13} /> Tier</dt>
          <dd style={{ color: TOKENS.text }}>{node.tier || node.properties?.tier || 'N/A'}</dd>
        </div>
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><History size={13} /> Lead time</dt>
          <dd style={{ color: TOKENS.text }}>{node.leadTime || node.properties?.leadTime || node.properties?.lead_time || 'N/A'}</dd>
        </div>
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Boxes size={13} /> Connections</dt>
          <dd className="node-meta-value" style={{ color: TOKENS.text }}>{node.connections !== undefined ? node.connections : (node.properties?.connections || 'N/A')}</dd>
        </div>
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Clock size={13} /> Last updated</dt>
          <dd style={{ color: TOKENS.text }}>{node.updated || node.properties?.updated || 'N/A'}</dd>
        </div>
        {node.prediction && (
          <>
            <div className="node-meta-row" style={{ borderTop: `1px dashed ${TOKENS.border}`, paddingTop: '8px', marginTop: '8px' }}>
              <dt className="node-meta-label" style={{ color: TOKENS.textDim, fontWeight: '600' }}>[Prediction Details]</dt>
              <dd style={{ color: TOKENS.textDim }}></dd>
            </div>
            <div className="node-meta-row">
              <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Activity size={13} /> Predicted Level</dt>
              <dd style={{ color: TOKENS.text }}>{node.prediction.predictedLevel || 'N/A'}</dd>
            </div>
            <div className="node-meta-row">
              <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Activity size={13} /> Predicted Risk</dt>
              <dd style={{ color: TOKENS.text }}>{node.prediction.predictedRisk !== undefined ? `${node.prediction.predictedRisk}%` : 'N/A'}</dd>
            </div>
            <div className="node-meta-row">
              <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Activity size={13} /> Confidence</dt>
              <dd style={{ color: TOKENS.text }}>{node.prediction.confidence !== undefined ? `${(node.prediction.confidence * 100).toFixed(0)}%` : 'N/A'}</dd>
            </div>
          </>
        )}
      </dl>

      <div className="node-actions-row">
        <button className="node-action-btn" style={{ color: TOKENS.riskLow, border: `1px solid ${TOKENS.riskLow}45`, backgroundColor: `${TOKENS.riskLow}14` }}>
          <Flag size={12} style={{ marginRight: 4, verticalAlign: "-2px" }} />
          Flag for review
        </button>
        <button className="node-action-btn" style={{ color: TOKENS.textDim, border: `1px solid ${TOKENS.border}`, backgroundColor: "transparent" }}>
          <History size={12} style={{ marginRight: 4, verticalAlign: "-2px" }} />
          View shipment history
        </button>
      </div>
    </div>
  );
}

function NodeDetailsPanel({ node }) {
  return (
    <aside className="node-panel" style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}>
      <h2 className="node-panel-heading" style={{ color: TOKENS.textDim }}>Node details</h2>
      <NodeDetailsBody node={node} />
    </aside>
  );
}

function NodeDetailsSheet({ node, onClose }) {
  if (!node) return null;
  return (
    <div className="node-sheet-overlay">
      <div className="node-sheet-backdrop" onClick={onClose} />
      <div className="node-sheet-panel" style={{ backgroundColor: TOKENS.surface, borderTop: `1px solid ${TOKENS.border}` }}>
        <div className="node-sheet-handle" style={{ backgroundColor: TOKENS.border }} />
        <div className="node-sheet-header">
          <h2 className="node-panel-heading" style={{ color: TOKENS.textDim, margin: 0 }}>Node details</h2>
          <button onClick={onClose} style={{ color: TOKENS.textDim }} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <NodeDetailsBody node={node} />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [fontsReady, setFontsReady] = useState(false);

  // Zoom and pan active states lifted from ControlsBar
  const [zoom, setZoom] = useState(100);
  const [panActive, setPanActive] = useState(false);
  const graphRef = useRef(null);

  // Retrieve development mode parameter from URL (?mode=mock|backend|large)
  const queryMode = new URLSearchParams(window.location.search).get('mode') || 'mock';

  // Network graph states from Shubham's version
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Prediction states for future integration
  const [predictions, setPredictions] = useState([]);
  const [_predictionsLoading, setPredictionsLoading] = useState(false);
  const [_predictionsError, setPredictionsError] = useState(null);

  useEffect(() => {
    let active = true;

    setLoading(true);
    setError(null);
    setData(null);
    setSelectedNode(null);
    setPredictions([]);
    setPredictionsLoading(true);
    setPredictionsError(null);

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

    getPredictionData(queryMode)
      .then((res) => {
        if (!active) return;
        setPredictions(res?.predictions || []);
        setPredictionsLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        console.warn("Failed to load predictions:", err);
        setPredictionsError(err.message || 'An error occurred while loading predictions.');
        setPredictionsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [queryMode]);

  useEffect(() => {
    const link = document.createElement("style");
    link.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
      :focus-visible { outline: 2px solid ${TOKENS.brand}; outline-offset: 2px; }
    `;
    document.head.appendChild(link);
    setFontsReady(true);
    return () => document.head.removeChild(link);
  }, []);

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key !== "Escape") return;
      if (selectedNode) setSelectedNode(null);
      else if (mobileNavOpen) setMobileNavOpen(false);
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedNode, mobileNavOpen]);

  const handleZoomIn = () => {
    graphRef.current?.zoomIn();
  };

  const handleZoomOut = () => {
    graphRef.current?.zoomOut();
  };

  const handleResetZoom = () => {
    graphRef.current?.resetZoom();
  };

  return (
    <div className="app-shell" style={{ backgroundColor: TOKENS.bg, fontFamily: "'Inter', sans-serif", opacity: fontsReady ? 1 : 0 }}>
      <Header onMenuClick={() => setMobileNavOpen(true)} />

      <div className="dashboard-body">
        <Sidebar mobileOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

        <main className="main-content">
          <ControlsBar
            onSelectNode={setSelectedNode}
            selectedNode={selectedNode}
            onClearSelection={() => setSelectedNode(null)}
            zoom={zoom}
            panActive={panActive}
            onPanActiveChange={setPanActive}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onResetZoom={handleResetZoom}
          />
          <div className="graph-canvas-wrapper" style={{ flex: 1, minHeight: 0, position: 'relative' }}>
            {loading && (
              <div
                className="graph-loading-overlay"
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: `1px solid ${TOKENS.border}`,
                  borderRadius: '6px',
                  background: TOKENS.surface,
                  color: TOKENS.text,
                  fontSize: '16px',
                  zIndex: 10
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div
                    style={{
                      width: '20px',
                      height: '20px',
                      border: `3px solid ${TOKENS.border}`,
                      borderTop: `3px solid ${TOKENS.brand}`,
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
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid #feb2b2',
                  borderRadius: '6px',
                  background: 'rgba(254, 178, 178, 0.1)',
                  color: '#c53030',
                  padding: '20px',
                  textAlign: 'center',
                  zIndex: 10
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
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: `1px solid ${TOKENS.border}`,
                  borderRadius: '6px',
                  background: TOKENS.surface,
                  color: TOKENS.text,
                  padding: '20px',
                  textAlign: 'center',
                  zIndex: 10
                }}
              >
                <div style={{ fontSize: '24px', marginBottom: '8px' }}>🔍</div>
                <div style={{ fontWeight: '600', marginBottom: '4px' }}>No Graph Data Available ({queryMode} mode)</div>
                <div style={{ fontSize: '14px' }}>The database query returned zero nodes or links.</div>
              </div>
            )}

            {!loading && !error && data && data.nodes.length > 0 && (
              <GraphCanvas 
                ref={graphRef}
                data={data} 
                selectedNodeId={selectedNode?.id} 
                onNodeClick={setSelectedNode} 
                predictions={predictions}
                onZoomLevelChange={setZoom}
                panActive={panActive}
              />
            )}
          </div>
        </main>

        <NodeDetailsPanel node={selectedNode} />
      </div>

      <NodeDetailsSheet node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  );
}
