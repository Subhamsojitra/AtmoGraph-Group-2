import React, { useState, useEffect, useRef } from "react";
import {
  Menu, X, Search, ZoomIn, ZoomOut, Maximize2, SlidersHorizontal, Hand,
  LayoutGrid, Bell, AlertTriangle, Activity, MapPin, Boxes, Clock,
  ChevronRight, Radio, FileText, Database, Settings, Wind,
} from "lucide-react";
import GraphCanvas from "../graph/GraphCanvas"; // TODO: confirm this relative path matches
                                                 // DashboardPage.jsx's real location in the repo
import "../DashboardPage.css";

/* ============================================================
   AtmoGraph — Week 1 Frontend Foundation (Yashaswini's scope)

   Changes for this PR, per teammate review:
   - Removed the local GraphCanvas placeholder + its simulated
     loading/error lifecycle. This now renders the shared
     src/graph/GraphCanvas.jsx directly; that component owns its
     own loading/empty/error/ready states during integration.
   - Removed Tailwind. All layout/spacing now lives in
     DashboardPage.css; per-instance colors (from TOKENS) stay as
     inline styles since they were never Tailwind classes.
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

const RISK_OPTIONS = ["All", "High", "Medium", "Low"];
const TYPE_OPTIONS = ["Supplier", "Warehouse", "Port"];

// Local mock dataset — stands in for a real node-search API in Week 1.
const MOCK_NODES = [
  {
    id: "rtm-04",
    name: "Rotterdam Port Terminal 4",
    type: "Port",
    region: "Europe · Netherlands",
    risk: "high",
    connections: 214,
    updated: "2 min ago",
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

function SearchBar({ onSelect }) {
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
    ? MOCK_NODES.filter((n) => n.name.toLowerCase().includes(query.trim().toLowerCase()))
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

function FilterPanel() {
  const [open, setOpen] = useState(false);
  const [risk, setRisk] = useState("All");
  const [types, setTypes] = useState([]);
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

function ControlsBar({ onSelectNode }) {
  const [panActive, setPanActive] = useState(false);

  return (
    <div className="controls-bar" style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}>
      <SearchBar onSelect={onSelectNode} />

      <div className="controls-zoom-group">
        <button
          onClick={() => setPanActive((p) => !p)}
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
        {[
          { icon: ZoomIn, label: "Zoom in" },
          { icon: ZoomOut, label: "Zoom out" },
          { icon: Maximize2, label: "Fit to screen" },
        ].map(({ icon: Icon, label }) => (
          <button key={label} title={label} aria-label={label} className="zoom-btn" style={{ color: TOKENS.textDim, border: `1px solid ${TOKENS.border}` }}>
            <Icon size={15} />
          </button>
        ))}
      </div>

      <FilterPanel />
    </div>
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
        <p className="node-title" style={{ color: TOKENS.text }}>{node.name}</p>
        <p className="node-subtitle" style={{ color: TOKENS.textDim }}>{node.type} · {node.region}</p>
      </div>

      <RiskBadge level={node.risk} />

      <div className="node-note-box" style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}`, color: TOKENS.textDim }}>
        {node.note}
      </div>

      <dl className="node-meta-list">
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Boxes size={13} /> Connections</dt>
          <dd className="node-meta-value" style={{ color: TOKENS.text }}>{node.connections}</dd>
        </div>
        <div className="node-meta-row">
          <dt className="node-meta-label" style={{ color: TOKENS.textDim }}><Clock size={13} /> Last updated</dt>
          <dd style={{ color: TOKENS.text }}>{node.updated}</dd>
        </div>
      </dl>
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

  return (
    <div className="app-shell" style={{ backgroundColor: TOKENS.bg, fontFamily: "'Inter', sans-serif", opacity: fontsReady ? 1 : 0 }}>
      <Header onMenuClick={() => setMobileNavOpen(true)} />

      <div className="dashboard-body">
        <Sidebar mobileOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

        <main className="main-content">
          <ControlsBar onSelectNode={setSelectedNode} />
          <div className="graph-canvas-wrapper">
            {/* Shared graph component — owns its own loading/empty/error/ready
                states during integration. Wire its node-click callback (once
                it exposes one) to setSelectedNode to replace the search-only
                selection path below. */}
            <GraphCanvas />
          </div>
        </main>

        <NodeDetailsPanel node={selectedNode} />
      </div>

      <NodeDetailsSheet node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  );
}
