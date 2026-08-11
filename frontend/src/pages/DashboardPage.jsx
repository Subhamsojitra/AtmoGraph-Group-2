import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Menu, X, Search, ZoomIn, ZoomOut, Maximize2, SlidersHorizontal, Hand,
  LayoutGrid, Bell, AlertTriangle, Activity, MapPin, Boxes, Clock,
  ChevronRight, Radio, FileText, Database, Settings, Wind, RefreshCw,
} from "lucide-react";

// Set to false once the real graph service is wired up — this only exists
// to demo the error/retry state without a backend.
const DEMO_SIMULATE_ERRORS = true;

/* ============================================================
   AtmoGraph — Week 1 Frontend Foundation (Yashaswini's scope)
   Dashboard layout, header/sidebar, controls bar (search + filter
   UI), node details panel (incl. mobile bottom sheet), loading/
   empty/error states, responsive foundation.

   Graph rendering, live data wiring, and the WebSocket/prediction
   pipeline are out of scope for Week 1 — that's the graph
   container placeholder below, owned by the architecture side.
   Search/filter here are UI-only per Week 1 scope: no backend
   query, just local mock data so the interaction shape is real.
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
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.16]"
      viewBox="0 0 800 120"
      preserveAspectRatio="none"
    >
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
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ backgroundColor: `${s.color}1A`, color: s.color, border: `1px solid ${s.color}40` }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: s.color }} />
      {s.label}
    </span>
  );
}

function Header({ onMenuClick }) {
  return (
    <header
      className="relative flex h-16 shrink-0 items-center justify-between overflow-hidden border-b px-4 sm:px-6"
      style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}
    >
      <ContourBackground />
      <div className="relative flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="rounded-md p-2 hover:bg-white/5 md:hidden"
          style={{ color: TOKENS.textDim }}
          aria-label="Toggle navigation"
        >
          <Menu size={20} />
        </button>
        <div
          className="flex h-9 w-9 items-center justify-center rounded-lg"
          style={{ backgroundColor: `${TOKENS.brand}22`, border: `1px solid ${TOKENS.brand}55` }}
        >
          <Wind size={18} style={{ color: TOKENS.brand }} />
        </div>
        <div className="leading-tight">
          <h1
            className="text-[15px] font-semibold tracking-tight"
            style={{ color: TOKENS.text, fontFamily: "'Space Grotesk', sans-serif" }}
          >
            AtmoGraph
          </h1>
          <p className="hidden text-xs sm:block" style={{ color: TOKENS.textDim }}>
            Supply Chain Ripple Monitor
          </p>
        </div>
      </div>

      <div className="relative flex items-center gap-2 sm:gap-4">
        <div
          className="hidden items-center gap-1.5 rounded-full px-3 py-1.5 text-xs sm:flex"
          style={{ backgroundColor: `${TOKENS.flow}14`, border: `1px solid ${TOKENS.flow}40`, color: TOKENS.flow }}
        >
          <Radio size={12} />
          Live feed synced
        </div>
        <button
          className="relative rounded-md p-2 hover:bg-white/5"
          style={{ color: TOKENS.textDim }}
          aria-label="Notifications, 1 unread"
        >
          <Bell size={18} />
          <span
            className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: TOKENS.riskHigh }}
            aria-hidden="true"
          />
        </button>
        <div
          className="flex h-8 w-8 items-center justify-center rounded-full text-xs font-medium"
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
      <nav className="flex flex-col gap-1 px-3">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.label}
            className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors"
            style={{
              backgroundColor: item.active ? `${TOKENS.brand}1A` : "transparent",
              color: item.active ? TOKENS.text : TOKENS.textDim,
              border: item.active ? `1px solid ${TOKENS.brand}40` : "1px solid transparent",
            }}
          >
            <item.icon size={17} />
            {item.label}
            {item.active && <ChevronRight size={14} className="ml-auto opacity-60" />}
          </button>
        ))}
      </nav>

      <div className="mt-6 px-3">
        <p className="mb-2 px-3 text-[11px] font-medium uppercase tracking-wider" style={{ color: TOKENS.textDim }}>
          Network snapshot
        </p>
        <div className="flex flex-col gap-2">
          {stats.map((s) => (
            <div
              key={s.label}
              className="flex items-center gap-3 rounded-lg px-3 py-2.5"
              style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}` }}
            >
              <s.icon size={15} style={{ color: s.color }} />
              <span className="text-xs" style={{ color: TOKENS.textDim }}>{s.label}</span>
              <span className="ml-auto font-mono text-sm font-medium" style={{ color: TOKENS.text }}>
                {s.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </>
  );

  return (
    <>
      <aside
        className="hidden w-60 shrink-0 flex-col overflow-y-auto border-r py-5 md:flex"
        style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}
      >
        {body}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={onClose} />
          <aside
            className="absolute left-0 top-0 flex h-full w-64 flex-col overflow-y-auto py-5"
            style={{ backgroundColor: TOKENS.surface, borderRight: `1px solid ${TOKENS.border}` }}
          >
            <div className="mb-4 flex items-center justify-between px-4">
              <span className="text-sm font-medium" style={{ color: TOKENS.text }}>Menu</span>
              <button onClick={onClose} style={{ color: TOKENS.textDim }}>
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
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setFocused(false);
      }
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
    <div ref={containerRef} className="relative min-w-[160px] flex-1">
      <div
        className="flex items-center gap-2 rounded-lg px-3 py-2"
        style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${focused ? TOKENS.brand + "80" : TOKENS.border}` }}
      >
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
          className="w-full bg-transparent text-sm outline-none placeholder:text-[13px]"
          style={{ color: TOKENS.text }}
        />
        {query && (
          <button onClick={() => setQuery("")} style={{ color: TOKENS.textDim }} aria-label="Clear search">
            <X size={14} />
          </button>
        )}
      </div>

      {showDropdown && (
        <div
          id="node-search-results"
          role="listbox"
          aria-label="Search results"
          className="absolute left-0 right-0 top-[calc(100%+6px)] z-30 max-h-64 overflow-y-auto rounded-lg py-1 shadow-lg"
          style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}` }}
        >
          {results.length === 0 ? (
            <p className="px-3 py-3 text-xs" style={{ color: TOKENS.textDim }} role="status">
              No nodes match “{query.trim()}”.
            </p>
          ) : (
            results.map((n) => (
              <button
                key={n.id}
                role="option"
                aria-selected="false"
                onClick={() => {
                  onSelect(n);
                  setQuery(n.name);
                  setFocused(false);
                }}
                className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left hover:bg-white/5 focus-visible:bg-white/5"
              >
                <span>
                  <span className="block text-xs font-medium" style={{ color: TOKENS.text }}>{n.name}</span>
                  <span className="block text-[11px]" style={{ color: TOKENS.textDim }}>{n.region}</span>
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
    <div ref={containerRef} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="true"
        aria-expanded={open}
        aria-controls="filter-panel"
        className="flex items-center gap-1.5 rounded-md px-3 py-2 text-xs font-medium"
        style={{
          color: activeCount ? TOKENS.brand : TOKENS.textDim,
          border: `1px solid ${activeCount ? TOKENS.brand + "55" : TOKENS.border}`,
          backgroundColor: activeCount ? `${TOKENS.brand}14` : "transparent",
        }}
      >
        <SlidersHorizontal size={14} aria-hidden="true" />
        Filter
        {activeCount > 0 && (
          <span
            className="flex h-4 w-4 items-center justify-center rounded-full text-[10px]"
            style={{ backgroundColor: TOKENS.brand, color: TOKENS.bg }}
          >
            {activeCount}
          </span>
        )}
      </button>

      {open && (
        <div
          id="filter-panel"
          className="absolute right-0 top-[calc(100%+6px)] z-30 w-56 rounded-lg p-3 shadow-lg"
          style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}` }}
        >
          <fieldset className="mb-3 border-0 p-0 m-0">
            <legend className="mb-2 text-[11px] font-medium uppercase tracking-wider" style={{ color: TOKENS.textDim }}>
              Risk
            </legend>
            <div className="flex flex-col gap-1.5">
              {RISK_OPTIONS.map((r) => (
                <label key={r} className="flex items-center gap-2 text-xs" style={{ color: TOKENS.text }}>
                  <input
                    type="radio"
                    name="risk"
                    checked={risk === r}
                    onChange={() => setRisk(r)}
                    className="accent-current"
                    style={{ color: TOKENS.brand }}
                  />
                  {r}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="mb-3 border-0 p-0 m-0">
            <legend className="mb-2 text-[11px] font-medium uppercase tracking-wider" style={{ color: TOKENS.textDim }}>
              Node type
            </legend>
            <div className="flex flex-col gap-1.5">
              {TYPE_OPTIONS.map((t) => (
                <label key={t} className="flex items-center gap-2 text-xs" style={{ color: TOKENS.text }}>
                  <input
                    type="checkbox"
                    checked={types.includes(t)}
                    onChange={() => toggleType(t)}
                    className="accent-current"
                    style={{ color: TOKENS.brand }}
                  />
                  {t}
                </label>
              ))}
            </div>
          </fieldset>

          <div className="flex items-center justify-end gap-2 border-t pt-2" style={{ borderColor: TOKENS.border }}>
            <button
              onClick={() => { setRisk("All"); setTypes([]); }}
              className="rounded-md px-2.5 py-1.5 text-xs"
              style={{ color: TOKENS.textDim }}
            >
              Clear
            </button>
            <button
              onClick={() => setOpen(false)}
              className="rounded-md px-2.5 py-1.5 text-xs font-medium"
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
    <div
      className="flex flex-wrap items-center gap-2 border-b px-4 py-3 sm:px-6"
      style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}
    >
      <SearchBar onSelect={onSelectNode} />

      <div className="flex items-center gap-1">
        <button
          onClick={() => setPanActive((p) => !p)}
          title="Pan"
          aria-label="Toggle pan mode"
          aria-pressed={panActive}
          className="rounded-md p-2 hover:bg-white/5"
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
          <button
            key={label}
            title={label}
            aria-label={label}
            className="rounded-md p-2 hover:bg-white/5"
            style={{ color: TOKENS.textDim, border: `1px solid ${TOKENS.border}` }}
          >
            <Icon size={15} />
          </button>
        ))}
      </div>

      <FilterPanel />
    </div>
  );
}

function GraphCanvas({ state, onRetry }) {
  if (state === "loading") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3">
        <div
          className="h-10 w-10 animate-spin rounded-full border-2 border-t-transparent"
          style={{ borderColor: `${TOKENS.flow}33`, borderTopColor: TOKENS.flow }}
        />
        <p className="text-sm" style={{ color: TOKENS.textDim }}>Pulling the latest network state…</p>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <div
          className="flex h-14 w-14 items-center justify-center rounded-2xl"
          style={{ backgroundColor: `${TOKENS.riskHigh}14`, border: `1px solid ${TOKENS.riskHigh}40` }}
        >
          <AlertTriangle size={24} style={{ color: TOKENS.riskHigh }} />
        </div>
        <p className="text-sm font-medium" style={{ color: TOKENS.text }}>Unable to load network</p>
        <p className="max-w-xs text-xs" style={{ color: TOKENS.textDim }}>
          We couldn't retrieve the supply-chain network. Check your connection and try again.
        </p>
        <button
          onClick={onRetry}
          className="mt-1 flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium"
          style={{ backgroundColor: `${TOKENS.riskHigh}18`, color: TOKENS.riskHigh, border: `1px solid ${TOKENS.riskHigh}45` }}
        >
          <RefreshCw size={13} />
          Retry
        </button>
      </div>
    );
  }

  if (state === "empty") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <div
          className="flex h-14 w-14 items-center justify-center rounded-2xl"
          style={{ backgroundColor: `${TOKENS.brand}14`, border: `1px solid ${TOKENS.brand}40` }}
        >
          <Wind size={24} style={{ color: TOKENS.brand }} />
        </div>
        <p className="text-sm font-medium" style={{ color: TOKENS.text }}>No network graph loaded yet</p>
        <p className="max-w-xs text-xs" style={{ color: TOKENS.textDim }}>
          Once the graph service connects, the supply chain map renders here — nodes, routes,
          and live disruption ripples.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center">
      <p className="text-xs" style={{ color: TOKENS.textDim }}>
        Graph canvas mounts here — reserved for React Flow / D3 rendering.
      </p>
    </div>
  );
}

function NodeDetailsBody({ node }) {
  if (!node) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 py-6 text-center">
        <MapPin size={22} style={{ color: TOKENS.textDim }} />
        <p className="text-sm" style={{ color: TOKENS.text }}>No node selected</p>
        <p className="text-xs" style={{ color: TOKENS.textDim }}>
          Search for a node above, or click one on the map once the graph is connected.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="text-sm font-medium" style={{ color: TOKENS.text }}>{node.name}</p>
        <p className="text-xs" style={{ color: TOKENS.textDim }}>{node.type} · {node.region}</p>
      </div>

      <RiskBadge level={node.risk} />

      <div
        className="rounded-lg p-3 text-xs leading-relaxed"
        style={{ backgroundColor: TOKENS.surface2, border: `1px solid ${TOKENS.border}`, color: TOKENS.textDim }}
      >
        {node.note}
      </div>

      <dl className="flex flex-col gap-2 text-xs">
        <div className="flex items-center justify-between">
          <dt className="flex items-center gap-1.5" style={{ color: TOKENS.textDim }}>
            <Boxes size={13} /> Connections
          </dt>
          <dd className="font-mono" style={{ color: TOKENS.text }}>{node.connections}</dd>
        </div>
        <div className="flex items-center justify-between">
          <dt className="flex items-center gap-1.5" style={{ color: TOKENS.textDim }}>
            <Clock size={13} /> Last updated
          </dt>
          <dd style={{ color: TOKENS.text }}>{node.updated}</dd>
        </div>
      </dl>
    </div>
  );
}

function NodeDetailsPanel({ node }) {
  return (
    <aside
      className="hidden w-80 shrink-0 flex-col overflow-y-auto border-l p-5 lg:flex"
      style={{ backgroundColor: TOKENS.surface, borderColor: TOKENS.border }}
    >
      <h2 className="mb-4 text-xs font-medium uppercase tracking-wider" style={{ color: TOKENS.textDim }}>
        Node details
      </h2>
      <NodeDetailsBody node={node} />
    </aside>
  );
}

function NodeDetailsSheet({ node, onClose }) {
  if (!node) return null;
  return (
    <div className="fixed inset-0 z-40 lg:hidden">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div
        className="absolute inset-x-0 bottom-0 max-h-[75vh] overflow-y-auto rounded-t-2xl p-5 pb-8"
        style={{ backgroundColor: TOKENS.surface, borderTop: `1px solid ${TOKENS.border}` }}
      >
        <div className="mx-auto mb-4 h-1 w-10 rounded-full" style={{ backgroundColor: TOKENS.border }} />
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xs font-medium uppercase tracking-wider" style={{ color: TOKENS.textDim }}>
            Node details
          </h2>
          <button onClick={onClose} style={{ color: TOKENS.textDim }} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <NodeDetailsBody node={node} />
      </div>
    </div>
  );
}

export default function AtmoGraphDashboard() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [canvasState, setCanvasState] = useState("loading");
  const [fontsReady, setFontsReady] = useState(false);

  useEffect(() => {
    const link = document.createElement("style");
    link.textContent = `
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
      :focus-visible { outline: 2px solid ${TOKENS.brand}; outline-offset: 2px; border-radius: 4px; }
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

  // Simulates the real lifecycle this will follow once the graph service
  // exists: loading -> empty (no live data yet) or error (connection failed).
  // DEMO_SIMULATE_ERRORS gates the fake failure — flip to false (or delete
  // this branch) once real fetch/socket status drives canvasState instead.
  const attemptLoad = useCallback(() => {
    setCanvasState("loading");
    const outcome = DEMO_SIMULATE_ERRORS && Math.random() < 0.15 ? "error" : "empty";
    setTimeout(() => setCanvasState(outcome), 900);
  }, []);

  useEffect(() => {
    attemptLoad();
  }, [attemptLoad]);

  return (
    <div
      className="flex h-screen w-full flex-col"
      style={{ backgroundColor: TOKENS.bg, fontFamily: "'Inter', sans-serif", opacity: fontsReady ? 1 : 0 }}
    >
      <Header onMenuClick={() => setMobileNavOpen(true)} />

      <div className="flex min-h-0 flex-1">
        <Sidebar mobileOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

        <main className="flex min-w-0 flex-1 flex-col">
          <ControlsBar onSelectNode={setSelectedNode} />
          <div className="min-h-0 flex-1">
            <GraphCanvas state={canvasState} onRetry={attemptLoad} />
          </div>
        </main>

        <NodeDetailsPanel node={selectedNode} />
      </div>

      <NodeDetailsSheet node={selectedNode} onClose={() => setSelectedNode(null)} />
    </div>
  );
}