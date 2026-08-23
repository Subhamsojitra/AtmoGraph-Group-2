# AtmoGraph — Frontend Dev Log

## Status

**Week 2 completed** — Node click interaction UI, zoom/pan controls,
selected-node styling contract, supplier/node details panel, and a real
search/filter interface implemented. Graph rendering and backend
integration remain with the architecture owner.

## Week 2 — Interactive Graph, UI side (Yashaswini)

**Scope owned this week:** node click interaction UI, zoom/pan controls
UI, selected-node styling, supplier/node details panel, search/filter
interface. Graph data/state, the FastAPI connection, and large-graph
rendering/performance stay with the architecture side.

### Delivered

- **Node click interaction UI** — a selection chip in the controls bar
  reflects the currently selected node (name + clear button). Wired
  today from the search dropdown's `onSelectNode`; the same setter is
  ready for the real graph's click callback once it exists.
- **Zoom/pan controls UI** — zoom is real local state now: −/+ buttons
  clamped 25–200%, a live percentage readout, and a "fit to screen"
  button that resets to 100%. Pan is a toggle (`aria-pressed`). Added
  an `onZoomChange` callback prop on `ControlsBar`/`DashboardPage` as
  the hook point for wiring to the actual graph transform.
- **Selected-node styling** — new `nodeStates.js` exports the class
  names (`graph-node`, `graph-node--hover`, `graph-node--selected`,
  `graph-node--risk-high`, `graph-node--risk-low`) and
  `DashboardPage.css` has the matching CSS (hover brightness, selected
  outline, risk-pulse animations, reduced-motion fallback). This is a
  contract for the shared `GraphCanvas` to apply to its own nodes.
- **Supplier/node details panel** — added Category, Tier, and Lead
  time fields to the mock node data and details view, plus a
  presentational actions row ("Flag for review", "View shipment
  history").
- **Search/filter interface** — filters (risk + node type) now
  actually narrow search results instead of sitting inert. Active
  filters show as removable chips under the controls bar, next to a
  risk-color legend (Stable / Elevated / High risk) that doubles as a
  style reference for how risk should read across the dashboard.

### Known gaps / intentionally deferred

- Node selection still only comes from search — real graph-node click
  wiring is the architecture side's job once `GraphCanvas` exposes a
  click callback.
- `onZoomChange` isn't connected to anything yet — it fires with the
  new zoom percentage but nothing consumes it until the real graph
  transform is wired in.
- Filter/search still run against the local mock dataset, not a real
  query.

### Handoff notes for the architecture side

- Apply `NODE_STATE_CLASS` / `NODE_RISK_CLASS` from `nodeStates.js` to
  each node so hover/selected/risk states match the rest of the UI.
- Call the dashboard's node-selection setter (currently exposed via
  `ControlsBar`'s `onSelectNode`) from the graph's click handler.
- Consume `onZoomChange(zoomLevel)` to drive the real zoom transform,
  or tell me if zoom should instead be owned entirely on your side —
  happy to drop the local state if so.

### Next up (Week 3, my side)

- At-risk node visual states (colors/indicators) once prediction data
  exists.
- Risk legend refinements to match whatever the GNN's risk output
  actually looks like.
- Prediction details UI in the node panel.
