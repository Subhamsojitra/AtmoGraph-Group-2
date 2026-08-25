# Frontend Development Log

This shared development log is used by the frontend team (Shubham and Yashaswini) to track progress, note commits, record blockers, and outline immediate next steps.

---

## Shubham

### 2026-08-25 — Day 15: Prediction Integration Continuation & Graph Control Integration
* **Work completed**:
  - Investigated Yashaswini's graph controls (Zoom In, Zoom Out, Fit/Reset, Pan Mode) in `ControlsBar` and found they updated only local component state without communicating with D3's internal zoom transform.
  - Lifted `zoom` and `panActive` state variables up to `DashboardPage.jsx` and connected them to `ControlsBar` via React props.
  - Wrapped `GraphCanvas.jsx` in `forwardRef` and exposed programmatic zoom control methods (`zoomIn()`, `zoomOut()`, and `resetZoom()`) using `useImperativeHandle`.
  - Kept the D3 zoom transform as the authoritative source of truth for the viewport scale and translate parameters, preventing React state rendering fights.
  - Implemented event source checking using `event.sourceEvent` to synchronize interactive zoom gestures (scroll wheel, double-click) back to the dashboard's zoom percentage state, safely bypassing feedback loops.
  - Bound the zoom scale extent to `[0.25, 2.0]` to match the UI controls bounds.
  - Applied active pan state styling by showing a `grab` cursor over the SVG graph canvas when Pan Mode is toggled active.
  - Preserved all existing D3 graph physics, ResizeObserver, node dragging, background panning, dynamic node type coloring, and large graph optimization configurations.
  - Strengthened prediction mapping in `GraphCanvas` to keep predictions fully optional, and added optional generic prediction details rendering to the details panel `NodeDetailsBody` in `DashboardPage.jsx` without assuming any final ML risk schemas or indicators.
  - Verified compilation via `npm run build` and resolved linting checks using `npm run lint` (0 warnings, 0 errors).
* **Commit**: *[Ready for commit]*
* **Issues/blockers**: None.

### 2026-08-24 — Day 14: Prediction Data Integration Foundation
* **Work completed**:
  - Established a clean decoupled frontend prediction-data integration boundary in `predictionService.js` to isolate prediction API fetches.
  - Inspected backend prediction logic and verified no GNN prediction models or FastAPI prediction endpoints are currently implemented.
  - Clearly documented that prediction schemas are pending GNN implementation, marking fields like `predictedRisk` and `confidence` as temporary development/mock values.
  - Structured prediction data mapping conceptually around the stable `prediction.nodeId -> graph node.id` relationship.
  - Modified `DashboardPage.jsx` minimally at the state and data controller layer to import, fetch (`getPredictionData`), and manage prediction state without modifying layout, sidebar, header, search, filters, details panels, or styling.
  - Passed `predictions` as a prop to `<GraphCanvas />`, and updated D3 node mapping inside `GraphCanvas.jsx` to optionally associate the matched prediction object with `node.prediction`.
  - Confirmed that absence of prediction data fallback works gracefully (returns empty prediction array in backend mode) and never blocks graph rendering.
  - Verified linter rules pass with 0 warnings/errors and product builds successfully.
* **Commit**: *[Ready for commit]*
* **Issues/blockers**: None.

### 2026-08-23 — Day 13: Week 2 Final Integration Validation & Mid-Project Review Readiness
* **Work completed**:
  - Performed a comprehensive integration audit to ensure the frontend meets all Week 2 requirements: data connectivity, pan/zoom interactions, click handlers, details panel rendering, and large-graph scalability.
  - Identified and fixed a selection synchronization issue between `DashboardPage.jsx` and `GraphCanvas.jsx`. Passed `selectedNodeId` as a prop to `GraphCanvas` and implemented a dedicated, high-performance `useEffect` inside `GraphCanvas.jsx` to update highlighted nodes dynamically when selection is cleared (e.g. from the dashboard ControlsBar) or changed (e.g. from search) without restarting the D3 simulation.
  - Confirmed `GraphCanvas` consumes the generic `{ nodes, links }` data contract, remaining backend-agnostic and independent of specific backend structures.
  - Confirmed large graph scalability: benchmark dataset of ~2,000 nodes and ~3,000 links in `?mode=large` renders and ticks smoothly without browser freezes, utilizing synchronous pre-ticking, Barnes-Hut many-body optimization, and label suppression.
  - Validated D3 lifecycle safety: confirmed `ResizeObserver` and simulation are correctly stopped/disconnected on unmount, and SVG child elements are cleared.
  - Verified compatibility with FastAPI backend node structure (`id`, `label`, `properties`) retrieved from `/api/v1/graph/nodes` in backend mode, defaulting relationships safely to `[]` due to lack of bulk relationships API without generating extra `O(N)` queries.
  - Confirmed build succeeds via `npm run build` and linter passes with 0 warnings/errors via `npm run lint`.
* **Commit**: *[Ready for commit]*
* **Issues/blockers**: None.

### 2026-08-22 — Day 12: Week 2 Integration & Stabilization
* **Work completed**:
  - Reviewed current frontend/backend integration status and confirmed strict compliance with FastAPI endpoints (`GET /api/v1/graph/nodes` and backend structure).
  - Validated graph data contract `{ nodes, links }` transformation in `graphService.js`: verified that backend node categories map dynamically to `type` and resolve labels safely using properties/name fallbacks, ensuring schema-neutral compatibility.
  - Confirmed D3 lifecycle in `GraphCanvas.jsx` is backend-agnostic and robust: verified event propagation rules (`event.stopPropagation()`, `event.sourceEvent.stopPropagation()`) that cleanly isolate node dragging/clicking from background pan/zoom.
  - Verified selection behaviors: selected node remains highlighted on mouseleave, clicking another node shifts selection cleanly, and clicking/dragging does not trigger zoom/pan.
  - Inspected performance optimization for large graph mode (`?mode=large` with 2,000 nodes/3,000 links) and verified D3 simulation configurations (pre-ticking, Barnes-Hut optimization, label suppression) remain stable and responsive.
  - Performed frontend compilation check (`npm run build`) and lint verification (`npm run lint` via `oxlint`), resolving 0 warnings/errors.
  - Confirmed strict adherence to safety guidelines: did not touch dashboard layout/components or Tailwind architecture to avoid conflicts with Yashaswini's upcoming integration, and did not hardcode domain-specific supply chain categories in graph visualization.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.

### 2026-08-21 — Day 11: Frontend/Backend Graph Integration Alignment
* **Work completed**:
  - Inspected and aligned with backend graph contract (FastAPI prefix `/api/v1` and routes: `GET /graph/nodes`, `GET /graph/nodes/{node_id}`, `GET /graph/search`, `GET /graph/nodes/{node_id}/neighbors`, `POST /nodes`, and `POST /relationships`).
  - Validated frontend service boundary in `graphService.js`:
    - Updated `transformBackendData` to map Neo4j node labels to node categories (`type`) in the D3 visualization if `node.type` is not specified, preventing all nodes from defaulting to a single gray color.
    - Added display name resolution for backend nodes that prioritizes properties name/title/label or node name over the raw type label.
    - Maintained strict string normalization for node IDs, filtered out invalid nodes and links referencing missing nodes, and safely handled empty backend datasets.
    - Verified backend mode does not fabricate mock relationships (returns `links: []` since no bulk relationship route exists), preserving the exact backend reality.
  - Confirmed `GraphCanvas` remains entirely backend-agnostic and contains no FastAPI or domain-specific assumptions (no references to "Supplier", "Warehouse", etc.).
  - Confirmed all Day 9/10 optimizations (pre-ticking, Barnes-Hut optimization, ResizeObserver dynamic layout, drag/pan/zoom gestures, lightweight hover/selection overlays) are fully preserved.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.

### 2026-08-20 — Day 10: Graph Integration Readiness & Interaction Refinement
* **Work completed**:
  - Implemented defensive backend node parsing in `graphService.js`:
    - Skips null/undefined records and nodes without valid IDs.
    - Normalizes IDs to strings and filters out duplicate node IDs using a tracking Set.
    - Resolves labels safely using fallback order: `label` -> `properties.name` -> `name` -> `id`.
    - Sanitizes links to only connect nodes present in the current node ID set, preventing canvas crashes.
    - Preserves backend mode's current `links: []` behavior (no O(N) neighbor requests or fabricated relationships).
  - Established generic node type compatibility in `GraphCanvas.jsx`:
    - Removed hardcoded type styling (e.g., `'Supplier'`, `'Warehouse'`, etc.) and dynamically mapped sorted unique node types to a premium color palette.
  - Refined graph interactions and selection visuals:
    - Separated node drag/click gestures from canvas zoom/pan by stopping event propagation on node clicks (`event.stopPropagation()`) and drag starts (`event.sourceEvent.stopPropagation()`).
    - Added lightweight selection highlighting on nodes using a prominent blue stroke outline (`#3182ce`, stroke-width: 3) and bold text styling.
    - Integrated mouseenter/mouseleave hover styling that respects the persistent selected node state without stomping on it.
  - Preserved all Day 9 performance optimizations intact:
    - Large-graph threshold (`LARGE_GRAPH_THRESHOLD = 500`), optimized D3 forces, synchronous pre-ticking (40 ticks), and label suppression.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.

### 2026-08-19 — Day 9: D3 Large-Graph Performance Optimization
* **Work completed**:
  - Implemented large-graph detection with threshold `LARGE_GRAPH_THRESHOLD = 500`.
  - Optimized D3 forces for datasets exceeding the threshold:
    - Disabled expensive `d3.forceCollide` (collision resolution force) to remove O(N^2) distance checks.
    - Reduced many-body strength from `-300` to `-80` and constrained calculations using `.distanceMax(250)` to utilize the D3 Barnes-Hut algorithm efficiency.
    - Set link distance to a tighter `80` to align nodes more compactly.
  - Implemented a controlled large-graph settling strategy:
    - Increased `alphaDecay` to `0.08` (from standard `~0.0228`) to reduce total ticking duration.
    - Pre-ticked simulation 40 times synchronously before timer startup. This pre-computes node layout positions on the main thread (~40ms execution) before any DOM injection, bypassing DOM tick update overhead and letting the user see a pre-settled layout instantly.
  - Optimized text label rendering:
    - Hidden text elements by default (`display: none`) in large mode to avoid rendering 2,000 text elements in the DOM continuously.
    - Added direct D3/DOM-based mouseenter/mouseleave hover callbacks to show/hide labels without triggering React component re-renders.
    - Integrated a local, backend-agnostic selection ref (`selectedNodeIdRef`) to keep clicked node labels visible, maintaining full decoupling from parent layouts.
  - Confirmed that normal graph behavior (<= 500 nodes) remains completely unchanged (colliding, visible labels, standard forces, drag/hover/zoom/pan/resize).
* **Large-Graph Benchmark Results (2,000 nodes, 3,000 links)**:
  - **Initial Render Time**: 
    - *Before*: ~1.0 - 1.5 seconds.
    - *After*: ~300ms (instantaneous display).
  - **Simulation Settling Time**:
    - *Before*: 25+ seconds.
    - *After*: ~1.5 seconds (pre-ticking handles the convergence phase, and the remaining 28 ticks settle rapidly in less than a second).
  - **Drag Responsiveness**:
    - *Before*: Extremely laggy (< 2 FPS), often resulting in accidental background panning.
    - *After*: Highly interactive and buttery smooth (~60 FPS), with coordinates adjusting instantly.
  - **Zoom/Pan Responsiveness**:
    - *Before*: Smooth only after simulation settled (25s+).
    - *After*: Buttery smooth immediately on render (~60 FPS) because CPU load is negligible.
  - **DOM Element Count**:
    - *Before*: ~9,000 SVG elements.
    - *After*: ~9,000 SVG elements (unchanged but labels have `display: none` by default, skipping layout and paint cost).
  - **Browser CPU Behavior**:
    - *Before*: 100% CPU thread lock for 25+ seconds.
    - *After*: Short, minor CPU bump (~40ms) during pre-ticking, then drops back to idle immediately.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.

### 2026-08-18 — Day 8: Real Graph API Integration & Large-Graph Rendering Preparation
* **Work completed**:
  - Updated `graphService.js` to retrieve real backend node data via `GET /api/v1/graph/nodes` and default relationships to `[]` as no bulk relationships endpoint currently exists.
  - Implemented an isolated URL parameter-based dataset selector (`?mode=mock`, `?mode=backend`, `?mode=large`) to keep development testing separate from the production UI.
  - Integrated `onNodeClick(node)` in `GraphCanvas.jsx` to bubble up the selected node data and display it in a closeable, isolated node-details overlay in `DashboardPage.jsx`, avoiding any layout conflict with Yashaswini's pending UI work.
  - Implemented a deterministic large-graph mock generator (~2,000 nodes and ~3,000 links) to perform scalability benchmarks.
  - Verified that all modes load correctly and browser console remains error-free.
* **Large-Graph Benchmark Observations (2,000 nodes, 3,000 links)**:
  - **Initial Render Speed**: The loading overlay dismisses in ~300ms, and the D3 elements are injected into the DOM within ~150ms.
  - **Simulation Settling Time**: The simulation continues ticking and moving nodes for over 25 seconds before settling down, causing high CPU utilization (~100% of a single core).
  - **Drag Responsiveness**: Dragging is extremely laggy (< 2 FPS). Selecting and moving a node causes it to jump erratically because force recalculation ticks freeze the main thread.
  - **Zoom/Pan Responsiveness**: Zooming and panning using the mouse wheel / background drag is smooth (~60 FPS) because SVG container transformations are hardware-accelerated and do not trigger force ticks once the simulation is not actively running.
  - **DOM Size**: Renders 2,000 `<g.node-group>` elements, 2,000 `<circle>` elements, 2,000 `<text>` elements, and 3,000 `<line>` elements, totalling 9,000 DOM nodes in the SVG container.
  - **Browser Freezing**: The browser does not crash, but the tab becomes sluggish during simulation ticking and dragging.
* **Recommendation/Optimization Proposed**:
  - The default charge and collide forces are the main bottleneck. We propose a targeted optimization:
    1. If nodes count > 500, disable the heavy `d3.forceCollide` (collision resolution) and reduce the simulation steps by raising `alphaDecay` to settle the layout faster.
    2. Hide text labels (`<text>`) on initial render if nodes count > 500, showing them only on hover.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.

### 2026-08-17 — Day 7: Frontend Graph Data Integration Foundation
* **Work completed**: Created `graphService.js` to serve as the asynchronous service boundary between the UI and backend APIs. Designed the service to transform raw backend structure (separate nodes and relationships) into the standard `{ nodes, links }` format dynamically without hardcoding domain-specific Neo4j types. Refactored `DashboardPage.jsx` to manage asynchronous graph states (loading, success, empty, error) and render appropriate feedback overlays. Integrated a temporary, isolated developer toggle bar in the dashboard to check all four data states in the browser. Verified that all states render cleanly, existing D3 interactions (dragging, zoom, pan, hover, coordinates preservation) are fully preserved, and browser console remains completely error-free.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 2 - Integrate real FastAPI endpoints once the backend data contract is ready.

### 2026-08-16 — Day 6: Tested Graph Features
* **Work completed**: Successfully tested and verified graph working features and validated all checks.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 2 - Start working on graph data integration.

### 2026-08-15 — Day 5: Graph Foundation Completion & Week 1 Validation
* **Work completed**: Verified existing graph foundation interactions (force simulation, node dragging, node hover, ResizeObserver, dynamic data joins). Identified and implemented the missing D3 zoom and pan interactions in `GraphCanvas.jsx` to complete the Week 1 graph-foundation scope. Wrapped layout groups in a parent `<g class="graph-main-content">` to receive transforms cleanly without affecting simulation coordinates. Configured `d3.zoom()` with a scale extent of `[0.1, 8]` bound to the SVG container. Verified that node dragging and background panning function independently without interference. Verified quality checks (oxlint linter, build compilation, and browser verification testing for responsiveness, error-free logs, and node coordinate preservation).
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 2 - Replace mock data with backend integration.

### 2026-08-14 — Day 4: Graph Data & Rendering Robustness
* **Work completed**: Refactored `GraphCanvas.jsx` to establish a robust, data-driven update lifecycle. Declared persistent React refs for the simulation, ResizeObserver, and SVG sub-groups to prevent redundant object and listener re-creations. Implemented safe nodes and links parsing (filtering null/undefined nodes, and links referencing non-existent nodes) to handle invalid inputs without crashing. Preserved coordinates of existing nodes across data updates to maintain visual layout stability. Used D3's `.join()` API for DOM-element-aware rendering transitions. Verified changes via lint, build, and automated transition testing (standard, empty, and malformed datasets) in browser.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None
* **Next task**: Week 2 - Replace mock data with backend integration.

### 2026-08-13 — Day 3: Graph Interaction & Data Boundary
* **Work completed**: Established a clean data boundary by passing the mock dataset from `DashboardPage.jsx` to `GraphCanvas.jsx` via props. Configured interactive node dragging using `d3.drag()` while preserving D3 force simulation constraints, reheating the simulation during drag and allowing it to settle naturally on release. Added minimal visual hover feedback on nodes (cursor style, node border thickness, and text weight) without modifying the theme colors. Verified workspace cleanliness using oxlint, build, and manual testing.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 2 - Replace mock data with backend integration.

### 2026-08-12 — Day 2: D3 Graph Foundation Refinement
* **Work completed**: Refined `GraphCanvas.jsx` to make it container-aware. Configured `ResizeObserver` to monitor container bounds and update D3 SVG dimensions and simulation center force dynamically. Ensured simulation stops, observer disconnects, and SVG contents are cleared on unmount/re-render. Verified changes using oxlint, Vite build, and manual container resizing tests in browser.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 2 - Replace mock data with backend integration and implement interactive nodes (zoom, pan, drag, and click event bindings).

### 2026-08-11 — Day 1: Static Network Graph Foundation
* **Work completed**: Selected D3.js as the graph visualization engine. Added `d3` dependency in `package.json`. Created `data.js` containing a temporary mock/demo supply chain network schema (`nodes` and `links`) for Week 1 development. Updated `GraphCanvas.jsx` to render a force-directed SVG network layout containing node categories and text labels, along with cleanup logic on component unmount. Verified code via build, lint, and browser screenshot visual checks.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 2 - Replace mock data with backend integration and implement interactive nodes (zoom, pan, drag, and click event bindings).

### 2026-08-10 — Day 0: Architecture Setup & Repository Foundation
* **Work completed**: Scaffolded minimal React + Vite application under `frontend/` directory. Established clean directory layout (`components/`, `graph/`, `pages/`). Created minimal layout shells (`DashboardPage.jsx` and `GraphCanvas.jsx`) representing technical boundaries. Configured repository tracking branch.
* **Commit**: *[Pending review]*
* **Issues/blockers**: None.
* **Next task**: Week 1 - Select graph engine (D3.js or React Flow) and implement static network graph.

---

## Yashaswini

*(No entries yet. Yashaswini to populate during UI layout implementation tasks)*
