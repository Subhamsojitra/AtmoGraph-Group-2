# Frontend Development Log

This shared development log is used by the frontend team (Shubham and Yashaswini) to track progress, note commits, record blockers, and outline immediate next steps.

---

## Shubham

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
