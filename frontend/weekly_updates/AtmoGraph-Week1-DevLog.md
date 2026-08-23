# AtmoGraph — Frontend Dev Log

## Status

**Week 1 completed** — Dashboard UI foundation, responsive layout,
search/filter UI, node details (incl. mobile bottom sheet), and
loading/empty/error states implemented. Graph rendering and backend
integration remain with the architecture owner.

## Week 1 — UI Foundation (Yashaswini)

**Scope owned this week:** dashboard layout, header/sidebar, controls area,
node details panel UI, loading/empty/error states, responsive foundation.
Graph rendering, API integration, prediction logic, and WebSocket/state
wiring stay out of scope — that belongs to the architecture/graph side.

### Delivered

- **Dashboard shell** — header, collapsible sidebar, main content area,
  right-side node details panel. Dark ops-console palette (not a template
  default) with a drifting isobar motif in the header, tying the visual
  language to "Atmo".
- **Header** — brand mark, live-sync indicator, notifications, avatar.
- **Sidebar** — nav items, network snapshot stats (active disruptions,
  nodes at risk, nodes monitored — placeholder values pending real data).
- **Controls bar**
  - Search: local mock dataset, dropdown of matches, clear button,
    no-results state. Not wired to a backend yet — matches Week 1 scope.
  - Filter: popover with risk radio group + node-type checkboxes,
    Clear/Apply. UI only, no filtering logic applied yet.
  - Zoom in/out/fit buttons — presentational; the architecture side wires
    these to actual graph zoom/pan.
- **Graph canvas placeholder** — reserved frame for React Flow/D3.
  Cycles through `loading` → `empty` (typical, since no live graph exists
  yet) or `error` on a simulated ~15% failure, with a Retry action. This
  is a stand-in for the real fetch/socket lifecycle the architecture side
  will drive later.
- **Node details panel** — empty state and populated state (risk badge,
  note, connections, last updated). Selecting a search result populates
  it; real graph-node click wiring is the architecture side's job.
- **Mobile responsiveness** — sidebar becomes a drawer below `md`; node
  details becomes a bottom sheet below `lg` (was previously just hidden
  on mobile — fixed after review).
- **Pan control** — explicit toggle button next to zoom in/out/fit
  (`aria-pressed`); actual pan behavior wires up once the graph exists.
- **Accessibility pass** — `aria-label`s on all icon-only buttons,
  `aria-expanded`/`aria-haspopup` on the filter popover, `fieldset`/
  `legend` grouping for filter options, combobox semantics + listbox
  role on search results, visible `:focus-visible` outline globally,
  Escape closes the search dropdown, filter popover, mobile nav drawer,
  and node details sheet.
- **Error simulation gated** — the random 15% failure used to demo the
  error/retry state is now behind a single `DEMO_SIMULATE_ERRORS` flag
  at the top of the file, so it's a one-line change (or deletion) once
  real fetch/socket status drives `GraphCanvas`'s state instead.

### Known gaps / intentionally deferred

- Search and filter are UI-only — no real query or backend filtering.
- Network snapshot numbers in the sidebar are placeholders.
- Graph canvas has no real graph in it by design (Week 1 doesn't require
  this — architecture side owns React Flow/D3 integration in Week 2).

### Handoff notes for the architecture side

- `GraphCanvas` takes a `state` prop (`loading` / `empty` / `error` /
  `ready`) — swap the simulated lifecycle in `AtmoGraphDashboard` for the
  real fetch/socket status.
- `NodeDetailsPanel` / `NodeDetailsSheet` expect a node shaped like:
  `{ id, name, type, region, risk, connections, updated, note }` — confirm
  this matches (or gets mapped from) the real API response.
- Node selection is currently driven by the search dropdown
  (`onSelectNode`) — real graph clicks should call the same setter.

### Manual test checklist (do before merging)

- [ ] Desktop, laptop, tablet, and mobile breakpoints
- [ ] Mobile sidebar drawer open/close
- [ ] Mobile node-details bottom sheet open/close
- [ ] Loading → empty state on load
- [ ] Loading → error state + Retry (force via `DEMO_SIMULATE_ERRORS`)
- [ ] Search: typing, result click, clear button, no-results message
- [ ] Filter: open/close, select risk + type, Clear, Apply
- [ ] Keyboard-only pass: tab order, Escape closes popovers/drawers/sheet,
      focus ring visible throughout
- [ ] Set `DEMO_SIMULATE_ERRORS = false` (or remove the branch) once the
      real graph service is connected

### Next up (Week 2, my side)

- Node click interaction styling (selected-node treatment).
- Zoom/pan controls UI refinement once real zoom values exist.
- Supplier/node details panel refinements based on real data shape.
- Search/filter interface refinements once backend query is available.
