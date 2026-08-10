# AtmoGraph Frontend Scaffold (Day 0)

This directory contains the frontend architecture scaffold for **AtmoGraph: Supply Chain Ripple Effect Predictor**, isolated to the `frontend/` directory.

## Current Directory Structure

The Day 0 architecture contains only the necessary configuration and scaffold entry points:

```text
frontend/
├── package.json
├── vite.config.js
├── index.html
├── public/
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── index.css
    ├── assets/          # Static assets (images, logos)
    ├── components/      # Common UI components (Empty directory for Week 1 layout work)
    ├── graph/           # Graph components directory
    │   └── GraphCanvas.jsx  # Minimal graph canvas mount boundary scaffold
    └── pages/           # High-level page containers
        └── DashboardPage.jsx  # Minimal dashboard page container
```

## Architectural Component Boundaries

1. **`src/pages/DashboardPage.jsx`**
   Serves as the high-level page container shell. It mounts the network graph boundary. It contains no state logic, headers, sidebars, slider controls, or mock data.

2. **`src/graph/GraphCanvas.jsx`**
   Serves as the technical component boundary for the graph engine. It provides the DOM insertion point where D3.js or React Flow will mount in Week 1, with no rendering libraries loaded.

## Setup & Running

Install dependencies:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```

Build verification:
```bash
npm run build
```
