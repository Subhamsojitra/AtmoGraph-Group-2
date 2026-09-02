/* ============================================================
   Shared node visual-state contract — Week 2 (Yashaswini)

   GraphCanvas owns rendering the actual nodes; this file just
   defines the class names + CSS (in DashboardPage.css, "Graph
   node states" section) that should be applied so a node's
   hover/selected/risk treatment looks consistent with the rest
   of the dashboard. Apply these as className on whatever DOM/SVG
   element represents a node — they're plain CSS, no framework
   dependency.

   Example:
     import { NODE_STATE_CLASS, NODE_RISK_CLASS } from "./nodeStates";
     <g className={`${NODE_STATE_CLASS.default} ${isSelected ? NODE_STATE_CLASS.selected : ""} ${NODE_RISK_CLASS[node.risk]}`}>
   ============================================================ */

export const NODE_STATE_CLASS = {
  default: "graph-node",
  hover: "graph-node--hover",
  selected: "graph-node--selected",
};

export const NODE_RISK_CLASS = {
  high: "graph-node--risk-high",
  medium: "graph-node--risk-medium",
  low: "graph-node--risk-low",
  none: "graph-node--risk-none",
};
