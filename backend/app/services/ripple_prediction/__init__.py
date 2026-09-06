"""Ripple prediction service package (Module 17, Part 3).

Orchestrates the existing risk propagation and GNN prediction services to
compute the ripple effect of a source entity's disruption through the supply
chain. This service is transport-agnostic: it returns a structured Python
result and knows nothing about WebSocket connections.
"""
