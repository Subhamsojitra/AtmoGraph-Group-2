content = '''
    def _resolve_risk_score(
        self,
        request: WebSocketRipplePredictionRequest,
        node: object,
    ) -> float:
        """Determine the source risk score.

        Prefers the request-supplied score. When omitted, falls back to the
        entity's persisted risk_score property. If neither is available,
        defaults to 0.0 (LOW) — never invents a non-zero risk.
        """
        if request.risk_score is not None:
            return float(request.risk_score)

        persisted = self._extract_persisted_risk_score(node)
        if persisted is not None:
            return persisted

        logger.info(
            "Ripple prediction: no risk score provided or persisted; "
            "defaulting to 0.0",
            extra={"entity_id": request.entity_id},
        )
        return 0.0

    @staticmethod
    def _extract_persisted_risk_score(node: object) -> Optional[float]:
        """Extract the persisted risk_score from a graph node response."""
        properties: Optional[dict] = None

        if hasattr(node, "properties"):
            candidate = getattr(node, "properties", None)
            if isinstance(candidate, dict):
                properties = candidate
        elif isinstance(node, dict):
            candidate = node.get("properties")
            if isinstance(candidate, dict):
                properties = candidate
            elif "risk_score" in node:
                properties = node

        if not properties:
            return None

        raw = properties.get("risk_score")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        return None
'''.lstrip()

with open('backend/app/services/ripple_prediction/ripple_prediction_service.py', 'a') as f:
    f.write(content)

print('Part 5 done')
