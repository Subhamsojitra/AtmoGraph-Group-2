"""Domain exceptions for the risk state update module (Module 9).

The API layer maps these exceptions to HTTP status codes:

- ``EntityNotFoundError``    -> 404 (the entity node does not exist in the graph)
"""

from __future__ import annotations


class RiskServiceError(Exception):
    """Base class for all risk update pipeline errors."""


class EntityNotFoundError(RiskServiceError):
    """Raised when the target entity does not exist as a graph node.

    Module 9 never creates new nodes: when the entity referenced by the update
    request is not present in Neo4j, this error is raised and the request
    fails with a "not found" response.
    """