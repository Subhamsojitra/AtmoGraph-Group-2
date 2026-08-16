"""Neo4j database connection manager for AtmoGraph.

Provides a reusable Neo4j Python driver with proper connection handling,
session management, and integration with FastAPI application lifecycle.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase, Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Neo4jDatabase:
    """Manages Neo4j database connections and sessions.

    This class implements a singleton pattern to ensure only one driver
    instance is created per application lifetime. It provides methods for:
    - Driver initialization and configuration
    - Connection verification
    - Session management (both sync and async)
    - Query execution (read and write)
    - Proper resource cleanup
    """

    _instance: Optional["Neo4jDatabase"] = None
    _driver: Optional[Driver] = None
    _async_driver: Optional[AsyncDriver] = None

    def __new__(cls) -> "Neo4jDatabase":
        """Ensure only one instance of Neo4jDatabase exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Neo4j database manager.

        Actual driver creation is deferred until initialize() is called
        to allow for proper configuration and error handling.
        """
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return

        self._initialized = False
        logger.debug("Neo4jDatabase instance created")

    @property
    def is_initialized(self) -> bool:
        """Return True when a driver has been initialized.

        This does *not* imply the database is reachable; use
        :meth:`verify_connectivity` for a real connectivity check.
        """
        return self._initialized

    @staticmethod
    def _close_async_driver_sync(async_driver: Optional[AsyncDriver]) -> None:
        """Best-effort close of an async driver from a synchronous context.

        ``AsyncDriver.close()`` returns a coroutine. When no event loop is
        running (plain scripts, pytest) the coroutine is driven to completion
        with ``asyncio.run``. When a loop is already running (e.g. inside the
        FastAPI lifespan) the coroutine is scheduled on that loop instead,
        because ``asyncio.run`` cannot be used from a running loop. This keeps
        synchronous callers from leaking an un-awaited coroutine warning.
        """
        if async_driver is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            loop.create_task(async_driver.close())
        else:
            asyncio.run(async_driver.close())

    def initialize(self) -> None:
        """Initialize the Neo4j driver with configuration from settings.

        Creates both synchronous and asynchronous drivers based on the
        Neo4j configuration in the application settings.

        The driver is created even if Neo4j is unavailable, allowing
        the application to start. Connection verification is performed
        separately when needed.
        """
        if self._initialized:
            logger.debug("Neo4j driver already initialized")
            return

        # We'll create the drivers in a try block so we can clean up on failure
        sync_driver = None
        async_driver = None
        try:
            logger.info("Initializing Neo4j driver...")

            # Create synchronous driver
            sync_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
                database=settings.neo4j_database,
            )

            # Create asynchronous driver
            async_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
                database=settings.neo4j_database,
            )

            # If we get here, both drivers were created successfully
            self._driver = sync_driver
            self._async_driver = async_driver
            self._initialized = True
            logger.info(
                "Neo4j driver initialized",
                extra={
                    "uri": settings.neo4j_uri,
                    "database": settings.neo4j_database,
                    "username": settings.neo4j_username,
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Neo4j driver",
                extra={"error": str(e)},
                exc_info=True
            )
            # Clean up any drivers that were created
            if sync_driver is not None:
                sync_driver.close()
            if async_driver is not None:
                try:
                    # Close the async driver without leaking an un-awaited coroutine.
                    self._close_async_driver_sync(async_driver)
                except Exception as close_error:
                    logger.warning(
                        "Error closing async driver during init failure",
                        extra={"error": str(close_error)}
                    )
            # Don't raise the exception - allow application to start
            # Connection status will be reported via health endpoint

    def verify_connectivity(self) -> bool:
        """Verify connectivity to the Neo4j database.

        Executes a simple query to confirm the database is reachable
        and responding correctly.

        Returns:
            True if connection is successful, False otherwise
        """
        if not self._driver:
            logger.warning("Neo4j driver not initialized")
            return False

        try:
            with self._driver.session() as session:
                result = session.run("RETURN 1 AS result")
                record = result.single()
                if record and record["result"] == 1:
                    logger.debug("Neo4j connectivity verified successfully")
                    return True
                else:
                    logger.warning("Unexpected response from Neo4j connectivity check")
                    return False
        except Exception as e:
            logger.warning(
                "Neo4j connectivity verification failed",
                extra={"error": str(e)},
                exc_info=True
            )
            return False

    @contextmanager
    def get_session(self) -> Generator[Any, None, None]:
        """Provide a Neo4j session in a context manager.

        Ensures the session is properly closed after use.

        Yields:
            A Neo4j session object

        Raises:
            ServiceUnavailable: If driver is not initialized
            Neo4jError: If session cannot be created (database unavailable)

        Example:
            with neo4j_db.get_session() as session:
                result = session.run("MATCH (n) RETURN n LIMIT 1")
        """
        if not self._driver:
            raise ServiceUnavailable("Neo4j driver not initialized")

        session = None
        try:
            session = self._driver.session()
            # Lightweight round-trip so connectivity errors surface immediately
            # instead of on the caller's first real query.
            session.run("RETURN 1").consume()
        except Exception as e:
            if session is not None:
                session.close()
            logger.error(
                "Failed to create Neo4j session",
                extra={"error": str(e)},
                exc_info=True
            )
            raise ServiceUnavailable(f"Unable to create a Neo4j session: {e}") from e

        try:
            yield session
        finally:
            session.close()

    def execute_read(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None
    ) -> Any:
        """Execute a read-only Cypher query.

        Args:
            query: The Cypher query to execute
            parameters: Optional dictionary of query parameters

        Returns:
            Query result object

        Raises:
            ServiceUnavailable: If driver is not initialized
            Neo4jError: For query execution errors
        """
        if not self._driver:
            raise ServiceUnavailable("Neo4j driver not initialized")

        with self._driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                return list(result)
            except Exception as e:
                logger.error(
                    "Read query execution failed",
                    extra={
                        "query": query,
                        "parameters": parameters,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise

    def execute_write(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None
    ) -> Any:
        """Execute a write Cypher query.

        Args:
            query: The Cypher query to execute
            parameters: Optional dictionary of query parameters

        Returns:
            Query result object

        Raises:
            ServiceUnavailable: If driver is not initialized
            Neo4jError: For query execution errors
        """
        if not self._driver:
            raise ServiceUnavailable("Neo4j driver not initialized")

        with self._driver.session() as session:
            try:
                # For write operations, we use explicit transactions
                with session.begin_transaction() as tx:
                    result = tx.run(query, parameters or {})
                    tx.commit()
                    return list(result)
            except Exception as e:
                logger.error(
                    "Write query execution failed",
                    extra={
                        "query": query,
                        "parameters": parameters,
                        "error": str(e)
                    },
                    exc_info=True
                )
                raise

    async def initialize_async(self) -> None:
        """Initialize the asynchronous Neo4j driver.

        This method is provided for completeness, though the main
        initialize() method already creates both sync and async drivers.
        """
        if self._async_driver is not None:
            return

        try:
            logger.info("Initializing asynchronous Neo4j driver...")

            self._async_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
                database=settings.neo4j_database,
            )

            logger.info("Asynchronous Neo4j driver initialized")

        except Exception as e:
            logger.error(
                "Failed to initialize asynchronous Neo4j driver",
                extra={"error": str(e)},
                exc_info=True
            )
            await self.close_async()
            # Don't raise - allow application to continue

    async def verify_connectivity_async(self) -> bool:
        """Verify connectivity to the Neo4j database asynchronously.

        Returns:
            True if connection is successful, False otherwise
        """
        if not self._async_driver:
            logger.warning("Neo4j async driver not initialized")
            return False

        try:
            async with self._async_driver.session() as session:
                result = await session.run("RETURN 1 AS result")
                record = await result.single()
                if record and record["result"] == 1:
                    logger.debug("Neo4j async connectivity verified successfully")
                    return True
                else:
                    logger.warning("Unexpected response from Neo4j async connectivity check")
                    return False
        except Exception as e:
            logger.warning(
                "Neo4j async connectivity verification failed",
                extra={"error": str(e)},
                exc_info=True
            )
            return False

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[Any, None]:
        """Provide an asynchronous Neo4j session in an async context manager.

        Yields:
            An asynchronous Neo4j session object

        Raises:
            ServiceUnavailable: If driver is not initialized
        """
        if not self._async_driver:
            raise ServiceUnavailable("Neo4j async driver not initialized")

        async with self._async_driver.session() as session:
            yield session

    def close(self) -> None:
        """Close the Neo4j driver and release all resources.

        This method should be called during application shutdown in a synchronous context.
        It attempts to close both drivers, handling the async driver as best as possible
        in a synchronous context.
        """
        if self._driver:
            logger.info("Closing Neo4j sync driver...")
            self._driver.close()
            self._driver = None

        if self._async_driver:
            logger.info("Closing Neo4j async driver...")
            async_driver = self._async_driver
            self._async_driver = None
            try:
                # In a synchronous context the async driver cannot be awaited;
                # close it without leaking an un-awaited coroutine.
                self._close_async_driver_sync(async_driver)
            except Exception as close_error:
                logger.warning(
                    "Error closing async driver in sync context",
                    extra={"error": str(close_error)}
                )

        self._initialized = False
        logger.info("Neo4j driver closed")

    def close_sync(self) -> None:
        """Close the Neo4j sync driver and release its resources.

        This method should be called during application shutdown in a synchronous context.
        """
        self.close()

    async def close_async(self) -> None:
        """Close the Neo4j driver asynchronously.

        This method should be called during application shutdown in an asynchronous context.
        It closes both the async and sync drivers.
        """
        if self._async_driver:
            logger.info("Closing asynchronous Neo4j driver...")
            await self._async_driver.close()
            self._async_driver = None

        if self._driver:
            logger.info("Closing Neo4j sync driver...")
            self._driver.close()
            self._driver = None

        self._initialized = False
        logger.info("Neo4j driver closed")


# Global instance for easy access throughout the application
neo4j_db = Neo4jDatabase()


def get_neo4j_database() -> Neo4jDatabase:
    """Dependency injection provider for Neo4jDatabase.

    Returns the global Neo4jDatabase instance for use in FastAPI dependencies.

    Returns:
        Neo4jDatabase: The global Neo4j database manager instance
    """
    return neo4j_db