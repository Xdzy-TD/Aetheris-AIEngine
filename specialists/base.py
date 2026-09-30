"""
Abstract base class for all Aetheris specialist modules.

Every specialist implements the same ``run()`` interface so the controller's
tool registry can invoke any tool uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BaseSpecialist(ABC):
    """Base class for specialist model modules.

    Subclasses must implement :meth:`run` and :meth:`load_model`.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._model_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights / initialise the specialist.

        Called lazily on first invocation or eagerly at startup.
        """

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the specialist's core function.

        Args:
            **kwargs: Tool-schema-defined arguments.

        Returns:
            A dict matching the tool's ``returns`` JSON schema.
        """

    def ensure_loaded(self) -> None:
        """Lazy-load the model if it hasn't been loaded yet."""
        if not self._model_loaded:
            logger.info("loading_model", specialist=self.name)
            self.load_model()
            self._model_loaded = True
            logger.info("model_loaded", specialist=self.name)

    def __repr__(self) -> str:
        status = "loaded" if self._model_loaded else "not loaded"
        return f"<{self.__class__.__name__} ({status})>"
