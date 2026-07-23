"""Emit compact JSON application logs with consistent fields."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format one log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record.

        Parameters
        ----------
        record : logging.LogRecord
            Record to serialize.

        Returns
        -------
        str
            Compact JSON with timestamp, level, logger, message, and context.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": getattr(record, "context", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=str
        )


def structured_logger(component: str) -> logging.Logger:
    """Return one configured application logger.

    Parameters
    ----------
    component : str
        Stable component name included in the logger name.

    Returns
    -------
    logging.Logger
        Non-propagating INFO logger that writes one JSON object per line.
    """
    logger = logging.getLogger(f"fml.{component}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
