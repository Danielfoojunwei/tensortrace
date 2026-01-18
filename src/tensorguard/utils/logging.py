import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # OpenTelemetry integration (optional)
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                log_entry["trace_id"] = format(span.get_span_context().trace_id, '032x')
                log_entry["span_id"] = format(span.get_span_context().span_id, '016x')
        except ImportError:
            pass

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra attributes if provided
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)

def get_logger(name: str) -> logging.Logger:
    """Returns a structured logger instance."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # In production, use JSON. In dev, use standard format.
        try:
            from .config import settings
            env = settings.ENVIRONMENT
            lvl = settings.LOG_LEVEL
        except (ImportError, AttributeError):
            env = "dev"
            lvl = "INFO"
        
        if env == "production":
            handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
        logger.addHandler(handler)
        logger.setLevel(lvl)
        logger.propagate = False
        
    return logger

# Global default logger
logger = get_logger("tensorguard")
