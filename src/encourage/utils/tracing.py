"""Tracing utilities for MLflow."""

from functools import wraps
from typing import Any, Callable

_TRACING_ENABLED = False


def enable_mlflow_tracing() -> None:
    """Enable MLflow tracing for the decorated class."""
    global _TRACING_ENABLED
    _TRACING_ENABLED = True


def enable_tracing(span_name: str | None = None) -> Callable:
    """Decorator to trace a function with MLflow, with an optional custom span name."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import mlflow

            # Check global tracing flag
            if not _TRACING_ENABLED:
                return func(*args, **kwargs)

            cls_instance = args[0] if args else None
            current_span_name = span_name or getattr(func, "__name__", str(func))
            trace_logic = getattr(cls_instance, "_trace_logic", None)

            # Start the span with the dynamic name
            func_input = args
            func_return = func(*args, **kwargs)

            with mlflow.start_span(current_span_name) as span:
                if trace_logic is not None and callable(trace_logic):
                    trace_logic(span, func_input, func_return)

            return func_return

        return wrapper

    return decorator


def mlflow_trace(func: Callable) -> Callable:
    """Apply mlflow.trace only if _TRACING_ENABLED is True."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import mlflow

        if _TRACING_ENABLED:
            return mlflow.trace(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper
