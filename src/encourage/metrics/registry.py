"""Registry for metrics."""

from typing import Any, Callable, Dict, Type, TypeVar

from encourage.metrics.metric import Metric

T = TypeVar("T", bound=Metric)

METRIC_REGISTRY: Dict[str, Callable[..., Metric]] = {}


def register_metric(name: str) -> Callable[[Type[T]], Type[T]]:
    """Register a metric in the registry."""

    def wrapper(cls: Type[T]) -> Type[T]:
        METRIC_REGISTRY[name.lower()] = cls
        return cls

    return wrapper


def get_metric_from_registry(name: str, **kwargs: Any) -> Metric:
    """Get a metric from the registry."""
    if name.lower() not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")
    return METRIC_REGISTRY[name.lower()](**kwargs)
