"""Factory for creating RAG pipeline instances."""

from typing import TYPE_CHECKING, Type, TypeVar, cast

from encourage.rag.base.enum import RAGMethod

if TYPE_CHECKING:
    from encourage.rag.base.config import BaseRAGConfig  # Adjust if needed
    from encourage.rag.base_impl import BaseRAG

# Generic type vars
RAGType = TypeVar("RAGType", bound="BaseRAG")
ConfigType = TypeVar("ConfigType", bound="BaseRAGConfig")


class RAGFactory:
    """Factory to create RAG pipeline instances from registered RAG classes and configs."""

    # Registry mapping: method → (Config subclass, RAG subclass)
    registry: dict[RAGMethod, tuple[Type[ConfigType], Type[RAGType]]] = {}  # pyright: ignore[reportGeneralTypeIssues]

    @classmethod
    def register(cls, name: RAGMethod, config_cls: Type[ConfigType]) -> Type[RAGType]:  # pyright: ignore[reportInvalidTypeVarUse]
        """Decorator to register a RAG class with its config type."""

        def decorator(rag_cls: Type[RAGType]) -> Type[RAGType]:
            cls.registry[name] = (config_cls, rag_cls)
            return rag_cls

        return decorator  # pyright: ignore[reportReturnType]

    @classmethod
    def create(cls, name: RAGMethod, config: ConfigType) -> RAGType:  # pyright: ignore[reportInvalidTypeVarUse]
        """Instantiate the correct RAG subclass using the registry and config."""
        if name not in cls.registry:
            raise KeyError(f"RAG method '{name}' is not registered.")
        config_cls, rag_cls = cls.registry[name]
        if not isinstance(config, config_cls):
            raise TypeError(
                f"Expected config of type {config_cls.__name__}, got {type(config).__name__}"
            )
        return cast(RAGType, rag_cls(config))
