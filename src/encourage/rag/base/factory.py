"""Factory for creating RAG pipeline instances."""

from typing import TYPE_CHECKING, Type

from encourage.rag.base.config import BaseRAGConfig
from encourage.rag.base.enum import RAGMethod

if TYPE_CHECKING:
    from encourage.rag.base_impl import BaseRAG


class RAGFactory:
    """Factory to create RAG pipeline instances from registered RAG classes and configs.

    It uses the Strategy and Factory patterns to dynamically register and instantiate
    various RAG implementations using a decorator-based registry.
    """

    # Registry mapping method name → (config class, RAG class)
    registry: dict[RAGMethod, tuple[Type[BaseRAGConfig], Type["BaseRAG"]]] = {}

    @classmethod
    def register(cls, name: RAGMethod, config_cls: Type[BaseRAGConfig]):
        """Register a new RAG method to the factory.

        Args:
            name: Unique string identifier for the RAG method.
            config_cls: Pydantic config class for validating initialization args.

        Returns:
            A decorator that registers the corresponding RAG class.

        """

        def decorator(rag_cls: Type["BaseRAG"]):
            cls.registry[name] = (config_cls, rag_cls)
            return rag_cls

        return decorator

    @classmethod
    def create(cls, name: RAGMethod, config: BaseRAGConfig) -> "BaseRAG":
        """Instantiate a registered RAG class using a validated config object.

        Args:
            name: Registered RAG method name.
            config: Instance of a subclass of BaseRAGConfig.

        Returns:
            Instance of the specified RAG class.

        Raises:
            KeyError: If the method name is not registered.
            TypeError: If config type does not match registered config class.

        """
        if name not in cls.registry:
            raise KeyError(f"RAG method '{name}' is not registered.")
        config_cls, rag_cls = cls.registry[name]
        if not isinstance(config, config_cls):
            raise TypeError(
                f"Expected config of type {config_cls.__name__}, got {type(config).__name__}"
            )
        return rag_cls(config)
