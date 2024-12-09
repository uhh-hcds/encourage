"""Module for defining the Context class, which represents the context for a prompt."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents a single document with its content and associated score.

    Attributes:
        content (str): The content of the document.
        score (float): The relevance or quality score of the document.

    """

    content: str
    score: float | None


@dataclass
class Context:
    """Represents the context for a prompt, including documents and variables.

    Attributes:
        documents (List[Document]): A list of Document objects, each containing content and score.
        prompt_vars (Dict[str, str]): A dictionary of variables used for rendering the prompt.

    """

    documents: list[Document] = field(default_factory=list)
    prompt_vars: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, documents: list[Document | str]) -> "Context":
        """Create a Context instance from a list of documents.

        Args:
            documents (list[Document | str]): A list of Document objects or strings.

        Returns:
            Context: A new Context instance with the processed documents.

        """
        processed_documents = cls._process_documents(documents)
        return cls(documents=processed_documents)

    def add_document(self, document: Document | str) -> None:
        """Add a single document to the context.

        Args:
            document (Document | str): The document to add.
            Can be a Document instance or a string to be converted into a Document.

        """
        self.documents.append(self._process_single_document(document))

    def add_documents(self, documents: list[Document | str]) -> None:
        """Add a list of documents to the context.

        Args:
            documents (list[Document | str]): A list of Document objects or strings
                                              to be added to the context.

        """
        self.documents.extend(self._process_documents(documents))

    @staticmethod
    def _process_single_document(document: Document | str) -> Document:
        """Process a single document, converting it to a Document instance if needed.

        Args:
            document (Document | str): The document to process.
            Can be:
            - A string: Converted to a Document with default score.
            - A dict: Converted to a Document using 'content' and 'score' keys.
            - A Document: Returned as is.

        Returns:
            Document: A processed Document instance.

        """
        if isinstance(document, str):
            return Document(content=document, score=None)
        elif isinstance(document, dict):
            return Document(content=document.get("content", ""), score=document.get("score", None))
        return document

    @classmethod
    def _process_documents(cls, documents: list[Document | str]) -> list[Document]:
        """Process a list of documents, converting each to a Document instance if needed.

        Args:
            documents (list[Document | str]): A list of documents to process.
            Each document can be:
            - A string
            - A dict with 'content' and 'score' keys
            - A Document instance

        Returns:
            list[Document]: A list of processed Document instances.

        """
        return [cls._process_single_document(doc) for doc in documents]

    @classmethod
    def from_prompt_vars(cls, prompt_vars: dict[str, Any]) -> "Context":
        """Create a Context instance from a dictionary of prompt variables.

        Args:
            prompt_vars (Dict[str, str]): A dictionary of variables used for rendering the prompt.

        Returns:
            Context: A new Context instance with the given prompt variables.

        """
        return cls(prompt_vars=prompt_vars)

    def add_prompt_vars(self, prompt_vars: dict[str, str]) -> None:
        """Add a dictionary of variables to the context.

        Args:
            prompt_vars (Dict[str, str]): A dictionary of variables to add to the context.

        """
        self.prompt_vars.update(prompt_vars)
