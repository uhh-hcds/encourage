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
    def from_documents(cls, documents: list[Document] | list[str]) -> "Context":
        """Create a Context instance from a list of documents.

        Args:
            documents (list[Document | str]): A list of Document objects or strings.

        Returns:
            Context: A new Context instance with the given documents.

        """
        processed_documents = []
        for doc in documents:
            if isinstance(doc, str):
                processed_documents.append(Document(content=doc, score=None))
            elif isinstance(doc, dict):
                processed_documents.append(
                    Document(content=doc.get("content", ""), score=doc.get("score", None))
                )
            else:
                processed_documents.append(doc)
        return cls(documents=processed_documents)

    @classmethod
    def from_prompt_vars(cls, prompt_vars: dict[str, Any]) -> "Context":
        """Create a Context instance from a dictionary of prompt variables.

        Args:
            prompt_vars (Dict[str, str]): A dictionary of variables used for rendering the prompt.

        Returns:
            Context: A new Context instance with the given prompt variables.

        """
        return cls(prompt_vars=prompt_vars)

    def add_document(self, document: Document | str) -> None:
        """Add a document to the context.

        Args:
            document (Document | str): The document to add to the context.

        """
        if isinstance(document, str):
            self.documents.append(Document(content=document, score=None))
        else:
            self.documents.append(document)

    def add_documents(self, documents: list[Document] | list[str]) -> None:
        """Add a list of documents to the context.

        Args:
            documents (List[Document | str]): A list of Document objects or strings.

        """
        for doc in documents:
            self.add_document(doc)

    def add_prompt_vars(self, prompt_vars: dict[str, str]) -> None:
        """Add a dictionary of variables to the context.

        Args:
            prompt_vars (Dict[str, str]): A dictionary of variables to add to the context.

        """
        self.prompt_vars.update(prompt_vars)
