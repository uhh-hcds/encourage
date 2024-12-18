"""Module for defining the Context class, which represents the context for a prompt."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from encourage.prompts.meta_data import MetaData


@dataclass
class Document:
    """Represents a single document with its content and associated score.

    Attributes:
        content (str): The content of the document.
        score (float): The relevance or quality score of the document.

    """

    content: str
    score: float = 0.0
    distance: float | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    meta_data: MetaData = field(default_factory=MetaData)


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
    def from_documents(
        cls,
        documents: list[Document] | list[str],
        meta_datas: list[MetaData] = [MetaData()],
        ids: list[uuid.UUID] | None = None,
    ) -> "Context":
        """Create a Context instance from a list of documents.

        Args:
            documents (list[Document | str]): A list of Document objects or strings.
            meta_datas (list[MetaData] | None): A list of MetaData objects.
            ids (list[uuid.UUID] | None): A list of UUIDs for the documents.

        Returns:
            Context: A new Context instance with the processed documents.

        """
        if meta_datas != [MetaData()] and len(documents) != len(meta_datas):
            raise ValueError("The length of documents and meta_datas must be equal.")
        elif ids is not None and len(documents) != len(ids):
            raise ValueError("The length of documents and ids must be equal.")

        processed_documents = cls._process_documents(documents, meta_datas, ids)
        return cls(documents=processed_documents)

    def add_document(self, document: Document | str, meta_data: MetaData = MetaData()) -> None:
        """Add a single document to the context.

        Args:
            document (Document | str): The document to add.
            Can be a Document instance or a string to be converted into a Document.
            meta_data (MetaData): The meta data associated with the document.

        """
        self.documents.append(self._process_single_document(document, meta_data))

    def add_documents(
        self, documents: list[Document] | list[str], meta_datas: list[MetaData] = [MetaData()]
    ) -> None:
        """Add a list of documents to the context.

        Args:
            documents (list[Document | str]): A list of Document objects or strings
                                              to be added to the context.
            meta_datas (list[MetaData]): A list of MetaData objects associated with the documents.

        """
        self.documents.extend(self._process_documents(documents, meta_datas))

    @staticmethod
    def _process_single_document(
        document: Document | str,
        meta_data: MetaData = MetaData(),
        doc_id: uuid.UUID | None = None,
    ) -> Document:
        """Process a single document, converting it to a Document instance if needed.

        Args:
            document (Document | str): The document to process.
            Can be:
            - A string: Converted to a Document with default score.
            - A dict: Converted to a Document using 'content' and 'score' keys.
            - A Document: Returned as is.
            meta_data (dict): The meta data associated with the document.
            doc_id (UUID | None): The UUID of the document.

        Returns:
            Document: A processed Document instance.

        """
        if doc_id is None:
            doc_id = uuid.uuid4()

        if isinstance(document, str):
            return Document(content=document, score=0.0, id=doc_id, meta_data=meta_data)
        elif isinstance(document, dict):
            return Document(
                id=document.get("id", None),
                content=document.get("content", ""),
                score=document.get("score", None),
                meta_data=meta_data,
                distance=document.get("distance", None),
            )
        return document

    @classmethod
    def _process_documents(
        cls,
        documents: list[Document] | list[str],
        meta_datas: list[MetaData] = [MetaData()],
        ids: list[uuid.UUID] | None = None,
    ) -> list[Document]:
        """Process a list of documents, converting each to a Document instance if needed.

        Args:
            documents (list[Document | str]): A list of documents to process.
            Each document can be:
            - A string
            - A dict with 'content' and 'score' keys
            - A Document instance
            meta_datas (list[MetaData] | list[str] | None): A list of MetaData objects or strings.
            ids (list[uuid.UUID] | None): A list of UUIDs for the documents.

        Returns:
            list[Document]: A list of processed Document instances.

        """
        if meta_datas == [MetaData()]:
            meta_datas = [MetaData()] * len(documents)
        if ids is None:
            ids = [uuid.uuid4() for _ in range(len(documents))]
        return [
            cls._process_single_document(doc, meta, doc_id)  # type: ignore
            for doc, meta, doc_id in zip(documents, meta_datas, ids)
        ]

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

    def to_dict(self) -> dict[str, Any]:
        """Convert the Context instance to a dictionary."""
        return {
            "documents": [
                {
                    "content": doc.content,
                    "score": doc.score,
                    "distance": doc.distance,
                    "id": str(doc.id),
                    "meta_data": doc.meta_data.to_dict(),
                }
                for doc in self.documents
            ],
            "prompt_vars": self.prompt_vars,
        }

    def __getitem__(self, key: int) -> Document:
        return self.documents[key]

    def __setitem__(self, key: int, value: Document) -> None:
        self.documents[key] = value

    def __iter__(self) -> Iterator:
        return iter(self.documents)
