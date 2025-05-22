import unittest
import uuid
from contextlib import suppress

from chromadb.errors import NotFoundError

from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.vector_store.chroma import ChromaClient


class TestChromaClient(unittest.TestCase):
    """Integration tests for ChromaClient with an in-memory ChromaDB."""

    def setUp(self):
        """Setup an in-memory ChromaDB client and the ChromaClient."""
        # Use an in-memory client
        self.chroma_client = ChromaClient()
        self.chroma_client.create_collection(collection_name="test_collection", overwrite=True)
        self.documents = [
            Document(content="This is document 1", meta_data=MetaData({"source": "test1"})),
            Document(content="This is document 2", meta_data=MetaData({"source": "test2"})),
            Document(content="This is document 3", meta_data=MetaData({"source": "test3"})),
        ]

    def tearDown(self):
        """Clean up after tests."""
        with suppress(NotFoundError):
            self.chroma_client.delete_collection("test_collection")

    def test_create_collection(self):
        """Test collection creation."""

        collection = self.chroma_client.client.get_collection("test_collection")
        self.assertIsNotNone(collection)
        print("Collection created successfully.")

    def test_insert_documents(self):
        """Test inserting documents into a collection."""

        # Insert documents
        self.chroma_client.insert_documents("test_collection", self.documents)

        # Verify insertion
        collection = self.chroma_client.client.get_collection("test_collection")
        result = collection.query(query_texts=["This is document 1"], n_results=1)

        self.assertGreater(len(result["documents"]), 0)  # type: ignore
        self.assertEqual(len(result["documents"][0]), 1)  # type: ignore
        self.assertEqual(result["documents"][0][0], "This is document 1")  # type: ignore
        print("Documents inserted successfully.")

    def test_meta_data(self):
        """Test inserting documents with metadata into a collection."""

        # Insert documents
        self.chroma_client.insert_documents("test_collection", self.documents)

        # Verify insertion
        collection = self.chroma_client.client.get_collection("test_collection")
        result = collection.query(query_texts=["Document with metadata"], n_results=1)

        self.assertEqual(len(result["documents"][0]), 1)  # type: ignore
        self.assertEqual(result["documents"][0][0], "This is document 1")  # type: ignore
        print("Documents with metadata inserted successfully.")

    def test_query(self):
        """Test querying documents in a collection."""
        self.chroma_client.insert_documents("test_collection", self.documents)

        # Perform a query
        query_result = self.chroma_client.query(
            "test_collection", query=["Document for querying"], top_k=1
        )

        # Verify results
        self.assertEqual(len(query_result[0]), 1)  # Ensure only one document is returned

        # Check that the first result is a valid Document
        doc = query_result[0][0]
        self.assertIsInstance(doc, Document)
        self.assertIsInstance(doc.id, uuid.UUID)
        self.assertIsInstance(doc.content, str)
        self.assertIsInstance(doc.meta_data, MetaData)
        self.assertIsInstance(doc.distance, (int, float))

        # Verify the content of the document
        self.assertEqual(doc.content, "This is document 1")  # Check that the content is correct

        print("Query executed successfully.")

    def test_query_multiple_documents(self):
        """Test querying multiple documents in a collection."""
        self.chroma_client.insert_documents("test_collection", self.documents)

        # Perform a query
        query_result = self.chroma_client.query(
            "test_collection", query=["Document for querying"], top_k=3
        )

        # Verify results
        self.assertEqual(len(query_result[0]), 3)

        # Check that the first three results are valid Documents
        for doc in query_result[0]:
            self.assertIsInstance(doc, Document)
            self.assertIsInstance(doc.id, uuid.UUID)
            self.assertIsInstance(doc.content, str)
            self.assertIsInstance(doc.meta_data, MetaData)
            self.assertIsInstance(doc.distance, (int, float))

        # Verify the content of the documents
        self.assertIn(
            query_result[0][0].content,
            ["This is document 1", "This is document 2", "This is document 3"],
        )
        self.assertIn(
            query_result[0][1].content,
            ["This is document 1", "This is document 2", "This is document 3"],
        )
        self.assertIn(
            query_result[0][2].content,
            ["This is document 1", "This is document 2", "This is document 3"],
        )

        print("Query for multiple documents executed successfully.")

    def test_delete_collection(self):
        """Test deleting a collection."""

        # Delete collection
        self.chroma_client.delete_collection("test_collection")

        with self.assertRaises(NotFoundError):
            self.chroma_client.client.get_collection("test_collection")
        print("Collection deleted successfully.")


if __name__ == "__main__":
    unittest.main()
