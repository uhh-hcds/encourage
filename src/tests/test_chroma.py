import unittest
from contextlib import suppress

from chromadb.errors import InvalidCollectionException

from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.vector_store.chroma import ChromaClient
from encourage.vector_store.vector_store import VectorStoreBatch


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
        ]

    def tearDown(self):
        """Clean up after tests."""
        with suppress(ValueError):
            self.chroma_client.delete_collection("test_collection")

    def test_create_collection(self):
        """Test collection creation."""

        collection = self.chroma_client.client.get_collection("test_collection")
        self.assertIsNotNone(collection)
        print("Collection created successfully.")

    def test_insert_documents(self):
        """Test inserting documents into a collection."""
        batch = VectorStoreBatch(documents=self.documents)

        # Insert documents
        self.chroma_client.insert_documents("test_collection", batch)

        # Verify insertion
        collection = self.chroma_client.client.get_collection("test_collection")
        result = collection.query(query_texts=["This is document 1"], n_results=1)

        self.assertEqual(len(result["documents"][0]), 1)
        self.assertEqual(result["documents"][0][0], "This is document 1")
        print("Documents inserted successfully.")

    def test_meta_data(self):
        """Test inserting documents with metadata into a collection."""
        batch = VectorStoreBatch(documents=self.documents)

        # Insert documents
        self.chroma_client.insert_documents("test_collection", batch)

        # Verify insertion
        collection = self.chroma_client.client.get_collection("test_collection")
        result = collection.query(query_texts=["Document with metadata"], n_results=1)

        self.assertEqual(len(result["documents"][0]), 1)
        self.assertEqual(result["documents"][0][0], "This is document 1")
        print("Documents with metadata inserted successfully.")

    def test_query(self):
        """Test querying documents in a collection."""
        batch = VectorStoreBatch(documents=self.documents)
        self.chroma_client.insert_documents("test_collection", batch)

        # Perform a query
        query_result = self.chroma_client.query(
            "test_collection", query=["Document for querying"], top_k=1
        )

        # Verify results
        self.assertEqual(len(query_result["documents"][0]), 1)
        self.assertIn("This is document 1", query_result["documents"][0][0])
        print("Query executed successfully.")

    def test_delete_collection(self):
        """Test deleting a collection."""

        # Delete collection
        self.chroma_client.delete_collection("test_collection")

        with self.assertRaises(InvalidCollectionException):
            self.chroma_client.client.get_collection("test_collection")
        print("Collection deleted successfully.")


if __name__ == "__main__":
    unittest.main()
