"""
Unit tests for the ingestion service (services/ingest.py).

Testing philosophy
------------------
These tests verify the **business logic** of the ingestion pipeline:
  - A new, valid PDF is fully parsed, embedded, and stored in ChromaDB.
  - A duplicate document (already ingested) is detected and skipped.
  - An empty PDF (no tables) raises IngestionError, not a silent failure.

All external I/O is mocked out so the tests:
  - Run without a real PDF file on disk.
  - Run without a real ChromaDB database.
  - Run without loading the SentenceTransformer ML model.
  - Cover the logic branches inside ingest_file in isolation.

ChromaDB stub (module-level)
-----------------------------
Same technique as test_api.py: inject a fake `services.store` into
`sys.modules` before importing the module under test so that no real
database or ML model is ever initialised during the test run.
"""

import unittest   # Standard library: test framework
import types      # Standard library: create lightweight module-like objects
import sys        # Standard library: access the Python module registry
from unittest.mock import Mock, patch  # Mock replaces real objects with fakes


# ---------------------------------------------------------------------------
# ChromaDB / store stub — must be registered BEFORE importing services.ingest
# ---------------------------------------------------------------------------
# Create a minimal fake `services.store` module.  The real module imports
# chromadb and sentence_transformers at call time; this stub avoids that.
fake_store = types.ModuleType("services.store")
fake_store.get_collection = lambda: None        # Won't be called; patched in tests
fake_store.get_embedding_model = lambda: None   # Won't be called; patched in tests
sys.modules.setdefault("services.store", fake_store)

# Now safe to import — ingest.py will import get_collection and get_embedding_model
# from the fake store module above.
from services.ingest import IngestionError, ingest_file


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestIngestService(unittest.TestCase):
    """
    Tests for the ingest_file() function in services/ingest.py.

    Each test method patches the collaborators (parse_pdf, get_collection,
    get_embedding_model) independently so it can control exactly what each
    dependency returns and verify the resulting behaviour.
    """

    @patch("services.ingest.get_embedding_model")   # Patch the embedding model factory
    @patch("services.ingest.get_collection")         # Patch the ChromaDB collection factory
    @patch("services.ingest.parse_pdf")              # Patch the PDF parser
    def test_ingest_success(self, mock_parse_pdf, mock_get_collection, mock_get_model):
        """
        Happy path: a new PDF with two transactions should be fully ingested.

        Verifies that:
          - The returned status is "ingested".
          - transactions_added equals the number of parsed rows.
          - collection.add() is called exactly once with the correct arguments.
          - Each record has the correct ID format and metadata fields.
        """

        # --- Configure mock return values ---

        # Simulate parse_pdf successfully extracting two transaction strings.
        mock_parse_pdf.return_value = ["txn1", "txn2"]

        # Simulate a ChromaDB collection that has no existing records for this
        # document (empty "ids" list → deduplication check passes).
        mock_collection = Mock()  # Mock() creates an object where any attribute/method works
        mock_collection.get.return_value = {"ids": []}  # No existing records for this doc
        mock_get_collection.return_value = mock_collection

        # Simulate the embedding model returning two embedding vectors.
        # The mock chain handles: model.encode(txns).tolist() -> [[...], [...]]
        mock_model = Mock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_model.return_value = mock_model

        # --- Execute ---
        result = ingest_file("uploads/sample.pdf", "sample.pdf")

        # --- Assertions ---

        # The function should report a successful ingestion.
        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["transactions_added"], 2)

        # collection.add() must have been called exactly once (one batch insert).
        mock_collection.add.assert_called_once()

        # Inspect the keyword arguments passed to collection.add() to verify
        # the IDs and metadata were constructed correctly.
        call_kwargs = mock_collection.add.call_args.kwargs

        # IDs should be "<doc_id>_<index>" for each transaction.
        self.assertEqual(call_kwargs["ids"], ["sample.pdf_0", "sample.pdf_1"])

        # Metadata for the first record should carry source + positional info.
        self.assertEqual(call_kwargs["metadatas"][0]["source"], "sample.pdf")
        self.assertEqual(call_kwargs["metadatas"][0]["chunk_index"], 0)

    @patch("services.ingest.get_collection")
    @patch("services.ingest.parse_pdf")
    def test_ingest_duplicate_document_returns_skipped(self, mock_parse_pdf, mock_get_collection):
        """
        Re-uploading an already-ingested document should return status "skipped".

        Verifies that:
          - When collection.get() returns existing IDs, ingest_file returns
            {"status": "skipped", "transactions_added": 0}.
          - collection.add() is NOT called (no duplicate writes).

        This test does NOT patch get_embedding_model because the embedding
        step is reached only AFTER the deduplication check passes.  Since
        we simulate that the document already exists, the function should
        return before ever calling the embedding model.
        """

        # Simulate one parsed transaction (the PDF itself is valid).
        mock_parse_pdf.return_value = ["txn1"]

        # Simulate a ChromaDB collection that ALREADY has a record for this doc.
        # A non-empty "ids" list triggers the deduplication guard.
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["sample.pdf_0"]}  # Already exists!
        mock_get_collection.return_value = mock_collection

        # --- Execute ---
        result = ingest_file("uploads/sample.pdf", "sample.pdf")

        # --- Assertions ---
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["transactions_added"], 0)

        # collection.add() must NOT have been called — duplicates are never written.
        mock_collection.add.assert_not_called()

    @patch("services.ingest.parse_pdf")
    def test_ingest_raises_when_no_transactions_found(self, mock_parse_pdf):
        """
        A PDF that contains no extractable rows should raise IngestionError.

        This handles the case of scanned (image-only) PDFs or completely
        blank documents.  The error should be raised before any database
        or embedding calls are made.
        """

        # Simulate parse_pdf returning an empty list (no tables found in the PDF).
        mock_parse_pdf.return_value = []

        # assertRaises verifies that the callable raises the specified exception.
        # The `with` block syntax is preferred because it also captures the
        # exception object for further inspection if needed.
        with self.assertRaises(IngestionError):
            ingest_file("uploads/empty.pdf", "empty.pdf")


if __name__ == "__main__":
    # Allows running this file directly with `python tests/test_ingest.py`
    # as well as via `python -m unittest discover`.
    unittest.main()
