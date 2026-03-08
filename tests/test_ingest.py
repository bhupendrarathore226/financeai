import unittest
import types
import sys
from unittest.mock import Mock, patch


# Provide a lightweight stub so importing services.ingest does not require Chroma.
fake_store = types.ModuleType("services.store")
fake_store.get_collection = lambda: None
fake_store.get_embedding_model = lambda: None
sys.modules.setdefault("services.store", fake_store)

from services.ingest import IngestionError, ingest_file


class TestIngestService(unittest.TestCase):
    @patch("services.ingest.get_embedding_model")
    @patch("services.ingest.get_collection")
    @patch("services.ingest.parse_pdf")
    def test_ingest_success(self, mock_parse_pdf, mock_get_collection, mock_get_model):
        mock_parse_pdf.return_value = ["txn1", "txn2"]

        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        mock_model = Mock()
        mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_model.return_value = mock_model

        result = ingest_file("uploads/sample.pdf", "sample.pdf")

        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["transactions_added"], 2)
        mock_collection.add.assert_called_once()

        call_kwargs = mock_collection.add.call_args.kwargs
        self.assertEqual(call_kwargs["ids"], ["sample.pdf_0", "sample.pdf_1"])
        self.assertEqual(call_kwargs["metadatas"][0]["source"], "sample.pdf")
        self.assertEqual(call_kwargs["metadatas"][0]["chunk_index"], 0)

    @patch("services.ingest.get_collection")
    @patch("services.ingest.parse_pdf")
    def test_ingest_duplicate_document_returns_skipped(self, mock_parse_pdf, mock_get_collection):
        mock_parse_pdf.return_value = ["txn1"]

        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["sample.pdf_0"]}
        mock_get_collection.return_value = mock_collection

        result = ingest_file("uploads/sample.pdf", "sample.pdf")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["transactions_added"], 0)
        mock_collection.add.assert_not_called()

    @patch("services.ingest.parse_pdf")
    def test_ingest_raises_when_no_transactions_found(self, mock_parse_pdf):
        mock_parse_pdf.return_value = []

        with self.assertRaises(IngestionError):
            ingest_file("uploads/empty.pdf", "empty.pdf")


if __name__ == "__main__":
    unittest.main()
