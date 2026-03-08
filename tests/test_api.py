import tempfile
import unittest
import types
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient


# Provide a lightweight stub so importing api.main does not require Chroma.
fake_store = types.ModuleType("services.store")
fake_store.get_collection = lambda: None
fake_store.get_embedding_model = lambda: None
sys.modules.setdefault("services.store", fake_store)

import api.main as api_main


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api_main.app)

    @patch("api.main.ingest_file")
    def test_upload_pdf_success(self, mock_ingest_file):
        mock_ingest_file.return_value = {
            "status": "ingested",
            "reason": None,
            "transactions_added": 3,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("api.main.config.PDF_FOLDER", temp_dir):
                response = self.client.post(
                    "/upload",
                    files={"file": ("statement.pdf", b"%PDF-1.4 mock", "application/pdf")},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ingested")
        self.assertEqual(payload["filename"], "statement.pdf")

    def test_upload_rejects_non_pdf(self):
        response = self.client.post(
            "/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF files", response.text)

    @patch("api.main.ask_llm")
    def test_chat_success(self, mock_ask_llm):
        mock_ask_llm.return_value = "You spent 1200 on dining."

        response = self.client.post("/chat", json={"question": "How much did I spend on dining?"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "You spent 1200 on dining.")

    def test_chat_validation_error(self):
        response = self.client.post("/chat", json={"question": "hi"})
        self.assertEqual(response.status_code, 422)

    @patch("api.main.get_all_transactions")
    def test_transactions_success(self, mock_get_all_transactions):
        mock_get_all_transactions.return_value = [
            {
                "id": "sample.pdf_0",
                "document": "2026-03-01 | Grocery | -54.20",
                "metadata": {"source": "sample.pdf", "chunk_index": 0, "total_chunks": 1},
            }
        ]

        response = self.client.get("/transactions?limit=10")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(len(data["transactions"]), 1)


if __name__ == "__main__":
    unittest.main()
