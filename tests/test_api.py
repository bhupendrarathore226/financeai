"""
Unit tests for the FastAPI HTTP layer (api/main.py).

Testing philosophy
------------------
These tests verify that the API **routes** behave correctly:
  - Correct HTTP status codes are returned for valid and invalid inputs.
  - Request validation (file type, question length) rejects bad input.
  - Successful responses contain the expected JSON structure.

Critically, these tests do NOT test the service layer (ingest, query, store).
All service calls are replaced with `unittest.mock.patch` fakes so that:
  1. Tests run without a real ChromaDB database or OpenAI API key.
  2. Tests are fast (no network calls, no ML model loading).
  3. Each test validates exactly one behaviour in isolation.

ChromaDB stub (module-level)
-----------------------------
Before any test code imports `api.main`, we inject a fake `services.store`
module into `sys.modules`.  This prevents the real ChromaDB client and
SentenceTransformer model from being imported, which would fail in CI
environments without those packages installed or without a writable database.

This technique ("module stubbing") is preferred over mocking individual
functions because it works regardless of how deeply the real module is imported
during the test run.
"""

import tempfile    # Standard library: create temporary files/directories in tests
import unittest   # Standard library: the test framework
import types      # Standard library: used to create lightweight fake module objects
import sys        # Standard library: access to Python's module import machinery
from unittest.mock import patch  # Replace real objects with controllable fakes

# FastAPI's built-in test client wraps the application in an HTTP-like interface
# so we can call endpoints without starting a real web server.
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# ChromaDB / store stub — must be registered BEFORE importing api.main
# ---------------------------------------------------------------------------
# Create a minimal fake module that satisfies the `from services.store import ...`
# statement inside api.main.  The lambda functions return None, which is
# sufficient because the actual service calls are patched out in each test.
fake_store = types.ModuleType("services.store")
fake_store.get_collection = lambda: None       # Prevents ChromaDB initialisation
fake_store.get_embedding_model = lambda: None  # Prevents model loading

# Register the fake only if no real module has been loaded yet.
# setdefault is used (instead of direct assignment) so that other test modules
# that run before this one and already registered the fake are not overwritten.
sys.modules.setdefault("services.store", fake_store)

# Safe to import now — api.main will see the fake store, not the real one.
import api.main as api_main


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestApiEndpoints(unittest.TestCase):
    """
    Test class grouping all HTTP endpoint tests.

    setUp() creates a fresh TestClient before each test method so that
    state from one test cannot leak into another.
    """

    def setUp(self):
        """
        Prepare resources needed by every test in this class.

        TestClient wraps the FastAPI `app` and exposes `.get()`, `.post()`, etc.
        methods that behave like a real HTTP client but run entirely in-process.
        """
        self.client = TestClient(api_main.app)

    # ------------------------------------------------------------------
    # /upload tests
    # ------------------------------------------------------------------

    @patch("api.main.ingest_file")
    def test_upload_pdf_success(self, mock_ingest_file):
        """
        A valid PDF upload should return HTTP 200 with the ingestion summary.

        The `ingest_file` service is patched to return a canned success
        response so this test validates only the API layer's behaviour:
          - It calls ingest_file with the correct arguments.
          - It maps the service result to the expected JSON structure.
          - It returns HTTP 200.
        """
        # Configure the mock to return a successful ingestion result.
        # This simulates ingest_file running successfully without doing real work.
        mock_ingest_file.return_value = {
            "status": "ingested",
            "reason": None,
            "transactions_added": 3,
        }

        # Use a real temporary directory so the upload handler can write the file
        # to disk (it saves the file before calling ingest_file).
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the upload folder to our temporary directory so no permanent
            # files are created on the developer's machine during testing.
            with patch("api.main.config.PDF_FOLDER", temp_dir):
                response = self.client.post(
                    "/upload",
                    # `files` sends a multipart/form-data request.
                    # Tuple format: (filename, content_bytes, content_type)
                    files={"file": ("statement.pdf", b"%PDF-1.4 mock", "application/pdf")},
                )

        # Assert the HTTP status code is 200 OK.
        self.assertEqual(response.status_code, 200)

        # Assert the JSON body contains the expected keys and values.
        payload = response.json()
        self.assertEqual(payload["status"], "ingested")
        self.assertEqual(payload["filename"], "statement.pdf")

    def test_upload_rejects_non_pdf(self):
        """
        Uploading a non-PDF file should return HTTP 400 with a descriptive error.

        This test verifies the content-type guard in the upload handler.
        No patching is needed because the handler rejects the request before
        ever calling ingest_file.
        """
        response = self.client.post(
            "/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        # Expect a 400 Bad Request.
        self.assertEqual(response.status_code, 400)

        # The error detail should mention PDF so developers know why it failed.
        self.assertIn("Only PDF files", response.text)

    # ------------------------------------------------------------------
    # /chat tests
    # ------------------------------------------------------------------

    @patch("api.main.ask_llm")
    def test_chat_success(self, mock_ask_llm):
        """
        A valid chat request should return HTTP 200 with the generated answer.

        `ask_llm` is patched so the test does not require a real OpenAI key
        or a populated ChromaDB database.
        """
        # Define the answer our fake LLM will return.
        mock_ask_llm.return_value = "You spent 1200 on dining."

        response = self.client.post("/chat", json={"question": "How much did I spend on dining?"})

        self.assertEqual(response.status_code, 200)

        data = response.json()
        # Verify both the answer and the echoed question are in the response.
        self.assertEqual(data["answer"], "You spent 1200 on dining.")

    def test_chat_validation_error(self):
        """
        A question shorter than 3 characters should be rejected with HTTP 422.

        FastAPI returns 422 Unprocessable Entity automatically when Pydantic
        validation fails (here: `Field(min_length=3)` on ChatRequest.question).
        This test confirms that the validation constraint is active.
        """
        response = self.client.post("/chat", json={"question": "hi"})

        # 422 = FastAPI's automatic validation-failure status code.
        self.assertEqual(response.status_code, 422)

    # ------------------------------------------------------------------
    # /transactions tests
    # ------------------------------------------------------------------

    @patch("api.main.get_all_transactions")
    def test_transactions_success(self, mock_get_all_transactions):
        """
        GET /transactions should return HTTP 200 with a list of transaction objects.

        `get_all_transactions` is patched to return a single sample record so
        the test validates the JSON structure without needing a real database.
        """
        # Provide a realistic-looking transaction record as the mock return value.
        mock_get_all_transactions.return_value = [
            {
                "id": "sample.pdf_0",
                "document": "2026-03-01 | Grocery | -54.20",
                "metadata": {"source": "sample.pdf", "chunk_index": 0, "total_chunks": 1},
            }
        ]

        # Pass `limit=10` as a query parameter to exercise parameter parsing.
        response = self.client.get("/transactions?limit=10")

        self.assertEqual(response.status_code, 200)

        data = response.json()
        # The `count` field should reflect the number of returned items.
        self.assertEqual(data["count"], 1)
        # The `transactions` list should contain the mocked record.
        self.assertEqual(len(data["transactions"]), 1)


if __name__ == "__main__":
    # Allow running this test file directly with `python tests/test_api.py`
    # in addition to the standard `python -m unittest discover` approach.
    unittest.main()
