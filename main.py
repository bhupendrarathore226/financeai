"""
Root entrypoint for the FinanceAI application.

Purpose
-------
This file exists purely as a convenience shim (a thin re-export layer).
It imports the FastAPI `app` object from `api/main.py` and re-exposes it at
the top-level package so that the development server can be started with
either of the two commands below — whichever the developer prefers.

How to run the server
---------------------
    # Using the top-level shim (this file):
    uvicorn main:app --reload

    # Using the api module directly:
    uvicorn api.main:app --reload

Both commands start the exact same FastAPI application; there is no
behavioural difference between them.

Notes for juniors
-----------------
- `uvicorn` is an ASGI web server — it reads the `app` variable and
  handles all incoming HTTP requests.
- The `--reload` flag tells uvicorn to watch source files and automatically
  restart the server when you save a change (useful during development).
- Never add business logic to this file; keep it as a pure re-export.
"""

# Re-export the FastAPI application instance from the `api` package.
# Any route, middleware, or configuration registered in api/main.py is
# automatically picked up here because Python objects are passed by reference.
from api.main import app  # noqa: F401  (imported but not used directly here)
