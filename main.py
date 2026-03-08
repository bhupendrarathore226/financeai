from api.main import app

# Compatibility entrypoint so both commands work:
# uvicorn main:app --reload
# uvicorn api.main:app --reload
