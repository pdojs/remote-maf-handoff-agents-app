"""ASGI entrypoint: `uvicorn app.main:app`."""

from .server import create_app

app = create_app()
