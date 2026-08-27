# Standalone image for the MAF handoff bridge server (WS1). Runs independently of OpenCode —
# `curl localhost:8000/agents/manifest` must succeed with no other services present, per
# design-proposal.md WS1's Definition of Done.
FROM python:3.12-slim

WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
