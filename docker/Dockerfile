FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
WORKDIR /app
# Bake the bootstrap corpus in at build time (falls back to the sample
# corpus since no raw PDF is shipped in the image) — swap for a
# DVC-pulled real corpus once Module 2's data versioning is wired in.
RUN python -m prodml.corpus && chown -R appuser:appuser /app/data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
