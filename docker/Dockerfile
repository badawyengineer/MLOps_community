FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
WORKDIR /app
# The image is self-contained: corpus + eval questions + the PRE-BUILT vector index are baked in
# at build time, so a container starts serving with no external store and no model download.
COPY data/processed/ ./data/processed/
COPY data/eval/ ./data/eval/
RUN python -m prodml.validate_corpus \
 && python -m prodml.indexing \
 && chown -R appuser:appuser /app/data
USER appuser
ENV PRODML_RETRIEVER=hybrid-dense
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
