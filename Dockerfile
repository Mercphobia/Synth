FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Create non-root user
RUN adduser --disabled-password --gecos '' synth && chown -R synth:synth /app
USER synth

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD synth --version

ENTRYPOINT ["synth"]