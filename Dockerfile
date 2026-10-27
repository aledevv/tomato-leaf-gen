# Environment of the experiments (Python 3.12, the versions of uv.lock) with the weights and the
# evaluation data of the v0.1.0 release, checked against SHA256SUMS. See "Install" in README.md.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /bin/
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl unzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# dependencies first, so that a change in the code does not reinstall them
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

ARG RELEASE=https://github.com/aledevv/tomato-leaf-gen/releases/download/v0.1.0
ARG WITH_DATA=1
RUN curl -fsSL -O "$RELEASE/weights.zip" -O "$RELEASE/SHA256SUMS" \
 && if [ "$WITH_DATA" = 1 ]; then curl -fsSL -O "$RELEASE/data.zip"; fi \
 && sha256sum -c --ignore-missing SHA256SUMS \
 && for f in *.zip; do unzip -q "$f"; done \
 && rm -f ./*.zip SHA256SUMS

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "generate.py"]
