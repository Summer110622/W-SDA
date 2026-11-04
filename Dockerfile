FROM python:3.11-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install torch==2.10.0 --index-url ${TORCH_INDEX_URL}
COPY pyproject.toml README.md LICENSE ./
COPY wsda ./wsda
COPY configs ./configs
RUN python -m pip install .
RUN useradd --uid 10001 --create-home wsda && mkdir -p /state /data && chown -R wsda:wsda /state /app
USER wsda
EXPOSE 8000
ENTRYPOINT ["wsda"]
CMD ["--help"]
