FROM ghcr.io/astral-sh/uv:0.8.22-debian

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN uv venv \
    && uv pip install .

COPY . .

EXPOSE 5000

CMD ["uv", "run", "src/web/app.py"]