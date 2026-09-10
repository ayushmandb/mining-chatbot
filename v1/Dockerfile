FROM python:3.11-slim

WORKDIR /app

# build-essential: some deps (e.g. faiss-cpu) need it to build wheels on
# certain platforms. curl: used by the healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Where persistent data (chat history db + knowledge base) lives inside
# the container. On Render, mount a Persistent Disk at this exact path
# so it survives restarts/redeploys — see README.md.
ENV DATA_DIR=/app/data
RUN mkdir -p /app/data/books

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
