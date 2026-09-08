# ---------------------------------------------------------
# VisionaryQ production container
# ---------------------------------------------------------

FROM python:3.12-slim


# Prevent Python from writing .pyc files and ensure logs
# are written directly to the container output stream.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app


# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------
#
# Install CPU-only PyTorch first to avoid pulling CUDA
# dependencies that VisionaryQ does not use.
#
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.0


# Install the exact dependency lock.
COPY requirements-lock.txt ./requirements-lock.txt

RUN python -m pip install --no-cache-dir \
    -r requirements-lock.txt


# ---------------------------------------------------------
# Application source
# ---------------------------------------------------------

COPY app ./app
COPY templates ./templates
COPY static ./static


EXPOSE 8000


# ---------------------------------------------------------
# Runtime
# ---------------------------------------------------------
#
# Runtime values such as Pinecone credentials and the
# Ollama endpoint are supplied when the container starts.
#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]