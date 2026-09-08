# VisionaryQ — Video Question Answering with RAG

VisionaryQ is an AI-powered video question-answering system that converts video speech into searchable text and answers user questions using Retrieval-Augmented Generation (RAG).

The production application uses FastAPI, Faster-Whisper, SentenceTransformers, Pinecone, and a locally hosted Ollama LLM.

## Architecture

VisionaryQ processes a video through the following pipeline:

```text
Video Upload
    ↓
Faster-Whisper ASR
    ↓
Timestamped Transcript
    ↓
Text Chunking
    ↓
SentenceTransformer Embeddings
    ↓
Pinecone Vector Database
    ↓
Semantic Retrieval
    ↓
Retrieved Video Context
    ↓
Ollama (llama3.2:3b)
    ↓
Grounded Answer
```

The system is designed to answer questions using information retrieved from the uploaded video rather than relying on unrestricted LLM knowledge.

If relevant information cannot be found in the video, VisionaryQ returns a response indicating that the information is unavailable in the video.

## Technology Stack

* Python 3.12
* FastAPI
* Uvicorn
* Faster-Whisper
* SentenceTransformers
* Pinecone
* Ollama
* Llama 3.2 3B
* PyTorch CPU
* Docker
* Pytest

## Project Structure

```text
VisionaryQ/
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── services/
│   └── main.py
├── static/
│   ├── script.js
│   └── styles.css
├── templates/
│   └── index.html
├── tests/
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── requirements-lock.txt
└── requirements-dev.txt
```

## Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Configure the local `.env` file with the required values.

Example:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=visionaryq

WHISPER_MODEL=small.en

RAG_MIN_SCORE=0.30

OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
```

The `.env` file is ignored by Git and must never be committed.

## Ollama Setup

Install Ollama and pull the configured model:

```bash
ollama pull llama3.2:3b
```

Verify that Ollama is running:

```bash
ollama list
```

VisionaryQ uses:

```text
llama3.2:3b
```

by default.

## Local Development

Create and activate a Python virtual environment.

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start the API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

## Web Interface

VisionaryQ includes a browser-based interface served directly by FastAPI.

After starting the application, open:

```text
http://localhost:8000

## API Endpoints

### Health Check

```http
GET /health
```

### API Status

```http
GET /api/v1/status
```

### Video Ingestion

```http
POST /api/v1/video/ingest
```

The endpoint accepts a video using `multipart/form-data` with a field named:

```text
file
```

Supported formats include:

```text
.mp4
.avi
.mkv
.mov
.webm
```

A successful ingestion returns information including:

```json
{
  "video_id": "<generated-id>",
  "filename": "video.mp4",
  "status": "success",
  "language": "en",
  "segment_count": 84,
  "chunk_count": 11,
  "upserted_count": 11
}
```

The returned `video_id` is used when querying the video.

### Query Video

```http
POST /api/v1/query
```

Example request:

```json
{
  "video_id": "<video-id>",
  "query": "What is generative AI?",
  "top_k": 5
}
```

`top_k` must be between `1` and `20`.

The response contains the generated answer, retrieved context, source chunks, similarity scores, and retrieval threshold.

## Running with Docker

Build the production image:

```powershell
docker build -t visionaryq:local .
```

When Ollama runs directly on the Windows or macOS host, start VisionaryQ with:

```powershell
docker run -d `
  --name visionaryq `
  -p 8000:8000 `
  --env-file .env `
  -e OLLAMA_HOST=http://host.docker.internal:11434 `
  visionaryq:local
```

Check the application:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/health" `
  -Method Get
```

Expected response:

```text
healthy
```

`host.docker.internal` is required because `localhost` inside the container refers to the container itself rather than to Ollama running on the host machine.

## Testing

Run the automated test suite with:

```powershell
pytest -q
```

The validated test suite currently contains 77 passing tests covering configuration, API behavior, ingestion, RAG, LLM integration logic, text processing, and vector services.

The application has also been validated using a real Docker end-to-end workflow covering:

```text
Video upload
→ Faster-Whisper transcription
→ Transcript chunking
→ Embedding generation
→ Pinecone vector storage
→ Semantic retrieval
→ Ollama generation
→ Grounded response
```

Additional runtime validation includes invalid request handling, unsupported video formats, retrieval persistence after replacing the Docker container, semantic retrieval, and out-of-context question rejection.

## RAG Grounding

VisionaryQ uses a configurable similarity threshold:

```env
RAG_MIN_SCORE=0.30
```

Retrieved chunks below the threshold are excluded.

If no sufficiently relevant transcript context is found, VisionaryQ does not answer using unrelated LLM knowledge and instead returns:

```text
I could not find that information in the video.
```

## Security

Sensitive configuration such as the Pinecone API key belongs only in the local `.env` file.

The following are excluded from version control and Docker build context:

* `.env`
* Python virtual environments
* Python caches
* local media
* temporary files
* logs

Never commit credentials or API keys to the repository.

## Current Production Configuration

```text
ASR             Faster-Whisper small.en
Embeddings      all-MiniLM-L6-v2
Vector Store    Pinecone
LLM             llama3.2:3b
LLM Runtime     Ollama
API             FastAPI / Uvicorn
Container       Docker
Python          3.12
PyTorch         CPU-only
```