from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="VisionaryQ API",
    description="Production-style RAG API for video understanding and semantic search.",
    version="1.0.0",
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "application": "VisionaryQ",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }