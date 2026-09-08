from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="VisionaryQ API",
    description=(
        "Production-style RAG application for "
        "video understanding and semantic search."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)


@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def root(request: Request):
    """
    Serve the VisionaryQ web interface.
    """

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# ---------------------------------------------------------
# API
# ---------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }