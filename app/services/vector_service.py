from typing import Any, Dict, List

from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings
from app.services.embedding_service import EmbeddingService


class VectorService:
    """
    Pinecone-backed vector database service.

    Responsibilities:
    1. Initialize Pinecone
    2. Create the required index if it does not exist
    3. Generate embeddings
    4. Store document vectors
    5. Perform semantic search
    """

    def __init__(
        self,
        embedding_service=None
    ):
        settings = get_settings()

        self.api_key = (
            settings.pinecone_api_key
        )

        self.index_name = (
            settings.pinecone_index_name
        )

        self.embedding_service = (
            embedding_service
            or EmbeddingService()
        )

        self.client = None
        self.index = None
        self.initialized = False

    def initialize(self):
        """
        Initialize Pinecone and connect to the index.
        """

        if not self.api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not configured."
            )

        print("Initializing Pinecone...")

        self.client = Pinecone(
            api_key=self.api_key
        )

        existing_indexes = [
            index.name
            for index in self.client.list_indexes()
        ]

        if self.index_name not in existing_indexes:

            print(
                f"Creating Pinecone index: "
                f"{self.index_name}"
            )

            dimension = (
                self.embedding_service.get_dimension()
            )

            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = self.client.Index(
            self.index_name
        )

        self.initialized = True

        print(
            f"Pinecone initialized successfully: "
            f"{self.index_name}"
        )

    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ):
        """
        Generate embeddings and store documents
        in Pinecone.

        Each document should contain:

        {
            "id": "...",
            "text": "...",
            "metadata": {...}
        }
        """

        if not self.initialized:
            raise RuntimeError(
                "Vector service has not been initialized."
            )

        if not documents:
            return {
                "upserted_count": 0
            }

        texts = [
            document["text"]
            for document in documents
        ]

        embeddings = (
            self.embedding_service.embed_documents(
                texts
            )
        )

        vectors = []

        for document, embedding in zip(
            documents,
            embeddings
        ):

            vector_id = document.get(
                "id"
            )

            if not vector_id:
                raise ValueError(
                    "Every document must contain an 'id'."
                )

            metadata = document.get(
                "metadata",
                {}
            )

            metadata["text"] = document["text"]

            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                }
            )

        self.index.upsert(
            vectors=vectors
        )

        return {
            "upserted_count": len(vectors)
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        video_id: str | None = None
    ):
        """
        Perform semantic search against Pinecone.
        """

        if not self.initialized:
            raise RuntimeError(
                "Vector service has not been initialized."
            )

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = (
            self.embedding_service.embed_text(
                query
            )
        )

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"video_id": video_id}
        )

        documents = []

        for match in results.matches:

            metadata = match.metadata or {}

            documents.append(
                {
                    "id": match.id,
                    "score": float(match.score),
                    "text": metadata.get(
                        "text",
                        ""
                    ),
                    "metadata": metadata
                }
            )

        return documents