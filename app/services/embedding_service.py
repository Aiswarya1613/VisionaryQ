from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Local text embedding service using Sentence Transformers.

    The embedding model runs locally, so no paid API is required.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = None

    def initialize(self):
        """
        Load the embedding model.
        """

        if self.model is None:
            print(
                f"Loading embedding model: {self.model_name}"
            )

            self.model = SentenceTransformer(
                self.model_name
            )

            print("Embedding model loaded successfully.")

    def embed_text(
        self,
        text: str
    ) -> List[float]:
        """
        Generate an embedding for a single text.
        """

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        if self.model is None:
            self.initialize()

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(
            np.float32
        ).tolist()

    def embed_documents(
        self,
        documents: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        """

        if not documents:
            return []

        if self.model is None:
            self.initialize()

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.astype(
            np.float32
        ).tolist()

    def get_dimension(self) -> int:
        """
        Return the embedding vector dimension.
        """

        if self.model is None:
            self.initialize()

        return self.model.get_embedding_dimension()