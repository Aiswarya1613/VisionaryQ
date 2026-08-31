import os
from typing import List, Dict, Any

from dotenv import load_dotenv


load_dotenv()


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Responsibilities:

    1. Query processing
    2. Vector retrieval
    3. Context construction
    4. LLM generation
    5. Response formatting
    """

    def __init__(
        self,
        vector_service=None,
        llm_service=None,
        min_score: float | None = None
    ):
        self.vector_service = vector_service
        self.llm_service = llm_service

        if min_score is None:
            min_score = float(
                os.getenv(
                    "RAG_MIN_SCORE",
                    "0.30"
                )
            )

        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        video_id: str | None = None
    ):
        """
        Retrieve relevant documents from the vector database.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.vector_service is None:
            raise RuntimeError(
                "Vector service has not been configured."
            )

        return self.vector_service.search(
            query=query,
            top_k=top_k,
            video_id=video_id
        )

    def build_context(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Build context from retrieved documents.
        """

        if not documents:
            return ""

        context_parts = []

        for document in documents:

            text = document.get(
                "text",
                ""
            )

            if text:
                context_parts.append(
                    text.strip()
                )

        return "\n\n".join(
            context_parts
        )

    def generate_answer(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate an answer using the configured LLM.
        """

        if self.llm_service is None:
            raise RuntimeError(
                "LLM service has not been configured."
            )

        return self.llm_service.generate(
            query=query,
            context=context
        )

    def query(
        self,
        query: str,
        top_k: int = 5,
        video_id: str | None = None
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.

        Pipeline:
            Query
            -> Vector retrieval
            -> Relevance filtering
            -> Context construction
            -> LLM generation
            -> Response
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        # ---------------------------------------------
        # Step 1: Retrieve candidate documents
        # ---------------------------------------------

        documents = self.retrieve(
            query=query,
            top_k=top_k,
            video_id=video_id
        )

        # ---------------------------------------------
        # Step 2: Remove weak retrieval results
        # ---------------------------------------------

        relevant_documents = [
            document
            for document in documents
            if document.get("score", 0.0)
            >= self.min_score
        ]

        # ---------------------------------------------
        # Step 3: Handle no relevant information
        # ---------------------------------------------

        if not relevant_documents:

            return {
                "query": query,
                "video_id": video_id,
                "answer": (
                    "I could not find that information "
                    "in the video."
                ),
                "context": "",
                "sources": [],
                "retrieval_threshold": self.min_score
            }

        # ---------------------------------------------
        # Step 4: Build context
        # ---------------------------------------------

        context = self.build_context(
            relevant_documents
        )

        if not context:

            return {
                "query": query,
                "video_id": video_id,
                "answer": (
                    "I could not find that information "
                    "in the video."
                ),
                "context": "",
                "sources": [],
                "retrieval_threshold": self.min_score
            }

        # ---------------------------------------------
        # Step 5: Generate grounded answer
        # ---------------------------------------------

        answer = self.generate_answer(
            query=query,
            context=context
        )

        return {
            "query": query,
            "video_id": video_id,
            "answer": answer,
            "context": context,
            "sources": relevant_documents,
            "retrieval_threshold": self.min_score
        }