from typing import Optional

import ollama


class LLMService:
    """
    Local LLM service using Ollama.

    The model receives the user's query together
    with retrieved context from the RAG pipeline.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b"
    ):
        self.model = model

    def generate(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate a grounded answer using retrieved context.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not context or not context.strip():
            return (
                "I could not find relevant information "
                "in the available video content."
            )

        prompt = f"""
You are VisionaryQ, a video question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find that information in the video."

Do not invent facts.
Do not use outside knowledge.
Keep the answer concise and clear.

Context:
{context}

Question:
{query}

Answer:
"""

        response = ollama.generate(
            model=self.model,
            prompt=prompt
        )

        return response["response"].strip()