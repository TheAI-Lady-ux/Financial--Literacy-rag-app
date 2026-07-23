"""
AI-Powered Personal Finance RAG Assistant

This file connects:
1. The saved FAISS vector database
2. The Hugging Face embedding model
3. The Ollama language model
4. The user's question
5. The retrieved document sources

Run from the main project folder with:

    python rag_assistant.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# ---------------------------------------------------------
# Project paths and settings
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "vector_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2"
DEFAULT_TOP_K = 4


# ---------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------

RAG_PROMPT = PromptTemplate.from_template(
    """
You are an educational personal-finance assistant.

Answer the user's question using only the retrieved context below.

Rules:
1. Do not invent information.
2. Do not use information that is not supported by the context.
3. Explain the answer clearly and in simple language.
4. Do not provide professional investment, tax, legal, or credit advice.
5. If the answer is not available in the context, say:
   "I cannot find this information in the uploaded documents."
6. When appropriate, provide practical educational suggestions based
   on the retrieved documents.

Retrieved context:
{context}

User question:
{question}

Answer:
"""
)

RAG_PROMPT = PromptTemplate.from_template(
    """
You are a supportive personal finance and financial-literacy assistant.

Use clear and simple language.
Maintain an encouraging and nonjudgmental tone.
Provide practical, step-by-step advice.
Answer using only the retrieved course-document context.

If the answer cannot be found in the context, say:
"I cannot find this information in the uploaded course materials.
Please ask another question or provide additional documents."

Context:
{context}

Question:
{question}

Answer:
"""
)
# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the same embedding model used to build the FAISS index."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------
# Load FAISS vector database
# ---------------------------------------------------------

def load_vector_store() -> FAISS:
    """Load the saved FAISS vector database."""
    if not VECTOR_STORE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vector-store folder not found: {VECTOR_STORE_DIRECTORY}"
        )

    index_file = VECTOR_STORE_DIRECTORY / "index.faiss"
    metadata_file = VECTOR_STORE_DIRECTORY / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "FAISS index files were not found. Run vector_store.py first."
        )

    print("Loading embedding model...")
    embeddings = get_embedding_model()

    print("Loading FAISS vector database...")
    store = FAISS.load_local(
        str(VECTOR_STORE_DIRECTORY),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("FAISS vector database loaded successfully.")
    return store


# ---------------------------------------------------------
# Load Ollama
# ---------------------------------------------------------

def get_ollama_model() -> OllamaLLM:
    """Create the local Ollama language model."""
    return OllamaLLM(
        model=OLLAMA_MODEL_NAME,
        temperature=0,
        num_predict=500,
    )


# ---------------------------------------------------------
# Retrieve relevant chunks
# ---------------------------------------------------------

def retrieve_documents(
    question: str,
    vector_store: FAISS,
    top_k: int = DEFAULT_TOP_K,
) -> list[Document]:
    """Retrieve the most relevant document chunks."""
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("The question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return retriever.invoke(clean_question)


# ---------------------------------------------------------
# Format retrieved context
# ---------------------------------------------------------

def format_context(documents: list[Document]) -> str:
    """Combine retrieved document chunks into one context string."""
    if not documents:
        return "No relevant information was retrieved from the uploaded documents."

    context_sections: list[str] = []

    for position, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "source_file",
            document.metadata.get("source", "Unknown document"),
        )

        page_number = document.metadata.get(
            "page_number",
            document.metadata.get("page", "Unknown"),
        )

        chunk_id = document.metadata.get("chunk_id", "Unknown")

        context_sections.append(
            f"Source {position}\n"
            f"Document: {source}\n"
            f"Page: {page_number}\n"
            f"Chunk: {chunk_id}\n"
            f"Text:\n{document.page_content}"
        )

    return "\n\n".join(context_sections)


# ---------------------------------------------------------
# Prepare source information
# ---------------------------------------------------------

def create_source_list(documents: list[Document]) -> list[dict[str, Any]]:
    """Create a list of source details for display."""
    sources: list[dict[str, Any]] = []

    for document in documents:
        source = document.metadata.get(
            "source_file",
            document.metadata.get("source", "Unknown document"),
        )

        page_number = document.metadata.get(
            "page_number",
            document.metadata.get("page", "Unknown"),
        )

        chunk_id = document.metadata.get("chunk_id", "Unknown")

        preview = document.page_content[:300]
        if len(document.page_content) > 300:
            preview += "..."

        sources.append(
            {
                "source": source,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "preview": preview,
            }
        )

    return sources


# ---------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------

def ask_rag(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run retrieval, prompting, and local Ollama generation."""
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Please enter a question.")

    store = load_vector_store()

    retrieved_documents = retrieve_documents(
        question=clean_question,
        vector_store=store,
        top_k=top_k,
    )

    context = format_context(retrieved_documents)

    final_prompt = RAG_PROMPT.format(
        context=context,
        question=clean_question,
    )

    ollama_model = get_ollama_model()
    answer = ollama_model.invoke(final_prompt)

    return {
        "question": clean_question,
        "answer": answer,
        "sources": create_source_list(retrieved_documents),
        "retrieved_documents": retrieved_documents,
        "top_k": top_k,
    }


# ---------------------------------------------------------
# Display the result in the terminal
# ---------------------------------------------------------

def display_result(result: dict[str, Any]) -> None:
    """Print the RAG answer and retrieved sources."""
    print("\n" + "=" * 70)
    print("AI PERSONAL FINANCE RAG ASSISTANT")
    print("=" * 70)

    print("\nQUESTION")
    print("-" * 70)
    print(result["question"])

    print("\nANSWER")
    print("-" * 70)
    print(result["answer"])

    print("\nRETRIEVED SOURCES")
    print("-" * 70)

    sources = result.get("sources", [])

    if not sources:
        print("No document sources were retrieved.")

    for number, source in enumerate(sources, start=1):
        print(f"\nSource {number}")
        print(f"Document: {source['source']}")
        print(f"Page: {source['page_number']}")
        print(f"Chunk: {source['chunk_id']}")
        print(f"Preview: {source['preview']}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------
# Interactive terminal test
# ---------------------------------------------------------

def run_interactive_test() -> None:
    """Run the RAG assistant in the terminal."""
    print("\nAI Personal Finance RAG Assistant")
    print("Ask a financial-literacy question.")
    print("Type 'exit' when you are finished.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() in {"exit", "quit", "stop"}:
            print("\nRAG assistant closed.")
            break

        try:
            result = ask_rag(question=question, top_k=DEFAULT_TOP_K)
            display_result(result)

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            print(
                "\nPlease check that:"
                "\n1. Ollama is installed and running."
                "\n2. The llama3.2 model is installed."
                "\n3. vector_store/index.faiss and index.pkl exist."
                "\n4. Your virtual environment is active."
            )


if __name__ == "__main__":
    run_interactive_test()
RAG_PROMPT = PromptTemplate.from_template(
    """
You are a supportive personal finance and financial-literacy assistant.

Use clear and simple language.
Maintain an encouraging and nonjudgmental tone.
Provide practical, step-by-step advice.
Answer using only the retrieved course-document context.

If the answer cannot be found in the context, say:
"I cannot find this information in the uploaded course materials.
Please ask another question or provide additional documents."

Context:
{context}

Question:
{question}

Answer:
"""
)
   