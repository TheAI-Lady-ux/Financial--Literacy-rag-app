from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"

VECTOR_STORE_DIRECTORY = PROJECT_ROOT / "data" / "vector_store"


# ---------------------------------------------------------
# Load PDF documents
# ---------------------------------------------------------

def load_pdf_documents(
    documents_directory: Path
) -> List[Document]:
    """
    Load all PDF files from the documents directory.
    """

    documents_directory = Path(documents_directory)

    if not documents_directory.exists():
        raise FileNotFoundError(
            f"Documents directory does not exist: "
            f"{documents_directory}"
        )

    pdf_files = sorted(
        documents_directory.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in: "
            f"{documents_directory}"
        )

    all_documents: List[Document] = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(
            str(pdf_file)
        )

        pages = loader.load()

        all_documents.extend(pages)

    print("Document preparation completed successfully.")

    return all_documents


# ---------------------------------------------------------
# Split documents into chunks
# ---------------------------------------------------------

def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split loaded PDF pages into smaller overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks


# ---------------------------------------------------------
# Create and save FAISS vector store
# ---------------------------------------------------------

def create_vector_store() -> FAISS:
    """
    Load documents, split them, create embeddings,
    build the FAISS vector store, and save it locally.
    """

    print("\nLoading PDF documents...")

    documents = load_pdf_documents(
        DOCUMENTS_DIRECTORY
    )

    print(f"PDF pages loaded: {len(documents)}")

    print("\nSplitting documents into chunks...")

    chunks = split_documents(
        documents=documents,
        chunk_size=500,
        chunk_overlap=50
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the documents."
        )

    print("Selected chunk size: 500 characters")
    print("Selected overlap: 50 characters")
    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Creating FAISS vector database...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    VECTOR_STORE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_STORE_DIRECTORY)
    )

    print(
        "\nVector store saved successfully to:"
    )
    print(VECTOR_STORE_DIRECTORY)

    return vector_store


# ---------------------------------------------------------
# Test the saved vector store
# ---------------------------------------------------------

def test_retrieval(
    vector_store: FAISS
) -> None:
    """
    Run a test query against the vector store.
    """

    question = (
        "How can someone improve their spending "
        "and saving habits?"
    )

    print("\nTesting vector-store retrieval...")
    print(f"Question: {question}")

    results = vector_store.similarity_search(
        question,
        k=3
    )

    if not results:
        print("No matching document chunks were retrieved.")
        return

    print(f"Retrieved documents: {len(results)}")

    for number, document in enumerate(
        results,
        start=1
    ):
        print("\n" + "=" * 60)
        print(f"Result {number}")
        print("=" * 60)

        content = document.page_content.strip()

        print(content[:700])

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        print(f"\nSource: {source}")
        print(f"Page: {page}")


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:
    try:
        vector_store = create_vector_store()

        test_retrieval(vector_store)

        print(
            "\nVector-store creation completed successfully."
        )

    except Exception as error:
        print(
            f"\nVector-store creation failed: {error}"
        )


if __name__ == "__main__":
    main()