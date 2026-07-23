"""
document_loader.py

This module loads PDF documents, cleans extracted text,
and divides the text into smaller chunks for the RAG system.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Find the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Default location of the financial-literacy PDFs.
DEFAULT_DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"


def clean_text(text: str) -> str:
    """
    Clean text extracted from a PDF.

    The function:
    - removes null characters;
    - joins words separated by PDF line-break hyphenation;
    - reduces repeated spaces;
    - reduces excessive blank lines.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Remove null characters that occasionally appear in PDF text.
    text = text.replace("\x00", " ")

    # Join words that were divided by a hyphen and line break.
    # Example: "finan-\\ncial" becomes "financial".
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace single line breaks with spaces.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Reduce three or more line breaks to two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Reduce repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def find_pdf_files(
    documents_directory: Path = DEFAULT_DOCUMENTS_DIRECTORY,
) -> List[Path]:
    """
    Find all PDF files in the documents directory.

    Args:
        documents_directory: Folder containing the PDF documents.

    Returns:
        A sorted list of PDF paths.

    Raises:
        FileNotFoundError: If the documents folder does not exist.
        ValueError: If no PDF files are found.
    """
    documents_directory = Path(documents_directory)

    if not documents_directory.exists():
        raise FileNotFoundError(
            f"Documents directory does not exist: "
            f"{documents_directory.resolve()}"
        )

    if not documents_directory.is_dir():
        raise NotADirectoryError(
            f"The supplied path is not a directory: "
            f"{documents_directory.resolve()}"
        )

    pdf_files = sorted(documents_directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files were found in: "
            f"{documents_directory.resolve()}"
        )

    return pdf_files


def load_pdf_documents(
    documents_directory: Path = DEFAULT_DOCUMENTS_DIRECTORY,
) -> List[Document]:
    """
    Load and clean every PDF in the documents directory.

    PyPDFLoader normally creates one LangChain Document object
    for each PDF page.

    Args:
        documents_directory: Folder containing the PDF documents.

    Returns:
        A list of cleaned LangChain Document objects.
    """
    pdf_files = find_pdf_files(documents_directory)

    all_documents: List[Document] = []

    print(f"\nFound {len(pdf_files)} PDF file(s).")

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

        except Exception as error:
            print(f"Could not load {pdf_path.name}: {error}")
            continue

        valid_pages = 0

        for page_document in pages:
            cleaned_content = clean_text(page_document.page_content)

            # Do not add pages that contain no usable text.
            if not cleaned_content:
                continue

            page_document.page_content = cleaned_content

            # Add clearer metadata for later citations and retrieval.
            page_number = page_document.metadata.get("page", 0)

            page_document.metadata.update(
                {
                    "source": pdf_path.name,
                    "file_path": str(pdf_path.resolve()),
                    "page_number": page_number + 1,
                    "document_title": pdf_path.stem.replace("_", " "),
                }
            )

            all_documents.append(page_document)
            valid_pages += 1

        print(
            f"Loaded {valid_pages} readable page(s) "
            f"from {pdf_path.name}."
        )

    if not all_documents:
        raise ValueError(
            "The PDFs were found, but no readable text was extracted. "
            "Check whether the documents are scanned images."
        )

    print(
        f"\nTotal readable pages loaded: {len(all_documents)}"
    )

    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Divide loaded PDF pages into smaller text chunks.

    Args:
        documents: PDF page documents.
        chunk_size: Maximum approximate number of characters per chunk.
        chunk_overlap: Number of characters repeated between neighboring chunks.

    Returns:
        A list of chunked LangChain Document objects.

    Raises:
        ValueError: If chunk settings are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    # Give each chunk a unique ID and preserve useful metadata.
    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"chunk_{chunk_number:04d}"
        chunk.metadata["chunk_size_characters"] = len(
            chunk.page_content
        )

    print(f"Total chunks created: {len(chunks)}")
    print(f"Selected chunk size: {chunk_size} characters")
    print(f"Selected overlap: {chunk_overlap} characters")

    return chunks


def display_document_summary(
    documents: List[Document],
    chunks: List[Document],
) -> None:
    """
    Print a summary of loaded documents and generated chunks.

    Args:
        documents: Loaded PDF page documents.
        chunks: Text chunks created from the documents.
    """
    sources = sorted(
        {
            document.metadata.get("source", "Unknown source")
            for document in documents
        }
    )

    total_page_characters = sum(
        len(document.page_content) for document in documents
    )

    total_chunk_characters = sum(
        len(chunk.page_content) for chunk in chunks
    )

    print("\n" + "=" * 60)
    print("DOCUMENT-LOADING SUMMARY")
    print("=" * 60)
    print(f"PDF files loaded: {len(sources)}")
    print(f"Readable PDF pages: {len(documents)}")
    print(f"Text chunks created: {len(chunks)}")
    print(f"Characters in page documents: {total_page_characters:,}")
    print(f"Characters across chunks: {total_chunk_characters:,}")

    print("\nSources:")

    for source in sources:
        source_pages = sum(
            1
            for document in documents
            if document.metadata.get("source") == source
        )

        source_chunks = sum(
            1
            for chunk in chunks
            if chunk.metadata.get("source") == source
        )

        print(
            f"- {source}: "
            f"{source_pages} readable page(s), "
            f"{source_chunks} chunk(s)"
        )

    print("=" * 60)


def preview_chunks(
    chunks: List[Document],
    number_of_chunks: int = 3,
) -> None:
    """
    Display a preview of the first few chunks.

    Args:
        chunks: Generated text chunks.
        number_of_chunks: Number of chunks to display.
    """
    if not chunks:
        print("No chunks are available for preview.")
        return

    preview_count = min(number_of_chunks, len(chunks))

    print(f"\nPreviewing {preview_count} chunk(s):")

    for position, chunk in enumerate(
        chunks[:preview_count],
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Preview {position}")
        print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Page: {chunk.metadata.get('page_number')}")
        print(
            f"Characters: "
            f"{chunk.metadata.get('chunk_size_characters')}"
        )
        print("-" * 60)

        preview_text = chunk.page_content[:500]

        if len(chunk.page_content) > 500:
            preview_text += "..."

        print(preview_text)


def prepare_documents(
    documents_directory: Path = DEFAULT_DOCUMENTS_DIRECTORY,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Run the complete document-preparation process.

    This is the main function that later parts of the RAG system
    will import.

    Args:
        documents_directory: Folder containing PDF files.
        chunk_size: Maximum approximate characters per chunk.
        chunk_overlap: Repeated characters between chunks.

    Returns:
        Prepared text chunks.
    """
    documents = load_pdf_documents(documents_directory)

    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    display_document_summary(
        documents=documents,
        chunks=chunks,
    )

    return chunks


if __name__ == "__main__":
    try:
        prepared_chunks = prepare_documents(
            chunk_size=1000,
            chunk_overlap=150,
        )

        preview_chunks(
            chunks=prepared_chunks,
            number_of_chunks=3,
        )

        print("\nDocument preparation completed successfully.")

    except Exception as error:
        print(f"\nDocument preparation failed: {error}")