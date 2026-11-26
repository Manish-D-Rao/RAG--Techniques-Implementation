import logging
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import(
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

def load_document(file_path: str | Path) -> List[Document]:
    """
    Load a document from disk into LangChain Document objects.

    Supports PDF, plain text, and DOCX files. The file extension determines
    which loader is used. Returns the parsed documents with their metadata.

    Args:
        file_path: Path to the file to load. Accepts str or pathlib.Path.

    Returns:
        A list of Document objects extracted from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not in SUPPORTED_EXTENSIONS.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} was not found.")

    if not path.is_file():
        raise ValueError(f"{path} is not a file.")
    
    extension = path.suffix.lower() # document.pdf -> .pdf 
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file '{extension}'."
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    elif extension == ".txt":
        loader = TextLoader(str(path), encoding = "utf-8")
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    documents = loader.load()
    cleaned_documents = []

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        if doc.page_content.strip():
            cleaned_documents.append(doc)

    
    if not cleaned_documents:
        logger.warning("No readable text found in %s. Skipping it.", path.name)
    
    logger.info("Loaded %d document(s) from %s", len(cleaned_documents), path.name)
    return add_metadata(cleaned_documents, path)


def load_documents_from_folder(folder_path: str | Path | None = None) -> List[Document]:
    """
    Load every supported document in a folder into LangChain Document objects.

    Scans the given folder (non-recursively) for files with extensions in
    SUPPORTED_EXTENSIONS, loads each one via load_document, and returns the
    combined list. Unsupported files are skipped.

    Args:
        folder_path: Folder to scan. Defaults to the project's ``data`` directory.

    Returns:
        A combined list of Document objects from all supported files.

    Raises:
        FileNotFoundError: If the folder does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if folder_path is None:
        folder_path = Path(__file__).resolve().parent.parent / "data"
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} was not found.")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path {folder} is not a directory.")

    all_documents: List[Document] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            documents = load_document(file_path)
            all_documents.extend(documents)
        except Exception as e:
            logger.exception("Failed to load %s. Skipping it: %s", file_path.name, e)

    logger.info(
        "Loaded %d document(s) from %d file(s) in %s",
        len(all_documents),
        len({doc.metadata.get("source") for doc in all_documents}),
        folder,
    )
    return all_documents

def clean_text(text: str) -> str:
    """
    Normalize and clean raw text extracted from a document for RAG chunking.

    Preserves line breaks (so headings/paragraphs survive for chunkers) while
    removing control characters and collapsing extra spaces and blank lines.

    Args:
        text: The raw text to clean.

    Returns:
        The cleaned text with structure preserved.
    """
    if not text:
        return ""

    text = text.replace("\x0c", "\n\n")          # page break -> paragraph break
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", "", text)  # drop control chars
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # cap blank lines
    return text.strip()


def add_metadata(documents: List[Document], path: Path) -> List[Document]:
    """
    Attach source file metadata to each document.

    Adds common provenance fields (source file name, full path, extension)
    while preserving any page-level metadata already present from the loader.

    Args:
        documents: The documents to annotate.
        path: The Path of the file the documents were loaded from.

    Returns:
        The same list of documents with enriched metadata.
    """
    for doc in documents:
        doc.metadata.update(
            {
                "source": str(path),
                "filename": path.name,
                "doc_type": path.suffix.lower().lstrip("."),
            }
        )

        page = doc.metadata.get("page")

        if page is not None:
            try:
                doc.metadata["page_number"] = int(page) + 1
            except (ValueError, TypeError):
                logger.warning("Invalid page metadata in %s", path.name)

    return documents

def guess_section(text: str) -> str:
    """
    Guess the section heading a piece of text belongs to, for citations.

    Scans the first 15 lines for a short heading-like line that is either
    ALL CAPS, Title Case, or numbered (e.g. "1. Purpose" / "2.3 Scope"),
    then returns it with any leading section number stripped. Body lines
    (long, or ending in sentence punctuation) are ignored.

    Args:
        text: The text to inspect, typically a document page or chunk.

    Returns:
        The cleaned section heading, or an empty string if none is found.
    """
    for raw_line in text.split("\n")[:15]:
        line = raw_line.strip()

        if not (3 < len(line) < 70) or line.endswith((".", ",", ";", ":")):
            continue

        if len(line.split()) > 8:
            continue

        looks_numbered = re.match(
            r"^\d+(?:\.\d+)*\.?\s+\S",
            line
        ) is not None

        if line.isupper() or line.istitle() or looks_numbered:
            return re.sub(
                r"^\d+(?:\.\d+)*\.?\s+",
                "",
                line
            ).strip()

    return ""