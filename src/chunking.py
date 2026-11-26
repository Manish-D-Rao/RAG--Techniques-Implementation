from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from src.data_ingestion import guess_section

def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter configured for structure-aware chunking.

    Splits on paragraph, line, sentence, word, then character boundaries (in that
    order) so chunks break at natural semantic seams. Chunk size is measured by
    character length.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ". ", " ", ""],
        length_function = len,
    )

def split_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    """
    Split documents into parent and child chunks for parent-child retrieval.

    Each input document is first split into large parent chunks (for context),
    then each parent is split into smaller child chunks (for precise retrieval).
    Every chunk receives a stable id and section metadata so children can be
    traced back to their parent and source.

    Args:
        documents: The documents to split.

    Returns:
        A tuple of (parents, children) document lists.
    """
    parent_splitter = make_splitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
    child_splitter = make_splitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP)

    parents: List[Document] = []
    children: List[Document] = []

    parent_per_file : Dict[str, int] = {}

    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")

        for parent_text in parent_splitter.split_text(doc.page_content):
            parent_number = parent_per_file.get(filename, 0)
            parent_per_file[filename] = parent_number+1
            parent_id = f"{filename}::{parent_number}"

            parent_metadata = dict(doc.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = guess_section(parent_text)

            parents.append(Document(page_content = parent_text, metadata = parent_metadata))

            for child_index, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = f"{parent_id}::{child_index}"
                children.append(Document(page_content=child_text, metadata = child_metadata))

    return parents, children

def build_parent_lookup(parents: List[Document]) -> Dict[str, Document]:
    """
    Build a lookup table mapping parent_id to its parent Document.

    Used to expand a retrieved child chunk back to its full parent context.

    Args:
        parents: The list of parent documents.

    Returns:
        A dict keyed by each parent's ``parent_id`` metadata value.
    """
    return {parent.metadata["parent_id"]: parent for parent in parents}
