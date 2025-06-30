import os
import uuid
import tempfile

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


def create_cache_dir(directory=None):
    """Create cache directory if it doesn't exist."""
    if not directory:
        directory = './.cache'

    os.makedirs(directory, exist_ok=True)
    return directory


def load_pdf(file_path):
    """Load a single PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_pdf_directory(directory):
    """Load all PDF files from a directory."""
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()


def split_pdf(pdfs: list[Document]):
    """
    Splits a list of Document objects into smaller chunks for retrieval purposes.
    
    Args:
        pdfs (List[Document]): List of LangChain Document objects.

    Returns:
        List[Document]: List of split Document chunks.
    """
    if not pdfs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False
    )

    return splitter.split_documents(pdfs)


def extract_pdf(uploaded_pdf):
    """
    Extract uploaded PDF files to a temporary directory with unique file names.
    
    Args:
        uploaded_pdf: Streamlit UploadedFile object or list of UploadedFile objects
        
    Returns:
        str: Path to the temporary directory containing extracted files
    """
    # Create a unique temporary directory
    temp_dir = tempfile.mkdtemp(prefix='pdf_extract_')
    
    # Normalize to a list if it's a single file
    if not isinstance(uploaded_pdf, list):
        uploaded_pdf = [uploaded_pdf]

    try:
        for file in uploaded_pdf:
            # Generate unique filename to avoid conflicts
            unique_id = str(uuid.uuid4())
            
            # Support both Streamlit UploadedFile and raw bytes
            if hasattr(file, "name") and hasattr(file, "getvalue"):
                # Use original filename with unique prefix
                safe_filename = f"{unique_id}_{file.name}"
                file_path = os.path.join(temp_dir, safe_filename)
                content = file.getvalue()
            else:
                # Assume raw bytes
                file_path = os.path.join(temp_dir, f"{unique_id}_uploaded.pdf")
                content = file

            with open(file_path, 'wb') as w:
                w.write(content)
                
    except Exception as e:
        # Clean up on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Failed to extract PDF: {str(e)}")

    return temp_dir

