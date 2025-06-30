from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index

from src.pdf_handler import extract_pdf, load_pdf_directory, split_pdf

import os
import shutil
from dotenv import load_dotenv

load_dotenv()


def setup_pinecone(index_name, embedding_model, embedding_dim, metric='cosine', use_serverless=True):
    """Setup Pinecone vector database with proper error handling and dimension checking."""
    try:
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        
        # Check if index exists and get its current dimensions
        index_exists = index_name in pc.list_indexes().names()
        current_dim = None
        
        if index_exists:
            try:
                index_stats = pc.describe_index(index_name)
                current_dim = index_stats.dimension
            except Exception as e:
                print(f"Warning: Could not get index stats: {e}")
        
        # If index exists but dimensions don't match, delete and recreate
        if index_exists and current_dim and current_dim != embedding_dim:
            print(f"Dimension mismatch detected: index has {current_dim} dimensions, but model requires {embedding_dim}")
            print("Deleting existing index and recreating with correct dimensions...")
            pc.delete_index(index_name)
            index_exists = False
        
        # Create index if it doesn't exist
        if not index_exists:
            if use_serverless:
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
            else:
                spec = PodSpec()

            pc.create_index(
                index_name,
                dimension=embedding_dim,
                metric=metric,
                spec=spec
            )
            print(f"Created Pinecone index '{index_name}' with {embedding_dim} dimensions")
        
        return PineconeVectorStore(index_name=index_name, embedding=embedding_model)
    except Exception as e:
        raise Exception(f"Failed to setup Pinecone: {str(e)}")


def setup_chroma(index_name, embedding_model, persist_directory=None):
    """Setup Chroma vector database."""
    if not persist_directory:
        persist_directory = './.cache/database'

    os.makedirs(persist_directory, exist_ok=True)

    db = Chroma(index_name, embedding_function=embedding_model, persist_directory=persist_directory)
    return db


class VectorDB:
    def __init__(self, db_name, index_name, cache_dir=None):
        from src.utils import load_config

        config = load_config()
        embedding = OpenAIEmbeddings(
            model=config['embedding_model']['model_name'],
            openai_api_key=os.environ.get('OPENAI_API_KEY')
        )

        if not cache_dir:
            cache_dir = './.cache/database'
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Use the embedding dimensions from config instead of hardcoded value
        embedding_dim = config['embedding_model'].get('dimensions', 1536)

        if db_name == 'pinecone':
            self.vectorstore = setup_pinecone(index_name, embedding, embedding_dim, 'cosine')
        else:
            self.vectorstore = setup_chroma(index_name, embedding, self.cache_dir)

        namespace = f'{db_name}/{index_name}'
        self.record_manager = SQLRecordManager(namespace,
                                               db_url=f'sqlite:///{self.cache_dir}/record_manager_cache.sql')
        self.record_manager.create_schema()

    def index(self, uploaded_file):
        """Index uploaded PDF file into the vector database."""
        try:
            directory = extract_pdf(uploaded_file)
            docs = load_pdf_directory(directory)
            chunks = split_pdf(docs)

            index(
                docs_source=chunks,
                record_manager=self.record_manager,
                vector_store=self.vectorstore,
                cleanup='full',
                source_id_key='source'
            )

            # Clean up temporary files
            for file in os.listdir(directory):
                os.remove(os.path.join(directory, file))
            os.rmdir(directory)
            
        except Exception as e:
            raise Exception(f"Failed to index document: {str(e)}")

    def as_retriever(self):
        return self.vectorstore.as_retriever()

    def __del__(self):
        try:
            # Try to close the SQLite connection gracefully first
            if hasattr(self.record_manager, 'engine'):
                self.record_manager.engine.dispose()
            # Don't remove the cache directory as it might be needed for persistence
        except Exception as e:
            print(f"Cleanup warning: {e}")
