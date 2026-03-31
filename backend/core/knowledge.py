import os
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def setup_knowledge_base(config):
    """
    Initializes the Supabase vector store and raw client.
    Returns:
        tuple (vector_db, supabase_client, embeddings)
    """
    print("Initializing Supabase Knowledge Base...")
    
    # 1. Initialize Supabase Client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        
    supabase_client = create_client(supabase_url, supabase_key)
    
    # 2. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # 3. Initialize Vector Store
    vector_db = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name=config.VECTOR_TABLE,
        query_name=config.VECTOR_QUERY
    )
    
    return vector_db, supabase_client, embeddings

def add_documents_to_knowledge_base(text: str, config, vector_db):
    """
    Chunks the input text and adds it to the Supabase vector store via pgvector.
    """
    print(f"Adding new content to Supabase (text length: {len(text)})")
    
    # 1. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    
    # 2. Add to Supabase
    vector_db.add_texts(chunks)
    
    print(f"Knowledge Base updated in Supabase table: {config.VECTOR_TABLE}")
    
    return vector_db
