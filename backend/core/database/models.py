from sqlalchemy import Column, Text, UUID, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    """
    SQLAlchemy model for storing document chunks and their embeddings.
    Using JSONB for metadata to support PostgreSQL containment operators (@>).
    """
    __tablename__ = 'documents'
    
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        server_default=func.gen_random_uuid()
    )
    content = Column(Text, nullable=False)
    metadata_ = Column(JSONB, name='metadata')  # 'metadata' is a reserved word in some contexts
    # 384 is the dimension for sentence-transformers/all-MiniLM-L6-v2
    embedding = Column(Vector(384))
