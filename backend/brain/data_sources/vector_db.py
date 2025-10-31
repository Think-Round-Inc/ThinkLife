"""
Vector Database Data Source
Provides vector database integration for semantic search
"""

import os
import logging
from typing import Dict, Any, List, Optional

from ..interfaces import IDataSource, DataSourceType
from .base_data_source import DataSourceConfig

logger = logging.getLogger(__name__)


class VectorDataSource(IDataSource):
    """Vector database data source implementation"""
    
    def __init__(self, vectorstore, config: DataSourceConfig):
        self.vectorstore = vectorstore
        self.config = config
        self._initialized = False
        self._is_external = False  # Track if this is an external agent-specific source
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize vector data source"""
        self._initialized = self.vectorstore is not None
        if self._initialized:
            logger.info(f"Vector data source {self.config.source_id} initialized")
        return self._initialized
    
    @classmethod
    async def create_from_path(cls, db_path: str, source_id: str = None):
        """
        Create a VectorDataSource from an external ChromaDB path (agent-specific)
        
        Args:
            db_path: Path to chroma.sqlite3 file or ChromaDB directory
            source_id: Optional identifier for this source
            
        Returns:
            VectorDataSource instance or None if creation fails
        """
        try:
            import chromadb
            from chromadb.config import Settings
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings
            
            # Determine the persist directory
            if db_path.endswith('chroma.sqlite3'):
                persist_directory = os.path.dirname(db_path)
            else:
                persist_directory = db_path
            
            if not os.path.exists(persist_directory):
                logger.error(f"External ChromaDB path does not exist: {persist_directory}")
                return None
            
            # Initialize ChromaDB client with external path
            chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                collections = chroma_client.list_collections()
                if collections:
                    collection_name = collections[0].name
                else:
                    collection_name = "default_collection"
            except:
                collection_name = "default_collection"
            
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create Chroma vectorstore
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            
            # Create config
            config = DataSourceConfig(
                source_id=source_id or f"external_vector_{os.path.basename(persist_directory)}",
                source_type=DataSourceType.VECTOR_DB,
                priority=10,
                config={"persist_directory": persist_directory, "external": True}
            )
            
            # Create instance
            instance = cls(vectorstore, config)
            instance._is_external = True
            await instance.initialize({})
            
            logger.info(f"Created external VectorDataSource from: {persist_directory}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create VectorDataSource from path {db_path}: {str(e)}")
            return None
    
    async def query(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query vector database"""
        if not self._initialized:
            return []
        
        try:
            k = kwargs.get("k", 5)
            filter_criteria = kwargs.get("filter")
            
            # Basic vector search
            docs = self.vectorstore.similarity_search(query, k=k, filter=filter_criteria)
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "vector_db",
                    "score": getattr(doc, 'score', None)
                }
                for doc in docs
            ]
            
        except Exception as e:
            logger.error(f"Vector query failed: {str(e)}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector database health"""
        try:
            if self.vectorstore and hasattr(self.vectorstore, '_collection'):
                count = self.vectorstore._collection.count()
                return {
                    "status": "healthy",
                    "document_count": count,
                    "initialized": self._initialized,
                    "is_external": self._is_external
                }
            else:
                return {
                    "status": "healthy" if self._initialized else "not_initialized",
                    "initialized": self._initialized,
                    "is_external": self._is_external
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }
    
    async def close(self) -> None:
        """Close vector database connection"""
        # Vector stores typically don't need explicit closing
        self._initialized = False
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.VECTOR_DB


# Factory function
def create_vector_data_source(vectorstore, config: DataSourceConfig = None) -> VectorDataSource:
    """Create a vector data source instance"""
    if config is None:
        config = DataSourceConfig(
            source_id="vector_db",
            source_type=DataSourceType.VECTOR_DB,
            priority=10
        )
    return VectorDataSource(vectorstore, config)

