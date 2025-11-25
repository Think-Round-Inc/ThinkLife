"""
Vector Database Data Source - Integration with vector databases for semantic search
"""

import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..specs import IDataSource, DataSourceType, DataSourceSpec
from .base_data_source import DataSourceConfig
from .data_source_registry import check_data_source_spec_availability

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB libraries not available. Install with: pip install chromadb langchain-chroma")
    CHROMA_AVAILABLE = False
    chromadb = None
    Chroma = None


class VectorDataSource(IDataSource):
    """Vector database data source for semantic search"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.vectorstore = None
        self._initialized = False
        self._is_external = False
    
    async def initialize(self, init_config: Dict[str, Any] = None) -> bool:
        """Validate config and initialize vector database"""
        if not CHROMA_AVAILABLE:
            logger.error("ChromaDB library not available")
            return False
        
        # Validate with registry
        if not self._validate_config():
            return False
        
        # Initialize vectorstore
        try:
            # Check if external db_path is provided
            db_path = self.config.config.get("db_path") or init_config.get("db_path")
            vectorstore = init_config.get("vectorstore")
            
            if vectorstore:
                # Use provided vectorstore
                self.vectorstore = vectorstore
            elif db_path:
                # Create from external path
                self.vectorstore = await self._create_from_path(db_path)
                self._is_external = True
            else:
                # Create default vectorstore with embeddings from data folder
                self.vectorstore = await self._create_default()
            
            if not self.vectorstore:
                return False
            
            self._initialized = True
            logger.info(f"Vector data source {self.config.source_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Vector data source initialization failed: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate configuration using data source registry"""
        spec = DataSourceSpec(
            source_type=DataSourceType.VECTOR_DB,
            query=None,
            filters={},
            limit=self.config.config.get("k", 5),
            enabled=self.config.enabled,
            config=self.config.config
        )
        
        is_valid, errors, _ = check_data_source_spec_availability(spec)
        if not is_valid:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            return False
        return True
    
    async def _create_default(self):
        """Create default vectorstore with embeddings from data folder"""
        try:
            # Get embeddings from data folder
            embeddings = self._load_embeddings()
            if not embeddings:
                logger.warning("No embeddings found, creating empty vectorstore")
            
            # Get data source directory
            data_sources_dir = Path(__file__).parent
            persist_dir = data_sources_dir / "data" / "chroma_db"
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = self.config.config.get("collection_name", "default_collection")
            
            # Create Chroma vectorstore
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create default vectorstore: {e}")
            return None
    
    async def _create_from_path(self, db_path: str):
        """Create vectorstore from external path"""
        try:
            # Determine persist directory
            if db_path.endswith('chroma.sqlite3'):
                persist_dir = os.path.dirname(db_path)
            else:
                persist_dir = db_path
            
            if not os.path.exists(persist_dir):
                logger.error(f"Database path does not exist: {persist_dir}")
                return None
            
            # Load embeddings
            embeddings = self._load_embeddings()
            if not embeddings:
                logger.warning("No embeddings found, using default")
            
            # Create ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get collection name
            try:
                collections = chroma_client.list_collections()
                collection_name = collections[0].name if collections else "default_collection"
            except:
                collection_name = self.config.config.get("collection_name", "default_collection")
            
            # Create Chroma vectorstore
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embeddings
            )
            
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create vectorstore from path {db_path}: {e}")
            return None
    
    def _load_embeddings(self):
        """Load embeddings from data folder"""
        try:
            from langchain_openai import OpenAIEmbeddings
            
            # Check for embeddings in data folder
            data_sources_dir = Path(__file__).parent
            embeddings_dir = data_sources_dir / "data" / "embeddings"
            
            # For now, use OpenAI embeddings (can be extended to load from file)
            # Check if OPENAI_API_KEY is available
            import os
            if os.getenv("OPENAI_API_KEY"):
                model = self.config.config.get("embedding_model", "text-embedding-ada-002")
                return OpenAIEmbeddings(model=model)
            else:
                logger.warning("OPENAI_API_KEY not found, embeddings may not work")
                return None
        except ImportError:
            logger.warning("OpenAI embeddings not available")
            return None
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None
    
    async def query(self, query: str, context: Dict[str, Any] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query vector database"""
        if not self._initialized:
            raise RuntimeError("Data source not initialized")
        
        try:
            k = kwargs.get("k", self.config.config.get("k", 5))
            filter_criteria = kwargs.get("filter") or context.get("filter") if context else None
            
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
            logger.error(f"Vector query failed: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector database health"""
        try:
            if not self._initialized:
                return {"status": "not_initialized", "initialized": False}
            
            if self.vectorstore and hasattr(self.vectorstore, '_collection'):
                count = self.vectorstore._collection.count()
                return {
                    "status": "healthy",
                    "document_count": count,
                    "initialized": True,
                    "is_external": self._is_external
                }
            else:
                return {
                    "status": "healthy",
                    "initialized": True,
                    "is_external": self._is_external
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "initialized": self._initialized}
    
    async def close(self) -> None:
        """Close vector database connection"""
        self._initialized = False
        self.vectorstore = None
        logger.info("Vector data source closed")
    
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.VECTOR_DB


def create_vector_data_source(config: Optional[DataSourceConfig] = None) -> VectorDataSource:
    """Create vector data source instance"""
    if config is None:
        config = DataSourceConfig(
            source_id="vector_db",
            source_type=DataSourceType.VECTOR_DB,
            priority=10
        )
    return VectorDataSource(config)
