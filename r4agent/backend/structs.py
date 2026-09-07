from pydantic import BaseModel, computed_field

# == Constructions ==========================

class UrlMap(BaseModel):
    root:str 
    
    @computed_field
    @property
    def query(self)->str:
        return self.root + "/send"
    
    @computed_field
    @property
    def rag_add(self)->str:
        return self.root + "/rag/add"
    
    @computed_field
    @property
    def rag_remove(self)->str:
        return self.root + "/rag/remove"
    
    @computed_field
    @property
    def rag_list(self)->str:
        return self.root + "/rag/list"

    @computed_field
    @property
    def provider_rebuild(self)->str:
        return self.root + "/providers/rebuild"
    
# == Packets ==========================

class ErrorCodes:
    INITIALIZATION = "INITIALIZATION."
    QUERY = "QUERY"
    RAG_ADD_CONTENT = "RAG_ADD_CONTENT"
    RAG_REMOVE_ELEMENTS = "RAG_REMOVE_ELEMENTS"
    RAG_LIST_ELEMENTS = "RAG_LIST_ELEMENTS"
    PROVIDER_REBUILD = "PROVIDER_REBUILD"
    

class ErrorDetail(BaseModel):
    code: str
    message: str

# root
class RootResponse(BaseModel):
    content: str
    error: ErrorDetail | None = None

# query
class QueryRequest(BaseModel):
    content: str

class QueryResponse(BaseModel):
    content: str
    thinking: str
    result: bool
    error: ErrorDetail | None = None

class QueryStreamChunk(BaseModel):
    """SSE stream'inde yalnızca yeni content ve thinking parçalarını taşır."""
    content: str
    thinking: str
    
# rag_add_content
class RagAddContentRequest(BaseModel):
    content: str
    document: str

class RagAddContentResponse(BaseModel):
    result: bool
    error: ErrorDetail | None = None

# rag_content_remove
class RagRemoveContentRequest(BaseModel):
    document: str
    
class RagRemoveContentResponse(BaseModel):
    result: bool
    error: ErrorDetail | None = None

# rag_list_elements
class RagListContentsResponse(BaseModel):
    result: bool
    elements: list[str] = []
    error: ErrorDetail | None = None

# provider rebuild
class ProviderRebuildRequest(BaseModel):
    context_model: str | None = None
    embed_model: str | None = None
    toolgen_model: str | None = None

class ProviderRebuildResponse(BaseModel):
    result: bool
    error: ErrorDetail | None = None