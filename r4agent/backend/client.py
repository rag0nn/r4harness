# Api'yi çalıştırmak
"""
python -m r4agent.backend.server
"""
import json
import requests
from .structs import (
    UrlMap, 
    RootResponse,
    QueryRequest, QueryResponse, QueryStreamChunk,
    RagAddContentRequest,  RagAddContentResponse,
    RagRemoveContentRequest, RagRemoveContentResponse,
    RagListContentsResponse
    )

# response handle
class R4Client:
    
    def __init__(self, url:str = "http://0.0.0.0:8000"):
        self.urlmap = UrlMap(root=url)
    
    def root(self):
        response = requests.get(self.urlmap.root)
        response.raise_for_status()    
        return RootResponse(**response.json())
    
    def query(self, text:str):
        """Query endpoint'ini çağırır ve SSE parçalarını client modeline dönüştürür."""
        request = QueryRequest(content=text)
        with requests.post(
            self.urlmap.query,
            json=request.model_dump(),
            stream=True,
        ) as resp:
            resp.raise_for_status()    
            
            if resp.headers.get("content-type", "").startswith("text/event-stream"):
                # SSE Stream chunkları
                for line in resp.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        yield QueryStreamChunk(**json.loads(line[6:]))
            else:
                yield QueryResponse(**resp.json())
    
    def rag_add_content(self, title:str, content:str):
        """Bir metni backend RAG koleksiyonuna ekler."""
        request = RagAddContentRequest(
            content=content,
            document = title
        )
        response = requests.post(
            self.urlmap.rag_add,
            json=request.model_dump()
        )
        response.raise_for_status()
        return RagAddContentResponse(**response.json())
    
    def rag_remove_elements(self, document:str):
        request = RagRemoveContentRequest(
            document=document
        )
        response = requests.delete(
            self.urlmap.rag_remove,
            json = request.model_dump()
        )
        response.raise_for_status()
        return RagRemoveContentResponse(**response.json())
    
    def rag_list_elements(self):
        """Backend'den RAG kayıtlarının özet listesini ister."""
        response = requests.get(self.urlmap.rag_list)
        response.raise_for_status()
        return RagListContentsResponse(
            **response.json()
        )
