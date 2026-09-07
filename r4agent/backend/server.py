from contextlib import asynccontextmanager
import asyncio
import json

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Union

from .. import R4Agent
from ..utils import setup_logging
from .structs import (
    UrlMap,
    ErrorDetail, ErrorCodes,
    RootResponse,
    QueryRequest, QueryResponse, QueryStreamChunk,
    RagAddContentRequest,  RagAddContentResponse,
    RagRemoveContentRequest, RagRemoveContentResponse,
    RagListContentsResponse,
    ProviderRebuildRequest, ProviderRebuildResponse,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.r4 = R4Agent(stream=True)
    try:
        yield
    finally:
        app.state.r4 = None

def create_app() -> FastAPI:
    urlmap = UrlMap(root="")
    app = FastAPI(title="R4Agent API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get(urlmap.root)
    async def root() -> RootResponse:
        return RootResponse(
            content="You reached the r4agent successfully!"
            "\n  Endpoints:"
            "\n".join([f" {k} =>  {v}" for k, v  in urlmap.model_dump().items()])
        )

    @app.post(urlmap.query, response_model=None)
    async def query(request: QueryRequest) -> Union[QueryResponse, StreamingResponse]:
        """Sorguyu agent'a iletir ve stream veya tamamlanmış cevap döndürür."""
        if app.state.r4 is None:
            return QueryResponse(
                content="", thinking= "",
                result=False,
                error=ErrorDetail(
                    code=ErrorCodes.INITIALIZATION,
                    message="R4Agent is not initialized"
                )
            )
        if app.state.r4.stream:
            # SSE Stream
            def generate():
                """Agent delta'larını SSE paketlerine dönüştürür."""
                request_agent = R4Agent(stream=True)
                for content_delta, thinking_delta in request_agent.send(request.content):
                    chunk = QueryStreamChunk(
                        content=content_delta,
                        thinking=thinking_delta,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            try:
                def collect_response() -> tuple[str, str]:
                    """Tek request'e ait agent çıktısını blocking thread'de toplar."""
                    request_agent = R4Agent(stream=False)
                    full_content, full_thinking = "", ""
                    for content_delta, thinking_delta in request_agent.send(request.content):
                        full_content += content_delta
                        full_thinking += thinking_delta
                    return full_content, full_thinking

                full_content, full_thinking = await asyncio.to_thread(collect_response)
                return QueryResponse(content=full_content, thinking=full_thinking, result=True)
            
            except Exception as e:
                return QueryResponse(
                    content="",
                    thinking="",
                    result=False,
                    error=ErrorDetail(
                        code=ErrorCodes.QUERY,
                        message= str(e)
                    )
                )
                
            
    @app.post(urlmap.rag_add)
    async def rag_add_content(request: RagAddContentRequest) -> RagAddContentResponse:  
        """Metni embedding'leyip RAG koleksiyonuna ekler."""
        try:    
            def add_content() -> None:
                import r4agent.providers as pv
                pv.DBCLIENT.add_content(request.content, request.document)

            await asyncio.to_thread(add_content)
            return RagAddContentResponse(
                result=True
            )
        except Exception as e:
            return RagAddContentResponse(
                result = False,
                error =  ErrorDetail(
                    code= ErrorCodes.RAG_ADD_CONTENT,
                    message=str(e)
                )
            )
        
    @app.get(urlmap.rag_list)
    async def rag_list_elements() -> RagListContentsResponse:
        """RAG koleksiyonundaki kayıt özetlerini döndürür."""
        try:    
            def list_elements():
                import r4agent.providers as pv
                return pv.DBCLIENT.list_elements()

            elems = await asyncio.to_thread(list_elements)
            return RagListContentsResponse(
                result=True,
                elements=([str(elem) for elem in elems])
            )
        except Exception as e:
            return RagListContentsResponse(
                result=False,
                error=ErrorDetail(
                    code= ErrorCodes.RAG_LIST_ELEMENTS,
                    message= str(e)
                )
            )
        
    @app.delete(urlmap.rag_remove)
    async def rag_remove_elements(request: RagRemoveContentRequest) -> RagRemoveContentResponse:
        """Belge başlığına göre RAG kayıtlarını siler."""
        try:
            def remove_elements() -> None:
                import r4agent.providers as pv
                pv.DBCLIENT.remove_elements(document=request.document)

            await asyncio.to_thread(remove_elements)
            return RagRemoveContentResponse(
                result=True
            )
        except Exception as e:
            return RagRemoveContentResponse(
                result=False,
                error=ErrorDetail(
                    code= ErrorCodes.RAG_REMOVE_ELEMENTS,
                    message= str(e)
                )
            )

    @app.post(urlmap.provider_rebuild)
    async def rebuild_providers(request: ProviderRebuildRequest) -> ProviderRebuildResponse:
        """Provider seçimini günceller ve agent kaynaklarını yeniden kurar."""
        try:
            def rebuild() -> None:
                import r4agent.providers as pv
                pv.configure(
                    context_model=request.context_model,
                    embed_model=request.embed_model,
                    toolgen_model=request.toolgen_model,
                )
                app.state.r4.rebuild()

            await asyncio.to_thread(rebuild)
            return ProviderRebuildResponse(result=True)
        except Exception as e:
            return ProviderRebuildResponse(
                result=False,
                error=ErrorDetail(
                    code=ErrorCodes.PROVIDER_REBUILD,
                    message=str(e),
                ),
            )
    
    return app


app = create_app()


def main() -> None:
    setup_logging(force=True)
    uvicorn.run("r4agent.backend.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()