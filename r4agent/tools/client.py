import asyncio
import atexit
import concurrent.futures
import os
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import logging

class MCPClient:
    _instance: "MCPClient | None" = None
    _lock = threading.Lock()

    SERVER_ROOT = Path(__file__).resolve().parents[2]

    def __new__(cls, *args, **kwargs) -> "MCPClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, embed_model: str = "cosmos"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        env = os.environ.copy()
        env["R4AGENT_EMBED_MODEL"] = embed_model
        self.SERVER_PARAMS = StdioServerParameters(
            command=sys.executable,
            args=["-m", "r4agent.tools.server", "--transport", "stdio"],
            env=env,
            cwd=str(self.SERVER_ROOT),
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mcp-client-loop",
            daemon=True,
        )
        self._stdio_context = None
        self._session_context = None
        self._session = None
        self._closed = False
        self._requests = None
        self._worker_done = None
        self._startup = concurrent.futures.Future()
        self._thread.start()
        self._startup.result()
        atexit.register(self.close)

    def _run_loop(self):
        """MCP asyncio döngüsünü ayrı thread'de başlatıp istek kuyruğunu tüketir."""
        asyncio.set_event_loop(self._loop)
        self._requests = asyncio.Queue()
        self._worker_done = self._loop.create_future()
        self._loop.create_task(self._worker())
        self._loop.run_forever()

    async def _worker(self):
        try:
            self._stdio_context = stdio_client(self.SERVER_PARAMS)
            read, write = await self._stdio_context.__aenter__()
            self._session_context = ClientSession(read, write)
            self._session = await self._session_context.__aenter__()
            await self._session.initialize()
            self._startup.set_result(None)

            while True:
                request = await self._requests.get()
                if request is None:
                    break
                coroutine, result = request
                try:
                    result.set_result(await coroutine)
                except Exception as error:
                    result.set_exception(error)
        except Exception as error:
            logging.exception("MCP Server başlatılırken hata oluştu:")
            if not self._startup.done():
                self._startup.set_exception(error)
        finally:
            if self._session_context is not None:
                await self._session_context.__aexit__(None, None, None)
                self._session_context = None
                self._session = None
            if self._stdio_context is not None:
                await self._stdio_context.__aexit__(None, None, None)
                self._stdio_context = None
            self._worker_done.set_result(None)

    async def _enqueue(self, coroutine, result):
        await self._requests.put((coroutine, result))

    async def _wait_worker(self):
        await self._worker_done

    def _run(self, coroutine, timeout: float = 30.0):
        """Senkron çağrıyı MCP thread'ine aktarır ve sonucunu bekler."""
        if self._closed:
            raise RuntimeError("MCPClient zaten kapatıldı")
        result = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(
            self._enqueue(coroutine, result), self._loop
        ).result()
        return result.result(timeout=timeout)

    def close(self):
        """MCP worker'ını kontrollü biçimde durdurup kaynaklarını serbest bırakır."""
        if self._closed:
            return
        self._closed = True
        asyncio.run_coroutine_threadsafe(
            self._requests.put(None), self._loop
        ).result()
        asyncio.run_coroutine_threadsafe(self._wait_worker(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def get_tools(self):
        """MCP sunucusundaki kullanılabilir tool şemasını getirir."""
        return self._run(self._session.list_tools())

    def call_tool(self, name, params: dict | None = None):
        """Bir MCP tool çağrısını çalıştırır ve metin sonucuna dönüştürür."""
        result = self._run(self._session.call_tool(name, arguments=params or {}))
        if result.is_error:
            error_text = "\n".join(
                getattr(block, "text", str(block)) if not isinstance(block, str) else block
                for block in result.content
            )
            logging.error(f"Tool hatası [{name}]: {error_text}")
            return f"Tool Hatası ({name}): {error_text}"
        text_parts = [
            getattr(block, "text", str(block)) if not isinstance(block, str) else block
            for block in result.content
        ]
        tool_text = "\n".join(text_parts)

        logging.info(f"[MCP tool result for '{name}']:\n{tool_text[:20]}...")
        return tool_text

    def get_insturactions(self):
        """MCP oturumunun model promptuna eklenecek talimatlarını döndürür."""
        return self._session.instructions