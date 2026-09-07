from abc import abstractmethod, ABC
from typing import Dict, Any, List, Tuple
from mcp import ListToolsResult

# == Abstraction ==========================

class ToolConvertions(ABC):
    
    @abstractmethod
    def from_mcp(tools: ListToolsResult) -> List[Dict[str, Any]]:
        """MCP toolunu alarak yeni formatta döndürür"""
        
    @abstractmethod
    def to_mcp(tools: list) -> List[Tuple[str, Dict[str, Any]]]:
        """Formatı alarak mcp formatına dönüştürür."""
    
# == Absolute ==========================
class ToolConvertionsOllama(ToolConvertions):
    
    @staticmethod
    def from_mcp(tools: ListToolsResult) -> List[Dict[str, Any]]:
        """MCP ListToolsResult nesnesini Ollama/OpenAI 'tools' formatına dönüştürür."""
        ollama_formatted_tools = []

        for tool in tools.tools:
            tool_name = tool.name or ""
            tool_desc = tool.description or ""
            input_schema = tool.input_schema or {}

            # MCP inputSchema -> Ollama function parameters
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": input_schema,
                },
            }
            ollama_formatted_tools.append(ollama_tool)

        return ollama_formatted_tools

    @staticmethod
    def to_mcp(tools) -> List[Tuple[str, Dict[str, Any]]]:
        """Ollama'nın döndürdüğü tool_calls listesini MCP için (name, params) tuple listesine dönüştürür.

        Ollama cevabı {"type": ...} alanı taşımaz; her çağrı ToolCall(function=Function(name, arguments))
        şeklindedir (dict de olabilir).
        """
        mcp_tuples = []

        for item in tools or []:
            fn = item["function"] if isinstance(item, dict) else getattr(item, "function", None)
            if fn is None:
                continue
            name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            params = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
            mcp_tuples.append((name, params or {}))

        return mcp_tuples