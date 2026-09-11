from r4agent.tools.client import MCPClient
from r4agent.tools.utils import  ToolConvertions, ToolConvertionsOllama
import pytest


@pytest.fixture
def mcpclient():
    client = MCPClient()
    return client

class TestMcpClient:

    def test_get_insturactions(self, mcpclient: MCPClient):
        assert isinstance(mcpclient.get_insturactions(), (str, None))
    
    def test_get_and_call_tools(self, mcpclient: MCPClient):
            tools_result = mcpclient.get_tools()
            assert len(tools_result.tools) >= 1

            for tool in tools_result.tools:
                # Tool'un kabul ettiği parametrelerin şemasını al
                properties = tool.input_schema.get("properties", {})
                required_params = tool.input_schema.get("required", [])

                mock_args = {}
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "string")

                    # Tipine göre uygun varsayılan değer ata
                    if param_type == "string":
                        mock_args[param_name] = "test"
                    elif param_type in ("integer", "number"):
                        mock_args[param_name] = 1
                    elif param_type == "boolean":
                        mock_args[param_name] = True
                    elif param_type == "array":
                        mock_args[param_name] = []
                    elif param_type == "object":
                        mock_args[param_name] = {}

                # Zorunlu parametre içermeyen tool'lar için sadece ilk opsiyonel parametreyi gönder
                if not required_params and mock_args:
                    first_key = next(iter(mock_args))
                    mock_args = {first_key: mock_args[first_key]}

                result = mcpclient.call_tool(tool.name, params=mock_args)

                assert isinstance(result, str)
                assert not result.startswith("Tool Hatası")
    
    def test_close(self, mcpclient: MCPClient):
        mcpclient.close()
        
    