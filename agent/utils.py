import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import settings


async def get_mcp_tools():
     with open(os.path.join(settings.mcp_json_dir, "mcp.json")) as f:
          mcp_settings = json.load(f)
     mcp_client = MultiServerMCPClient(mcp_settings["mcpServers"])
     return await mcp_client.get_tools()
