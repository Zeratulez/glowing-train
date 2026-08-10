import asyncio
from fastmcp import FastMCP

mcp = FastMCP("Knowledge base server")

@mcp.tool
async def search_knowledge_base(query: str):
    """Ищет информацию в векторной БД"""
    return f"Реузльтаты поиска MCP для запроса: {query}"

if __name__ == "__main__":
    mcp.run(transport="stdio")