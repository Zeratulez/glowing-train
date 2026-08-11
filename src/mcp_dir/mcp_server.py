import asyncio
import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from src.database import AsyncSessionLocal
from src.dependencies.ai_tools import search_knowledge_base


logger = logging.getLogger(__name__)

mcp = FastMCP("Knowledge base server")

@asynccontextmanager
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session

@mcp.tool
async def search_base(query: str):
    """Ищет информацию в векторной БД"""
    async with get_session() as session:
        return await search_knowledge_base(session, query)

if __name__ == "__main__":
    mcp.run(transport="stdio")