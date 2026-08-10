from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies.rag import prompt_embedding
from src.dependencies.utils import calculation
from src.schemas.test_schema import SearchArgs, CalcArgs

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Получение близкого по смыслу контекста из векторной БД",
            "parameters": SearchArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Калькулятор выполняющий математические вычисления",
            "parameters": CalcArgs.model_json_schema()
        }
    }
]

async def search_knowledge_base(session: AsyncSession, query: str):
    try:
        chunks = await prompt_embedding(session, query)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ошибка сервера: {e}")
    context_list = []
    for c in chunks:
        context_list.append(c[0].content)
    return "\n".join(context_list)


def calculator(expression: str):
    return calculation(expression)