import json
from typing import Annotated
from inspect import iscoroutinefunction

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.test_schema import Cookies
from src.dependencies.utils import get_cookie
from src.dependencies.ai_dependencies import _ai_request, message_history
from src.dependencies.ai_tools import TOOL_SCHEMAS, search_knowledge_base, calculator
from src.database import async_session
from src.core.logging import logger

router = APIRouter()

async def react_loop(session: AsyncSession, max_iterations: int, action_history: set, current_message: list):
    for iteration in range(1, max_iterations+1):
        logger.info(f"[Iteration {iteration}/{max_iterations}], request to LLM\n")
        response = await _ai_request(current_message, tools=TOOL_SCHEMAS)
        model_output = response.choices[0].message
        logger.info(f"LLM response: {model_output}\n")

        if model_output.tool_calls is None:
            logger.info("Final answer:\n")
            return model_output.content
        logger.info(f"chech history before: {current_message}\n")
        current_message.append(model_output)
        logger.info(f"check history after: {current_message}\n")
        
        for item in model_output.tool_calls:
            if item.type == "function":
                tool_name = item.function.name
                tool_args = item.function.arguments
                tool_call_id = item.id
                try:
                    args_dict = json.loads(tool_args)
                    norm_args = json.dumps(args_dict, sort_keys=True)
                except Exception:
                    args_dict = {}
                    norm_args = tool_args
                tool_call = (tool_name, norm_args)
                if tool_call in action_history:
                    logger.warning(f"Cycle detected for {tool_name} with arguments {tool_args}\n")
                    current_message.append({"role": "tool", "tool_call_id": tool_call_id, 
                                            "content": f"Ты зациклился и вызываешь один и тот же инструмент {tool_name} с теми же аргументами"})
                    logger.info(f"history after cycle: {current_message}\n")
                    continue

                action_history.add(tool_call)
                logger.info(f"Tool call: {tool_name} with args: {args_dict}")
                try:
                    if tool_name == "search_knowledge_base":
                        query = args_dict.get("query", "")
                        observation = await search_knowledge_base(session, query)
                    elif tool_name == "calculator":
                        expression = args_dict.get("expression", "")
                        observation = calculator(expression)
                    else:
                        observation = f"Ошибка, инструмент {tool_name} не существует"
                    if observation is None:
                        observation = f"Ошибка, инструмент вернул пустой ответ (None)"

                except Exception as e:
                    logger.error(f"Ошибка при вызове инструмента {e}")
                    observation = f"Ошибка при выполнении инструмента {str(e)}"

                logger.info(f"Result: {observation}\n")
                current_message.append({"role": "tool", "tool_call_id": tool_call_id,
                                        "content": str(observation)})
                logger.info(f"history after observation {current_message}\n")
            else:
                logger.warning("No proper output")
                current_message.append({"role": "system", "content": "Ты не вернул ответ или он не подошел под формат"})
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Agent reached max iterations")
        

@router.post("/agent/chat")
async def agent_chat(prompt: str, session: Annotated[AsyncSession, Depends(async_session)], 
                     cookies: Annotated[Cookies, Depends(get_cookie)]):
    history = message_history.setdefault(cookies.session_id, [])
    current_messages = history + [{"role": "user", "content": prompt}]

    max_iterations = 5
    action_history = set()

    logger.info(f"Start ReAct cycle for request: {prompt}")
    final_answer = await react_loop(session, max_iterations, action_history, current_messages)
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": final_answer})
    logger.info(f"Main history {history}")
    return final_answer

