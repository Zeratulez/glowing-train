from fastapi import FastAPI

from src.endpoints import chat

app = FastAPI()

app.include_router(chat.router)