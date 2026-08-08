from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from src.core.settings import settings

async_engine = create_async_engine(url=settings.ASYNC_ENGINE_CONNECT)

AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)

async def async_session():
    async with AsyncSessionLocal() as session:
        yield session