import uuid

from pathlib import Path
from sqlalchemy import select, ScalarResult, func
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.models.chunks import Chunk
from src.core.settings import settings

model = OllamaEmbeddings(model="bge-m3", base_url=settings.OLLAMA_URL)

async def prompt_embedding(session: AsyncSession, text: str) -> list[tuple[Chunk, float]] | None:
    embeddings = await model.aembed_query(text)
    distance = Chunk.embedding.cosine_distance(embeddings)
    query = select(Chunk, distance).order_by(distance).limit(5)
    result = await session.execute(query)
    return result.all()

async def chunking(session: AsyncSession, document_id: str, texts: list[str], filename: str):
    if not texts:
        return None

    embeddings = await model.aembed_documents(texts)

    add_chunks = []
    for idx, (text_content, embedding) in enumerate(zip(texts, embeddings)):
        chunk = Chunk(
            document_id=document_id,
            content=text_content,
            embedding=embedding,
            metadata_info={
                "filename": filename,
                "chunk_number": idx + 1
            }
        )
        add_chunks.append(chunk)
    session.add_all(add_chunks)
    await session.commit()

def file_splitter(document):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    md_header_splits = markdown_splitter.split_text(document)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    splits = text_splitter.split_documents(md_header_splits)
    texts = [chunks.page_content for chunks in splits]

    return texts

async def file_get(session: AsyncSession,folderpath: str):
    directory = Path(folderpath)
    filepaths = list(directory.glob("*.md"))
    if not filepaths:
        return 
    
    for file in filepaths:
        if not await session.scalar(select(Chunk).filter(Chunk.metadata_info["filename"].astext == file.name)):
            with open(file, "r", encoding="utf-8") as f:
                full_text = f.read()
            chunks = file_splitter(full_text)
            document_id = str(uuid.uuid4())
            filename = file.name
            await chunking(session,document_id,chunks,filename)