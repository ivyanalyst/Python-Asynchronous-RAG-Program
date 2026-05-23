import asyncio
import json
import numpy as np
import aiofiles
from typing import List, Optional
import pydantic
from pydantic import BaseModel, ValidationError
import aiohttp
import httpx
from openai import AsyncOpenAI, base_url
import os
from dotenv import load_dotenv


http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=5, max_keepalive_connections=5),
    timeout=httpx.Timeout(10.0, connect=5.0)
)

load_dotenv()

client = AsyncOpenAI(
    http_client=http_client,
    base_url="https://router.huggingface.co/v1", #example base url
    api_key=os.getenv("HF_TOKEN")  
)   

class Document(BaseModel):
    question: str
    answer: str
    
async def load_documents(file_path: str) -> List[Document]:
    try:
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
            data = json.loads(content)
            return [Document(**item) for item in data]
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    
    except ValidationError as e:
        print(f"Data validation error: {e}")
        return []
    
MAX_CONCURRENT_REQUESTS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
async def get_embeddings(text: str) -> List[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

async def get_embeddings_with_semaphore(doc:Document) -> List[float]:
    async with semaphore:
        try:
            text_chunk = f"Question: {doc.question} Answer: {doc.answer}"
            emb = await get_embeddings(text_chunk)
            doc.embedding = emb
        except Exception as e:
            print(f"Error embedding document: {e}")

async def index_documents(documents: List[Document]) -> List[dict]:
    print(f"Embedding {len(documents)} chunk. Max concurrency: {MAX_CONCURRENT_REQUESTS}...")

    await asyncio.gather(*[get_embeddings_with_semaphore(doc) for doc in documents])   
    
    print("All chunks embedded.")
        
def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np. linalg.norm(v1) * np.linalg.norm(v2))

async def retrieve(query: str, documents: List[Document], top_k: int = 5) -> str:
    query_embedding = await get_embeddings(query)
    
    score_docs = []
    for doc in documents:
        score = compute_similarity(query_embedding, doc.embedding)
        score_docs.append((score, doc))
        
    score_docs.sort(key=lambda x: x[0], reverse=True)
    
    best_matches = score_docs[0][1]
    return best_matches.answer

async def generate_response(query: str, retrieved_answer: str) -> str:
    prompt = f"Use the following retrieved information to answer the question:\n\n{retrieved_answer}\n\nQuestion: {query}"
    
    response = await client.chat.completion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise and helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

async def main():
    db_path = "qa_pairs.json"
    print(f"Loading databases from {db_path}...")
    
    documents = await load_documents(db_path)
    
    if documents:
        print(f" Success Loaded {len(documents)} documents.")
        print(f"Fist Question: {documents[0].question}")
        print(f"First Answer: {documents[0].answer}")
        
    else: print(f"Failed to load documents from {db_path}.")
    
    await index_documents(documents)
    print("documents indexing completed. \n")
    
    user_query = "Who was England's longest-ruling monarch?"
    print(f"User Query: {user_query}")
    
    retrieved_answer = await retrieve(user_query, documents)
    print(f"Retrieved Answer: {retrieved_answer}\n")
    
    print("Generating response...")
    final_response = await generate_response(user_query, retrieved_answer)
    print(f"Final Response: {final_response}")
    
if __name__ == "__main__":
    asyncio.run(main())

