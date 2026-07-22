"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""
import os
from openai import OpenAI

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase
from rag_traced import RAGTraced

from dotenv import load_dotenv
load_dotenv()

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1") or os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
)
rag = RAGTraced(
    index=index, 
    llm_client=client, 
    model="openai/gpt-oss-20b"
    )

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)
