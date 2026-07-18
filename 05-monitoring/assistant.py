import sys
import os

from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_faq_data, build_index
from rag_helper import RAGBase


def create_assistant():
    load_dotenv()

    documents = load_faq_data()
    index = build_index(documents)

    openai_client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("GROQ_API_BASE_URL", "https://api.groq.ai/v1") or os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")  
        )

    return RAGBase(
        index=index,
        llm_client=openai_client,
        model="openai/gpt-oss-20b",
    )


if __name__ == "__main__":
     assistant = create_assistant()

     query = "How do I join the course?"
     
     if len(sys.argv) > 1:
        query = sys.argv[1]
        
     answer = assistant.rag(query)
     print(answer)
