import json
import os

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry

class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str

judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()


def evaluate_relevance(question, answer, client=None):
    if client is None:
        client = OpenAI(
             api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("GROQ_API_BASE_URL", "https://api.groq.ai/v1") or os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")  
        )

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
        model="openai/gpt-oss-20b"
    )

    return result.relevance, result.explanation

if __name__ == "__main__":
    load_dotenv()

    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)