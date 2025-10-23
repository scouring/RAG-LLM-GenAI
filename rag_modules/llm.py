# -*- coding: utf-8 -*-
from openai import OpenAI

def query_llm(api_key: str, user_prompt: str, search_results: list):
    """Generate an LLM response based on retrieved context."""
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": (
                "You are a wine specialist chatbot. "
                "Your job is to recommend amazing wines to users."
            )},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(search_results)}
        ]
    )
    return completion.choices[0].message
