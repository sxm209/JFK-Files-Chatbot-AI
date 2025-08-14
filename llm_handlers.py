import os
from openai import OpenAI
import requests

# Constants
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
LOW_TEMPERATURE = 0.2  # For factual responses
MAX_TOKENS = 500

def ask_openai(question: str, context_chunks: list, api_key: str, model: str = DEFAULT_OPENAI_MODEL, prompt_override: str = None) -> str:
    """
    Query OpenAI's API to answer a question using provided context.
    """
    client = OpenAI(api_key=api_key)

    if prompt_override:
        user_content = prompt_override
    else:
        context_text = "\n\n".join(chunk["chunk"] for chunk in context_chunks)
        user_content = f"Context:\n{context_text}\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful research assistant. Your job is to answer user questions "
                        "based strictly on the provided context, which comes from government documents. "
                        "Do not make up any information. Only use what is relevant from the context. "
                        "Always cite the sources you used at the end of your answer in the format: "
                        "Sources: [FileName] (Page X), [FileName] (Page Y). If the answer cannot be found "
                        "in the context, say 'The information is not available in the provided documents.'"
                    )
                },
                {"role": "user", "content": user_content}
            ],
            temperature=LOW_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI Error] {e}"
    
def ask_groq(question: str, context_chunks: list, api_key: str, model: str = DEFAULT_GROQ_MODEL, prompt_override: str = None) -> str:
    """
    Query Groq's API to answer a question using provided context.

    Args:
        question (str): The user's question.
        context_chunks (list): List of context chunks to provide as input.
        api_key (str): Groq API key.
        model (str): Groq model to use (default: llama-3.3-70b-versatile).
        prompt_override (str): Optional custom prompt to override default behavior.

    Returns:
        str: The response from Groq or an error message.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if prompt_override:
        user_content = prompt_override
    else:
        context_text = "\n\n".join(chunk["chunk"] for chunk in context_chunks)
        user_content = f"Context:\n{context_text}\n\nQuestion: {question}"

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. Your job is to answer user questions "
                    "based strictly on the provided context, which comes from government documents. "
                    "Do not make up any information. Only use what is relevant from the context. "
                    "Always cite the sources you used at the end of your answer in the format: "
                    "Sources: [FileName] (Page X), [FileName] (Page Y). If the answer cannot be found "
                    "in the context, say 'The information is not available in the provided documents.'"
                )
            },
            {"role": "user", "content": user_content}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Groq API Error] {e}"