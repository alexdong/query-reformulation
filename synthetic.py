from typing import Tuple
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

prompt = open("PROMPT.md").read()


def generate_output(question: str) -> Tuple[str, int]:
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"input: {question}"},
        {"role": "assistant", "content": "output: ", "prefix": True},
    ]
    response = client.chat.completions.create(
        model="deepseek-reasoner", messages=messages, stream=False
    )
    token_usage = (
        response.usage.total_tokens - response.usage.prompt_tokens_details.cached_tokens
    )
    message = response.choices[0].message
    print(message.content)
    return message.content, token_usage


if __name__ == "__main__":
    # question = "Which genus of moth in the world's seventh-largest country contains only one species?"
    question = "What nationality was James Henry Miller's wife?"
    # question = "What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?"
    print(generate_output(question))
