from __future__ import annotations

from typing import List
import requests
from config import OPENAI_API_KEY, EMBEDDING_MODEL


def embedding(inputs: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """Get embeddings from the embedding service"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "input": inputs,
        "model": model
    }

    response = requests.post(model, headers=headers, json=data)
    outputs = [output['embedding'] for output in response.json()['data']]
    return outputs


if __name__ == '__main__':
    test_inputs = ["Hello world!", "How are you?"]
    test_outputs = embedding(test_inputs)
    test_output = test_outputs[0]
    print(test_output)
    print("Dim: ", len(test_output))
