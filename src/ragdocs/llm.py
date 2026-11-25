import os

from .retry import with_retry
from google import genai

def make_client():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMNIN API KEY IS NOT SET")
    return genai.Client(api_key=key)

def generate(client, model: str, prompt: str) -> str:
    interaction = with_retry(lambda: client.interactions.create(model=model, input=prompt))
    return interaction.output_text


