import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents="The quick brown fox jumps over the lazy dog.",
    config=types.EmbedContentConfig(output_dimensionality=1536),
)

print(len(response.embeddings[0].values))
