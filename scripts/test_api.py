import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["The quick brown fox jumps over the lazy dog."] * 5,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
)

print([len(e.values) for e in response.embeddings])
