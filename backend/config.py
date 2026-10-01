import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
POKEAPI_BASE_URL = os.getenv("POKEAPI_BASE_URL", "https://pokeapi.co/api/v2")
# Upper bound on how many Pokémon profiles are fed to the LLM per chat turn.
MAX_POKEMON_PER_CHAT = int(os.getenv("MAX_POKEMON_PER_CHAT", "6"))
