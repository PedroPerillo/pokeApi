"""Application logic shared by the FastAPI routes and the in-process Streamlit backend.

Every function returns plain JSON-shaped dicts so both transports expose the same data.
"""

import asyncio
from typing import Any

import httpx

from backend import config
from backend.llm import GroqChat, team_analysis
from backend.pokeapi_client import PokeAPIClient, PokeAPIError, normalize_name

MAX_CARDS = 8
CARD_FIELDS = ("id", "name", "display_name", "types", "sprite")


async def create_pokeapi(http: httpx.AsyncClient) -> PokeAPIClient:
    client = PokeAPIClient(http, config.POKEAPI_BASE_URL)
    try:
        await client.load_index()
    except PokeAPIError:
        pass  # resolve() falls back to direct lookups until the index is available
    return client


def create_llm() -> GroqChat:
    return GroqChat(config.GROQ_API_KEY, config.GROQ_MODEL)


def to_card(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: profile[k] for k in CARD_FIELDS}


async def gather_many(client: PokeAPIClient, names: list[str]) -> tuple[list[dict], list[str]]:
    """Gather profiles for unique names; return (profiles, names that don't exist)."""
    unique = list(dict.fromkeys(n for n in names if normalize_name(n)))
    results = await asyncio.gather(*(client.gather(n) for n in unique))
    profiles, missing, seen = [], [], set()
    for name, profile in zip(unique, results):
        if profile is None:
            missing.append(name)
        elif profile["name"] not in seen:
            seen.add(profile["name"])
            profiles.append(profile)
    return profiles, missing


async def exists(client: PokeAPIClient, name: str) -> dict[str, Any]:
    resolved = await client.resolve(name)
    return {"query": name, "exists": resolved is not None, "name": resolved}


async def team_report(client: PokeAPIClient, names: list[str]) -> dict[str, Any]:
    team, missing = await gather_many(client, names)
    return {**team_analysis(team), "pokemon": [to_card(p) for p in team], "not_found": missing}


async def chat(
    client: PokeAPIClient, llm: GroqChat, messages: list[dict[str, str]], team_names: list[str]
) -> dict[str, Any]:
    """Grounded chat turn. Raises LLMError if the model call fails."""
    # 1. Verify the team and every Pokémon mentioned in the conversation (newest first).
    team, not_found = await gather_many(client, team_names)
    on_team = {p["name"] for p in team}
    mentioned: list[str] = []
    for m in reversed(messages):
        if m["role"] == "user":
            mentioned += client.find_mentions(m["content"])
    mentioned = [n for n in dict.fromkeys(mentioned) if n not in on_team][: config.MAX_POKEMON_PER_CHAT]
    context, _ = await gather_many(client, mentioned)

    # 2. Ask the model, letting it look up any other Pokémon it needs.
    reply, _ = await llm.reply(messages, context, team, lookup=client.gather)

    # 3. Cards for the Pokémon the answer talks about (fall back to what the user asked about).
    # Profiles are cached, so re-gathering Pokémon already seen this turn is free.
    shown, _ = await gather_many(client, client.find_mentions(reply)[:MAX_CARDS])
    return {"reply": reply, "pokemon": [to_card(p) for p in shown or context], "not_found": not_found}
