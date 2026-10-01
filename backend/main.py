import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend import config
from backend.llm import TEAM_SIZE, GroqChat, LLMError, team_analysis
from backend.pokeapi_client import PokeAPIClient, PokeAPIError, normalize_name

MAX_CARDS = 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as http:
        app.state.pokeapi = PokeAPIClient(http, config.POKEAPI_BASE_URL)
        try:
            await app.state.pokeapi.load_index()
        except PokeAPIError:
            pass  # resolve() falls back to direct lookups until the index is available
        app.state.llm = GroqChat(config.GROQ_API_KEY, config.GROQ_MODEL)
        yield


app = FastAPI(title="PokéChat API", lifespan=lifespan)


def get_pokeapi(request: Request) -> PokeAPIClient:
    return request.app.state.pokeapi


def get_llm(request: Request) -> GroqChat:
    return request.app.state.llm


# ------------------------------------------------------------------ schemas


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    team: list[str] = Field(default_factory=list, max_length=TEAM_SIZE)


class TeamRequest(BaseModel):
    team: list[str] = Field(max_length=TEAM_SIZE)


class PokemonCard(BaseModel):
    id: int
    name: str
    display_name: str
    types: list[str]
    sprite: str | None


class ChatResponse(BaseModel):
    reply: str
    pokemon: list[PokemonCard]
    not_found: list[str]


def to_card(profile: dict[str, Any]) -> PokemonCard:
    return PokemonCard(**{k: profile[k] for k in PokemonCard.model_fields})


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


# ---------------------------------------------------------------- routes


@app.exception_handler(PokeAPIError)
async def pokeapi_error_handler(_: Request, exc: PokeAPIError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
async def health(client: PokeAPIClient = Depends(get_pokeapi)):
    return {"status": "ok", "index_loaded": client.index_loaded, "model": config.GROQ_MODEL}


@app.get("/pokemon/{name}/exists")
async def pokemon_exists(name: str, client: PokeAPIClient = Depends(get_pokeapi)):
    resolved = await client.resolve(name)
    return {"query": name, "exists": resolved is not None, "name": resolved}


@app.get("/pokemon/{name}")
async def pokemon_profile(name: str, client: PokeAPIClient = Depends(get_pokeapi)):
    profile = await client.gather(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"'{name}' does not exist in PokeAPI")
    return profile


@app.post("/team/analysis")
async def analyze_team(req: TeamRequest, client: PokeAPIClient = Depends(get_pokeapi)):
    team, missing = await gather_many(client, req.team)
    return {**team_analysis(team), "pokemon": [to_card(p) for p in team], "not_found": missing}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client: PokeAPIClient = Depends(get_pokeapi),
    llm: GroqChat = Depends(get_llm),
):
    # 1. Verify the team and every Pokémon mentioned in the conversation (newest first).
    team, not_found = await gather_many(client, req.team)
    team_names = {p["name"] for p in team}
    mentioned: list[str] = []
    for m in reversed(req.messages):
        if m.role == "user":
            mentioned += client.find_mentions(m.content)
    mentioned = [n for n in dict.fromkeys(mentioned) if n not in team_names][: config.MAX_POKEMON_PER_CHAT]
    context, _ = await gather_many(client, mentioned)

    # 2. Ask the model, letting it look up any other Pokémon it needs.
    try:
        reply, _ = await llm.reply(
            [m.model_dump() for m in req.messages], context, team, lookup=client.gather
        )
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # 3. Cards for the Pokémon the answer talks about (fall back to what the user asked about).
    # Profiles are cached, so re-gathering Pokémon already seen this turn is free.
    shown, _ = await gather_many(client, client.find_mentions(reply)[:MAX_CARDS])
    return ChatResponse(reply=reply, pokemon=[to_card(p) for p in shown or context], not_found=not_found)
