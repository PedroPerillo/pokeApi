from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend import config, service
from backend.llm import TEAM_SIZE, GroqChat, LLMError
from backend.pokeapi_client import PokeAPIClient, PokeAPIError


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as http:
        app.state.pokeapi = await service.create_pokeapi(http)
        app.state.llm = service.create_llm()
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


# ---------------------------------------------------------------- routes


@app.exception_handler(PokeAPIError)
async def pokeapi_error_handler(_: Request, exc: PokeAPIError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
async def health(client: PokeAPIClient = Depends(get_pokeapi)):
    return {"status": "ok", "index_loaded": client.index_loaded, "model": config.GROQ_MODEL}


@app.get("/pokemon/{name}/exists")
async def pokemon_exists(name: str, client: PokeAPIClient = Depends(get_pokeapi)):
    return await service.exists(client, name)


@app.get("/pokemon/{name}")
async def pokemon_profile(name: str, client: PokeAPIClient = Depends(get_pokeapi)):
    profile = await client.gather(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"'{name}' does not exist in PokeAPI")
    return profile


@app.post("/team/analysis")
async def analyze_team(req: TeamRequest, client: PokeAPIClient = Depends(get_pokeapi)):
    return await service.team_report(client, req.team)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client: PokeAPIClient = Depends(get_pokeapi),
    llm: GroqChat = Depends(get_llm),
):
    try:
        return await service.chat(client, llm, [m.model_dump() for m in req.messages], req.team)
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
