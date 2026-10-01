"""How the Streamlit app reaches the backend.

- HttpBackend: talks to the FastAPI server (local development, BACKEND_URL set).
- LocalBackend: runs the same service layer in-process (single-process hosting such as
  Streamlit Community Cloud). It owns one asyncio loop on a background thread so the
  PokeAPI cache, which is tied to that loop, is shared by every session.
"""

import asyncio
import threading
from typing import Any, Coroutine

import httpx

from backend import service
from backend.llm import LLMError
from backend.pokeapi_client import PokeAPIError

TIMEOUT_S = 120


class BackendError(Exception):
    pass


class HttpBackend:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = httpx.request(method, f"{self._base_url}{path}", timeout=TIMEOUT_S, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"Can't reach the backend at {self._base_url} ({exc}). Is FastAPI running?") from exc
        if resp.status_code >= 400:
            raise BackendError(resp.json().get("detail", resp.text))
        return resp.json()

    def exists(self, name: str) -> dict:
        return self._request("GET", f"/pokemon/{name}/exists")

    def profile(self, name: str) -> dict:
        return self._request("GET", f"/pokemon/{name}")

    def team_report(self, team: list[str]) -> dict:
        return self._request("POST", "/team/analysis", json={"team": team})

    def chat(self, messages: list[dict], team: list[str]) -> dict:
        return self._request("POST", "/chat", json={"messages": messages, "team": team})


class LocalBackend:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="pokechat-backend").start()
        self._pokeapi = self._run(self._create_pokeapi())
        self._llm = service.create_llm()

    async def _create_pokeapi(self):
        # Lives for the whole process, so the client is intentionally never closed.
        http = httpx.AsyncClient(timeout=20, follow_redirects=True)
        return await service.create_pokeapi(http)

    def _run(self, coro: Coroutine) -> Any:
        try:
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=TIMEOUT_S)
        except (LLMError, PokeAPIError) as exc:
            raise BackendError(str(exc)) from exc

    def exists(self, name: str) -> dict:
        return self._run(service.exists(self._pokeapi, name))

    def profile(self, name: str) -> dict:
        profile = self._run(self._pokeapi.gather(name))
        if profile is None:
            raise BackendError(f"'{name}' does not exist in PokeAPI")
        return profile

    def team_report(self, team: list[str]) -> dict:
        return self._run(service.team_report(self._pokeapi, team))

    def chat(self, messages: list[dict], team: list[str]) -> dict:
        return self._run(service.chat(self._pokeapi, self._llm, messages, team))
