import httpx
import pytest

from backend.llm import LLMError
from backend.main import app, get_llm, get_pokeapi


class FakeLLM:
    def __init__(self, reply="Try mr-mime alongside pikachu.", error=None):
        self.reply_text, self.error, self.calls = reply, error, []

    async def reply(self, messages, profiles, team, lookup):
        self.calls.append({"messages": messages, "profiles": profiles, "team": team})
        if self.error:
            raise self.error
        return self.reply_text, []


@pytest.fixture
async def make_client(pokeapi):
    async def make(llm=None):
        app.dependency_overrides[get_pokeapi] = lambda: pokeapi
        app.dependency_overrides[get_llm] = lambda: llm or FakeLLM()
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    yield make
    app.dependency_overrides.clear()


async def test_exists_endpoint(make_client):
    async with await make_client() as c:
        assert (await c.get("/pokemon/Deoxys/exists")).json() == {"query": "Deoxys", "exists": True, "name": "deoxys-normal"}
        assert (await c.get("/pokemon/fakemon/exists")).json()["exists"] is False


async def test_profile_endpoint_404s_for_unknown_pokemon(make_client):
    async with await make_client() as c:
        assert (await c.get("/pokemon/pikachu")).json()["types"] == ["electric"]
        assert (await c.get("/pokemon/fakemon")).status_code == 404


async def test_team_analysis_flags_missing_members(make_client):
    async with await make_client() as c:
        body = (await c.post("/team/analysis", json={"team": ["pikachu", "fakemon"]})).json()
    assert body["members"] == ["pikachu"]
    assert body["not_found"] == ["fakemon"]
    assert body["open_slots"] == 5


async def test_chat_grounds_mentions_and_returns_cards_for_reply(make_client):
    llm = FakeLLM()
    async with await make_client(llm) as c:
        resp = await c.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Tell me about deoxys"},
                    {"role": "assistant", "content": "It is psychic."},
                    {"role": "user", "content": "And pikachu?"},
                ],
                "team": ["fakemon"],
            },
        )
    body = resp.json()
    assert resp.status_code == 200
    # newest mention first, species resolved to its default form
    assert [p["name"] for p in llm.calls[0]["profiles"]] == ["pikachu", "deoxys-normal"]
    assert [p["name"] for p in body["pokemon"]] == ["mr-mime", "pikachu"]
    assert body["pokemon"][0]["sprite"] == "https://img/122.png"
    assert body["not_found"] == ["fakemon"]


async def test_chat_rejects_oversized_team(make_client):
    async with await make_client() as c:
        resp = await c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}], "team": ["pikachu"] * 7})
    assert resp.status_code == 422


async def test_chat_surfaces_rate_limits(make_client):
    async with await make_client(FakeLLM(error=LLMError("slow down", status_code=429))) as c:
        resp = await c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 429
    assert resp.json()["detail"] == "slow down"
