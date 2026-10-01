import json
from types import SimpleNamespace as NS

import pytest

from backend.llm import GroqChat, LLMError, build_context, team_fit


def profile(name, types, weak=(), resist=(), immune=(), weak4=()):
    return {
        "id": 1,
        "name": name,
        "types": list(types),
        "base_stats": {"hp": 100},
        "base_stat_total": 500,
        "level_up_moves": [{"move": "tackle", "level": 1}],
        "sprite": "https://img/x.png",
        "type_matchups": {
            "weak_to_4x": list(weak4),
            "weak_to_2x": list(weak),
            "resists_0.5x": list(resist),
            "resists_0.25x": [],
            "immune_to": list(immune),
        },
    }


CHARIZARD = profile("charizard", ["fire", "flying"], weak=["electric", "water"], weak4=["rock"], immune=["ground"])
SWAMPERT = profile("swampert", ["water", "ground"], weak4=["grass"], resist=["rock", "fire"], immune=["electric"])
BLASTOISE = profile("blastoise", ["water"], weak=["electric", "grass"], resist=["water", "fire"])


def test_team_fit_reports_covered_and_shared_weaknesses():
    result = team_fit([CHARIZARD], [SWAMPERT, BLASTOISE])
    swampert, blastoise = result["candidates"]
    assert swampert["covers_team_weaknesses"] == ["electric", "rock"]
    assert swampert["shares_team_weaknesses"] == []
    assert swampert["facts"] == (
        "swampert (water/ground, base stat total 500) covers team weaknesses: electric, rock; "
        "shares team weaknesses: none; weak to: grass (4x); resists: rock, fire; immune to: electric."
    )
    assert blastoise["covers_team_weaknesses"] == ["water"]
    assert blastoise["shares_team_weaknesses"] == ["electric"]
    assert result["team_after"]["members"] == ["charizard", "swampert", "blastoise"]


def test_context_includes_team_and_strips_sprites():
    ctx = build_context([SWAMPERT], [CHARIZARD])
    assert "charizard" in ctx and "swampert" in ctx
    assert "https://img/x.png" not in ctx
    assert "tackle@1" in ctx


def completion(content=None, tool_calls=None, finish_reason="stop"):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason=finish_reason)], usage=None)


def tool_call(name, args, id_="call_1"):
    return NS(id=id_, function=NS(name=name, arguments=json.dumps(args)))


class FakeGroq:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


async def lookup(name):
    return {"swampert": SWAMPERT, "charizard": CHARIZARD}.get(name)


async def test_reply_runs_team_fit_tool_then_answers():
    fake = FakeGroq(
        [
            completion(tool_calls=[tool_call("check_team_fit", {"candidates": ["swampert", "missingno"]})]),
            completion("Add swampert."),
        ]
    )
    reply, looked_up = await GroqChat("k", "m", client=fake).reply(
        [{"role": "user", "content": "help"}], [], [CHARIZARD], lookup
    )
    assert reply == "Add swampert."
    assert [p["name"] for p in looked_up] == ["swampert"]
    tool_msg = json.loads(fake.requests[1]["messages"][-1]["content"])
    assert tool_msg["candidates"][0]["covers_team_weaknesses"] == ["electric", "rock"]
    assert tool_msg["not_in_pokeapi"] == ["missingno"]


async def test_reply_reports_nonexistent_pokemon_to_model():
    fake = FakeGroq([completion(tool_calls=[tool_call("get_pokemon", {"name": "fakemon"})]), completion("Nope.")])
    await GroqChat("k", "m", client=fake).reply([{"role": "user", "content": "fakemon?"}], [], [], lookup)
    assert "does not exist" in fake.requests[1]["messages"][-1]["content"]


async def test_empty_answer_raises():
    fake = FakeGroq([completion("", finish_reason="length")])
    with pytest.raises(LLMError, match="empty answer"):
        await GroqChat("k", "m", client=fake).reply([{"role": "user", "content": "hi"}], [], [], lookup)
