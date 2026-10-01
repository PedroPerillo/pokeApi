"""Fixtures: a fake PokeAPI served through httpx.MockTransport (no real network)."""

import httpx
import pytest

from backend.pokeapi_client import PokeAPIClient

BASE = "https://pokeapi.test/api/v2"


def named(name: str, kind: str = "pokemon") -> dict:
    return {"name": name, "url": f"{BASE}/{kind}/{name}/"}


def type_doc(name: str, double=(), half=(), none=()) -> dict:
    return {
        "name": name,
        "damage_relations": {
            "double_damage_from": [named(t, "type") for t in double],
            "half_damage_from": [named(t, "type") for t in half],
            "no_damage_from": [named(t, "type") for t in none],
        },
    }


def pokemon_doc(pid: int, name: str, types: list[str], species: str | None = None) -> dict:
    return {
        "id": pid,
        "name": name,
        "height": 4,
        "weight": 60,
        "types": [{"slot": i + 1, "type": named(t, "type")} for i, t in enumerate(types)],
        "abilities": [{"ability": named("static", "ability"), "is_hidden": False}],
        "stats": [{"stat": {"name": s}, "base_stat": v} for s, v in [("hp", 35), ("attack", 55), ("speed", 90)]],
        "moves": [
            {
                "move": {"name": "thunder-shock"},
                "version_group_details": [{"level_learned_at": 1, "move_learn_method": {"name": "level-up"}}],
            },
            {
                "move": {"name": "thunderbolt"},
                "version_group_details": [{"level_learned_at": 0, "move_learn_method": {"name": "machine"}}],
            },
        ],
        "species": named(species or name, "pokemon-species"),
        "sprites": {"front_default": None, "other": {"official-artwork": {"front_default": f"https://img/{pid}.png"}}},
    }


def species_doc(name: str, default_variety: str, others=()) -> dict:
    return {
        "name": name,
        "names": [{"language": {"name": "en"}, "name": name.replace("-", " ").title()}],
        "genera": [{"language": {"name": "en"}, "genus": "Test Pokémon"}],
        "generation": {"name": "generation-i"},
        "flavor_text_entries": [{"language": {"name": "en"}, "flavor_text": "A\ntest\fentry."}],
        "is_legendary": False,
        "is_mythical": False,
        "habitat": None,
        "color": {"name": "yellow"},
        "capture_rate": 190,
        "egg_groups": [],
        "evolution_chain": {"url": f"{BASE}/evolution-chain/1/"},
        "varieties": [{"is_default": True, "pokemon": named(default_variety)}]
        + [{"is_default": False, "pokemon": named(o)} for o in others],
    }


EVOLUTION = {
    "chain": {
        "species": {"name": "pichu"},
        "evolves_to": [
            {
                "species": {"name": "pikachu"},
                "evolution_details": [
                    {"trigger": {"name": "level-up"}, "min_happiness": 220, "version_group": {"name": "x"}, "item": None}
                ],
                "evolves_to": [
                    {
                        "species": {"name": "raichu"},
                        "evolution_details": [{"trigger": {"name": "use-item"}, "item": {"name": "thunder-stone"}}],
                        "evolves_to": [],
                    }
                ],
            }
        ],
    }
}

ROUTES = {
    "pokemon-species?limit=100000": {"results": [named(n, "pokemon-species") for n in ["pikachu", "mr-mime", "deoxys"]]},
    "pokemon?limit=100000": {"results": [named(n) for n in ["pikachu", "mr-mime", "deoxys-normal", "deoxys-attack"]]},
    "pokemon/pikachu": pokemon_doc(25, "pikachu", ["electric"]),
    "pokemon/mr-mime": pokemon_doc(122, "mr-mime", ["psychic", "fairy"]),
    "pokemon/deoxys-normal": pokemon_doc(386, "deoxys-normal", ["psychic"], species="deoxys"),
    "pokemon-species/pikachu": species_doc("pikachu", "pikachu"),
    "pokemon-species/mr-mime": species_doc("mr-mime", "mr-mime"),
    "pokemon-species/deoxys": species_doc("deoxys", "deoxys-normal", others=["deoxys-attack"]),
    "type/electric": type_doc("electric", double=["ground"], half=["electric", "flying", "steel"]),
    "type/psychic": type_doc("psychic", double=["bug", "dark", "ghost"], half=["fighting", "psychic"]),
    "type/fairy": type_doc("fairy", double=["poison", "steel"], half=["fighting", "bug", "dark"], none=["dragon"]),
    "ability/static": {"effect_entries": [{"language": {"name": "en"}, "short_effect": "May paralyze on contact."}]},
    "evolution-chain/1": EVOLUTION,
}


@pytest.fixture
def requests_log() -> list[str]:
    return []


def fake_pokeapi_transport(requests_log: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = str(request.url).removeprefix(BASE + "/").rstrip("/")
        requests_log.append(path)
        if path in ROUTES:
            return httpx.Response(200, json=ROUTES[path])
        return httpx.Response(404, text="Not Found")

    return httpx.MockTransport(handler)


@pytest.fixture
async def pokeapi(requests_log):
    async with httpx.AsyncClient(transport=fake_pokeapi_transport(requests_log)) as http:
        client = PokeAPIClient(http, BASE)
        await client.load_index()
        yield client
