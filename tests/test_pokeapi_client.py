import httpx
import pytest

from backend.pokeapi_client import PokeAPIClient, PokeAPIError, normalize_name, type_matchups
from tests.conftest import type_doc


@pytest.mark.parametrize(
    "raw, slug",
    [
        ("Pikachu", "pikachu"),
        ("  Mr. Mime ", "mr-mime"),
        ("Farfetch'd", "farfetchd"),
        ("Nidoran♀", "nidoran-f"),
        ("Type: Null", "type-null"),
        ("Flabébé", "flabebe"),
        ("", ""),
    ],
)
def test_normalize_name_produces_pokeapi_slugs(raw, slug):
    assert normalize_name(raw) == slug


async def test_resolve_returns_canonical_name_for_existing_pokemon(pokeapi):
    assert await pokeapi.resolve("PIKACHU") == "pikachu"
    assert await pokeapi.resolve("Mr. Mime") == "mr-mime"


async def test_resolve_maps_species_to_default_form(pokeapi):
    assert await pokeapi.resolve("deoxys") == "deoxys-normal"


async def test_resolve_rejects_unknown_names_without_extra_requests(pokeapi, requests_log):
    before = len(requests_log)
    assert await pokeapi.resolve("fakemon") is None
    assert await pokeapi.exists("") is False
    assert len(requests_log) == before


def test_find_mentions_detects_multiword_names_in_order(pokeapi):
    text = "Is Mr. Mime better than pikachu? What about PIKACHU and deoxys-attack, or my cat?"
    assert pokeapi.find_mentions(text) == ["mr-mime", "pikachu", "deoxys-attack"]


def test_find_mentions_returns_nothing_for_plain_text(pokeapi):
    assert pokeapi.find_mentions("build me a balanced team please") == []


async def test_gather_builds_complete_profile(pokeapi):
    p = await pokeapi.gather("pikachu")
    assert p["name"] == "pikachu" and p["id"] == 25
    assert p["types"] == ["electric"]
    assert p["type_matchups"]["weak_to_2x"] == ["ground"]
    assert p["abilities"] == [{"name": "static", "hidden": False, "effect": "May paralyze on contact."}]
    assert p["base_stat_total"] == 180
    assert p["height_m"] == 0.4 and p["weight_kg"] == 6.0
    assert p["pokedex_entry"] == "A test entry."
    assert p["level_up_moves"] == [{"move": "thunder-shock", "level": 1}]
    assert p["total_learnable_moves"] == 2
    assert p["evolution_chain"] == [
        "pichu -> pikachu via level-up (min happiness: 220)",
        "pikachu -> raichu via use-item (item: thunder-stone)",
    ]
    assert p["sprite"] == "https://img/25.png"


async def test_gather_returns_none_for_unknown_pokemon(pokeapi):
    assert await pokeapi.gather("fakemon") is None


async def test_gather_caches_every_request(pokeapi, requests_log):
    await pokeapi.gather("pikachu")
    first = len(requests_log)
    await pokeapi.gather("pikachu")
    assert len(requests_log) == first


async def test_upstream_errors_raise_and_are_not_cached():
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(500) if len(calls) == 1 else httpx.Response(200, json={"name": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = PokeAPIClient(http, "https://pokeapi.test/api/v2")
        with pytest.raises(PokeAPIError):
            await client._get("pokemon/ok")
        assert await client._get("pokemon/ok") == {"name": "ok"}


def test_type_matchups_multiplies_dual_types():
    psychic = type_doc("psychic", double=["bug", "dark", "ghost"], half=["fighting", "psychic"])
    fairy = type_doc("fairy", double=["poison", "steel"], half=["fighting", "bug", "dark"], none=["dragon"])
    m = type_matchups([psychic, fairy])
    assert m["weak_to_2x"] == ["ghost", "poison", "steel"]
    assert m["resists_0.25x"] == ["fighting"]
    assert m["resists_0.5x"] == ["psychic"]
    assert m["immune_to"] == ["dragon"]
    # bug and dark cancel out to neutral
    assert "bug" not in sum(m.values(), []) and "dark" not in sum(m.values(), [])
