"""Async PokeAPI client: existence checks, name detection and profile gathering.

PokeAPI data is effectively static, so every response is cached in memory for
the lifetime of the process.
"""

import asyncio
import re
import unicodedata
from typing import Any

import httpx

MAX_LEVEL_UP_MOVES = 20
MAX_MENTION_WORDS = 3
MAX_EVOLUTION_METHODS = 3
# Evolution-detail fields that are bookkeeping rather than evolution conditions.
EVOLUTION_NOISE = {"trigger", "version_group", "is_default", "required_pokemon_form", "evolved_pokemon_form"}


class PokeAPIError(Exception):
    """PokeAPI is unreachable or returned an unexpected error."""


def normalize_name(raw: str) -> str:
    """Turn user input like "Mr. Mime" or "Farfetch'd" into a PokeAPI slug."""
    text = raw.strip().lower().replace("♀", "-f").replace("♂", "-m")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"['.:]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


class PokeAPIClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str):
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._cache: dict[str, asyncio.Task] = {}
        self._species_names: set[str] = set()
        self._pokemon_names: set[str] = set()

    # ----------------------------------------------------------------- HTTP

    async def _fetch(self, url: str) -> dict[str, Any] | None:
        try:
            resp = await self._http.get(url)
        except httpx.HTTPError as exc:
            raise PokeAPIError(f"Could not reach PokeAPI: {exc}") from exc
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise PokeAPIError(f"PokeAPI returned {resp.status_code} for {url}")
        return resp.json()

    async def _get(self, path_or_url: str) -> dict[str, Any] | None:
        """GET with an in-memory cache shared by concurrent callers."""
        url = path_or_url if path_or_url.startswith("http") else f"{self._base_url}/{path_or_url.lstrip('/')}"
        task = self._cache.get(url)
        if task is None:
            task = asyncio.ensure_future(self._fetch(url))
            self._cache[url] = task
        try:
            return await asyncio.shield(task)
        except PokeAPIError:
            self._cache.pop(url, None)  # don't cache transient failures
            raise

    # ---------------------------------------------------------- name index

    async def load_index(self) -> None:
        """Load every species and Pokémon name so lookups/mentions are cheap."""
        species, pokemon = await asyncio.gather(
            self._get("pokemon-species?limit=100000"),
            self._get("pokemon?limit=100000"),
        )
        self._species_names = {r["name"] for r in (species or {}).get("results", [])}
        self._pokemon_names = {r["name"] for r in (pokemon or {}).get("results", [])}

    @property
    def index_loaded(self) -> bool:
        return bool(self._species_names)

    async def resolve(self, name: str) -> str | None:
        """Return the PokeAPI `pokemon` name for user input, or None if it doesn't exist.

        Accepts Pokédex numbers, species names ("deoxys" -> "deoxys-normal") and
        form names ("charizard-mega-x").
        """
        slug = normalize_name(name)
        if not slug:
            return None
        if slug in self._pokemon_names:
            return slug
        if slug in self._species_names or slug.isdigit() or not self.index_loaded:
            species = await self._get(f"pokemon-species/{slug}")
            if species:
                default = next(v for v in species["varieties"] if v["is_default"])
                return default["pokemon"]["name"]
            if not self.index_loaded:
                pokemon = await self._get(f"pokemon/{slug}")
                return pokemon["name"] if pokemon else None
        return None

    async def exists(self, name: str) -> bool:
        return await self.resolve(name) is not None

    def find_mentions(self, text: str) -> list[str]:
        """Pokémon names mentioned in free text, in order of first appearance."""
        words = [normalize_name(w) for w in re.findall(r"[\w'.:♀♂-]+", text)]
        words = [w for w in words if w]
        found: list[str] = []
        i = 0
        while i < len(words):
            for size in range(MAX_MENTION_WORDS, 0, -1):  # longest match first: "mr mime"
                candidate = "-".join(words[i : i + size])
                if candidate in self._species_names or candidate in self._pokemon_names:
                    if candidate not in found:
                        found.append(candidate)
                    i += size
                    break
            else:
                i += 1
        return found

    # ----------------------------------------------------------- profiles

    async def gather(self, name: str) -> dict[str, Any] | None:
        """Collect everything useful about a Pokémon into one compact profile."""
        resolved = await self.resolve(name)
        if resolved is None:
            return None
        pokemon = await self._get(f"pokemon/{resolved}")
        if pokemon is None:
            return None

        species_task = self._get(pokemon["species"]["url"])
        type_tasks = [self._get(t["type"]["url"]) for t in pokemon["types"]]
        ability_tasks = [self._get(a["ability"]["url"]) for a in pokemon["abilities"]]
        species, *rest = await asyncio.gather(species_task, *type_tasks, *ability_tasks)
        types_data = rest[: len(type_tasks)]
        abilities_data = rest[len(type_tasks) :]

        evolution = None
        if species and species.get("evolution_chain"):
            evolution = await self._get(species["evolution_chain"]["url"])

        stats = {s["stat"]["name"]: s["base_stat"] for s in pokemon["stats"]}
        artwork = (pokemon["sprites"].get("other") or {}).get("official-artwork") or {}
        return {
            "id": pokemon["id"],
            "name": pokemon["name"],
            "display_name": _english(species, "names", "name") or pokemon["name"].title(),
            "genus": _english(species, "genera", "genus"),
            "generation": (species or {}).get("generation", {}).get("name"),
            "types": [t["type"]["name"] for t in pokemon["types"]],
            "abilities": [
                {
                    "name": a["ability"]["name"],
                    "hidden": a["is_hidden"],
                    "effect": _english(data, "effect_entries", "short_effect"),
                }
                for a, data in zip(pokemon["abilities"], abilities_data)
            ],
            "base_stats": stats,
            "base_stat_total": sum(stats.values()),
            "height_m": pokemon["height"] / 10,
            "weight_kg": pokemon["weight"] / 10,
            "type_matchups": type_matchups([t for t in types_data if t]),
            "evolution_chain": describe_evolution_chain(evolution["chain"]) if evolution else [],
            "level_up_moves": level_up_moves(pokemon["moves"]),
            "total_learnable_moves": len(pokemon["moves"]),
            "pokedex_entry": _latest_flavor_text(species),
            "is_legendary": (species or {}).get("is_legendary", False),
            "is_mythical": (species or {}).get("is_mythical", False),
            "habitat": ((species or {}).get("habitat") or {}).get("name"),
            "color": ((species or {}).get("color") or {}).get("name"),
            "capture_rate": (species or {}).get("capture_rate"),
            "egg_groups": [g["name"] for g in (species or {}).get("egg_groups", [])],
            "other_forms": [
                v["pokemon"]["name"] for v in (species or {}).get("varieties", []) if not v["is_default"]
            ],
            "sprite": artwork.get("front_default") or pokemon["sprites"].get("front_default"),
        }


# --------------------------------------------------------------- helpers


def _english(data: dict | None, key: str, field: str) -> str | None:
    for entry in (data or {}).get(key, []):
        if entry["language"]["name"] == "en":
            return " ".join(entry[field].split())
    return None


def _latest_flavor_text(species: dict | None) -> str | None:
    entries = [e for e in (species or {}).get("flavor_text_entries", []) if e["language"]["name"] == "en"]
    return " ".join(entries[-1]["flavor_text"].split()) if entries else None


def type_matchups(defending_types: list[dict]) -> dict[str, list[str]]:
    """Damage multipliers taken from each attacking type, grouped by multiplier."""
    multipliers: dict[str, float] = {}
    for t in defending_types:
        rel = t["damage_relations"]
        for key, factor in (("double_damage_from", 2), ("half_damage_from", 0.5), ("no_damage_from", 0)):
            for attacker in rel[key]:
                name = attacker["name"]
                multipliers[name] = multipliers.get(name, 1) * factor
    groups = {"4x": [], "2x": [], "0.5x": [], "0.25x": [], "0x": []}
    labels = {4: "4x", 2: "2x", 0.5: "0.5x", 0.25: "0.25x", 0: "0x"}
    for attacker, m in sorted(multipliers.items()):
        if m in labels:
            groups[labels[m]].append(attacker)
    return {
        "weak_to_4x": groups["4x"],
        "weak_to_2x": groups["2x"],
        "resists_0.5x": groups["0.5x"],
        "resists_0.25x": groups["0.25x"],
        "immune_to": groups["0x"],
    }


def _describe_conditions(detail: dict) -> str:
    parts = []
    for key, value in detail.items():
        if key in EVOLUTION_NOISE or value in (None, "", False, []):
            continue
        if isinstance(value, dict):
            value = value.get("name")
        parts.append(f"{key.replace('_', ' ')}: {value}")
    trigger = (detail.get("trigger") or {}).get("name", "unknown")
    return trigger + (f" ({', '.join(parts)})" if parts else "")


def describe_evolution_chain(chain: dict) -> list[str]:
    """Flatten an evolution chain into lines like 'bulbasaur -> ivysaur via level-up (min level: 16)'."""
    lines: list[str] = []

    def walk(node: dict) -> None:
        for child in node["evolves_to"]:
            details = child["evolution_details"]
            methods = list(dict.fromkeys(_describe_conditions(d) for d in details))[:MAX_EVOLUTION_METHODS]
            how = "; or ".join(methods) if methods else "unknown"
            lines.append(f"{node['species']['name']} -> {child['species']['name']} via {how}")
            walk(child)

    walk(chain)
    return lines or [f"{chain['species']['name']} does not evolve"]


def level_up_moves(moves: list[dict]) -> list[dict[str, Any]]:
    """Level-up moves using the most recent game's learnset for each move."""
    result = []
    for m in moves:
        levels = [
            d["level_learned_at"] for d in m["version_group_details"] if d["move_learn_method"]["name"] == "level-up"
        ]
        if levels:
            result.append({"move": m["move"]["name"], "level": levels[-1]})
    result.sort(key=lambda x: (x["level"], x["move"]))
    return result[:MAX_LEVEL_UP_MOVES]
