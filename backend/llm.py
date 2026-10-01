"""Groq chat with PokeAPI grounding.

The model is "trained" on PokeAPI data by grounding: verified profiles are
injected into the system prompt, and the model can call tools to pull real data
for any other Pokémon. Type arithmetic for team building is done in code
(`team_fit`) because small models get matchups wrong when left to reason alone.
"""

import asyncio
import json
import logging
from collections import Counter
from typing import Any, Awaitable, Callable

from groq import APIError, AsyncGroq, AuthenticationError, RateLimitError

log = logging.getLogger("uvicorn.error")

MAX_TOOL_ROUNDS = 4
# gpt-oss is a reasoning model; reasoning tokens count against this budget.
MAX_COMPLETION_TOKENS = 3072
TEAM_SIZE = 6

SYSTEM_PROMPT = """You are PokéChat, an expert Pokémon assistant and team builder.

Ground rules:
- Facts about Pokémon (types, stats, abilities, matchups, evolutions, moves) must come from the
  PokeAPI data given below or returned by your tools. Never invent numbers or type relationships.
- If you need data about a Pokémon that is not below, call `get_pokemon`. If a tool says a
  Pokémon does not exist, tell the user it isn't in PokeAPI.
- Team building (max 6 members): pick candidates, then call `check_team_fit` ONCE with ALL of them.
  It computes, from real type data, which of the team's weaknesses each candidate covers and which
  it shares. Prefer candidates with a non-empty `covers_team_weaknesses` and an empty
  `shares_team_weaknesses`; if a pick is poor, call `check_team_fit` again with replacements.
  If the user has no team yet and wants one built around a Pokémon, pass it as `core`.
  Prefer fully evolved Pokémon with a high `base_stat_total`, never pick two Pokémon from the same
  evolution line, and suggest exactly as many members as asked. Give each pick a role (sweeper, wall, support...).
  When explaining picks, take every type claim ONLY from that candidate's `facts` sentence (copy
  its wording; do not add resistances or immunities that are not in it), and end with the team's
  remaining `unresisted_weaknesses` from `team_after`. Never print tool or field names.
- Never tell the user to call tools themselves.
- Refer to Pokémon by their PokeAPI names (e.g. "charizard", "mr-mime") at least once so the app
  can show their sprites.
- Be concise and use Markdown (short lists/tables) where it helps."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pokemon",
            "description": "Look up verified PokeAPI data for one Pokémon: types, base stats, abilities, "
            "type matchups, evolution chain and level-up moves. Returns an error if it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Pokémon name or Pokédex number"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_team_fit",
            "description": "Evaluate candidate team members against a team using real type data: for each "
            "candidate, which team weaknesses it covers (resists/immune) and which it shares, plus the "
            "team's remaining weaknesses if all candidates were added.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Pokémon names to evaluate (up to 6)",
                    },
                    "core": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Pokémon already on the team. Omit to use the user's current team.",
                    },
                },
                "required": ["candidates"],
            },
        },
    },
]

Lookup = Callable[[str], Awaitable[dict[str, Any] | None]]


class LLMError(Exception):
    """The LLM provider failed (bad key, rate limit, outage...)."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Strip UI-only fields and flatten moves so profiles cost fewer tokens."""
    data = {k: v for k, v in profile.items() if k not in ("sprite", "level_up_moves")}
    data["level_up_moves"] = ", ".join(f"{m['move']}@{m['level']}" for m in profile["level_up_moves"])
    data["type_matchups"] = {k: v for k, v in profile["type_matchups"].items() if v}
    return data


def team_analysis(team: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic team summary so the model doesn't have to do the arithmetic."""
    weak, resist = Counter(), Counter()
    for p in team:
        m = p["type_matchups"]
        weak.update(m["weak_to_4x"] + m["weak_to_2x"])
        resist.update(m["resists_0.5x"] + m["resists_0.25x"] + m["immune_to"])
    return {
        "members": [p["name"] for p in team],
        "open_slots": max(0, TEAM_SIZE - len(team)),
        "types_on_team": sorted({t for p in team for t in p["types"]}),
        "weakness_counts": dict(weak.most_common()),
        "shared_weaknesses_2plus": [t for t, n in weak.most_common() if n >= 2],
        "unresisted_weaknesses": [t for t in weak if resist[t] == 0],
    }


def _matchup_summary(profile: dict[str, Any]) -> dict[str, list[str]]:
    m = profile["type_matchups"]
    return {
        "weak_to": [f"{t} (4x)" for t in m["weak_to_4x"]] + m["weak_to_2x"],
        "resists": [f"{t} (0.25x)" for t in m["resists_0.25x"]] + m["resists_0.5x"],
        "immune_to": m["immune_to"],
    }


def _join(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def team_fit(core: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """How well each candidate patches the core team's weaknesses (pure type arithmetic)."""
    core_weak = Counter(t for p in core for t in p["type_matchups"]["weak_to_4x"] + p["type_matchups"]["weak_to_2x"])
    evaluated = []
    for c in candidates:
        m = c["type_matchups"]
        covered = set(m["resists_0.5x"] + m["resists_0.25x"] + m["immune_to"])
        weak = set(m["weak_to_4x"] + m["weak_to_2x"])
        covers = sorted(t for t in core_weak if t in covered)
        shares = sorted(t for t in core_weak if t in weak)
        summary = _matchup_summary(c)
        evaluated.append(
            {
                # A ready-made sentence: small models copy prose far more faithfully than they read JSON.
                "facts": (
                    f"{c['name']} ({'/'.join(c['types'])}, base stat total {c['base_stat_total']}) "
                    f"covers team weaknesses: {_join(covers)}; shares team weaknesses: {_join(shares)}; "
                    f"weak to: {_join(summary['weak_to'])}; resists: {_join(summary['resists'])}; "
                    f"immune to: {_join(summary['immune_to'])}."
                ),
                "name": c["name"],
                "base_stats": c["base_stats"],
                "covers_team_weaknesses": covers,
                "shares_team_weaknesses": shares,
            }
        )
    return {"core": [p["name"] for p in core], "candidates": evaluated, "team_after": team_analysis(core + candidates)}


def build_context(profiles: list[dict[str, Any]], team: list[dict[str, Any]]) -> str:
    sections = [SYSTEM_PROMPT]
    if team:
        members = {p["name"]: {"types": p["types"], **_matchup_summary(p)} for p in team}
        sections.append(
            "## User's current team (verified)\n" + json.dumps({"members": members, "analysis": team_analysis(team)})
        )
    if profiles:
        sections.append(
            "## PokeAPI data for Pokémon in this conversation\n"
            + "\n".join(json.dumps(compact_profile(p)) for p in profiles)
        )
    return "\n\n".join(sections)


class GroqChat:
    def __init__(self, api_key: str, model: str, client: AsyncGroq | None = None):
        self._client = client or AsyncGroq(api_key=api_key)
        self._model = model

    async def reply(
        self,
        messages: list[dict[str, str]],
        profiles: list[dict[str, Any]],
        team: list[dict[str, Any]],
        lookup: Lookup,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return the assistant reply and any profiles the model looked up via tools."""
        convo: list[dict[str, Any]] = [{"role": "system", "content": build_context(profiles, team)}, *messages]
        looked_up: list[dict[str, Any]] = []

        for round_ in range(MAX_TOOL_ROUNDS + 1):
            use_tools = round_ < MAX_TOOL_ROUNDS
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=convo,
                    tools=TOOLS if use_tools else None,
                    tool_choice="auto" if use_tools else None,
                    reasoning_effort="low",
                    temperature=0.4,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
            except AuthenticationError as exc:
                raise LLMError(
                    "Groq rejected the API key. Check GROQ_API_KEY in .env (or the app's Secrets on Streamlit Cloud).",
                    status_code=502,
                ) from exc
            except RateLimitError as exc:
                raise LLMError(
                    "Groq rate limit reached (the free tier allows ~8k tokens/minute). Wait a moment and retry.",
                    status_code=429,
                ) from exc
            except APIError as exc:
                raise LLMError(f"LLM request failed: {exc}") from exc

            msg = resp.choices[0].message
            log.info("groq round %d: %d tool call(s), usage=%s", round_, len(msg.tool_calls or []), resp.usage)
            if not msg.tool_calls:
                if not (msg.content or "").strip():
                    raise LLMError(f"The model returned an empty answer (finish_reason={resp.choices[0].finish_reason}).")
                return msg.content.strip(), looked_up

            convo.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                        for c in msg.tool_calls
                    ],
                }
            )
            for call in msg.tool_calls:
                result, profiles = await self._run_tool(call.function.name, call.function.arguments, team, lookup)
                looked_up += profiles
                convo.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

        raise LLMError("The model did not produce an answer.")

    @staticmethod
    async def _run_tool(
        name: str, arguments: str, team: list[dict[str, Any]], lookup: Lookup
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Execute a tool call; returns (result for the model, profiles fetched)."""
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}, []

        async def fetch(names: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
            results = await asyncio.gather(*(lookup(str(n)) for n in names))
            return [p for p in results if p], [str(n) for n, p in zip(names, results) if not p]

        if name == "get_pokemon":
            query = str(args.get("name", ""))
            profile = await lookup(query)
            if not profile:
                return {"error": f"'{query}' does not exist in PokeAPI"}, []
            return compact_profile(profile), [profile]

        if name == "check_team_fit":
            candidates, missing = await fetch(list(args.get("candidates") or [])[:TEAM_SIZE])
            core = team
            if args.get("core"):
                core, missing_core = await fetch(list(args["core"])[:TEAM_SIZE])
                missing += missing_core
            result = team_fit(core, candidates)
            if missing:
                result["not_in_pokeapi"] = missing
            return result, candidates

        return {"error": f"Unknown tool {name}"}, []
