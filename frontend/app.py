"""PokéChat Streamlit frontend. All data and LLM calls go through the FastAPI backend."""

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TEAM_SIZE = 6
TYPE_COLORS = {
    "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0", "electric": "#F7D02C",
    "grass": "#7AC74C", "ice": "#96D9D6", "fighting": "#C22E28", "poison": "#A33EA1",
    "ground": "#E2BF65", "flying": "#A98FF3", "psychic": "#F95587", "bug": "#A6B91A",
    "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC", "dark": "#705746",
    "steel": "#B7B7CE", "fairy": "#D685AD",
}
EXAMPLE_PROMPTS = [
    "What are Gengar's weaknesses?",
    "How does Eevee evolve into Umbreon?",
    "Build me a balanced team around Pikachu",
]

st.set_page_config(page_title="PokéChat", page_icon="⚡", layout="wide")

ss = st.session_state
ss.setdefault("messages", [])  # {"role", "content", "pokemon": [cards]}
ss.setdefault("team", [])  # cards
ss.setdefault("selected", None)  # full profile from the Pokédex lookup
ss.setdefault("pending_prompt", None)


# ------------------------------------------------------------------- API


class BackendError(Exception):
    pass


def api(method: str, path: str, **kwargs):
    try:
        resp = httpx.request(method, f"{BACKEND_URL}{path}", timeout=90, **kwargs)
    except httpx.HTTPError as exc:
        raise BackendError(f"Can't reach the backend at {BACKEND_URL} ({exc}). Is FastAPI running?") from exc
    if resp.status_code >= 400:
        raise BackendError(resp.json().get("detail", resp.text))
    return resp.json()


# ------------------------------------------------------------- rendering


def type_badges(types: list[str]) -> str:
    return " ".join(
        f"<span style='background:{TYPE_COLORS.get(t, '#888')};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.75rem;font-weight:600'>{t.upper()}</span>"
        for t in types
    )


def add_to_team(card: dict) -> None:
    if any(p["name"] == card["name"] for p in ss.team):
        st.toast(f"{card['display_name']} is already on your team")
    elif len(ss.team) >= TEAM_SIZE:
        st.toast("Your team is full (6 Pokémon)")
    else:
        ss.team.append(card)
        st.toast(f"Added {card['display_name']} to your team")


def render_card(card: dict, key: str, show_add: bool = True) -> None:
    with st.container(border=True):
        if card.get("sprite"):
            st.image(card["sprite"], width=110)
        st.markdown(f"**#{card['id']} {card['display_name']}**")
        st.markdown(type_badges(card["types"]), unsafe_allow_html=True)
        on_team = any(p["name"] == card["name"] for p in ss.team)
        if show_add and not on_team:
            st.button("➕ Team", key=key, on_click=add_to_team, args=(card,), width="stretch")


def render_cards(cards: list[dict], key_prefix: str) -> None:
    if not cards:
        return
    cols = st.columns(min(len(cards), 4))
    for i, card in enumerate(cards):
        with cols[i % len(cols)]:
            render_card(card, key=f"{key_prefix}-{card['name']}")


def render_profile(p: dict) -> None:
    if p.get("sprite"):
        st.image(p["sprite"], width="stretch")
    st.markdown(f"### #{p['id']} {p['display_name']}")
    st.caption(" · ".join(filter(None, [p.get("genus"), (p.get("generation") or "").replace("-", " ").title()])))
    st.markdown(type_badges(p["types"]), unsafe_allow_html=True)
    if p.get("pokedex_entry"):
        st.write(f"_{p['pokedex_entry']}_")
    st.markdown(f"**Height** {p['height_m']} m · **Weight** {p['weight_kg']} kg")
    st.markdown("**Abilities:** " + ", ".join(a["name"] + (" (hidden)" if a["hidden"] else "") for a in p["abilities"]))
    for stat, value in p["base_stats"].items():
        st.progress(min(value / 200, 1.0), text=f"{stat.replace('-', ' ').title()}: {value}")
    st.caption(f"Base stat total: {p['base_stat_total']}")
    weak = [f"{t} ×4" for t in p["type_matchups"]["weak_to_4x"]] + p["type_matchups"]["weak_to_2x"]
    if weak:
        st.markdown("**Weak to:** " + ", ".join(weak))
    card = {k: p[k] for k in ("id", "name", "display_name", "types", "sprite")}
    if not any(m["name"] == p["name"] for m in ss.team):
        st.button("➕ Add to team", key=f"lookup-add-{p['name']}", on_click=add_to_team, args=(card,), type="primary")


# ---------------------------------------------------------------- sidebar


def lookup(name: str) -> None:
    name = name.strip()
    if not name:
        return
    try:
        check = api("GET", f"/pokemon/{name}/exists")
        if not check["exists"]:
            ss.selected = None
            st.sidebar.error(f"“{name}” doesn't exist in PokeAPI.")
            return
        ss.selected = api("GET", f"/pokemon/{check['name']}")
    except BackendError as exc:
        st.sidebar.error(str(exc))


with st.sidebar:
    st.header("🔎 Pokédex")
    with st.form("lookup", clear_on_submit=True, border=False):
        query = st.text_input("Pokémon name or number", placeholder="e.g. pikachu, 25, mr. mime")
        submitted = st.form_submit_button("Look up", width="stretch")
    if submitted:
        lookup(query)
    if ss.selected:
        render_profile(ss.selected)

    st.divider()
    st.header(f"🎒 My team ({len(ss.team)}/{TEAM_SIZE})")
    if not ss.team:
        st.caption("Add Pokémon from the Pokédex or from chat answers.")
    cols = st.columns(3)
    for i, member in enumerate(list(ss.team)):
        with cols[i % 3]:
            if member.get("sprite"):
                st.image(member["sprite"], width="stretch")
            st.caption(member["display_name"])
            st.button("✖", key=f"remove-{member['name']}", help=f"Remove {member['display_name']}",
                      on_click=lambda n=member["name"]: ss.update(team=[p for p in ss.team if p["name"] != n]))
    if ss.team:
        try:
            analysis = api("POST", "/team/analysis", json={"team": [p["name"] for p in ss.team]})
            if analysis["shared_weaknesses_2plus"]:
                st.warning("Shared weaknesses: " + ", ".join(analysis["shared_weaknesses_2plus"]))
            if analysis["unresisted_weaknesses"]:
                st.caption("Nobody resists: " + ", ".join(analysis["unresisted_weaknesses"]))
        except BackendError as exc:
            st.caption(f"Team analysis unavailable: {exc}")
        b1, b2 = st.columns(2)
        if b1.button("🤖 Rate my team", width="stretch"):
            ss.pending_prompt = "Rate my current team and suggest how to fill the open slots."
        b2.button("Clear team", width="stretch", on_click=lambda: ss.update(team=[]))

    st.divider()
    st.button("🗑️ New chat", width="stretch", on_click=lambda: ss.update(messages=[]))


# ------------------------------------------------------------------- chat

st.title("⚡ PokéChat")
st.caption("Ask anything about Pokémon or get help building a team. Answers are grounded in live PokeAPI data.")

if not ss.messages:
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, prompt in zip(cols, EXAMPLE_PROMPTS):
        if col.button(prompt, width="stretch"):
            ss.pending_prompt = prompt

for i, msg in enumerate(ss.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("not_found"):
                st.warning("Not found in PokeAPI: " + ", ".join(msg["not_found"]))
            render_cards(msg.get("pokemon", []), key_prefix=f"msg{i}")

prompt = st.chat_input("Ask about a Pokémon or your team…") or ss.pending_prompt
ss.pending_prompt = None

if prompt:
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking PokeAPI and thinking…"):
            try:
                data = api(
                    "POST",
                    "/chat",
                    json={
                        "messages": [{"role": m["role"], "content": m["content"]} for m in ss.messages],
                        "team": [p["name"] for p in ss.team],
                    },
                )
            except BackendError as exc:
                ss.messages.pop()  # let the user retry the same question
                st.error(str(exc))
                st.stop()
        ss.messages.append(
            {"role": "assistant", "content": data["reply"], "pokemon": data["pokemon"], "not_found": data["not_found"]}
        )
    st.rerun()
