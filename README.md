# PokéChat

Chat with an LLM about Pokémon and get help building a team. Every fact it uses comes from
[PokeAPI](https://pokeapi.co/docs/v2).

- **Frontend:** Streamlit (`frontend/app.py`). It has a chat, a Pokédex lookup with sprites and stats, and a team builder.
- **Backend:** FastAPI (`backend/`). It checks that each Pokémon exists, gathers its data from PokeAPI, and calls Groq.
- **LLM:** `openai/gpt-oss-20b` on [Groq](https://console.groq.com).

## How it works

1. **Existence check:** at startup the backend loads the index of all PokeAPI species and Pokémon
   names. Lookups (`/pokemon/{name}/exists`) and name detection in chat messages are then instant. Species names
   resolve to their default form (`deoxys` → `deoxys-normal`). Input like `Mr. Mime` or `Farfetch'd` is normalized.
2. **Data gathering:** for each verified Pokémon the backend assembles one profile. It covers types, base stats,
   abilities with their effects, type matchups (computed from type damage relations), evolution chain,
   level-up moves, Pokédex entry and artwork. Every PokeAPI response is cached in memory.
3. **Grounded chat:** profiles of the Pokémon mentioned in the conversation, plus your team, go into the
   system prompt. The model can also call tools:
   - `get_pokemon(name)`: verified data for any other Pokémon.
   - `check_team_fit(candidates, core?)`: computes in code which of the team's weaknesses each candidate
     covers or shares. Small models get type math wrong, so the model picks and explains while the code does the arithmetic.
4. The response includes cards (sprite and types) for the Pokémon the answer mentions. One click adds a card to your team.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then set GROQ_API_KEY
```

## Run

```bash
./run.sh
```

Or run each piece in its own terminal:

```bash
.venv/bin/uvicorn backend.main:app --reload --port 8000
.venv/bin/streamlit run frontend/app.py
```

Open http://localhost:8501. API docs are at http://localhost:8000/docs.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/pokemon/{name}/exists` | Whether a Pokémon exists, and its canonical name |
| GET | `/pokemon/{name}` | Full gathered profile (404 if it doesn't exist) |
| POST | `/team/analysis` | `{"team": [...]}`: shared and unresisted weaknesses |
| POST | `/chat` | `{"messages": [...], "team": [...]}`: reply, Pokémon cards, names not found |

## Tests

```bash
.venv/bin/python -m pytest
```

Tests use a fake PokeAPI and a fake LLM, so they never touch the network.

## Notes

- Groq's free tier allows about 8k tokens per minute for `gpt-oss-20b`, and reasoning tokens count toward that limit.
  When you hit it, the API returns 429 and the UI asks you to retry.
- `.env` is git-ignored. Never commit your API key.
