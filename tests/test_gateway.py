import httpx
import pytest

from backend import service
from backend.llm import LLMError
from backend.pokeapi_client import PokeAPIClient
from frontend.gateway import BackendError, LocalBackend
from tests.conftest import BASE, fake_pokeapi_transport
from tests.test_api import FakeLLM


@pytest.fixture
def local_backend(monkeypatch):
    """The in-process backend used on Streamlit Community Cloud, wired to fakes."""
    llm = FakeLLM()

    async def create_pokeapi(_http):
        client = PokeAPIClient(httpx.AsyncClient(transport=fake_pokeapi_transport([])), BASE)
        await client.load_index()
        return client

    monkeypatch.setattr(service, "create_pokeapi", create_pokeapi)
    monkeypatch.setattr(service, "create_llm", lambda: llm)
    return LocalBackend(), llm


def test_local_backend_checks_existence_and_gathers_profiles(local_backend):
    backend, _ = local_backend
    assert backend.exists("Deoxys")["name"] == "deoxys-normal"
    assert backend.exists("fakemon")["exists"] is False
    assert backend.profile("pikachu")["types"] == ["electric"]
    with pytest.raises(BackendError, match="does not exist"):
        backend.profile("fakemon")


def test_local_backend_chat_matches_http_response_shape(local_backend):
    backend, _ = local_backend
    body = backend.chat([{"role": "user", "content": "pikachu?"}], ["fakemon"])
    assert set(body) == {"reply", "pokemon", "not_found"}
    assert [p["name"] for p in body["pokemon"]] == ["mr-mime", "pikachu"]
    assert body["not_found"] == ["fakemon"]


def test_local_backend_turns_llm_failures_into_backend_errors(local_backend):
    backend, llm = local_backend
    llm.error = LLMError("rate limited", status_code=429)
    with pytest.raises(BackendError, match="rate limited"):
        backend.chat([{"role": "user", "content": "hi"}], [])
