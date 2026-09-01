from src.config import get_settings
from src.index_manager import IndexManager


def test_verify_missing_index_is_non_destructive(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "index"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "missing")
    settings = get_settings(load_environment=False)

    status = IndexManager(settings).verify()

    assert status.exists is False
    assert status.document_count == 0


def test_loading_an_empty_index_requires_an_explicit_build(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "index"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "existing")
    settings = get_settings(load_environment=False)
    manager = IndexManager(settings)
    manager._client().create_collection(settings.collection_name)

    try:
        manager.create_retriever()
    except RuntimeError as exc:
        assert "unavailable or empty" in str(exc)
    else:
        raise AssertionError("Empty index should not be loaded by the application")
