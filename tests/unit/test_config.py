from src.config import get_settings


def test_settings_use_one_default_for_reranking(monkeypatch):
    monkeypatch.delenv("ENABLE_RERANKING", raising=False)
    settings = get_settings(load_environment=False)
    assert settings.enable_reranking is False


def test_legacy_list_prediction_setting_is_supported(monkeypatch):
    monkeypatch.delenv("LIST_NUM_PREDICT", raising=False)
    monkeypatch.setenv("OLLAMA_LIST_NUM_PREDICT", "777")
    assert get_settings(load_environment=False).list_num_predict == 777


def test_explicit_list_prediction_setting_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_LIST_NUM_PREDICT", "777")
    monkeypatch.setenv("LIST_NUM_PREDICT", "888")
    assert get_settings(load_environment=False).list_num_predict == 888
