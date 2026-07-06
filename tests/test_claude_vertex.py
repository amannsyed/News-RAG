from news_ingest.ml.claude_vertex import BedrockSettings, DEFAULT_BEDROCK_MODEL, DEFAULT_CLAUDE_MODEL, build_rag_messages, load_bedrock_settings, load_chat_settings, load_claude_settings, with_model_override, _parse_follow_up_questions, thinking_budget_tokens, _thinking_payload


def test_build_rag_messages_includes_citations_and_grounding_instruction() -> None:
    system, messages = build_rag_messages(
        question="What happened?",
        citations=[{"article_id": 7, "title": "Title", "provider": "newsapi", "published_at": "2026-07-05", "url": "https://example.com", "snippet": "Evidence text."}],
    )

    assert "Answer only from the provided article context" in system
    assert "clean, readable Markdown" in system
    assert "inline citation marker like [1]" in system
    assert messages[0]["role"] == "user"
    assert "[1] source_type=rag_article source_id=7 | Title" in messages[0]["content"]
    assert "Evidence text." in messages[0]["content"]


def test_build_rag_messages_labels_web_search_citations() -> None:
    _, messages = build_rag_messages(
        question="What is new?",
        citations=[{"source_type": "web_search", "web_search_id": 9, "title": "Web Title", "provider": "web:bing_news", "published_at": None, "url": "https://example.com/news", "snippet": "Web evidence."}],
    )

    assert "[1] source_type=web_search source_id=9 | Web Title" in messages[0]["content"]
    assert "Web evidence." in messages[0]["content"]


def test_load_claude_settings_uses_vertex_project_env(monkeypatch) -> None:
    monkeypatch.setenv("VERTEX_PROJECT_ID", "project-1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    settings = load_claude_settings()

    assert settings.project_id == "project-1"
    assert settings.location == "global"
    assert settings.model == DEFAULT_CLAUDE_MODEL


def test_load_bedrock_settings_uses_json_key_region(monkeypatch) -> None:
    monkeypatch.delenv("AWS_BEDROCK_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_MODEL_ID", raising=False)
    monkeypatch.setenv("AWS_BEDROCK_KEY", '{"region":"us-west-2"}')

    settings = load_bedrock_settings()

    assert settings.provider == "bedrock"
    assert settings.region == "us-west-2"
    assert settings.model == DEFAULT_BEDROCK_MODEL


def test_load_chat_settings_prefers_bedrock_when_key_exists(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_PROVIDER", raising=False)
    monkeypatch.setenv("AWS_BEDROCK_KEY", "token-value")

    settings = load_chat_settings()

    assert settings.provider == "bedrock"


def test_parse_follow_up_questions_handles_fenced_json() -> None:
    raw = """```json
["What changed?", "Who is affected?", "What happens next?"]
```"""

    assert _parse_follow_up_questions(raw) == ["What changed?", "Who is affected?", "What happens next?"]


def test_thinking_budget_levels(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_THINKING_LOW_TOKENS", raising=False)
    monkeypatch.delenv("CLAUDE_THINKING_MEDIUM_TOKENS", raising=False)
    monkeypatch.delenv("CLAUDE_THINKING_MAX_TOKENS", raising=False)
    assert thinking_budget_tokens(thinking_enabled=False, thinking_level="max") is None
    assert thinking_budget_tokens(thinking_enabled=True, thinking_level="low") == 512
    assert thinking_budget_tokens(thinking_enabled=True, thinking_level="medium") == 1024
    assert thinking_budget_tokens(thinking_enabled=True, thinking_level="max") == 2048


def test_thinking_payload_is_anthropic_shape() -> None:
    assert _thinking_payload(1024) == {"type": "enabled", "budget_tokens": 1024}
    assert _thinking_payload(None) is None


def test_thinking_budget_levels_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_THINKING_LOW_TOKENS", "700")
    monkeypatch.setenv("CLAUDE_THINKING_MEDIUM_TOKENS", "1300")
    monkeypatch.setenv("CLAUDE_THINKING_MAX_TOKENS", "2600")

    assert thinking_budget_tokens(thinking_enabled=True, thinking_level="low") == 700
    assert thinking_budget_tokens(thinking_enabled=True, thinking_level="medium") == 1300
    assert thinking_budget_tokens(thinking_enabled=True, thinking_level="max") == 2600


def test_with_model_override_preserves_bedrock_settings() -> None:
    settings = BedrockSettings(region="eu-west-2", model="old-model", max_tokens=900, temperature=0.4)

    overridden = with_model_override(settings, "anthropic.claude-sonnet-5")

    assert isinstance(overridden, BedrockSettings)
    assert overridden.region == "eu-west-2"
    assert overridden.model == "anthropic.claude-sonnet-5"
    assert overridden.max_tokens == 900
    assert overridden.temperature == 0.4


def test_with_model_override_ignores_blank_model() -> None:
    settings = BedrockSettings(region="eu-west-2", model="old-model")

    assert with_model_override(settings, "  ") is settings
