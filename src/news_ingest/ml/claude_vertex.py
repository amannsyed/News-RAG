from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal


DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5@20251001"
DEFAULT_BEDROCK_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_MAX_TOKENS = 1800
DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class ClaudeSettings:
    project_id: str
    location: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    provider: Literal["vertex"] = "vertex"


@dataclass(frozen=True)
class BedrockSettings:
    region: str
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    provider: Literal["bedrock"] = "bedrock"


ChatSettings = ClaudeSettings | BedrockSettings
ThinkingLevel = Literal["low", "medium", "max"]
DEFAULT_THINKING_BUDGET_TOKENS: dict[ThinkingLevel, int] = {"low": 512, "medium": 1024, "max": 2048}


def load_claude_settings() -> ClaudeSettings:
    project_id = os.getenv("VERTEX_PROJECT_ID", "").strip() or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.getenv("VERTEX_LOCATION", "global").strip() or "global"
    model = os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
    max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    temperature = float(os.getenv("CLAUDE_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    if not project_id:
        raise ValueError("Missing VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT")
    return ClaudeSettings(project_id=project_id, location=location, model=model, max_tokens=max_tokens, temperature=temperature)


def load_bedrock_settings() -> BedrockSettings:
    key_config = _aws_bedrock_key_config()
    region = (
        os.getenv("AWS_BEDROCK_REGION", "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or str(key_config.get("region", "")).strip()
        or "us-east-1"
    )
    model = os.getenv("AWS_BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL).strip() or DEFAULT_BEDROCK_MODEL
    max_tokens = int(os.getenv("CLAUDE_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    temperature = float(os.getenv("CLAUDE_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    return BedrockSettings(region=region, model=model, max_tokens=max_tokens, temperature=temperature)


def load_chat_settings() -> ChatSettings:
    provider = os.getenv("CHAT_PROVIDER", "").strip().lower()
    if not provider:
        provider = "bedrock" if os.getenv("AWS_BEDROCK_KEY", "").strip() else "vertex"
    if provider == "bedrock":
        return load_bedrock_settings()
    if provider == "vertex":
        return load_claude_settings()
    raise ValueError("CHAT_PROVIDER must be 'bedrock' or 'vertex'")


def with_model_override(settings: ChatSettings, model: str | None) -> ChatSettings:
    model = (model or "").strip()
    if not model:
        return settings
    if isinstance(settings, BedrockSettings):
        return BedrockSettings(region=settings.region, model=model, max_tokens=settings.max_tokens, temperature=settings.temperature)
    return ClaudeSettings(project_id=settings.project_id, location=settings.location, model=model, max_tokens=settings.max_tokens, temperature=settings.temperature)


def build_rag_messages(
    *,
    question: str,
    citations: list[dict[str, Any]],
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    context = _citation_context(citations)
    memory_context = _memory_context(conversation_summary=conversation_summary, recent_turns=recent_turns or [])
    system = (
        "You are a news RAG assistant. Answer only from the provided article context and conversation memory. "
        "Use conversation memory only to resolve references, prior user preferences, and continuity; do not use it as factual news evidence. "
        "Write the answer in clean, readable Markdown. Use short headings and bullets when useful. "
        "Every factual claim from article context must include an inline citation marker like [1] or [2]. "
        "Use only citation markers that appear in the provided article context. "
        "If the article context is insufficient, say what is missing instead of guessing. "
        "Do not include follow-up questions and do not include a separate sources section."
    )
    user = f"Question:\n{question}\n\nConversation memory:\n{memory_context}\n\nArticle context:\n{context}"
    return system, [{"role": "user", "content": user}]


def generate_chat_answer(
    *,
    question: str,
    citations: list[dict[str, Any]],
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, str]] | None = None,
    thinking_enabled: bool = False,
    thinking_level: ThinkingLevel = "medium",
    settings: ChatSettings | None = None,
) -> tuple[str, ChatSettings]:
    settings = settings or load_chat_settings()
    if isinstance(settings, BedrockSettings):
        return generate_bedrock_answer(question=question, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, thinking_enabled=thinking_enabled, thinking_level=thinking_level, settings=settings), settings
    return generate_claude_answer(question=question, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, thinking_enabled=thinking_enabled, thinking_level=thinking_level, settings=settings), settings


def generate_claude_answer(
    *,
    question: str,
    citations: list[dict[str, Any]],
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, str]] | None = None,
    thinking_enabled: bool = False,
    thinking_level: ThinkingLevel = "medium",
    settings: ClaudeSettings | None = None,
) -> str:
    settings = settings or load_claude_settings()
    system, messages = build_rag_messages(question=question, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns)
    thinking_budget = thinking_budget_tokens(thinking_enabled=thinking_enabled, thinking_level=thinking_level)
    return _invoke_vertex_text(settings=_settings_for_thinking(settings, thinking_budget), system=system, messages=messages, thinking_budget_tokens=thinking_budget).strip()


def generate_bedrock_answer(
    *,
    question: str,
    citations: list[dict[str, Any]],
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, str]] | None = None,
    thinking_enabled: bool = False,
    thinking_level: ThinkingLevel = "medium",
    settings: BedrockSettings | None = None,
) -> str:
    settings = settings or load_bedrock_settings()
    system, messages = build_rag_messages(question=question, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns)
    thinking_budget = thinking_budget_tokens(thinking_enabled=thinking_enabled, thinking_level=thinking_level)
    return _invoke_bedrock_text(settings=_settings_for_thinking(settings, thinking_budget), system=system, user_content=str(messages[0]["content"]), thinking_budget_tokens=thinking_budget).strip()


def generate_chat_summary(
    *,
    previous_summary: str | None,
    recent_turns: list[dict[str, str]],
    latest_user_message: str,
    latest_assistant_answer: str,
    settings: ChatSettings | None = None,
) -> tuple[str, ChatSettings]:
    settings = settings or load_chat_settings()
    system = (
        "You maintain a compact rolling summary for a news RAG conversation. "
        "Summarize durable user intent, constraints, entities, open questions, and conclusions. "
        "Do not include citations. Keep it under 180 words."
    )
    turns = _memory_context(conversation_summary=previous_summary, recent_turns=recent_turns)
    user_content = (
        f"Previous memory and recent turns:\n{turns}\n\n"
        f"Latest user message:\n{latest_user_message}\n\n"
        f"Latest assistant answer:\n{latest_assistant_answer}\n\n"
        "Return only the updated rolling summary."
    )
    return _invoke_with_settings(settings=settings, system=system, user_content=user_content, max_tokens=500).strip(), settings


def generate_follow_up_questions(
    *,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    conversation_summary: str | None = None,
    recent_turns: list[dict[str, str]] | None = None,
    settings: ChatSettings | None = None,
) -> tuple[list[str], ChatSettings]:
    settings = settings or load_chat_settings()
    system = (
        "Generate exactly three concise follow-up questions for a news RAG chat. "
        "The questions should be useful next queries the user can ask. "
        "Return only a JSON array of three strings."
    )
    user_content = (
        f"User question:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        f"Conversation memory:\n{_memory_context(conversation_summary=conversation_summary, recent_turns=recent_turns or [])}\n\n"
        f"Available citation titles:\n{_citation_titles(citations)}"
    )
    raw = _invoke_with_settings(settings=settings, system=system, user_content=user_content, max_tokens=350).strip()
    return _parse_follow_up_questions(raw), settings


def _invoke_with_settings(*, settings: ChatSettings, system: str, user_content: str, max_tokens: int | None = None) -> str:
    effective_settings = _with_max_tokens(settings, max_tokens) if max_tokens else settings
    if isinstance(effective_settings, BedrockSettings):
        return _invoke_bedrock_text(settings=effective_settings, system=system, user_content=user_content)
    return _invoke_vertex_text(settings=effective_settings, system=system, messages=[{"role": "user", "content": user_content}])


def _with_max_tokens(settings: ChatSettings, max_tokens: int | None) -> ChatSettings:
    if max_tokens is None or max_tokens >= settings.max_tokens:
        return settings
    if isinstance(settings, BedrockSettings):
        return BedrockSettings(region=settings.region, model=settings.model, max_tokens=max_tokens, temperature=settings.temperature)
    return ClaudeSettings(project_id=settings.project_id, location=settings.location, model=settings.model, max_tokens=max_tokens, temperature=settings.temperature)


def thinking_budget_tokens(*, thinking_enabled: bool, thinking_level: ThinkingLevel = "medium") -> int | None:
    if not thinking_enabled:
        return None
    return _thinking_budget_config().get(thinking_level, _thinking_budget_config()["medium"])


def _thinking_budget_config() -> dict[ThinkingLevel, int]:
    return {
        "low": _env_int("CLAUDE_THINKING_LOW_TOKENS", DEFAULT_THINKING_BUDGET_TOKENS["low"]),
        "medium": _env_int("CLAUDE_THINKING_MEDIUM_TOKENS", DEFAULT_THINKING_BUDGET_TOKENS["medium"]),
        "max": _env_int("CLAUDE_THINKING_MAX_TOKENS", DEFAULT_THINKING_BUDGET_TOKENS["max"]),
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _settings_for_thinking(settings: ChatSettings, thinking_budget: int | None) -> ChatSettings:
    if not thinking_budget:
        return settings
    # Anthropic's max_tokens includes both internal thinking and visible output.
    max_tokens = max(settings.max_tokens, thinking_budget + settings.max_tokens)
    if isinstance(settings, BedrockSettings):
        return BedrockSettings(region=settings.region, model=settings.model, max_tokens=max_tokens, temperature=1.0)
    return ClaudeSettings(project_id=settings.project_id, location=settings.location, model=settings.model, max_tokens=max_tokens, temperature=1.0)


def _thinking_payload(thinking_budget_tokens: int | None) -> dict[str, Any] | None:
    if not thinking_budget_tokens:
        return None
    return {"type": "enabled", "budget_tokens": thinking_budget_tokens}


def _invoke_vertex_text(*, settings: ClaudeSettings, system: str, messages: list[dict[str, Any]], thinking_budget_tokens: int | None = None) -> str:
    try:
        from anthropic import AnthropicVertex
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("anthropic package is required for Claude Vertex chat") from exc

    credentials = _load_google_credentials_in_memory()
    client = AnthropicVertex(project_id=settings.project_id, region=settings.location, credentials=credentials)
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "system": system,
        "messages": messages,
    }
    thinking = _thinking_payload(thinking_budget_tokens)
    if thinking:
        kwargs["thinking"] = thinking
    response = client.messages.create(**kwargs)
    return "".join(getattr(block, "text", "") or "" for block in response.content)


def _invoke_bedrock_text(*, settings: BedrockSettings, system: str, user_content: str, thinking_budget_tokens: int | None = None) -> str:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("boto3 is required for AWS Bedrock chat") from exc

    os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    client = boto3.client("bedrock-runtime", region_name=settings.region, **_bedrock_client_kwargs())
    kwargs: dict[str, Any] = {
        "modelId": settings.model,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user_content}]}],
        "inferenceConfig": {"maxTokens": settings.max_tokens, "temperature": settings.temperature},
    }
    thinking = _thinking_payload(thinking_budget_tokens)
    if thinking:
        kwargs["additionalModelRequestFields"] = {"thinking": thinking}
    response = client.converse(**kwargs)
    content = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(part.get("text", "") for part in content)


def _memory_context(*, conversation_summary: str | None, recent_turns: list[dict[str, str]]) -> str:
    lines: list[str] = []
    if conversation_summary:
        lines.append(f"Rolling summary:\n{conversation_summary.strip()}")
    if recent_turns:
        turn_lines = []
        for turn in recent_turns[-10:]:
            role = turn.get("role", "message")
            content = " ".join((turn.get("content") or "").split())
            if len(content) > 900:
                content = content[:897].rstrip() + "..."
            turn_lines.append(f"{role}: {content}")
        lines.append("Recent turns:\n" + "\n".join(turn_lines))
    return "\n\n".join(lines) if lines else "No prior conversation memory."


def _citation_context(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "No retrieved article context."
    lines = []
    for index, citation in enumerate(citations, start=1):
        marker = citation.get("citation_marker") or f"[{index}]"
        source_type = citation.get("source_type") or "rag_article"
        source_id = citation.get("article_id") if source_type == "rag_article" else citation.get("web_search_id")
        title = citation.get("title") or "Untitled article"
        provider = citation.get("provider") or "unknown source"
        published_at = citation.get("published_at") or "unknown date"
        url = citation.get("url") or ""
        snippet = citation.get("snippet") or "No snippet available."
        lines.append(f"{marker} source_type={source_type} source_id={source_id} | {title} ({provider}, {published_at})\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)


def _citation_titles(citations: list[dict[str, Any]]) -> str:
    titles = []
    for citation in citations[:10]:
        marker = citation.get("citation_marker") or ""
        title = citation.get("title") or "Untitled source"
        titles.append(f"{marker} {title}".strip())
    return "\n".join(titles) if titles else "No citations."


def _parse_follow_up_questions(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        questions = [str(item).strip() for item in parsed if str(item).strip()]
    else:
        questions = []
        for line in raw.splitlines():
            cleaned = line.strip().lstrip("-*").strip()
            if ". " in cleaned[:4]:
                cleaned = cleaned.split(". ", 1)[1].strip()
            if cleaned:
                questions.append(cleaned.strip('"'))
    deduped = []
    seen = set()
    for question in questions:
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(question)
        if len(deduped) == 3:
            break
    return deduped


def _load_google_credentials_in_memory():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return None

    credentials_json = (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
        or os.getenv("VERTEX_CREDENTIALS_JSON", "").strip()
        or os.getenv("GOOGLE_VERTEX_AI_KEY", "").strip()
    )
    if not credentials_json or not credentials_json.startswith("{"):
        return None

    try:
        from google.oauth2 import service_account  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-auth is required for in-memory Vertex credentials") from exc

    info = json.loads(credentials_json)
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _aws_bedrock_key_config() -> dict[str, Any]:
    value = os.getenv("AWS_BEDROCK_KEY", "").strip()
    if not value or not value.startswith("{"):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _bedrock_client_kwargs() -> dict[str, str]:
    key_config = _aws_bedrock_key_config()
    raw_key = os.getenv("AWS_BEDROCK_KEY", "").strip()
    delimited_key = raw_key.split(":") if raw_key and not raw_key.startswith("{") else []
    access_key = (
        os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        or str(key_config.get("aws_access_key_id", "")).strip()
        or str(key_config.get("access_key_id", "")).strip()
        or (delimited_key[0].strip() if len(delimited_key) >= 2 else "")
    )
    secret_key = (
        os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        or os.getenv("AWS_BEDROCK_SECRET", "").strip()
        or str(key_config.get("aws_secret_access_key", "")).strip()
        or str(key_config.get("secret_access_key", "")).strip()
        or (delimited_key[1].strip() if len(delimited_key) >= 2 else "")
    )
    session_token = (
        os.getenv("AWS_SESSION_TOKEN", "").strip()
        or str(key_config.get("aws_session_token", "")).strip()
        or (delimited_key[2].strip() if len(delimited_key) >= 3 else "")
    )
    kwargs: dict[str, str] = {}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        kwargs["aws_session_token"] = session_token
    return kwargs
