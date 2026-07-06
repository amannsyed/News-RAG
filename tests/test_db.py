from news_ingest.db import content_hash, normalize_content, normalize_url, url_hash


def test_normalize_url_removes_fragment_and_utm_params() -> None:
    assert (
        normalize_url("HTTPS://Example.COM/story?utm_source=x&a=1#section")
        == "https://example.com/story?a=1"
    )


def test_url_hash_is_stable_for_tracking_variants() -> None:
    assert url_hash("https://example.com/story?utm_campaign=x") == url_hash("https://example.com/story")


def test_normalize_content_collapses_whitespace() -> None:
    assert normalize_content("  Same\n\tarticle   content  ") == "Same article content"


def test_content_hash_is_stable_for_whitespace_variants() -> None:
    assert content_hash("Same\narticle content") == content_hash(" Same article   content ")
