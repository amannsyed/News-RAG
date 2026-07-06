from news_ingest.ml.web_search import DuckDuckGoHTMLParser, WebSearchResult, bing_news_search, build_web_document, result_to_citation


def test_duckduckgo_parser_extracts_results_and_snippets() -> None:
    parser = DuckDuckGoHTMLParser()
    parser.feed('<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">Example &amp; Title</a><a class="result__snippet">Snippet text</a>')

    assert len(parser.results) == 1
    assert parser.results[0].title == "Example & Title"
    assert parser.results[0].url == "https://example.com/article"
    assert parser.results[0].snippet == "Snippet text"


def test_web_result_to_citation_has_web_source_identity() -> None:
    citation = result_to_citation(WebSearchResult(id=12, title="T", url="https://example.com", snippet="S", score=1.0), rank=2)

    assert citation["source_type"] == "web_search"
    assert citation["article_id"] is None
    assert citation["web_search_id"] == 12
    assert citation["stream_ranks"] == {"web_search": 2}


def test_build_web_document_normalizes_title_snippet_and_url() -> None:
    document = build_web_document(WebSearchResult(id=None, title=" T ", url="https://example.com", snippet=" S  text "))

    assert document == "T S text https://example.com"


def test_bing_news_search_parses_rss(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return b"""<rss><channel><item><title>News title</title><link>http://www.bing.com/news/apiclick.aspx?url=https%3A%2F%2Fexample.com%2Fnews</link><description><![CDATA[<b>Snippet</b> text]]></description><pubDate>Sun, 05 Jul 2026 00:00:00 GMT</pubDate></item></channel></rss>"""

    monkeypatch.setattr("news_ingest.ml.web_search.urlopen", lambda request, timeout: Response())

    results = bing_news_search(query="ai", limit=3)

    assert len(results) == 1
    assert results[0].provider == "bing_news"
    assert results[0].title == "News title"
    assert results[0].url == "https://example.com/news"
    assert results[0].snippet == "Snippet text"
