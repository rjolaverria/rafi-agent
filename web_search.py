import os

from exa_py import Exa

_client: Exa | None = None


def _get_client() -> Exa:
    global _client
    if _client is None:
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            raise ValueError("EXA_API_KEY environment variable is not set.")
        _client = Exa(api_key=api_key)
    return _client


def search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Search the web using Exa and return results with title, url, and highlights."""
    response = _get_client().search_and_contents(
        query,
        type="auto",
        num_results=num_results,
        highlights={"max_characters": 4000},
    )

    results: list[dict[str, str]] = []
    for r in response.results:
        results.append(
            {
                "title": r.title or "",
                "url": r.url,
                "highlights": "\n".join(r.highlights) if r.highlights else "",
            }
        )
    return results
