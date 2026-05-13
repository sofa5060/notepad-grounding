from __future__ import annotations

import requests

API_URL = "https://jsonplaceholder.typicode.com/posts"


class ApiError(RuntimeError):
    """Raised when the JSONPlaceholder API call fails."""


def fetch_posts(*, limit: int = 10) -> list[dict]:
    """Fetch the first N posts from JSONPlaceholder.

    Returns a list of dicts with keys: id, title, body.
    """
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ApiError(f"Failed to fetch posts from {API_URL}: {exc}") from exc

    posts = response.json()
    if not isinstance(posts, list):
        raise ApiError(f"Unexpected API response shape: {type(posts).__name__}")

    return posts[:limit]
