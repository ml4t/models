"""Verify that public documentation identifies the release being published."""

from __future__ import annotations

import json
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

USER_AGENT = "ml4t-models-release-verifier/1.0"
REQUIRED_META = ("ml4t-library", "ml4t-version", "ml4t-commit")


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        name = values.get("name")
        content = values.get("content")
        if name in REQUIRED_META and content is not None:
            self.values[name] = content


def html_identity_failures(html: str, expected: dict[str, str], source: str) -> list[str]:
    """Return missing or mismatched documentation identity fields."""
    parser = _MetadataParser()
    parser.feed(html)
    expected_meta = {
        "ml4t-library": expected["library"],
        "ml4t-version": expected["version"],
        "ml4t-commit": expected["commit"],
    }
    return [
        f"{source}: {name} is {parser.values.get(name)!r}, expected {value!r}"
        for name, value in expected_meta.items()
        if parser.values.get(name) != value
    ]


def verify_site(site: Path, expected: dict[str, str]) -> None:
    """Verify that every rendered page exposes the expected release identity."""
    pages = sorted(
        page for page in site.rglob("*.html") if "overrides" not in page.relative_to(site).parts
    )
    if not pages:
        raise RuntimeError(f"{site}: no rendered HTML pages found")
    failures = [
        failure
        for page in pages
        for failure in html_identity_failures(
            page.read_text(encoding="utf-8"), expected, str(page.relative_to(site))
        )
    ]
    if failures:
        raise RuntimeError("rendered documentation identity did not match:\n" + "\n".join(failures))


def _read_identity(url: str, commit: str, attempt: int) -> object:
    query = urlencode({"commit": commit, "attempt": attempt})
    request = Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _read_page(url: str, commit: str, attempt: int) -> str:
    query = urlencode({"commit": commit, "attempt": attempt})
    request = Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def verify(
    identity_urls: tuple[str, ...],
    expected: dict[str, str],
    *,
    page_urls: tuple[str, ...] = (),
    attempts: int = 24,
    retry_seconds: float = 10,
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be positive")

    observed: list[object] = []
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            observed = [_read_identity(url, expected["commit"], attempt) for url in identity_urls]
            page_failures = [
                failure
                for url in page_urls
                for failure in html_identity_failures(
                    _read_page(url, expected["commit"], attempt), expected, url
                )
            ]
            last_error = None
            if all(value == expected for value in observed) and not page_failures:
                return
        except (OSError, URLError, ValueError) as error:
            last_error = error

        if attempt + 1 < attempts:
            time.sleep(retry_seconds)

    raise RuntimeError(
        f"deployed documentation identity did not match {expected!r}; "
        f"last observed {observed!r}; last error {last_error!r}"
    )


def main() -> None:
    expected = {
        "commit": os.environ["RELEASE_COMMIT"],
        "library": "models",
        "version": os.environ["RELEASE_VERSION"],
    }
    site = os.environ.get("ML4T_DOCS_SITE")
    if site is not None:
        verify_site(Path(site), expected)
        return
    verify(
        (
            "https://www.ml4trading.io/docs/models/release.json",
            f"https://www.ml4trading.io/docs/models/releases/{expected['version']}/release.json",
        ),
        expected,
        page_urls=(
            "https://www.ml4trading.io/docs/models/",
            f"https://www.ml4trading.io/docs/models/releases/{expected['version']}/",
        ),
    )


if __name__ == "__main__":
    main()
